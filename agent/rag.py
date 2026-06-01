"""
agent/rag.py
Contextual Compression RAG for Tadarruj.

Loads knowledge files from data/knowledge/, embeds them using
sentence-transformers, stores in ChromaDB, and retrieves the most
relevant context for a given query.
"""

import os
import glob
from pathlib import Path
from langchain_core.tools import tool

# ── Lazy imports (heavy libraries loaded only when needed) ────
_chroma_client = None
_collection    = None
_embedder      = None

KNOWLEDGE_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")
COLLECTION_NAME = "tadarruj_knowledge"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"  # supports Arabic + English
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 100   # overlap between chunks


def _get_embedder():
    """Load sentence-transformers model (cached)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_collection():
    """Initialize ChromaDB and return the collection (cached)."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    import chromadb
    _chroma_client = chromadb.Client()  # in-memory, no disk required
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Pass the collection directly to avoid recursive call
    if _collection.count() == 0:
        _index_knowledge(_collection)

    return _collection


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks on paragraph boundaries first.
    Falls back to sentence boundaries if a paragraph is too long.
    This avoids cutting sentences in the middle.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                lines = [l.strip() for l in para.split("\n") if l.strip()]
                sub = ""
                for line in lines:
                    if len(sub) + len(line) + 1 <= chunk_size:
                        sub = (sub + "\n" + line).strip()
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = line
                current = sub if sub else ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 50]


def _index_knowledge(collection) -> None:
    """Read all .txt files from data/knowledge/ and index them in ChromaDB.

    Args:
        collection: Already-initialized ChromaDB collection (avoids recursive call).
    """
    embedder  = _get_embedder()
    txt_files = glob.glob(os.path.join(KNOWLEDGE_DIR, "*.txt"))

    if not txt_files:
        print(f"[RAG] No knowledge files found in {KNOWLEDGE_DIR}")
        return

    all_chunks = []
    all_ids    = []
    all_metas  = []

    for filepath in txt_files:
        filename = Path(filepath).stem
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_{i}")
            all_metas.append({"source": filename, "chunk_index": i})

    if not all_chunks:
        return

    # Embed in batches of 64
    batch_size     = 64
    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch      = all_chunks[i:i + batch_size]
        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )
    print(f"[RAG] Indexed {len(all_chunks)} chunks from {len(txt_files)} files.")


def _compress_context(query: str, chunks: list[str], max_length: int = 2000) -> str:
    """
    Contextual compression: trim each chunk to its most relevant sentences.
    Uses substring matching which works well with Arabic text.
    """
    compressed  = []
    query_terms = [w for w in query.split() if len(w) > 3]

    for chunk in chunks:
        sentences = []
        for s in chunk.replace("،", "،\n").replace(".", ".\n").split("\n"):
            s = s.strip()
            if len(s) > 25:
                sentences.append(s)

        scored = []
        for sentence in sentences:
            score = sum(1 for term in query_terms if term in sentence)
            scored.append((score, sentence))

        scored.sort(reverse=True)
        top = [s for _, s in scored[:4]] if scored[0][0] > 0 else [s for _, s in scored[:2]]
        compressed.extend(top)

    result = "\n".join(compressed)
    return result[:max_length] if len(result) > max_length else result


def search(query: str, n_results: int = 4, compress: bool = True) -> str:
    """
    Main RAG search function.

    Args:
        query:     The student's question or plan request.
        n_results: Number of chunks to retrieve before compression.
        compress:  Whether to apply contextual compression.

    Returns:
        A string with the most relevant knowledge context.
    """
    try:
        collection = _get_collection()
        embedder   = _get_embedder()

        if collection.count() == 0:
            return ""

        query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = results["documents"][0] if results["documents"] else []

        if not chunks:
            return ""

        if compress:
            return _compress_context(query, chunks)
        else:
            return "\n\n".join(chunks)

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return ""


def get_exam_context(subject: str, days_left: int) -> str:
    """Convenience: build a RAG query from exam details and return relevant study tips."""
    query = f"كيف أذاكر {subject} في {days_left} يوماً؟ استراتيجيات الدراسة والتحضير"
    return search(query)


def get_roadmap_context(grade: str, major_interest: str) -> str:
    """Convenience: get context for the 3-year roadmap feature."""
    query = f"خارطة الطريق الأكاديمية للطالب في الصف {grade} المهتم بـ {major_interest} ومتطلبات القبول الجامعي"
    return search(query)


# ── LangGraph tool wrappers ───────────────────────────────────

@tool
def study_context_tool(query: str) -> str:
    """Retrieve relevant study tips and exam strategies from the knowledge base."""
    return search(query)


@tool
def roadmap_context_tool(query: str) -> str:
    """Retrieve academic roadmap guidance and university admission info."""
    return search(query)