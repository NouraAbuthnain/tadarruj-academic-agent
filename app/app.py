"""
app/app.py — Tadarruj Phase 2 Streamlit App

Study Planning Engine with:
- Mode selection (study plan / 3-year roadmap)
- Structured study-plan form with AI-powered plan generation (RAG-enhanced)
- 3-year academic roadmap generator (RAG-enhanced)
- Chat-based recalibration with multi-turn memory
- Error handling with user-friendly Arabic messages
"""

# ── Standard library ──────────────────────────────────────────
import os, sys, datetime, base64
from typing import cast, Literal

# ── Fix import path so `agent` package is found ───────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Third-party ───────────────────────────────────────────────
import streamlit as st

# ── Internal ──────────────────────────────────────────────────
from agent import generate_plan, generate_roadmap, chat_reply
from agent.state import StudyPlanRequest, RoadmapRequest, TadarrujState, ChatMessage

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="تدرج | Tadarruj",
    page_icon="assets/page_icon.svg",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global styles ─────────────────────────────────────────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;700;800&display=swap');

        /* ── Color tokens ── */
        :root {
            --primary:        #1e46c0;   /* main blue */
            --primary-hover:  #1a3aa8;   /* darker blue for hover */
            --primary-light:  #5b8cf8;   /* lighter blue accent */
            --success:        #bbf7d0;   /* green button background */
            --success-hover:  #86efac;   /* green button hover */
            --muted:          #64748b;   /* secondary text */
            --border:         #dbe4f0;   /* card and input borders */
            --shadow:         rgba(30, 70, 192, 0.08); /* subtle card shadow */
            --bubble-bot-bg:  #e2e8f0;   /* bot chat bubble background */
        }

        /* ── Base: apply Cairo font and RTL direction ── */
        html, body, [class*="css"], .stMarkdown, h1, h2, h3, p {
            font-family: 'Cairo', sans-serif !important;
        }
        .stApp {
            direction: rtl;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 10rem !important;
            max-width: 720px;
        }

        /* ── Form labels ── */
        label,
        .stSelectbox label,
        .stNumberInput label,
        .stSlider label,
        .stDateInput label,
        .stCheckbox label {
            font-size: 1rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            display: block !important;
            margin-bottom: 8px !important;
        }

        /* ── Inputs ── */
        input, .stSelectbox div[data-baseweb="select"] {
            font-size: 1rem !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
        }

        /* ── Slider: force LTR so min/max values render correctly ── */
        .stSlider {
            direction: ltr;
        }
        .stSlider label {
            direction: rtl !important;
            text-align: right !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            display: block !important;
            margin-bottom: 8px !important;
        }

        /* ── Prevent horizontal overflow on small screens ── */
        .stTextInput, .stSelectbox, .stNumberInput,
        .stDateInput, .stRadio, .stSlider {
            max-width: 100% !important;
            overflow: hidden;
        }

        /* ── Radio buttons: stack vertically ── */
        .stRadio > div {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            gap: 6px !important;
        }
        .stRadio label {
            font-size: 1rem !important;
            font-weight: 700 !important;
            width: 100% !important;
            display: block !important;
            margin-bottom: 8px !important;
        }

        /* ── Expander used as the form card ── */
        section[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 20px var(--shadow) !important;
            padding: 0.5rem 1rem !important;
            overflow: hidden;
        }
        div[data-testid="stExpander"] summary {
            font-size: 1rem !important;
            font-weight: 700 !important;
            direction: rtl;
        }

        /* ── Section captions ── */
        div[data-testid="stCaption"] {
            font-size: 1rem !important;
            color: var(--muted) !important;
            direction: rtl;
            text-align: right;
        }

        /* ── Primary button (generate plan) ── */
        .stButton button[kind="primary"] {
            background-color: var(--primary) !important;
            border: none !important;
            color: white !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: var(--primary-hover) !important;
        }

        /* ── Secondary button (log hours) ── */
        .stButton button[kind="secondary"] {
            background-color: var(--success) !important;
            border: none !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
            color: black !important;
        }
        .stButton button[kind="secondary"]:hover {
            background-color: var(--success-hover) !important;
            color: inherit !important;
        }
        .stButton button {
            white-space: nowrap !important;
            font-size: clamp(0.75rem, 3vw, 1rem) !important;
        }

        /* ── Generated study plan display ── */
        .plan-box {
            border-right: 4px solid var(--primary);
            border-radius: 12px;
            padding: 1.5rem 1.8rem;
            font-size: 1rem;
            line-height: 2;
            white-space: pre-wrap;
            direction: rtl;
            text-align: right;
        }

        /* ── 3-year academic roadmap display ── */
        .roadmap-box {
            border-right: 4px solid #7c3aed;
            border-radius: 12px;
            padding: 1.5rem 1.8rem;
            font-size: 1rem;
            line-height: 2;
            white-space: pre-wrap;
            direction: rtl;
            text-align: right;
        }

        /* ── User chat bubble (right-aligned, no avatar) ── */
        .bubble-user {
            background: var(--primary);
            color: white;
            border-radius: 16px;
            padding: 1rem;
            max-width: 100%;
            font-size: 1rem;
            line-height: 1.8;
            direction: rtl;
            text-align: right;
            margin: 8px 0;
        }

        /* ── Bot chat row: avatar + bubble side by side ── */
        .chat-bot-row {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            margin: 6px 0;
        }
        .bot-avatar {
            width: 32px; height: 32px;
            border-radius: 50%;
            border: 1px solid var(--border);
            background: var(--bubble-bot-bg);
            display: flex; align-items: center; justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .bubble-bot {
            background: var(--bubble-bot-bg);
            color: black;
            border-radius: 16px;
            padding: 1rem;
            max-width: 100%;
            font-size: 1rem;
            line-height: 1.8;
            direction: rtl;
            text-align: right;
            margin: 8px 0;
        }

        /* ── Progress metrics: center values and labels ── */
        [data-testid="stMetric"] {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }
        [data-testid="stMetricValue"] {
            width: 100% !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricLabel"] {
            width: 100% !important;
            font-size: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ── Logo: load SVG as base64 so it renders inline ─────────────
with open("assets/page_icon.svg", "rb") as f:
    LOGO_SVG = base64.b64encode(f.read()).decode()

st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; direction:rtl; margin-bottom:0.5rem;">
        <img src="data:image/svg+xml;base64,{LOGO_SVG}" width="40">
        <h3 style="margin:0;">طريقك الهادئ نحو النجاح</h3>
    </div>
""", unsafe_allow_html=True)

st.write("حوّل موعد اختبارك إلى خطة يومية واضحة وقابلة للتنفيذ، بخطوات منظمة بعيدًا عن الضغط والفوضى لتصل بثقة إلى هدفك.")

# ── No model loading needed — using OpenRouter API ────────────

# ── Session state: initialize defaults on first run ───────────
for k, v in [
    ("mode", None),
    ("plan", None),
    ("request", None),
    ("roadmap_output", None),
    ("roadmap_request", None),
    ("chat_history", []),
    ("hours_done", 0.0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# MODE SELECTION — shown on first load
# ══════════════════════════════════════════════════════════════
if st.session_state.mode is None:
    st.caption(":compass: ماذا تريد اليوم؟")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 خطة دراسية لاختبار قادم", use_container_width=True, type="primary"):
            st.session_state.mode = "study_plan"
            st.rerun()
    with col2:
        if st.button("🗺️ خارطة طريق أكاديمية (3 سنوات)", use_container_width=True):
            st.session_state.mode = "roadmap"
            st.rerun()
    st.stop()

# ── Back button: reset state and return to mode selection ─────
if st.button("← العودة للرئيسية"):
    for k in ["mode", "plan", "request", "roadmap_output", "roadmap_request", "chat_history", "hours_done"]:
        st.session_state[k] = None if k not in ("chat_history", "hours_done") else ([] if k == "chat_history" else 0.0)
    st.rerun()

# ══════════════════════════════════════════════════════════════
# PATH A — STUDY PLAN
# ══════════════════════════════════════════════════════════════
if st.session_state.mode == "study_plan":

    with st.expander(":memo: معلومات الاختبار", expanded=(st.session_state.plan is None)):

        subject = st.selectbox("اختر المادة أو نوع الاختبار", [
            "قدرات (كمي + لفظي)", "قدرات كمي فقط", "قدرات لفظي فقط",
            "تحصيلي علمي", "تحصيلي أدبي",
            "SAT", "STEP", "IELTS",
            "رياضيات", "فيزياء", "كيمياء", "أحياء", "أخرى",
        ])

        # Allow free-text entry when subject is not in the list
        if subject == "أخرى":
            custom_subject = st.text_input("اكتب اسم المادة أو الاختبار")
            if custom_subject.strip():
                subject = custom_subject.strip()

        # cast() resolves the type ambiguity from st.date_input
        exam_date = cast(
            datetime.date,
            st.date_input(
                "حدد تاريخ الاختبار",
                value=datetime.date.today() + datetime.timedelta(days=45),
                min_value=datetime.date.today() + datetime.timedelta(days=1),
            )
        )
        days_left = (exam_date - datetime.date.today()).days
        st.caption(f":calendar: تبقّى على اختبارك **{days_left}** يومًا")

        content_type = st.radio(
            "كيف تود تحديد حجم المحتوى؟",
            ["حسب عدد الفصول", "حسب إجمالي ساعات المحتوى"],
        )

        chapters, total_hours = None, None
        if content_type == "حسب عدد الفصول":
            chapters = st.number_input("كم عدد الفصول المطلوب مذاكرتها؟", min_value=1, max_value=100, value=20, step=1)
        else:
            total_hours = st.number_input("كم ساعة يحتاج المحتوى كاملًا؟", min_value=1.0, max_value=200.0, value=20.0, step=0.5)

        hours_per_day = st.slider("كم ساعة يمكنك الدراسة يوميًا؟", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
        difficulty    = st.selectbox("ما مستوى صعوبة المحتوى بالنسبة لك؟", ["سهل", "متوسط", "صعب"])
        target_score  = st.number_input("ما درجتك المستهدفة؟", min_value=1, max_value=1600, value=85, step=1)

        took_before   = st.checkbox("سبق لك أداء هذا الاختبار؟")
        current_score = None
        if took_before:
            current_score = st.number_input("ما درجتك السابقة؟", min_value=1, max_value=1600, value=70, step=1)

        # Block generation if too few days remain
        if days_left < 7:
            st.warning(":warning: الوقت المتبقي قصير جدًا، نوصي بحد أدنى 7 أيام لبناء خطة فعّالة.")
            generate_btn = False
        else:
            generate_btn = st.button(":rocket: أنشئ خطتي الدراسية", type="primary", use_container_width=True)

    # ── Generate plan on button click ─────────────────────────
    if generate_btn:
        try:
            req = StudyPlanRequest(
                subject=subject or "",
                days_left=int(days_left),
                chapters=int(chapters) if chapters else None,
                total_hours=float(total_hours) if total_hours else None,
                hours_per_day=float(hours_per_day),
                difficulty=difficulty or "متوسط",
                current_score=int(current_score) if current_score else None,
                target_score=int(target_score),
            )
            with st.spinner(":hourglass: جارٍ إعداد خطتك الدراسية..."):
                plan = generate_plan(req)
            st.session_state.plan         = plan
            st.session_state.request      = req
            st.session_state.chat_history = []
            st.session_state.hours_done   = 0.0
            st.success(":white_check_mark: تم إنشاء خطتك الدراسية بنجاح!")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            st.error(f":x: {str(e)}")

    # ── Plan output: shown after plan is generated ────────────
    if st.session_state.plan:
        req = st.session_state.request
        st.divider()

        # ── Compute progress values ───────────────────────────
        total_hours_planned = req.hours_per_day * req.days_left
        hours_remaining     = max(0.0, total_hours_planned - st.session_state.hours_done)
        pct = (st.session_state.hours_done / total_hours_planned * 100) if total_hours_planned > 0 else 0

        # ── Progress metrics ──────────────────────────────────
        st.caption(":bar_chart: ملخص التقدّم")
        s1, s2, s3 = st.columns(3)
        for col, val, label in [
            (s1, f"{total_hours_planned:.0f}", "إجمالي ساعات الخطة"),
            (s2, f"{st.session_state.hours_done:.1f}", "الساعات المنجزة"),
            (s3, f"{hours_remaining:.1f}", "الساعات المتبقية"),
        ]:
            with col:
                st.metric(label=label, value=val)

        # ── Progress bar with percentage label ───────────────
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; margin-bottom:4px;'>"
            f"<span style='color:var(--muted); font-size:0.85rem;'>نسبة الإنجاز</span>"
            f"<span style='color:var(--primary); font-weight:700;'>{pct:.0f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(pct / 100, 1.0))

        # ── Log study hours ───────────────────────────────────
        st.divider()
        logged = st.number_input(
            "كم ساعة درست اليوم؟",
            min_value=0.0, max_value=24.0, step=0.5, value=0.0, key="log_input",
        )
        if st.button(":white_check_mark: إضافة الساعات", use_container_width=True):
            st.session_state.hours_done += logged
            st.rerun()

        # ── Today's focus: extract day-1 lines from the plan ─
        st.divider()
        st.caption(":dart: تركيز اليوم")
        focus_lines = [
            line.strip() for line in st.session_state.plan.split("\n")
            if any(k in line for k in ["يوم 1", "اليوم الأول", "المرحلة 1", "التعلم"])
        ]
        if focus_lines:
            for fl in focus_lines[:3]:
                st.info(fl)
        else:
            st.info(f"ابدأ بـ {req.hours_per_day} ساعة دراسة اليوم وفق خطتك أدناه.")

        # ── Full generated plan ───────────────────────────────
        st.divider()
        st.caption(":open_book: خطتك الدراسية")
        st.markdown(f'<div class="plan-box">{st.session_state.plan}</div>', unsafe_allow_html=True)

        # ── Recommendation based on completion percentage ─────
        st.divider()
        st.caption(":bulb: التوصية الحالية")
        if pct < 10:
            rec = "ابدأ بالمرحلة الأولى وركّز على بناء عادة يومية ثابتة."
        elif pct < 50:
            rec = "تقدّمك جيد. استمر على نفس الوتيرة وحافظ على انتظامك اليومي."
        elif pct < 80:
            rec = "أنت في مرحلة متقدمة. ابدأ بزيادة التطبيق وحل الأسئلة."
        else:
            rec = "اقترب موعد الاختبار. ركّز على المحاكاة والمراجعة الخفيفة والراحة الكافية."
        st.info(rec)

        # ══════════════════════════════════════════════════════
        # CHAT — recalibration and Q&A with multi-turn memory
        # ══════════════════════════════════════════════════════
        st.divider()
        st.caption(":thought_balloon: اسأل أو عدّل خطتك")
        st.write("يمكنك طلب تعديل الخطة إذا فاتتك أيام، أو تغيّرت ظروفك، أو أردت إضافة مادة جديدة.")

        # ── Render chat history ───────────────────────────────
        for msg in st.session_state.chat_history:
            if msg.role == "user":
                st.markdown(f'<div class="bubble-user">{msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bot-row">
                    <div class="bot-avatar">🤖</div>
                    <div class="bubble-bot">{msg.content}</div>
                </div>""", unsafe_allow_html=True)

        # ── Handle new user message ───────────────────────────
        user_chat = st.chat_input("اكتب سؤالك أو التعديل المطلوب")
        if user_chat:
            new_history = st.session_state.chat_history + [ChatMessage(role="user", content=user_chat)]
            agent_state = TadarrujState(
                request=st.session_state.request,
                plan=st.session_state.plan,
                chat_history=new_history,
            )
            with st.spinner(":hourglass: جارٍ تحديث خطتك..."):
                updated_state = chat_reply(agent_state)
            st.session_state.chat_history = updated_state.chat_history
            # Update plan in session if agent returned a revised version
            if updated_state.plan != st.session_state.plan:
                st.session_state.plan = updated_state.plan
            st.rerun()


# ══════════════════════════════════════════════════════════════
# PATH B — 3-YEAR ACADEMIC ROADMAP
# ══════════════════════════════════════════════════════════════
elif st.session_state.mode == "roadmap":

    with st.expander(":compass: معلوماتك الأكاديمية", expanded=(st.session_state.roadmap_output is None)):

        # Grade selector
        current_grade = st.selectbox(
            "في أي صف أنت الآن؟",
            ["10", "11", "12"],
            format_func=lambda x: f"الصف {x} (ثانوي {'أول' if x=='10' else 'ثاني' if x=='11' else 'ثالث'})",
        )

        major_interest = st.selectbox("ما المجال الذي تطمح إليه؟", [
            "طب وعلوم صحية", "هندسة وعلوم حاسب", "ذكاء اصطناعي وأمن سيبراني",
            "إدارة أعمال ومحاسبة", "حقوق وعلوم إنسانية", "تربية وتعليم", "أخرى",
        ])

        # Allow free-text entry for major
        if major_interest == "أخرى":
            custom_major = st.text_input("اكتب المجال الذي تطمح إليه")
            if custom_major.strip():
                major_interest = custom_major.strip()

        target_university = st.text_input(
            "جامعتك المستهدفة (اختياري)",
            placeholder="مثال: جامعة الملك سعود",
        )

        roadmap_btn = st.button(":map: أنشئ خارطة طريقي الأكاديمية", type="primary", use_container_width=True)

    # ── Generate roadmap on button click ──────────────────────
    if roadmap_btn:
        try:
            req = RoadmapRequest(
                grade=current_grade,                                               # ← fixed: was current_grade=
                major_interest=major_interest or "",
                target_university=target_university.strip() if target_university.strip() else None,
            )
            with st.spinner(":hourglass: جارٍ إنشاء خارطة طريقك الأكاديمية..."):
                roadmap = generate_roadmap(req)
            st.session_state.roadmap_output  = roadmap
            st.session_state.roadmap_request = req
            st.session_state.chat_history    = []
            st.success(":white_check_mark: تم إنشاء خارطة طريقك الأكاديمية بنجاح!")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            st.error(f":x: {str(e)}")

    # ── Roadmap output ────────────────────────────────────────
    if st.session_state.roadmap_output:
        st.divider()
        st.caption(":map: خارطة طريقك الأكاديمية")
        st.markdown(f'<div class="roadmap-box">{st.session_state.roadmap_output}</div>', unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # CHAT — Q&A about the roadmap with multi-turn memory
        # ══════════════════════════════════════════════════════
        st.divider()
        st.caption(":thought_balloon: اسأل عن خارطة طريقك")
        st.write("يمكنك سؤال تدرّج عن أي جزء من خطتك الأكاديمية.")

        # ── Render chat history ───────────────────────────────
        for msg in st.session_state.chat_history:
            if msg.role == "user":
                st.markdown(f'<div class="bubble-user">{msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-bot-row">
                    <div class="bot-avatar">🤖</div>
                    <div class="bubble-bot">{msg.content}</div>
                </div>""", unsafe_allow_html=True)

        # ── Handle new user message ───────────────────────────
        user_chat = st.chat_input("اكتب سؤالك عن خارطة طريقك")
        if user_chat:
            new_history = st.session_state.chat_history + [ChatMessage(role="user", content=user_chat)]
            agent_state = TadarrujState(
                roadmap=st.session_state.roadmap_output,
                chat_history=new_history,
            )
            with st.spinner(":hourglass: جارٍ البحث عن إجابتك..."):
                updated_state = chat_reply(agent_state)
            st.session_state.chat_history = updated_state.chat_history
            st.rerun()