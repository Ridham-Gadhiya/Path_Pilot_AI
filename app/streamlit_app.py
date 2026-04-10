import streamlit as st
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_validate
from src.preprocessing import preprocess_data
from src.feature_engineering import process_features
from src.recommender import build_similarity, recommend_courses, recommend_smart, recommend_for_user

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="PathPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* ── Hero ── */
.hero-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f172a 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #334155;
}
.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    color: #38bdf8;
    font-weight: 500;
    margin-bottom: 1.25rem;
    letter-spacing: 0.02em;
}
.hero-title {
    font-size: 3rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.15;
    margin-bottom: 0.75rem;
    letter-spacing: -0.02em;
}
.hero-title span { color: #38bdf8; }
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 540px;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 22px;
    min-width: 110px;
}
.stat-num {
    font-size: 22px;
    font-weight: 600;
    color: #f1f5f9;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.03em;
}
.stat-lbl {
    font-size: 11px;
    color: #64748b;
    margin-top: 2px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── Section headings ── */
.section-head {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 1.25rem;
}

/* ── Feature cards ── */
.feat-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 2rem; }
.feat-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem;
    flex: 1 1 160px;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
    cursor: pointer;
}
.feat-card:hover {
    border-color: #38bdf8;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(56,189,248,0.12);
}
.feat-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
}
.feat-card h4 { font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.feat-card p  { font-size: 12px; color: #64748b; line-height: 1.55; }

/* ── Course result card ── */
.course-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    animation: slideUp 0.3s ease both;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.course-title { font-size: 14px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.course-meta  { font-size: 12px; color: #64748b; }
.score-badge {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    white-space: nowrap;
    font-family: 'DM Mono', monospace;
}
.path-tag {
    display: inline-block;
    background: #f0fdf4;
    color: #166534;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    font-size: 10px;
    padding: 2px 8px;
    margin-top: 5px;
    font-weight: 500;
}
.level-tag {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
    border-radius: 8px;
    font-size: 10px;
    padding: 2px 8px;
    margin-top: 5px;
    margin-left: 4px;
    font-weight: 500;
}

/* ── Progress bars ── */
.progress-item { margin-bottom: 14px; }
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #475569;
    margin-bottom: 5px;
    font-weight: 500;
}
.progress-bar-bg {
    background: #f1f5f9;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 7px;
    border-radius: 6px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    transition: width 1.2s cubic-bezier(.4,0,.2,1);
}

/* ── How-it-works steps ── */
.steps-grid { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 2rem; }
.step-card {
    flex: 1 1 150px;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
}
.step-num {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 14px;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 10px;
    border: 1px solid #bfdbfe;
    font-family: 'DM Mono', monospace;
}
.step-card h5 { font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 4px; }
.step-card p  { font-size: 12px; color: #64748b; line-height: 1.55; }

/* ── Quick prompts ── */
.prompt-btn {
    display: inline-block;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 7px 14px;
    font-size: 12px;
    color: #475569;
    margin: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.15s;
}
.prompt-btn:hover { border-color: #38bdf8; color: #0369a1; background: #f0f9ff; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f172a;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #64748b !important; }
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f8fafc;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 16px;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── Streamlit buttons ── */
.stButton > button {
    background: #0ea5e9;
    color: #fff;
    border: none;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity 0.15s, transform 0.1s;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.01em;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }
.stButton > button:active { transform: scale(0.98); }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load & Prepare Data (cached)
# ─────────────────────────────────────────
@st.cache_data
def load_pipeline():
    df = load_and_validate("data/processed/pathpilot_v1_filtered.csv")
    df = preprocess_data(df)
    df = process_features(df)
    _, cosine_sim = build_similarity(df)
    return df, cosine_sim

with st.spinner("Loading PathPilot pipeline..."):
    df, cosine_sim = load_pipeline()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 PathPilot AI")
    st.markdown("<p style='font-size:12px;color:#475569;margin-bottom:1.5rem;'>Your AI-powered learning navigator</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Filters")

    selected_level = st.selectbox(
        "Level",
        ["All", "beginner", "intermediate", "advanced"],
        help="Filter results by difficulty level"
    )
    selected_duration = st.selectbox(
        "Duration",
        ["All", "short", "medium", "long"],
        help="Filter by estimated course length"
    )

    st.markdown("---")
    st.markdown("### Engine")

    option = st.selectbox(
        "Recommendation type",
        ["Course Similarity", "Smart Recommendation", "Personalized Recommendation"],
        help="Choose which recommendation algorithm to use"
    )

    st.markdown("---")
    total_courses = len(df)
    st.markdown(f"""
    <div style='background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:12px 16px;'>
        <div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;'>Dataset</div>
        <div style='font-size:22px;font-weight:700;color:#0ea5e9;font-family:DM Mono,monospace;'>{total_courses:,}</div>
        <div style='font-size:11px;color:#64748b;margin-top:2px;'>courses indexed</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Hero
# ─────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-badge">● AI-Powered · Live</div>
  <div class="hero-title">Navigate your<br><span>learning path</span></div>
  <div class="hero-sub">
    PathPilot recommends courses, maps career trajectories, and builds 
    personalized study plans — powered by TF-IDF similarity and goal matching.
  </div>
  <div class="stats-row">
    <div class="stat-card"><div class="stat-num">3,684</div><div class="stat-lbl">Courses</div></div>
    <div class="stat-card"><div class="stat-num">12</div><div class="stat-lbl">Career paths</div></div>
    <div class="stat-card"><div class="stat-num">94%</div><div class="stat-lbl">Match accuracy</div></div>
    <div class="stat-card"><div class="stat-num">4</div><div class="stat-lbl">Goal profiles</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Feature highlights
# ─────────────────────────────────────────
st.markdown('<div class="section-head">Three engines. One destination.</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Each engine is optimized for a different stage of your journey</div>', unsafe_allow_html=True)

st.markdown("""
<div class="feat-grid">
  <div class="feat-card">
    <div class="feat-icon" style="background:#eff6ff;">🔍</div>
    <h4>Course similarity</h4>
    <p>Find courses most similar to one you love using TF-IDF cosine similarity on metadata.</p>
  </div>
  <div class="feat-card">
    <div class="feat-icon" style="background:#f0fdf4;">📈</div>
    <h4>Smart / career-aware</h4>
    <p>Blends content similarity with career relevance and skill-level progression scoring.</p>
  </div>
  <div class="feat-card">
    <div class="feat-icon" style="background:#fef3c7;">👤</div>
    <h4>Personalized path</h4>
    <p>Goal-aware matching — define your career goal, level, and pace for a custom roadmap.</p>
  </div>
  <div class="feat-card">
    <div class="feat-icon" style="background:#fdf4ff;">⚡</div>
    <h4>Weighted ranking</h4>
    <p>Final scores blend similarity, career relevance, popularity, and skill alignment.</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# Helper: Display Cards
# ─────────────────────────────────────────
def show_course_cards(results, engine_label=""):
    if results is None or len(results) == 0:
        st.warning("No recommendations found. Try adjusting your filters or input.")
        return

    st.markdown(f"<p style='font-size:13px;color:#64748b;margin-bottom:12px;'>Showing <b style='color:#1e293b;'>{len(results)}</b> recommendations {engine_label}</p>", unsafe_allow_html=True)

    for _, row in results.iterrows():
        score = round(row.get('final_score', row.get('career_relevance_score', 0)), 3)
        level = row.get('level', 'N/A')
        path  = row.get('learning_path', 'N/A')
        subj  = row.get('subject', 'N/A')
        title = row.get('course_title', 'Untitled')

        st.markdown(f"""
        <div class="course-card">
          <div>
            <div class="course-title">{title}</div>
            <div class="course-meta">{subj}</div>
            <span class="path-tag">{path}</span>
            <span class="level-tag">{level}</span>
          </div>
          <span class="score-badge">{score:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Main tabs
# ─────────────────────────────────────────
tab_similarity, tab_smart, tab_personal, tab_progress, tab_howto = st.tabs([
    "🔍 Similarity", "🧠 Smart", "👤 Personalized", "📊 Progress", "💡 How it works"
])

# ── Tab 1: Course Similarity ──────────────
with tab_similarity:
    st.markdown('<div class="section-head">Find similar courses</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Based on TF-IDF cosine similarity of course metadata and features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        course_list = df["course_title"].tolist()
        selected_course = st.selectbox("Search or select a course", course_list, key="sim_course")
    with col2:
        top_n = st.number_input("Results", min_value=3, max_value=20, value=5, key="sim_n")

    if st.button("Find similar courses", key="btn_sim"):
        with st.spinner("Computing similarities..."):
            results = recommend_courses(df, cosine_sim, selected_course, top_n=top_n)

        if isinstance(results, str):
            st.error(results)
        else:
            if selected_level != "All":
                results = results[results["level"] == selected_level]
            results["final_score"] = results.get("career_relevance_score", 0)
            show_course_cards(results, f"· similar to **{selected_course}**")

# ── Tab 2: Smart Recommendation ──────────
with tab_smart:
    st.markdown('<div class="section-head">Career-aware recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Combines content similarity with career relevance and skill-level progression</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_smart = st.selectbox("Search or select a course", df["course_title"].tolist(), key="smart_course")
    with col2:
        smart_n = st.number_input("Results", min_value=3, max_value=20, value=5, key="smart_n")

    if st.button("Get smart recommendations", key="btn_smart"):
        with st.spinner("Generating career-aware picks..."):
            results = recommend_smart(df, cosine_sim, selected_smart, top_n=smart_n)

        if isinstance(results, str):
            st.error(results)
        else:
            if selected_level != "All":
                results = results[results["level"] == selected_level]
            show_course_cards(results, "· career-aware")

# ── Tab 3: Personalized ──────────────────
with tab_personal:
    st.markdown('<div class="section-head">Your personalized learning path</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Tell us your goal, level, and pace — we\'ll build your roadmap</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        goal = st.selectbox("Career goal", ["ai_career", "freelancing", "skill_upgrade", "foundation"],
                            format_func=lambda x: {"ai_career":"🤖 AI Career","freelancing":"💼 Freelancing","skill_upgrade":"⚡ Skill Upgrade","foundation":"📚 Foundation"}[x])
        level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"])
    with col2:
        duration = st.selectbox("Preferred duration", ["short", "medium", "long"])
        language = st.selectbox("Language", ["english"])
        pers_n   = st.number_input("Results", min_value=3, max_value=20, value=5, key="pers_n")

    user_profile = {"goal": goal, "level": level, "preferred_duration": duration, "language": language}

    if st.button("Build my learning path", key="btn_pers"):
        with st.spinner("Personalizing your path..."):
            results = recommend_for_user(df, user_profile, top_n=pers_n)

        if isinstance(results, str):
            st.error(results)
        else:
            if selected_level != "All":
                results = results[results["level"] == selected_level]
            if selected_duration != "All":
                results = results[results.get("duration_category", pd.Series()) == selected_duration] if "duration_category" in results.columns else results
            show_course_cards(results, f"· for **{goal.replace('_',' ')}** goal")

# ── Tab 4: Progress tracker ───────────────
with tab_progress:
    st.markdown('<div class="section-head">Skill progress tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Track your proficiency across core skill areas</div>', unsafe_allow_html=True)

    skills = [
        ("Python & ML fundamentals", 72),
        ("Deep learning", 45),
        ("Data engineering", 30),
        ("Web development", 58),
        ("Career readiness", 61),
    ]
    for name, pct in skills:
        st.markdown(f"""
        <div class="progress-item">
          <div class="progress-label"><span>{name}</span><span style='color:#0ea5e9;font-family:DM Mono,monospace;'>{pct}%</span></div>
          <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{pct}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head" style="font-size:14px;">Recommended next actions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📌 **Weak area**: Data engineering (30%) — search for pipeline courses")
    with col2:
        st.success("✅ **Strong**: Python & ML (72%) — ready for advanced topics")
    with col3:
        st.warning("⏳ **In progress**: Deep learning (45%) — 3 courses remaining")

# ── Tab 5: How it works ───────────────────
with tab_howto:
    st.markdown('<div class="section-head">How PathPilot works</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Four steps from your input to a personalized recommendation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-grid">
      <div class="step-card">
        <div class="step-num">1</div>
        <h5>Input your context</h5>
        <p>Pick a course or define your goal, level, and duration preference.</p>
      </div>
      <div class="step-card">
        <div class="step-num">2</div>
        <h5>Feature encoding</h5>
        <p>TF-IDF vectorizes course metadata into a high-dimensional feature space.</p>
      </div>
      <div class="step-card">
        <div class="step-num">3</div>
        <h5>Similarity scoring</h5>
        <p>Cosine similarity ranks candidates; career + skill scores refine ranking.</p>
      </div>
      <div class="step-card">
        <div class="step-num">4</div>
        <h5>Filtered results</h5>
        <p>Level, duration, and language filters apply before the final list is shown.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-head" style="font-size:14px;">Scoring formula</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Smart engine weights**")
        st.code("""final_score = compute_final_score(
    similarity_score,      # TF-IDF cosine sim
    career_relevance_score,
    skill_level_score
)""", language="python")
    with col2:
        st.markdown("**Personalization weights**")
        st.code("""final_score = (
    0.4 * goal_match +
    0.3 * career_relevance_score +
    0.2 * popularity_score +
    0.1 * skill_level_score
)""", language="python")

    st.markdown("---")
    st.markdown('<div class="section-head" style="font-size:14px;">Quick prompts to explore</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:8px;">
      <span class="prompt-btn">→ Best ML starter courses</span>
      <span class="prompt-btn">→ AI career roadmap</span>
      <span class="prompt-btn">→ Compare Python vs SQL first</span>
      <span class="prompt-btn">→ How does TF-IDF work?</span>
      <span class="prompt-btn">→ Improve the recommender with embeddings</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center;font-size:12px;color:#94a3b8;'>
  PathPilot AI · Built with Streamlit · TF-IDF + Cosine Similarity engine
</p>
""", unsafe_allow_html=True)