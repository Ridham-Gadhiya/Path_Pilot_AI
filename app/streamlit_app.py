import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_validate
from src.preprocessing import preprocess_data
from src.feature_engineering import process_features
from src.recommender import build_similarity, recommend_courses, recommend_smart, recommend_for_user

# -------------------------------
# 🔹 Load & Prepare Data (cached)
# -------------------------------
@st.cache_data
def load_pipeline():
    df = load_and_validate("data/processed/pathpilot_v1_filtered.csv")
    df = preprocess_data(df)
    df = process_features(df)
    _, cosine_sim = build_similarity(df)
    return df, cosine_sim

df, cosine_sim = load_pipeline()

# -------------------------------
# 🔹 UI Header
# -------------------------------
st.markdown("""
# PathPilot AI  
### Your AI-Powered Study & Career Assistant
""")

st.divider()

# -------------------------------
# 🔹 Sidebar Filters
# -------------------------------
st.sidebar.header("⚙️ Filters")

selected_level = st.sidebar.selectbox(
    "Filter by Level",
    ["All", "beginner", "intermediate", "advanced"]
)

selected_duration = st.sidebar.selectbox(
    "Filter by Duration",
    ["All", "short", "medium", "long"]
)

# -------------------------------
# 🔹 Recommendation Type
# -------------------------------
option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "Course Similarity",
        "Smart Recommendation",
        "Personalized Recommendation"
    ]
)

# -------------------------------
# 🔹 Helper: Display Cards
# -------------------------------
def show_course_cards(results):
    if results is None or len(results) == 0:
        st.warning("No recommendations found. Try different input.")
        return

    st.write(f"🎯 Showing {len(results)} recommendations")

    for _, row in results.iterrows():
        st.markdown(f"""
        ### 🎓 {row['course_title']}
        📘 **Subject:** {row['subject']}  
        📊 **Level:** {row['level']}  
        🧠 **Path:** {row['learning_path']}  
        ⭐ **Score:** {round(row.get('final_score', 0), 3)}
        ---
        """)

# -------------------------------
# 🔍 1. Course Similarity
# -------------------------------
if option == "Course Similarity":
    st.header("🔍 Find Similar Courses")

    course_list = df["course_title"].tolist()

    selected_course = st.selectbox(
        "Search or Select Course",
        course_list
    )

    if st.button("Recommend"):
        with st.spinner("Finding similar courses..."):
            results = recommend_courses(df, cosine_sim, selected_course)

            if isinstance(results, str):
                st.error(results)
            else:
                # Apply filters
                if selected_level != "All":
                    results = results[results["level"] == selected_level]

                show_course_cards(results)


# -------------------------------
# 🧠 2. Smart Recommendation
# -------------------------------
elif option == "Smart Recommendation":
    st.header("🧠 Career-Aware Recommendations")

    course_list = df["course_title"].tolist()

    selected_course = st.selectbox(
        "Search or Select Course",
        course_list
    )

    if st.button("Get Smart Recommendations"):
        with st.spinner("Generating smart recommendations..."):
            results = recommend_smart(df, cosine_sim, selected_course)

            if isinstance(results, str):
                st.error(results)
            else:
                # Apply filters
                if selected_level != "All":
                    results = results[results["level"] == selected_level]

                show_course_cards(results)


# -------------------------------
# 👤 3. Personalized Recommendation
# -------------------------------
elif option == "Personalized Recommendation":
    st.header("👤 Personalized Learning Path")

    goal = st.selectbox(
        "Your Goal",
        ["ai_career", "freelancing", "skill_upgrade", "foundation"]
    )

    level = st.selectbox(
        "Your Level",
        ["beginner", "intermediate", "advanced"]
    )

    duration = st.selectbox(
        "Preferred Duration",
        ["short", "medium", "long"]
    )

    language = st.selectbox(
        "Preferred Language",
        ["english"]
    )

    user_profile = {
        "goal": goal,
        "level": level,
        "preferred_duration": duration,
        "language": language
    }

    if st.button("Get Personalized Recommendations"):
        with st.spinner("Personalizing your learning path..."):
            results = recommend_for_user(df, user_profile)

            if isinstance(results, str):
                st.error(results)
            else:
                # Apply filters
                if selected_level != "All":
                    results = results[results["level"] == selected_level]

                if selected_duration != "All":
                    results = results[results["duration_category"] == selected_duration]

                show_course_cards(results)