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
# 🔹 UI Layout
# -------------------------------
st.title("🚀 PathPilot AI")
st.subheader("Your AI-Powered Study & Career Assistant")

option = st.sidebar.selectbox(
    "Choose Recommendation Type",
    [
        "Course Similarity",
        "Smart Recommendation",
        "Personalized Recommendation"
    ]
)

# -------------------------------
# 🔍 1. Course Similarity
# -------------------------------
if option == "Course Similarity":
    st.header("🔍 Find Similar Courses")

    course_input = st.text_input("Enter Course Name")

    if st.button("Recommend"):
        results = recommend_courses(df, cosine_sim, course_input)

        if isinstance(results, str):
            st.error(results)
        else:
            st.dataframe(results)


# -------------------------------
# 🧠 2. Smart Recommendation
# -------------------------------
elif option == "Smart Recommendation":
    st.header("🧠 Career-Aware Recommendations")

    course_input = st.text_input("Enter Course Name")

    if st.button("Get Smart Recommendations"):
        results = recommend_smart(df, cosine_sim, course_input)

        if isinstance(results, str):
            st.error(results)
        else:
            st.dataframe(results)


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
        results = recommend_for_user(df, user_profile)

        if isinstance(results, str):
            st.error(results)
        else:
            st.dataframe(results)