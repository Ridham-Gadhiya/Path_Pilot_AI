import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib


# -----------------------------------
# 🔹 1. Build TF-IDF + Similarity
# -----------------------------------
def build_similarity(df: pd.DataFrame):
    """
    Build TF-IDF matrix and cosine similarity matrix
    """

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return tfidf_matrix, cosine_sim


# -----------------------------------
# 🔹 2. Smart Title Matching
# -----------------------------------
def find_best_match(user_input: str, df: pd.DataFrame):
    """
    Find best matching course index using:
    - exact match
    - partial match
    - fuzzy match
    """

    user_input = user_input.lower().strip()

    # Exact match
    exact = df[df["title_clean"] == user_input]
    if not exact.empty:
        idx = exact.index[0]
        return idx, df.loc[idx, "course_title"]

    # Partial match
    partial = df[df["title_clean"].str.contains(user_input, na=False)]
    if not partial.empty:
        idx = partial.index[0]
        return idx, df.loc[idx, "course_title"]

    # Fuzzy match
    possible_titles = df["title_clean"].tolist()
    close = difflib.get_close_matches(user_input, possible_titles, n=1, cutoff=0.4)

    if close:
        matched_title = close[0]
        idx = df[df["title_clean"] == matched_title].index[0]
        return idx, df.loc[idx, "course_title"]

    return None, None


# -----------------------------------
# 🔹 3. Basic Recommendation
# -----------------------------------
def recommend_courses(df, cosine_sim, course_title, top_n=5):
    idx, matched_title = find_best_match(course_title, df)

    if idx is None:
        return "❌ Course not found!"

    print(f"🔍 Showing results for: {matched_title}")

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    course_indices = [i[0] for i in sim_scores]

    return df[[
        "course_title",
        "subject",
        "level",
        "learning_path",
        "career_relevance_score"
    ]].iloc[course_indices]


# -----------------------------------
# 🔹 4. Smart Recommendation (Career-Aware)
# -----------------------------------
def recommend_smart(df, cosine_sim, course_title, top_n=5):
    idx, matched_title = find_best_match(course_title, df)

    if idx is None:
        return "❌ Course not found!"

    print(f"🔍 Showing results for: {matched_title}")

    input_level = df.loc[idx, "level_num"]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:20]

    results = []

    for i, sim_score in sim_scores:
        row = df.iloc[i]

        if input_level <= row["level_num"] <= input_level + 1:
            final_score = (
                0.5 * sim_score +
                0.3 * row["career_relevance_score"] +
                0.2 * row["skill_level_score"]
            )

            results.append({
                "course_title": row["course_title"],
                "subject": row["subject"],
                "level": row["level"],
                "learning_path": row["learning_path"],
                "similarity_score": sim_score,
                "final_score": final_score
            })

    result_df = pd.DataFrame(results)

    return result_df.sort_values(by="final_score", ascending=False).head(top_n)


# -----------------------------------
# 🔹 5. Personalization Engine
# -----------------------------------
def recommend_for_user(df, user_profile, top_n=5):
    filtered_df = df.copy()

    level_map = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3
    }

    user_level = level_map[user_profile["level"].lower()]

    # Filter by level
    filtered_df = filtered_df[
        (filtered_df["level_num"] >= user_level) &
        (filtered_df["level_num"] <= user_level + 1)
    ]

    # Filter by duration
    filtered_df = filtered_df[
        filtered_df["duration_category"] == user_profile["preferred_duration"]
    ]

    # Language filter
    if "language" in user_profile:
        filtered_df = filtered_df[
            filtered_df["language"] == user_profile["language"].lower()
        ]

    # Goal match
    def goal_match(tags):
        return 1 if user_profile["goal"] in tags else 0

    filtered_df["goal_match"] = filtered_df["goal_alignment_tags"].apply(goal_match)

    # Final score
    filtered_df["final_score"] = (
        0.4 * filtered_df["goal_match"] +
        0.3 * filtered_df["career_relevance_score"] +
        0.2 * filtered_df["popularity_score"] +
        0.1 * filtered_df["skill_level_score"]
    )

    return filtered_df.sort_values(by="final_score", ascending=False)[[
        "course_title",
        "subject",
        "level",
        "learning_path",
        "final_score"
    ]].head(top_n)  