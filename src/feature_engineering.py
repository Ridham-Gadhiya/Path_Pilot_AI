import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add intelligent features to dataset.

    Includes:
    - career_relevance_score
    - duration_category
    - level_num
    - goal_alignment_tags

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame
    """

    df = df.copy()

    # -------------------------------
    # 🔹 1. Normalize numerical features
    # -------------------------------
    scaler = MinMaxScaler()

    numeric_cols = [
        "num_subscribers",
        "num_reviews",
        "rating",
        "popularity_score"
    ]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # -------------------------------
    # 🔹 2. Career Relevance Score
    # -------------------------------
    df["career_relevance_score"] = (
        0.35 * df["num_subscribers"] +
        0.25 * df["num_reviews"] +
        0.2 * df["rating"] +
        0.2 * df["popularity_score"]
    )

    # -------------------------------
    # 🔹 3. Duration Category
    # -------------------------------
    def duration_category(hours):
        if hours < 2:
            return "short"
        elif hours < 10:
            return "medium"
        else:
            return "long"

    df["duration_category"] = df["content_duration"].apply(duration_category)

    # -------------------------------
    # 🔹 4. Level Mapping
    # -------------------------------
    level_map = {
        "beginner": 1,
        "intermediate": 2,
        "advanced": 3
    }

    df["level_num"] = df["level"].map(level_map).fillna(0)

    # -------------------------------
    # 🔹 5. Goal Alignment Tags
    # -------------------------------
    def generate_goal_tags(row):
        tags = []

        level = row["level"]
        title = row["course_title"]
        subject = row["subject"]

        # Foundation vs Skill Upgrade
        if level == "beginner":
            tags.append("foundation")
        else:
            tags.append("skill_upgrade")

        # Career readiness
        if row["career_relevance_score"] > 0.7:
            tags.append("job_ready")

        # Domain-based tags
        if "web" in subject:
            tags.append("freelancing")

        if "machine learning" in title or "ai" in title:
            tags.append("ai_career")

        return ",".join(tags)

    df["goal_alignment_tags"] = df.apply(generate_goal_tags, axis=1)

    # -------------------------------
    # 🔹 6. Combined Features (for TF-IDF)
    # -------------------------------
    df["combined_features"] = (
        df["course_title"] + " " +
        df["subject"] + " " +
        df["level"] + " " +
        df["tags"] + " " +
        df["learning_path"] + " " +
        df["instructor"]
    )

    print("✅ Feature engineering completed")

    return df