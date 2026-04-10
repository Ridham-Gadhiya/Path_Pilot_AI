import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing on dataset.

    - Handle missing values
    - Normalize text columns
    - Create clean title column for search

    Args:
        df (pd.DataFrame): Raw dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    df = df.copy()

    # 🔹 Fill missing values (safety)
    text_cols = [
        "course_title",
        "subject",
        "level",
        "tags",
        "learning_path",
        "instructor",
        "language"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 🔹 Normalize text (lowercase + strip)
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()

    # 🔹 Create clean title for search
    df["title_clean"] = df["course_title"].str.lower().str.strip()

    # 🔹 Clean tags (important for NLP later)
    if "tags" in df.columns:
        df["tags"] = df["tags"].str.replace(",", " ", regex=False)

    print("✅ Preprocessing completed")

    return df