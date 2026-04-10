import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from given file path.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")

    df = pd.read_csv(file_path)

    print(f"✅ Data loaded successfully")
    print(f"Shape: {df.shape}")

    return df

def validate_columns(df: pd.DataFrame):
    required_columns = [
        'course_title',
        'subject',
        'level',
        'num_subscribers',
        'num_reviews',
        'rating',
        'content_duration',
        'tags',
        'learning_path',
        'skill_level_score',
        'popularity_score'
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    print("✅ All required columns present")

def load_and_validate(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    validate_columns(df)
    return df