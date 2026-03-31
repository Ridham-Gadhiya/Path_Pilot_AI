import pandas as pd


def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None