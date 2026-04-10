from src.data_loader import load_and_validate
from src.preprocessing import preprocess_data
from src.feature_engineering import process_features

df = load_and_validate("data/processed/your_file.csv")
df = preprocess_data(df)
df = process_features(df)