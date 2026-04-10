from src.data_loader import load_and_validate
from src.preprocessing import preprocess_data
from src.feature_engineering import process_features
from src.recommender import build_similarity, recommend_smart


df = load_and_validate("data/processed/your_file.csv")
df = preprocess_data(df)
df = process_features(df)

tfidf_matrix, cosine_sim = build_similarity(df)

print(recommend_smart(df, cosine_sim, "python"))