import os
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DATASET_FILE = "games.csv"


def create_sample_dataset():
    """Create a small illustrative dataset so the project can run immediately."""
    data = {
        "name": [
            "Counter-Strike 2", "Dota 2", "Grand Theft Auto V",
            "Elden Ring", "Cyberpunk 2077", "Minecraft",
            "Valorant", "The Witcher 3"
        ],
        "price": [0, 0, 29.99, 59.99, 59.99, 26.95, 0, 39.99],
        "tags": [
            "Action FPS Multiplayer",
            "MOBA Strategy Multiplayer",
            "Action Open World Adventure",
            "RPG Action Open World",
            "RPG Action Sci-Fi",
            "Sandbox Survival Adventure",
            "FPS Tactical Multiplayer",
            "RPG Adventure Open World"
        ],
        "positive_reviews": [
            800000, 1500000, 1200000, 750000,
            600000, 900000, 500000, 700000
        ],
        "negative_reviews": [
            150000, 250000, 200000, 100000,
            180000, 100000, 100000, 80000
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(DATASET_FILE, index=False)
    print(f"Created illustrative sample dataset: {DATASET_FILE}")
    return df


def load_dataset():
    if os.path.isfile(DATASET_FILE):
        try:
            df = pd.read_csv(DATASET_FILE)
            print(f"Loaded dataset: {DATASET_FILE}")
            return df
        except Exception as exc:
            print(f"Could not read {DATASET_FILE}: {exc}")
            return None
    print(f"{DATASET_FILE} was not found. Creating a sample dataset for demonstration.")
    return create_sample_dataset()


def standardize_columns(df):
    df.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    ]
    return df.drop_duplicates().copy()


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def run_analysis(df):
    print("\n" + "=" * 60)
    print("STEAM GAMES DATASET ANALYSIS")
    print("=" * 60)
    print(f"\nDataset shape (rows, columns): {df.shape}")
    print("\nColumns:", ", ".join(map(str, df.columns)))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values by column:")
    print(df.isnull().sum().to_string())

    name_col = find_column(df, ["name", "game_name", "title", "app_name"])
    price_col = find_column(df, ["price", "final_price", "discounted_price", "price_usd"])
    tags_col = find_column(df, ["tags", "genres", "categories", "genre"])
    positive_col = find_column(df, ["positive_reviews", "positive_ratings", "positive"])
    negative_col = find_column(df, ["negative_reviews", "negative_ratings", "negative"])
    rating_col = find_column(df, ["rating", "user_rating", "review_score"])

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        print("\n--- PRICE ANALYSIS ---")
        print(f"Average price: {df[price_col].mean():.2f}")
        print(f"Minimum price: {df[price_col].min()}")
        print(f"Maximum price: {df[price_col].max()}")
        print(f"Free games: {(df[price_col] == 0).sum()}")
        print(f"Paid games: {(df[price_col] > 0).sum()}")
        if name_col:
            print("\nFive most expensive games:")
            print(df.nlargest(5, price_col)[[name_col, price_col]].to_string(index=False))
    else:
        print("\n--- PRICE ANALYSIS ---\nNo recognized price column was found.")

    if positive_col and negative_col:
        df[positive_col] = pd.to_numeric(df[positive_col], errors="coerce")
        df[negative_col] = pd.to_numeric(df[negative_col], errors="coerce")
        df["_total_reviews"] = df[positive_col].fillna(0) + df[negative_col].fillna(0)
        denom = df["_total_reviews"].replace(0, float("nan"))
        df["_positive_pct"] = df[positive_col] / denom * 100
        print("\n--- REVIEW ANALYSIS ---")
        cols = ([name_col] if name_col else []) + [
            positive_col, negative_col, "_total_reviews", "_positive_pct"
        ]
        print("Most reviewed games:")
        print(df.nlargest(5, "_total_reviews")[cols].to_string(index=False))
        print("\nHighest positive-review percentage (among games with reviews):")
        reviewed = df[df["_total_reviews"] > 0]
        print(reviewed.nlargest(5, "_positive_pct")[cols].to_string(index=False))
    elif rating_col:
        print("\n--- RATING ANALYSIS ---")
        cols = ([name_col] if name_col else []) + [rating_col]
        print(df.nlargest(5, rating_col)[cols].to_string(index=False))
    else:
        print("\nNo recognized review/rating columns were found.")

    if tags_col:
        print("\n--- MOST COMMON TAGS ---")
        tag_counts = (
            df[tags_col].dropna().astype(str)
            .str.split(r"[,;|]")
            .explode().str.strip()
        )
        tag_counts = tag_counts[tag_counts != ""].value_counts()
        print(tag_counts.head(10).to_string())
    else:
        print("\nNo recognized tags/genres column was found.")

    return name_col, tags_col


def recommend_games(df, name_col, tags_col):
    print("\n--- GAME RECOMMENDATIONS ---")
    if not name_col or not tags_col:
        print("Recommendations require recognizable game-name and tags/genres columns.")
        return

    games = df[[name_col, tags_col]].copy()
    games[name_col] = games[name_col].fillna("").astype(str)
    games[tags_col] = games[tags_col].fillna("").astype(str)
    games = games.reset_index(drop=True)
    if len(games) < 2:
        print("At least two games are needed for recommendations.")
        return

    if not SKLEARN_AVAILABLE:
        print("Scikit-learn is not installed; skipping recommendation step.")
        print("Install dependencies with: pip install -r requirements.txt")
        return

    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        vectors = vectorizer.fit_transform(games[tags_col])
        similarity = cosine_similarity(vectors)
    except ValueError:
        print("Could not calculate similarity: tags may be empty.")
        return

    print("Available games:")
    for title in games[name_col].tolist():
        print(f" - {title}")
    selected = input("\nEnter a game name exactly as shown (or press Enter to skip): ").strip()
    if not selected:
        print("Recommendation input skipped.")
        return

    matches = games.index[games[name_col].str.casefold() == selected.casefold()].tolist()
    if not matches:
        print("Game not found. Please use a name from the list above.")
        return

    selected_index = matches[0]
    ranked = sorted(enumerate(similarity[selected_index]), key=lambda item: item[1], reverse=True)
    ranked = [(idx, score) for idx, score in ranked if idx != selected_index][:5]
    print(f"\nGames similar to {games.iloc[selected_index][name_col]}:")
    for idx, score in ranked:
        print(f" - {games.iloc[idx][name_col]} (tag similarity: {score:.2f})")


def main():
    df = load_dataset()
    if df is None or df.empty:
        print("No usable data found. Please check games.csv.")
        return
    df = standardize_columns(df)
    name_col, tags_col = run_analysis(df)
    recommend_games(df, name_col, tags_col)
    print("\nProject execution completed.")


if __name__ == "__main__":
    main()
