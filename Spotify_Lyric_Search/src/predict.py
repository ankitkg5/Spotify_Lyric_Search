import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("data/lyrics.csv")

# Keep required columns
df = df[["artist", "song", "text"]]
df.dropna(inplace=True)

# -------------------------------
# Preprocess Lyrics
# -------------------------------
df["clean_lyrics"] = df["text"].apply(preprocess)
df["clean_lyrics"] = df["clean_lyrics"].str[:2000]

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.85
)


X = vectorizer.fit_transform(df["clean_lyrics"])

# -------------------------------
# Prediction Function
# -------------------------------
def predict_song(snippet):
    snippet = preprocess(snippet)
    snippet_vector = vectorizer.transform([snippet])

    similarity = cosine_similarity(snippet_vector, X)
    index = similarity.argmax()

    return df.iloc[index][["song", "artist"]]

# -------------------------------
# Accuracy Evaluation
# -------------------------------
def evaluate_accuracy(samples=50):
    correct = 0

    for _ in range(samples):
        sample = df.sample(1)
        snippet = sample["text"].values[0][100:350]


        prediction = predict_song(snippet)

        if prediction["artist"] == sample["artist"].values[0]:
            correct += 1

    return correct / samples

# -------------------------------
# Run Test
# -------------------------------
if __name__ == "__main__":
    test_snippet = "hello darkness my old friend ive come to talk with you again"
    result = predict_song(test_snippet)

    print("Predicted Song :", result["song"])
    print("Predicted Artist :", result["artist"])
    print("Accuracy :", evaluate_accuracy())
