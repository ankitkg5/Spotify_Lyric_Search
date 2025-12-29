📌 Objective
Predict Song Title and Artist
Input: Small lyrics snippet

📂 Dataset
Kaggle Lyrics Dataset
Columns used: artist, song, text

🛠 Technologies
Python 3.10
NLTK
Scikit-learn
TF-IDF
Cosine Similarity
VS Code

🔄 Text Preprocessing
Lowercasing
Remove special characters
Stop-word removal
Lemmatization

🧠 Model
TF-IDF Vectorizer (uni, bi, tri-grams)
Cosine similarity for matching

▶️ Run Project
python src/predict.py

🖥 Sample Output
Predicted Song : Come Talk To Me
Predicted Artist : Peter Gabriel
Accuracy : 0.88

📊 Performance
Accuracy: ~85–90%
Based on lyric similarity