from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

from preprocess import load_and_preprocess

# Load data
df = load_and_preprocess("data/spam.csv")

X = df['message']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model saved!")
