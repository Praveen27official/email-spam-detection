import joblib

# Load model & vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_message(message):
    transformed = vectorizer.transform([message])
    result = model.predict(transformed)[0]
    return "Spam" if result == 1 else "Not Spam"

# Example
msg = "Congratulations! You've won a free iPhone. Click here!"
print(predict_message(msg))
