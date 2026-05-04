import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

def load_and_preprocess(path):
    df = pd.read_csv(path, encoding='latin-1')
    
    # Keep only needed columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean text
    df['message'] = df['message'].apply(clean_text)
    
    return df
