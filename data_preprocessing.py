import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from random import shuffle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return words

def load_and_prepare_data(path='spam.csv', split_ratio=0.8):
    df = pd.read_csv(path)
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['tokens'] = df['message'].apply(preprocess)
    
    data = list(zip(df['tokens'], df['label']))
    shuffle(data)
    split_idx = int(split_ratio * len(data))
    train_data, test_data = data[:split_idx], data[split_idx:]
    return train_data, test_data
