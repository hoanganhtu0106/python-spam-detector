import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_and_prepare_data
import pickle

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()

    def train(self, data):
        label_counts = Counter()
        word_counts = {0: defaultdict(int), 1: defaultdict(int)}
        total_words = {0: 0, 1: 0}

        for words, label in data:
            label_counts[label] += 1
            for word in words:
                self.vocab.add(word)
                word_counts[label][word] += 1
                total_words[label] += 1

        total_docs = len(data)
        self.class_probs = {label: count / total_docs for label, count in label_counts.items()}
        self.word_probs = {
            label: {
                word: (word_counts[label][word] + 1) / (total_words[label] + len(self.vocab))
                for word in self.vocab
            } for label in [0, 1]
        }

    def predict(self, words):
        scores = {}
        for label in [0, 1]:
            log_prob = np.log(self.class_probs[label])
            for word in words:
                if word in self.vocab:
                    log_prob += np.log(self.word_probs[label][word])
            scores[label] = log_prob
        return max(scores, key=scores.get)

    def save_model(self, filename='model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename='model.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()

    def train(self, data):
        label_counts = Counter()
        word_counts = {0: defaultdict(int), 1: defaultdict(int)}
        total_words = {0: 0, 1: 0}

        for words, label in data:
            label_counts[label] += 1
            for word in words:
                self.vocab.add(word)
                word_counts[label][word] += 1
                total_words[label] += 1

        total_docs = len(data)
        self.class_probs = {label: count / total_docs for label, count in label_counts.items()}
        self.word_probs = {
            label: {
                word: (word_counts[label][word] + 1) / (total_words[label] + len(self.vocab))
                for word in self.vocab
            } for label in [0, 1]
        }

    def predict(self, words):
        scores = {}
        for label in [0, 1]:
            log_prob = np.log(self.class_probs[label])
            for word in words:
                if word in self.vocab:
                    log_prob += np.log(self.word_probs[label][word])
            scores[label] = log_prob
        return max(scores, key=scores.get)

if __name__ == "__main__":
    train_data, test_data = load_and_prepare_data()

    nb = NaiveBayesClassifier()
    nb.train(train_data)
    nb.save_model('spam_model.pkl')
    y_true = [label for _, label in test_data]
    y_pred = [nb.predict(words) for words, _ in test_data]

    print("🔵 Naive Bayes Metrics:")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
