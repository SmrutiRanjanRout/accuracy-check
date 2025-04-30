# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and merge datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')
news = pd.read_csv('news.csv')
welfake = pd.read_csv('WELFake_Dataset.csv')
fake['label'] = 0
true['label'] = 1
df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
df['text'] = df['title'] + " " + df['text']

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# === ML Pipeline ===
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

ml_models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

results = {}

for name, model in ml_models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report['weighted avg']

# === Deep Learning Setup ===
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=200, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=200, padding='post')
vocab_size = len(tokenizer.word_index) + 1

# DL model builder
def build_and_train_model(model_type):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=200))
    
    if model_type == "LSTM":
        model.add(LSTM(64))
    elif model_type == "GRU":
        model.add(GRU(64))
    elif model_type == "BiLSTM":
        model.add(Bidirectional(LSTM(64)))
    elif model_type == "CNN":
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
    elif model_type == "CNN_BiLSTM":
        model.add(Conv1D(64, 5, activation='relu'))
        model.add(Bidirectional(LSTM(64)))
    else:
        return None  # Unsupported

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=3, verbose=0, validation_split=0.1)

    y_pred = (model.predict(X_test_pad) > 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report['weighted avg']

for name in ["LSTM", "GRU", "BiLSTM", "CNN", "CNN_BiLSTM"]:
    try:
        results[name] = build_and_train_model(name)
    except Exception as e:
        print(f"Failed on {name}: {e}")

# === Plot Results ===
results_df = pd.DataFrame(results).T[['precision', 'recall', 'f1-score', 'accuracy']]
results_df.plot(kind='bar', figsize=(12, 6), ylim=(0.5, 1), title="Model Performance Comparison")
plt.grid(True)
plt.show()
