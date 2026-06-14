import bz2
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_dataset(file_path, num_samples=40000):
    """Loads a specific number of lines from the fastText bz2 compressed file."""
    texts = []
    labels = []
    
    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            if line.startswith('__label__1'):
                labels.append('Negative')
            elif line.startswith('__label__2'):
                labels.append('Positive')
            else:
                continue
            
            text = line.split(' ', 1)[1].strip()
            texts.append(text)
            
    return pd.DataFrame({'review': texts, 'sentiment': labels})

print("Loading dataset samples... Please wait.")
train_df = load_dataset('data/train.ft.txt.bz2', num_samples=40000)
test_df = load_dataset('data/test.ft.txt.bz2', num_samples=10000)

def clean_text(text):
    """Performs lowercasing and removes punctuation symbols."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

print("Performing NLP Text Cleaning...")
train_df['clean_review'] = train_df['review'].apply(clean_text)
test_df['clean_review'] = test_df['review'].apply(clean_text)

print("Extracting features using TF-IDF and removing English stopwords...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train = vectorizer.fit_transform(train_df['clean_review'])
X_test = vectorizer.transform(test_df['clean_review'])

y_train = train_df['sentiment']
y_test = test_df['sentiment']

# Define the three mandatory machine learning models from the project guide
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine (SVM)': LinearSVC(max_iter=1000)
}

results = {}

print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, pos_label='Positive')
    rec = recall_score(y_test, predictions, pos_label='Positive')
    f1 = f1_score(y_test, predictions, pos_label='Positive')
    
    results[name] = {
        'model': model,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

print("\n--- Model Comparison Performance Metrics ---")
for name in results:
    print(f"\n[{name}]")
    print(f"Accuracy : {results[name]['Accuracy']:.4f}")
    print(f"Precision: {results[name]['Precision']:.4f}")
    print(f"Recall   : {results[name]['Recall']:.4f}")
    print(f"F1-Score : {results[name]['F1-Score']:.4f}")

# Select the best model based on accuracy
best_model_name = max(results, key=lambda k: results[k]['Accuracy'])
print(f"\nBest Model selected: {best_model_name}")

# Save the trained model and vectorizer for the Streamlit dashboard app
with open('best_model.pkl', 'wb') as f:
    pickle.dump(results[best_model_name]['model'], f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Saved the best model and vectorizer successfully as .pkl files!")