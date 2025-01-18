import pandas as pd
import numpy as np
data = pd.read_csv('Resources/spam.csv', encoding = 'latin-1')
data.head()
list = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
data = data.drop(list, axis=1)
nham, nspam = data['label'].value_counts()
print('nham: ',nham)
print('nspam: ',nspam)
data = pd.ExcelFile('Resources/D1.xls')
sheet = data.parse(0)
sheet.head()
nham, nspam = sheet['Label'].value_counts()
print('nham: ',nham)
print('nspam: ',nspam)
data = pd.ExcelFile('Resources/D2.xls')
sheet = data.parse(0)
sheet.head()
nham, nspam = sheet['Label'].value_counts()
print('nham: ',nham)
print('nspam: ',nspam)
data = pd.ExcelFile('Resources/D2.xls')
sheet = data.parse(0)
sheet.head()
nham, nspam = sheet['Label'].value_counts()
print('nham: ',nham)
print('nspam: ',nspam)
data = pd.ExcelFile('Resources/D3.csv')
sheet = data.parse(0)
sheet.head()
nham, nspam = sheet['Label'].value_counts()
print('nham: ',nham)
print('nspam: ',nspam)


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('spam_sms_dataset.csv')

# Preprocess the text data
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

# Split the data into training and testing sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Use the classifier to detect spam SMS
def detect_spam_sms(text):
    text = preprocess_text(text)
    text_tfidf = vectorizer.transform([text])
    prediction = clf.predict(text_tfidf)
    return prediction[0]

text = 'You have won a prize! Click here to claim it.'
print('Spam?' , detect_spam_sms(text))

