from newsapi import NewsApiClient
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import string 

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv(r'C:\Users\krujo\OneDrive\Desktop\BBC-Dataset-News-Classification\dataset\dataset.csv', encoding='ISO-8859-1')

def cleaned_desc_column(text):
    text = re.sub(r'\d', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing whitespaces

    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric and non-whitespace characters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]

    return ' '.join(filtered_tokens)

df['news_title'] = df['news'].apply(cleaned_desc_column)

X = df['news_title']
y = df['type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=90)

lr = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
              ])

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"Accuracy is: {accuracy_score(y_pred, y_test)}")