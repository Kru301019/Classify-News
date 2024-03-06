# Import necessary libraries
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Uncomment these lines if you haven't downloaded nltk resources yet
# nltk.download('wordnet')
# nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv(r'C:\Users\krujo\OneDrive\Desktop\BBC-Dataset-News-Classification\dataset\dataset.csv', encoding='ISO-8859-1')

# Function to clean and tokenize text
def preprocess_text(text):
    # Remove digits
    text = re.sub(r'\d', '', text)

    # Remove extra whitespaces, convert to lowercase, and strip leading/trailing whitespaces
    text = re.sub(r'\s+', ' ', text).lower().strip()

    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization and lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    
    # Initialize an empty list for lemmatized tokens
    lemmatized_tokens = []
    
    # Loop through each token and lemmatize
    for token in tokens:
        lemmatized_tokens.append(lemmatizer.lemmatize(token))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize an empty list for filtered tokens
    filtered_tokens = []
    
    # Loop through each lemmatized token and filter stopwords
    for token in lemmatized_tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    # Join the filtered tokens into a single string
    return ' '.join(filtered_tokens)

# Apply the text preprocessing function to the 'news' column
df['processed_news'] = df['news'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_news'], df['type'], test_size=0.30, random_state=90)

# Build and train the model using a simple pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced')),
])

classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = classifier.predict(X_test)

# Evaluate and print the accuracy of the model
accuracy = accuracy_score(predictions, y_test)
#print(f"Model Accuracy: {accuracy}")

def predict_category(news_text):
    preprocessed_text = preprocess_text(news_text)
    category_prediction = classifier.predict([preprocessed_text])
    return category_prediction[0]

# Example usage
new_article_text = """
Kylian Mbappe capped a fine display with two goals as Paris St-Germain comfortably moved into the quarter-finals of the Champions League at the expense of Real Sociedad.
With the trip to San Sebastian billed as a potentially tricky second leg for PSG to negotiate, all eyes were on Mbappe, who has agreed to join Real Madrid this summer.

And the France forward turned in a superb performance to register his 27th and 28th goals of the campaign for Luis Enrique's side.

The 25-year-old served an early warning to the hosts by accelerating past Hamari Traore to tee up a chance for Bradley Barcola, which was well saved by home goalkeeper Alex Remiro
And six minutes later Mbappe opened his account for the evening, latching on to Ousmane Dembele's pass and teasing home defender Igor Zubeldia, before whipping a right-foot effort into the bottom-right corner with such power and precision that it temporarily dislodged the netting.

Mbappe's second of the night arrived shortly after the break as he raced clear down the left to collect Lee Kang-in's pass and then emphatically dispatched a disguised low shot past Remiro at his near post.

It was no more than the Parisiens deserved with Imanol Alguacil's Sociedad team far too passive in the opening stages and only sparking into life after they had fallen four goals behind on aggregate.

By then it was far too late with Mikel Merino's driven effort in the closing minutes proving merely a consolation.
"""

predicted_category = predict_category(new_article_text)
print(f"Predicted Category: {predicted_category}")