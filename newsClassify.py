from newsapi import NewsApiClient
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
9
newsapi = NewsApiClient(api_key='733bed816c134dc299f9ae6d8b183989')

tech_articles = newsapi.get_everything(q='tech', language='en', \
page_size=100)
entertainment_articles = newsapi.get_everything(q='entertainment',\
language='en', page_size=100)
business_articles = newsapi.get_everything(q='business',\
language='en', page_size=100)
sports_articles = newsapi.get_everything(q='sports',\
language='en', page_size=100)
politics_articles = newsapi.get_everything(q='politics',\
language='en', page_size=100)
travel_articles = newsapi.get_everything(q='travel',\
language='en', page_size=100)
food_articles = newsapi.get_everything(q='food',\
language='en', page_size=100)
health_articles = newsapi.get_everything(q='health',\
language='en', page_size=100)

tech = pd.DataFrame(tech_articles['articles'])
tech['category'] = 'Tech'
entertainment = pd.DataFrame(entertainment_articles['articles'])
entertainment['category'] = 'Entertainment'
business = pd.DataFrame(business_articles['articles'])
business['category'] = 'Business'
sports = pd.DataFrame(sports_articles['articles'])
sports['category'] = 'Sports'
politics = pd.DataFrame(politics_articles['articles'])
politics['category'] = 'Politics'
travel = pd.DataFrame(travel_articles['articles'])
travel['category'] = 'Travel'
food = pd.DataFrame(food_articles['articles'])
food['category'] = 'Food'
health = pd.DataFrame(health_articles['articles'])
health['category'] = 'Health'

categories = [tech, entertainment, business, sports, politics, \
travel, food, health]
df = pd.concat(categories)
df.info()


def cleaned_desc_column(text):

    text = re.sub(r',', '', text)

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'\.', '', text)

    text = re.sub(r"['\"]", '', text)

    text = re.sub(r'\W', ' ', text)
    
    text_token = word_tokenize(text) 
    stop_words = set(stopwords.words('English'))