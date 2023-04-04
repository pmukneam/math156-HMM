# Importing libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

## Importing data

# Importing Steam Review
original_data = pd.read_csv("data/dataset.csv")
steam_review = original_data.sample(frac=0.0013)

# Selecting only the review and score column
steam_review = steam_review[["review_text", "review_score"]]

# Droping NULL/NA rows
steam_review = steam_review.dropna()

# Droping duplicate
steam_review = steam_review.drop_duplicates()

## Data Cleaning

# Removing trailing space and make all reviews lowercase
steam_review['review_text'] = [review.strip().lower() for review in steam_review['review_text']]

# Removing special characters
steam_review['review_text'] = steam_review['review_text'].replace(r"[^a-zA-Z\d\_\+\-\'\.\/\s]+", ' ', regex = True)
steam_review['review_text'] = steam_review['review_text'].replace(["./ ", "' ", " '"], " ", regex = True)

# Importing stopwords in English
nltk.download('stopwords')
stopwords = stopwords.words('english')

# Creating stopwords regex
pat = r'\b(?:{})\b'.format('|'.join(stopwords))

# Removing stopwords
steam_review['review_text'] = steam_review['review_text'].str.replace(pat, '', regex = True)
steam_review['review_text'] = steam_review['review_text'].str.replace(r'\s+', ' ', regex = True)

# Removing special char
steam_review['review_text'] = steam_review['review_text'].str.replace(pattern, ' ', regex =  True)

steam_review.to_csv("data/steam_review_10k.csv")
steam_review.head(1000).to_csv("data/steam_review_1k.csv")
steam_review.head(100).to_csv("data/steam_review_100.csv")
steam_review.head(10).to_csv("data/steam_review_10.csv")
