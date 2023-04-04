# Importing libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

## Importing data

# Importing Steam Review
original_data = pd.read_csv("data/steam_review_10k.csv")
#steam_review = original_data.sample(frac=0.0013)
steam_review = original_data


steam_review['review_text'] = steam_review['review_text'].str.replace('[^a-zA-Z]+', ' ', regex = True)

# get the words list of postive neg
pos_words = []
neg_words = []

with open('data/opinion-lexicon-English/positive-words.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Remove the newline character at the end of the line
        word = line.strip()
        # Append the word to the list
        pos_words.append(word)

with open('data/opinion-lexicon-English/negative-words.txt', 'r', encoding='ISO-8859-1') as file:
    # Read each line in the file
    for line in file:
        # Remove the newline character at the end of the line
        word = line.strip()
        # Append the word to the list
        neg_words.append(word)

# Combine both list
both_words = pos_words + neg_words

# Clearn words
pattern = '[^a-zA-Z]'

# Use a list comprehension to apply the regular expression to each string in the list
both_words = [re.sub(pattern, '', w) for w in both_words]
pos_words = [re.sub(pattern, '', w) for w in pos_words]
neg_words = [re.sub(pattern, '', w) for w in neg_words]

# Remove review that doesn't contain words in lexicon
steam_review = steam_review[steam_review['review_text'].str.contains('|'.join(both_words))]



def filter_adjectives(text):
    # Split the string into a list of words
    words = text.split()
    # Filter out any words that are not in the adjective list
    filtered_words = [word for word in words if word.lower() in both_words]
    # Join the filtered list of words back into a single string
    filtered_words
    return filtered_words


# Apply the string method separately and assign the result to a new column
steam_review['filtered_review_text'] = steam_review['review_text'].str.lower()

# Apply the function to the new column using .apply()
steam_review['filtered_review_text'] = steam_review['filtered_review_text'].apply(filter_adjectives)

# Assign sentiment sequence

def label_words(text_list):
    # Create a list of ones and zeros for each word in each string in text_list
    word_labels_list = [[1 if word.lower() in pos_words else 0 if word.lower() in neg_words else None for word in text.split()] for text in text_list]
    # Return the filtered list of ones and zeros
    return word_labels_list

def flatten_list(list):
    flat_list = [item for sublist in list for item in sublist]
    return flat_list

steam_review['sen_seq'] = steam_review['filtered_review_text'].apply(label_words)


steam_review['sen_seq'] = steam_review['sen_seq'].apply(flatten_list)

# Filter out rows where the 'word_labels' column contains an empty list
steam_review = steam_review[steam_review['sen_seq'].apply(lambda x: len(x) > 0)]


steam_review.head(10).to_csv("data/steam_review_tmp.csv")

# Count the occurance of each words in the data frame

# Initialize an empty dictionary to store the word counts
word_count = {}

num_words = 0
num_pos = 0
num_neg = 0

# Loop through each row in the DataFrame
for index, row in steam_review.iterrows():
    # Loop through each word in the 'review_words' list of the current row
    for word in row['filtered_review_text']:
        num_words += 1

        if word in pos_words:
            num_pos += 1
        else:
            num_neg += 1

        # If the word is already in the dictionary, increment its count
        if word in word_count:
            word_count[word] += 1
        # If the word is not in the dictionary, add it with a count of 1
        else:
            word_count[word] = 1

pi_dist = [num_pos/num_words, num_neg/num_words]


# Emission
#
#
# Initialize dictionaries to store the word counts for positive and negative sentiments
positive_word_count = {}
negative_word_count = {}

# Loop through each row in the DataFrame
for index, row in steam_review.iterrows():
    # Get the list of words and their corresponding sentiments
    words = row['filtered_review_text']
    sentiments = row['sen_seq']

    # Loop through the words and sentiments
    for word, sentiment in zip(words, sentiments):
        if sentiment == 1:  # Positive sentiment
            if word in positive_word_count:
                positive_word_count[word] += 1
            else:
                positive_word_count[word] = 1
        else:  # Negative sentiment
            if word in negative_word_count:
                negative_word_count[word] += 1
            else:
                negative_word_count[word] = 1

# Calculate the total number of words in positive and negative sentiments
total_positive_words = sum(positive_word_count.values())
total_negative_words = sum(negative_word_count.values())

# Calculate emission probabilities
emission_probabilities = {
    word: {
        'positive': positive_word_count.get(word, 0) / total_positive_words,
        'negative': negative_word_count.get(word, 0) / total_negative_words
    }
    for word in set(positive_word_count) | set(negative_word_count)
}

# transiiton
#


# Initialize a dictionary to store the state transition counts
state_transition_counts = {
    0: {0: 0, 1: 0},
    1: {0: 0, 1: 0}
}

# Loop through each row in the DataFrame
for index, row in steam_review.iterrows():
    # Get the list of sentiments
    sentiments = row['sen_seq']

    # Loop through the sentiments, skipping the last one since there's no transition after it
    for i in range(len(sentiments) - 1):
        current_state = sentiments[i]
        next_state = sentiments[i + 1]
        state_transition_counts[current_state][next_state] += 1

# Calculate the total number of transitions from positive and negative sentiments
total_positive_transitions = sum(state_transition_counts[1].values())
total_negative_transitions = sum(state_transition_counts[0].values())

# Calculate transition probabilities
transition_probabilities = {
    current_state: {
        next_state: state_transition_counts[current_state][next_state] / (total_positive_transitions if current_state == 1 else total_negative_transitions)
        for next_state in [0, 1]
    }
    for current_state in [0, 1]
}




