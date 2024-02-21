Applying NLP models (FinVADER and FinBERT) to predict stock movements

1. Data Preprocessing
2. Applying NLP models

   ##-- FinVADER sentiment analysis

   import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from SentiBignomics import lexicon1
from Henry import lexicon2

# Read the dataset from Excel
df = pd.read_excel('APPLE_SENTIMENT_ABLAGE.xlsx')

# Load lexicons
sentibignomics_lexicon = lexicon1()
henry_lexicon = lexicon2()

# Create a SentimentIntensityAnalyzer object
sid_obj = SentimentIntensityAnalyzer()

# Function to apply sentiment analysis
def apply_sentiment_analysis(text, use_sentibignomics=False, use_henry=False):
    if use_sentibignomics and use_henry:
        combined_lexicon = {**sentibignomics_lexicon, **henry_lexicon}
    elif use_sentibignomics:
        combined_lexicon = sentibignomics_lexicon
    elif use_henry:
        combined_lexicon = henry_lexicon
    else:
        combined_lexicon = {}

    # Update the VADER lexicon with the combined lexicon
    sid_obj.lexicon.update(combined_lexicon)

    # Get the compound score for the text
    sentiment_dict = sid_obj.polarity_scores(text)
    return sentiment_dict['compound']
    return sentiment_dict['positive']
    return sentiment_dict['negative']

# Apply sentiment analysis to each row in the 'clean_this' column
df['compound_score'] = df['clean_this'].apply(lambda x: apply_sentiment_analysis(x, use_sentibignomics=True, use_henry=True))

# Function to categorize sentiment
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment categorization to each row
df['sentiment_category'] = df['compound_score'].apply(categorize_sentiment)

# Save the DataFrame with sentiment analysis results to a new Excel file in the same directory
df.to_excel('sentiment_results_appleo.xlsx', index=False)


