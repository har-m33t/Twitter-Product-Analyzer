# Import all necessary libraries
import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
import re
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib
import logging
from nltk.corpus import stopwords
import nltk
import time

# Download Stopwords:uninformative words that don't add substance
nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
'''
Free Twitter API does not support web scraping for tweets
'''
class Scrape:
    def __init__(self, product):
        """
        Initialize the Scrape class with Twitter API credentials and product to search for.
        """
        print("Initializing Scrape class...")
        self._consumer_key = os.getenv('CONSUMER_KEY')
        self._consumer_secret = os.getenv('CONSUMER_SECRET')
        self._access_token = os.getenv('ACCESS_TOKEN')
        self._access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')
        self._bearer_token = os.getenv('BEARER_TOKEN')
        self._product = product

        # Check if API keys are loaded correctly
        if not all([self._consumer_key, self._consumer_secret, self._access_token, self._access_token_secret, self._bearer_token]):
            logging.error('API keys not properly loaded. Check your environment variables.')
        print("Scrape class initialized successfully.")

    def twitter_auth(self):
        """
        Authenticate with the Twitter API using the provided credentials.
        Returns:
            tweepy.API: Authenticated API object.
        """
        print("Authenticating with Twitter API...")
        auth = tweepy.OAuthHandler(self._consumer_key, self._consumer_secret)
        auth.set_access_token(self._access_token, self._access_token_secret)
        self._api = tweepy.API(auth, wait_on_rate_limit=True)
        print("Authentication successful.")
        return self._api

    def twitter_search(self, no_of_tweets=100) -> pd.DataFrame:
        """
        Search for tweets related to the specified product.
        Args:
            no_of_tweets (int): Number of tweets to fetch.
        Returns:
            pd.DataFrame: DataFrame containing tweet details.
        """
        print("Starting Twitter search for product:", self._product)
        try:
            search_query = f"{self._product} -filter:retweets AND -filter:replies AND -filter:links"
            tweets = self._api.search_tweets(q=search_query, lang="en", count=no_of_tweets, tweet_mode='extended')

            if not tweets:
                logging.info(f'No tweets found for product: {self._product}')
                return pd.DataFrame()

            print(f"Fetched {len(tweets)} tweets.")
            attributes_container = [
                [tweet.user.name, tweet.created_at, tweet.favorite_count, tweet.source, tweet.full_text]
                for tweet in tweets
            ]
            columns = ["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
            tweets_df = pd.DataFrame(attributes_container, columns=columns)
            print("Twitter search complete. Returning DataFrame.")
            return tweets_df
        except tweepy.errors.TooManyRequests:
            logging.warning("Rate limit reached. Sleeping for 15 minutes...")
            time.sleep(15 * 60)
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching tweets: {str(e)}")
            return pd.DataFrame()


class DataCleaning:
    def __init__(self):
        """
        Initialize the DataCleaning class with a set of stop words using NLTK.
        """
        print("Initializing DataCleaning class...")
        self.stop_words = set(stopwords.words('english'))
        print("DataCleaning class initialized.")

    def clean_text(self, text):
        """
        Clean the text by removing URLs, special punctuation, converting to lowercase, and removing stop words.
        Args:
            text (str): The text to clean.
        Returns:
            str: The cleaned text.
        """
        print("Cleaning text:", text[:30], "...")  # Display first 30 characters of the text
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text and remove stop words
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        cleaned_text = ' '.join(tokens)
        print("Text cleaned.")
        return cleaned_text

    def clean_data_frame(self, df):
        """
        Clean the 'Tweet' column in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing tweets.
        Returns:
            pd.DataFrame: The DataFrame with cleaned tweets.
        """
        if not df.empty:
            print(f"Cleaning {len(df)} tweets in DataFrame...")
            df['Cleaned Tweet'] = df['Tweet'].apply(self.clean_text)
            time.sleep(5)
            print("DataFrame cleaning complete.")
        else:
            logging.info('DataFrame is empty, no cleaning performed.')
        return df


class SentimentAnalysis:
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the text using TextBlob.
        Args:
            text (str): The text to analyze.
        Returns:
            float: The sentiment polarity score.
        """
        print(f"Analyzing sentiment for text: {text[:30]} ...")
        analysis = TextBlob(text)
        print(f"Sentiment analysis complete with polarity score: {analysis.sentiment.polarity}")
        return analysis.sentiment.polarity

    def categorize_sentiment(self, polarity):
        """
        Categorize the sentiment based on polarity score.
        Args:
            polarity (float): Sentiment polarity score.
        Returns:
            str: Categorized sentiment (Positive, Neutral, Negative).
        """
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_data_frame(self, df):
        """
        Analyze the sentiment of the 'Cleaned Tweet' column in the DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing cleaned tweets.
        Returns:
            pd.DataFrame: The DataFrame with sentiment scores and categories.
        """
        if not df.empty:
            print(f"Analyzing sentiment for {len(df)} cleaned tweets...")
            df['Sentiment'] = df['Cleaned Tweet'].apply(self.analyze_sentiment)
            df['Sentiment Category'] = df['Sentiment'].apply(self.categorize_sentiment)
            time.sleep(5)
            print("Sentiment analysis complete.")
        else:
            logging.info('DataFrame is empty, no sentiment analysis performed.')
        return df

    def plot_sentiment(self, df):
        """
        Plot a histogram and bar chart of the sentiment scores and categories.
        Args:
            df (pd.DataFrame): The DataFrame containing sentiment scores.
        """
        if not df.empty:
            print("Plotting sentiment analysis results...")

            # Switch to non Interactive Backend
            matplotlib.use('Agg')

            # Histogram of sentiment scores
            plt.figure(figsize=(10, 6))
            plt.hist(df['Sentiment'], bins=20, edgecolor='black', color='skyblue')
            plt.title('Sentiment Score Distribution')
            plt.xlabel('Sentiment Score')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('sentiment_histogram.png')
            plt.close()  # Close figure after plotting to avoid memory issues

            sentiment_counts = df['Sentiment Category'].value_counts()
            plt.figure(figsize=(8, 5))
            sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
            plt.title('Sentiment Category Distribution')
            plt.xlabel('Sentiment Category')
            plt.ylabel('Count')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig('sentiment_bar_chart.png')
            plt.close()  # Close figure after plotting to avoid memory issues
        else:
            logging.info('DataFrame is empty, no plots generated.')


if __name__ == "__main__":
    print("Starting the program...")

    # Set up mock data for testing instead of Twitter API
    try:
        mock_data = pd.read_csv("mock_tweets.csv")
    except pd.errors.ParserError as e:
        print("Error reading CSV file:", e)
    print("Loaded mock dataset. Here are the first few rows:")
    print(mock_data.head())
    print("Proceeding with data cleaning...")

    # Data cleaning
    cleaner = DataCleaning()
    cleaned_df = cleaner.clean_data_frame(mock_data)

    # Sentiment analysis
    analyzer = SentimentAnalysis()
    analyzed_df = analyzer.analyze_data_frame(cleaned_df)

    # Plot sentiment analysis results
    time.sleep(10)
    analyzer.plot_sentiment(analyzed_df)

    print("Program completed successfully.")