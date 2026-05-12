import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

logger = logging.getLogger(__name__)

class SentimentCorrelationAnalyzer:
    """
    Analyzes the relationship between financial news sentiment and stock price movements.
    
    This class handles date alignment between news and stock data, applies VADER sentiment 
    analysis, computes daily returns, and performs correlation analysis with visualizations.
    
    Attributes:
        ticker (str): Stock symbol being analyzed
        stock_df (pd.DataFrame): Pre-processed stock price DataFrame from FinancialAnalyzer
        combined_df (pd.DataFrame): Aligned DataFrame with sentiment scores and returns
        correlation (float): Pearson correlation between sentiment and daily returns
    """
    def __init__(self, ticker, stock_df):
        """
        Initialize the SentimentCorrelationAnalyzer.
        
        Args:
            ticker (str): Stock symbol (e.g., 'GOOG')
            stock_df (pd.DataFrame): Cleaned stock price DataFrame with 'Close' column
        """
        self.ticker = ticker
        self.stock_df = stock_df.copy()
        self.combined_df = None
        self.correlation = None

        # Initialize VADER Sentiment Analyzer
        # VADER was chosen for its excellent performance on short-form financial headlines,
        # speed, and ability to capture intensity (compound score).
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def _calculate_returns(self):
        """
        Calculate daily percentage returns from closing prices.
        """
    
        self.stock_df['Daily_Return'] = self.stock_df['Close'].pct_change() * 100
        logger.info("Calculated daily stock returns.")

    def _calculate_sentiment(self, df):
        """
        Apply VADER sentiment analysis to headlines.
        
        Args:
            df (pd.DataFrame): DataFrame containing news headlines
            
        Returns:
            pd.DataFrame: Original DataFrame with added 'sentiment_score' column
        """
        
        text_col = 'clean_headline' if 'clean_headline' in df.columns else 'headline'
        df['sentiment_score'] = df[text_col].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound']
        )
        return df

    def _align_and_aggregate(self):
        """
        Align news articles with trading days and compute average daily sentiment.
        
        Handles weekend/holiday alignment by forwarding news to the next trading day.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, '..', 'data', 'processed', 'analysed_text.csv')
        file_path = os.path.normpath(file_path)
        
        news_df = pd.read_csv(file_path)
        news_df = news_df[news_df['stock'] == self.ticker].copy()
        
        
        news_df = self._calculate_sentiment(news_df)
        
        
        news_df['date_cleaned'] = news_df['date'].astype(str).str.slice(0, 19)
        news_df['date'] = pd.to_datetime(news_df['date_cleaned'], errors='coerce')
    
        news_df['date'] = news_df['date'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        news_df['date'] = news_df['date'].dt.tz_localize(None) 

        news_df.drop(columns=['date_cleaned'], inplace=True)

        self.stock_df.index = pd.to_datetime(self.stock_df.index)
        
        trading_days = pd.DataFrame(index=self.stock_df.index.unique()).sort_index()
        trading_days['trading_day'] = trading_days.index

        
        aligned_news = pd.merge_asof(
            news_df.sort_values('date'),
            trading_days,
            left_on='date',
            right_on='trading_day',
            direction='forward'
        )

        
        daily_sentiment = aligned_news.groupby('trading_day')['sentiment_score'].mean()
        
        
        self.combined_df = self.stock_df[['Daily_Return']].merge(
            daily_sentiment, left_index=True, right_index=True, how='inner'
        ).dropna()


        self.combined_df['Category'] = pd.cut(
            self.combined_df['sentiment_score'], 
            bins=[-1, -0.1, 0.1, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )

    def _generate_visuals(self):
        """Generate correlation scatter plot and sentiment category bar chart."""

        plt.figure(figsize=(10, 6))
        sns.regplot(x='sentiment_score', y='Daily_Return', data=self.combined_df, 
                    scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
        plt.text(0.05, 0.95, f'Pearson Correlation: {self.correlation:.4f}', 
                 transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.title(f'Relationship: Daily Sentiment vs Stock Returns ({self.ticker})')
        plt.xlabel('Average Daily Sentiment Score')
        plt.ylabel('Daily Return (%)')
        plt.grid(True, alpha=0.3)
        plt.show()

        
        self.combined_df['Category'] = pd.cut(
            self.combined_df['sentiment_score'], 
            bins=[-1, -0.1, 0.1, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )
        avg_returns = self.combined_df.groupby('Category')['Daily_Return'].mean()
        
        plt.figure(figsize=(8, 5))
        colors = ['#e74c3c', '#95a5a6', '#2ecc71'] # Red, Gray, Green
        avg_returns.plot(kind='bar', color=colors)
        plt.title(f'Avg Daily Return per Sentiment Category ({self.ticker})')
        plt.ylabel('Average Daily Return (%)')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=0)
        plt.show()

    def _print_automated_insights(self):
        """Print automated interpretation of correlation results."""
        print("\n" + "="*50)
        print(f"AUTOMATED INSIGHTS: {self.ticker} SENTIMENT ANALYSIS")
        print("="*50)
        
        
        strength = "very weak" if abs(self.correlation) < 0.1 else "moderate" if abs(self.correlation) < 0.4 else "strong"
        direction = "positive" if self.correlation > 0 else "negative"
        
        print(f"1. Correlation Analysis: A {strength} {direction} correlation ({self.correlation:.4f}) was found.")
        
        # Categorical Insights
        cat_means = self.combined_df.groupby('Category')['Daily_Return'].mean()
        pos_ret = cat_means['Positive']
        neg_ret = cat_means['Negative']
        
        print(f"2. Return Impact: Days with Positive sentiment averaged {pos_ret:.4f}% returns.")
        print(f"3. Risk Context: Days with Negative sentiment averaged {neg_ret:.4f}% returns.")
        
        # Strategic Advice
        if pos_ret > neg_ret and self.correlation > 0:
            print("\nCONCLUSION: Sentiment is a directionally valid leading indicator for this asset.")
        else:
            print("\nCONCLUSION: Price action is likely driven by macro factors rather than headline sentiment.")
        print("="*50 + "\n")

    def run(self):
        """
        Execute the full sentiment-to-returns correlation pipeline.
        
        Returns:
            float: Pearson correlation coefficient between sentiment and returns
        """
        self._calculate_returns()
        self._align_and_aggregate()
        
        if self.combined_df is not None and not self.combined_df.empty:
           
            self.correlation = self.combined_df['sentiment_score'].corr(self.combined_df['Daily_Return'])
            
            self._print_automated_insights()
            self._generate_visuals()
            return self.correlation
        else:
            logger.error("Data alignment failed. Check ticker availability in the processed file.")
            return None