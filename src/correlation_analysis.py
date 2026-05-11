import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

logger = logging.getLogger(__name__)

class SentimentCorrelationAnalyzer:
    def __init__(self, ticker, stock_df):
        """
        :param ticker: Stock symbol
        :param stock_df: Dataframe from FinancialAnalyzer
        """
        self.ticker = ticker
        self.stock_df = stock_df.copy()
        self.combined_df = None
        self.correlation = None
        
        # Initialize VADER
        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def _calculate_returns(self):
        """Task 3.3: Compute daily percentage change in closing prices"""
        # (Close_t - Close_t-1) / Close_t-1 * 100
        self.stock_df['Daily_Return'] = self.stock_df['Close'].pct_change() * 100
        logger.info("Calculated daily stock returns.")

    def _calculate_sentiment(self, df):
        """Task 3.2: Apply VADER to assign numerical sentiment scores"""
        
        text_col = 'clean_headline' if 'clean_headline' in df.columns else 'headline'
        df['sentiment_score'] = df[text_col].apply(
            lambda x: self.sia.polarity_scores(str(x))['compound']
        )
        return df

    def _align_and_aggregate(self):
        """Task 3.1 & 3.4: Date alignment and daily sentiment averaging"""
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
        """Task 3.4: Separate visuals for scatter and bar chart"""

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

        # Visual 2: Categorical Bar Chart
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
        """Task 3.5: Automated interpretation of results"""
        print("\n" + "="*50)
        print(f"AUTOMATED INSIGHTS: {self.ticker} SENTIMENT ANALYSIS")
        print("="*50)
        
        # Interpret Correlation Strength
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
        """Execute the full encapsulated pipeline"""
        self._calculate_returns()
        self._align_and_aggregate()
        
        if self.combined_df is not None and not self.combined_df.empty:
            # Calculate Pearson correlation
            self.correlation = self.combined_df['sentiment_score'].corr(self.combined_df['Daily_Return'])
            
            self._print_automated_insights()
            self._generate_visuals()
            return self.correlation
        else:
            logger.error("Data alignment failed. Check ticker availability in the processed file.")
            return None