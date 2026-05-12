import talib
import pynance.tech as pn_tech
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """
    A modular class for loading, cleaning, and analyzing stock price data.
    
    It computes technical indicators using TA-Lib and PyNance, generates 
    financial metrics, and produces visualizations and investment insights.
    
    Attributes:
        ticker (str): Stock symbol to analyze (e.g., 'GOOG', 'AAPL')
        df (pd.DataFrame): Processed DataFrame containing price data and indicators
    """
    def __init__(self, ticker):
        """
        Initialize the FinancialAnalyzer with a stock ticker.
        
        Args:
            ticker (str): Stock symbol (e.g., 'GOOG')
        """
        self.ticker = ticker
        self.df=None
       
    
    def load_data(self):
        """
        Load stock price data from CSV with robust error handling.
        
        Expected file location: ../data/raw/{ticker}.csv
        
        Raises:
            FileNotFoundError: If the data file does not exist
            Exception: For other loading errors
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, '..', 'data', 'raw', f'{self.ticker}.csv')

            file_path = os.path.normpath(file_path)
    
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Could not find the file at: {file_path}")

            self.df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data for {self.ticker} Ticker")
            self.df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def data_preparation(self):
        """
        Clean and standardize the loaded stock data.
        
        Operations performed:
        - Convert 'Date' to datetime and set as index
        - Standardize column names (capitalize)
        - Convert price/volume columns to float
        - Forward-fill missing values and drop any remaining NaNs
        """
        try:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)
            self.df.sort_index(inplace=True)
            
            self.df.columns = [col.capitalize() for col in self.df.columns]
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(float)

            self.df = self.df.ffill().dropna()
            logger.info(f"Successfully cleaned and standardized data for {self.ticker}")
        except Exception as e:
            logger.error(f"Failed to clean and standardize data for {self.ticker}: {e}")
            raise
        


    def calculate_indicators(self):
        """
        Calculate technical indicators using TA-Lib and PyNance.
        
        Adds the following columns to self.df:
            - SMA_20, EMA_20
            - RSI
            - MACD, MACD_Signal, MACD_Hist
            - BB_Upper, BB_Mid, BB_Lower (Bollinger Bands)
        """
        try:
            close = self.df['Close']

            self.df['SMA_20'] = pn_tech.sma(close)
            self.df['EMA_20'] = talib.EMA(close, timeperiod=20)
            
            self.df['RSI'] = talib.RSI(close, timeperiod=14)
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = macd, signal, hist
            
            
            self.df['PyN_SMA'] = pn_tech.sma(self.df['Close'], period=20)
            print(self.df)
            logger.info("Successfully used PyNance for SMA")
            
            logger.info(f"Technical indicators calculated for {self.ticker}")
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        

    
    

    def finance_metrics(self):
        """
        Calculate financial metrics and generate a composite technical verdict.
        
        Adds the following columns:
            - Daily_Return, Cumulative_Return, Volatility
            - Signal_Score and Technical_Verdict (Strong Bullish → Strong Bearish)
        """
        try:
            self.df['Daily_Return'] = self.df['Close'].pct_change()
            self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod() - 1
            self.df['Volatility'] = self.df['Daily_Return'].rolling(window=20).std()
            self.df['Growth_Vol'] = self.df['Daily_Return'].rolling(window=60).std()
            self.df['Signal_Score'] = 0 
            self.df['PyN_OBV'] = talib.OBV(self.df.Close, self.df.Volume)
            

            upper, middle, lower = talib.BBANDS(
                self.df['Close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            
            self.df['BB_Upper'] = upper
            self.df['BB_Mid'] = middle
            self.df['BB_Lower'] = lower
            

            self.df.loc[self.df['Close'] > self.df['SMA_20'].fillna(0), 'Signal_Score'] += 1 
            self.df.loc[self.df['Close'] < self.df['SMA_20'].fillna(float('inf')), 'Signal_Score'] -= 1
            
            self.df.loc[self.df['RSI'] < 30, 'Signal_Score'] += 1
            self.df.loc[self.df['RSI'] > 70, 'Signal_Score'] -= 1
            
            self.df.loc[self.df['MACD'] > self.df['MACD_Signal'], 'Signal_Score'] += 1
            self.df.loc[self.df['MACD'] < self.df['MACD_Signal'], 'Signal_Score'] -= 1

            def classify_verdict(score):
                if score >= 2: return 'Strong Bullish'
                elif score == 1: return 'Bullish'
                elif score == -1: return 'Bearish'
                elif score <= -2: return 'Strong Bearish'
                return 'Neutral'

            self.df['Technical_Verdict'] = self.df['Signal_Score'].apply(classify_verdict)
            
            logger.info(f"Technical sentiment and metrics calculated for {self.ticker}")
            print(self.df['Technical_Verdict'].tail())

        except Exception as e:
            logger.error(f"Error in finance_metrics: {e}")


        

    def visualize(self):
        """Generate key technical analysis visualizations."""
    # 1. Price and SMA Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.7)
        plt.plot(self.df.index, self.df['PyN_SMA'], label='PyNance SMA 20', color='orange')
        plt.title(f"{self.ticker} Price Action & Moving Average")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    # 2. RSI Visualization
        plt.figure(figsize=(12, 4))
        plt.plot(self.df.index, self.df['RSI'], color='purple', label='RSI')
        plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.title(f"{self.ticker} Relative Strength Index")
        plt.ylabel('RSI Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    # 3. MACD Visualization
        plt.figure(figsize=(12, 5))
        plt.plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
        plt.plot(self.df.index, self.df['MACD_Signal'], label='Signal', color='red')
        plt.bar(self.df.index, self.df['MACD'] - self.df['MACD_Signal'], color='gray', alpha=0.3, label='Histogram')
        plt.title(f"{self.ticker} MACD Analysis")
        plt.xlabel('Date')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.show()

    #4 Price and Bollinger charts    
        plt.figure(figsize=(10, 5))
        plt.plot(self.df.index, self.df['Close'], label='Close Price', color='blue')
        plt.plot(self.df.index, self.df['BB_Upper'], 'r--', alpha=0.4, label='Upper Band')
        plt.plot(self.df.index, self.df['BB_Lower'], 'g--', alpha=0.4, label='Lower Band')
        plt.fill_between(self.df.index, self.df['BB_Lower'], self.df['BB_Upper'], alpha=0.1)
        plt.title(f"{self.ticker} Price & Bollinger Bands")
        plt.legend()
        plt.show()

    #5. Growth Volatility
        plt.figure(figsize=(10, 4))
        plt.plot(self.df.index, self.df['Growth_Vol'], color='teal')
        plt.title(f"{self.ticker} Growth Volatility (Stability Check)"); plt.show()

    def output_insights(self):
        """Generates a professional summary of technical and financial status."""
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        volatility_status = "High" if latest['Volatility'] > self.df['Volatility'].mean() else "Low"
        
        print("-" * 50)
        print(f"INVESTMENT INSIGHTS FOR: {self.ticker}")
        print("-" * 50)
        
        
        print(f"Current Price: ${latest['Close']:.2f} ({price_change:+.2f}%)")
        print(f"Technical Verdict: {latest['Technical_Verdict'].upper()}")
        
        print(f"\n--- Indicator Breakdown ---")
        print(f"• RSI: {latest['RSI']:.2f} ({'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'})")
        print(f"• Trend: {'Above' if latest['Close'] > latest['SMA_20'] else 'Below'} 20-Day SMA")
        print(f"• MACD: {'Bullish Crossover' if latest['MACD_Hist'] > 0 else 'Bearish Crossover'}")
        
        
        print(f"\n--- Risk Profile ---")
        print(f"• 20-Day Rolling Volatility: {latest['Volatility']:.4f} ({volatility_status})")
        print(f"• Cumulative Return (Period): {latest['Cumulative_Return']*100:.2f}%")
        print("-" * 50 + "\n")


    def run(self):
        """
        Execute the full analysis pipeline for the ticker.
        """

        self.load_data()
        self.data_preparation()
        self.calculate_indicators()
        self.finance_metrics()
        self.visualize()
        self.output_insights()
