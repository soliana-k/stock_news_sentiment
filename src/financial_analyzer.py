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
    def __init__(self, ticker):
        self.ticker = ticker
        self.df=None
       
    
    def load_data(self):
        """Loads data with error handling for missing files."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, '..', 'data', 'Data', f'{self.ticker}.csv')

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
        try:
            close = self.df['Close']

            self.df['SMA_20'] = talib.SMA(close, timeperiod=20)
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
        pass

    def visualize(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Price & PyNance SMA
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.7)
        ax1.plot(self.df.index, self.df['PyN_SMA'], label='PyNance Simple Moving Average 20', color='orange')
        ax1.set_title(f"{self.ticker} Technical Dashboard")
        ax1.legend(loc='best')
        
        # Relative Strength Index
        ax2.plot(self.df.index, self.df['RSI'], color='purple', label='RSI')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Relative Strength Index')
        
        # MACD
        ax3.plot(self.df.index, self.df['MACD'], label='Moving Average Convergence Divergence', color='blue')
        ax3.plot(self.df.index, self.df['MACD_Signal'], label='Signal', color='red')
        ax3.bar(self.df.index, self.df['MACD'] - self.df['MACD_Signal'], color='gray', alpha=0.3)
        ax3.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
        

    def run(self):
        self.load_data()
        self.data_preparation()
        self.calculate_indicators()
        self.visualize()
