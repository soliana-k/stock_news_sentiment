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
        try:
            self.df['Daily_Return'] = self.df['Close'].pct_change()
            self.df['Cumulative_Return'] = (1 + self.df['Daily_Return']).cumprod() - 1
            self.df['Volatility'] = self.df['Daily_Return'].rolling(window=20).std()
            self.df['Signal_Score'] = 0 
            self.df['PyN_OBV'] = pn_tech.obv(self.df.Close, self.df.Volume)
            upper, mid, lower = pn_tech.bollinger(self.df.Close, period=20, width=2)
            self.df['BB_Upper'], self.df['BB_Mid'], self.df['BB_Lower'] = upper, mid, lower
            self.df.loc[self.df['Close'] > self.df['SMA_20'], 'Signal_Score'] += 1 
            self.df.loc[self.df['Close'] < self.df['SMA_20'], 'Signal_Score'] -= 1
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
            print(self.df['Technical_Verdict'])
        except Exception as e:
            logger.error(f"Error in finance_metrics: {e}")


        

    def visualize(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        
        ax1.plot(self.df.index, self.df['Close'], label='Close Price', alpha=0.7)
        ax1.plot(self.df.index, self.df['PyN_SMA'], label='PyNance Simple Moving Average 20', color='orange')
        ax1.set_title(f"{self.ticker} Technical Dashboard")
        ax1.legend(loc='best')
        
        
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
        self.load_data()
        self.data_preparation()
        self.calculate_indicators()
        
        self.visualize()
        self.finance_metrics()
        self.output_insights()
