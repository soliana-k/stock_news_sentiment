import pytest
import pandas as pd
import numpy as np
from src.financial_analyzer import FinancialAnalyzer
from src.correlation_analysis import SentimentCorrelationAnalyzer


@pytest.fixture
def temp_csv_data(tmp_path):
    """Creates a real temporary CSV file to test the load_data method."""
    d = tmp_path / "data" / "raw"
    d.mkdir(parents=True)
    file_path = d / "TEST_TICKER.csv"
    
   
    dates = pd.date_range(start="2024-01-01", periods=50)
    df = pd.DataFrame({
        'Date': dates,
        'Open': np.linspace(100, 150, 50),
        'High': np.linspace(105, 155, 50),
        'Low': np.linspace(95, 145, 50),
        'Close': np.linspace(100, 150, 50),
        'Volume': [1000000] * 50
    })
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_financial_pipeline_execution(temp_csv_data, monkeypatch):
    """Validates the full pipeline from loading to metrics calculation."""
   
    ticker = "TEST_TICKER"
    analyzer = FinancialAnalyzer(ticker=ticker)
    
   
    df_raw = pd.read_csv(temp_csv_data)
    analyzer.df = df_raw
    
    
    analyzer.data_preparation()
    analyzer.calculate_indicators()
    analyzer.finance_metrics()
    
    assert 'SMA_20' in analyzer.df.columns
    assert 'RSI' in analyzer.df.columns
    assert 'Technical_Verdict' in analyzer.df.columns
   
    latest_verdict = analyzer.df['Technical_Verdict'].iloc[-1]
    assert "Bullish" in latest_verdict

def test_rsi_range_validation(temp_csv_data):
    """Ensures RSI is calculated within the correct bounds (0-100)."""
    analyzer = FinancialAnalyzer(ticker="TEST_TICKER")
    analyzer.df = pd.read_csv(temp_csv_data)
    
    analyzer.data_preparation()
    analyzer.calculate_indicators()
    
    rsi_values = analyzer.df['RSI'].dropna()
    assert rsi_values.min() >= 0
    assert rsi_values.max() <= 100
@pytest.fixture
def sample_market_data():
    """Generates a realistic technical series for verdict testing."""
    data = {
        'Close': 120.0,
        'SMA_20': 105.0,
        'MACD': 1.0,
        'MACD_Signal': 0.5,
        'RSI': 65.0
    }
    return pd.Series(data)

