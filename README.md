# Financial News Sentiment & Stock Price Correlation (FNSPID Analysis)

## 🏢 Business Objective

As a Data Analyst at **Nova Financial Solutions**, the primary objective of this project is to enhance predictive analytics capabilities to boost financial forecasting accuracy and operational efficiency. This is achieved through a two-fold analysis of the **Financial News and Stock Price Integration Dataset (FNSPID)**:

1. **Sentiment Analysis:** Utilizing Natural Language Processing (NLP) to quantify the tone of financial headlines and understand the emotional context surrounding specific stock symbols.
2. **Correlation Analysis:** Establishing statistical links between news sentiment and daily stock price movements to track the impact of media coverage on market performance.

The ultimate goal is to provide **Nova Financial Solutions** with actionable investment strategies that utilize news sentiment as a predictive tool for stock market trends.

---

## 📂 Directory Structure

```text
stock_news_sentiment/
├── .github/
│   └── workflows/
│       └── unittests.yml      
|
├── notebooks/
|  └── nlp/
│       └── sentiment_analysis.ipnb 
│   ├── AAPL.ipynb        
│   └── AMZN.ipynb   
├── src/
│   ├── __init__.py
│   └── financial_analyzer.py   
├── .gitignore                 
├── requirements.txt           
└── README.md                   

```

---

## 📊 Dataset Specifications

### 1. Financial News (Qualitative)

* **Headline:** Article titles detailing key financial actions (earnings, price targets, etc.).
* **Publisher:** Source of the news, used to analyze organizational contribution patterns.
* **Date:** Publication timestamp normalized to **UTC-4/US-Eastern**.
* **Stock:** Ticker symbols used as the primary key for data integration.

### 2. Historical Stock Prices (Quantitative)

* **Source:** Sourced via `YFinance`.
* **Metrics:** Daily Open, High, Low, Close, and Volume.
* **Adjusted Close:** Primary metric for return calculations to account for corporate actions (dividends/splits).

---

## 🛠 Project Roadmap & Methodology

### Task 1: Exploratory Data Analysis (EDA)

* **Headline Length Analysis:** Identified a right-skewed distribution ($Mean > Median$), revealing a long tail of descriptive news summaries vs. concise ticker alerts.
* **Publication Trends:** Mapped news frequency over a decade, identifying major outliers such as the **2020 COVID-19 volatility spike**.
* **Topic Modeling:** Employed **Latent Dirichlet Allocation (LDA)** to categorize 1.4M headlines into 6 operational domains (Earnings, Market Movers, Analyst Ratings, etc.).

### Task 2: Quantitative Technical Analysis

* **Indicator Implementation:** Leveraged `TA-Lib` and `PyNance` to compute **SMA, EMA, RSI, and MACD**.
* **Sentiment Classification:** Developed a modular `FinancialAnalyzer` class to generate a **Technical Verdict** (Strong Bullish to Strong Bearish) based on indicator crossovers and momentum.

### Task 3: Correlation & Predictive Strategy (In Progress)

* Aligning daily sentiment scores with stock returns to calculate Pearson correlation coefficients and predictive beta values.

---

## ⚙️ CI/CD & Pipeline Automation

This repository utilizes modern DevOps practices to ensure code reliability:

* **Continuous Integration:** GitHub Actions are configured to run automated linting on every push.
* **Dependency Management:** Strict version control via `requirements.txt` to ensure environment parity across different systems.
* **Modular Architecture:** Logic is encapsulated in a class-based structure for easy integration into automated reporting dashboards.

---

## 🚀 Installation & Usage

### 1. Prerequisites

* Python 3.9+
* **TA-Lib C-Library** 

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/soliana-k/stock_news_sentiment.git
cd stock_news_sentiment

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Execution

```python
from src.financial_analyzer import FinancialAnalyzer

# Example: Analysis for a specific ticker
analyzer = FinancialAnalyzer('AAPL')
analyzer.run()

```

---

## 📦 Core Dependencies

* `pandas` & `numpy` - Data Manipulation
* `TA-Lib` & `pynance` - Financial Indicators
* `scikit-learn` - Topic Modeling
* `matplotlib` & `seaborn` - Data Visualization

---

## 👨‍💻 Author

**Kalkidan Kassahun** *Data Analyst, Nova Financial Solutions* *Software Engineering & Financial Engineering Specialist*
