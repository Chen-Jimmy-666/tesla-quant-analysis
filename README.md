# TSLA Quant Analysis 

This is my first end-to-end project built from scratch while learning quantitative finance.

At the current stage, the project focuses on:  
**data collection, basic metrics calculation, and visual analysis**

Future extensions will include:
- Simple trading strategies  
- Backtesting framework  
- Factor-based research  

👉 This repository will be continuously updated to reflect my learning progress and growing understanding of quantitative research.

---

## 📊 Project Overview

Using TSLA (Tesla) price data over the past 5 years, with comparison to the S&P 500, this project covers:

### Data Processing
- Retrieved historical market data via Yahoo Finance API  
- Cleaned and aligned data to ensure comparability between TSLA and the index  

### Return Analysis
- Daily returns / log returns  
- Cumulative return comparison (TSLA vs S&P 500)  
- Annual returns  

### Risk Analysis
- Annualized volatility (based on log returns)  
- 30-day rolling volatility  
- Maximum drawdown  
- Sharpe ratio (risk-free rate = 0)  

### Visualizations (7 charts)
- Price + moving averages + volume  
- Return distribution (including skewness & kurtosis)  
- Cumulative return comparison  
- Volatility trends  
- Annual performance  
- Maximum drawdown  
- Combined dashboard  

---

## 📈 Key Findings (Current Version)

- TSLA 5-year CAGR ≈ 10.4%, slightly lower than the S&P 500  
- Annualized volatility ≈ 58.8%, about 3.5× the market  
- Maximum drawdown reached -73.6%  
- Sharpe ratio ≈ 0.17 → relatively poor risk-adjusted return  

👉 Current takeaway:  
TSLA behaves as a **high-volatility asset with insufficient risk compensation**

---

## 🛠 Tech Stack

- Python (core language)  
- Pandas / NumPy (data processing)  
- Matplotlib / Seaborn (visualization)  
- Requests (data retrieval)  

---

## 🤖 Development Approach

This project was developed with assistance from Claude Code for:
- Structuring code  
- Debugging issues  
- Improving development efficiency  

👉 The focus remains on **understanding the logic behind each step**, rather than simply relying on tools.

---

## 💡 Project Objective

The goal of this project is to build an intuitive understanding of quantitative analysis starting from a single asset, rather than just computing financial indicators.

Instead of focusing only on results, I aim to:
- Transform raw data into interpretable investment insights  
- Understand the financial meaning and limitations of each metric  
- Develop a basic framework for evaluating risk and return  

👉 This project will continue to evolve into strategy development, backtesting, and factor research, forming a long-term learning path into quantitative finance.
