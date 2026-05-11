# pakistan-ecommerce
End-to-end e-commerce analytics: Python pipeline (ARIMA forecasting + anomaly detection) + 4-page Power BI dashboard analyzing 60M+ sales across Pakistan | 10K records
 🇵🇰 Pakistan E-Commerce Analytics Dashboard
 
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.1.0-150458?style=flat-square&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![Records](https://img.shields.io/badge/Dataset-10K%20Records-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
 
---
 
## 📋 Problem Statement
 
Pakistan's rapidly growing e-commerce sector generates massive volumes of transactional data, yet businesses lack the tools to:
 
1. **Monitor sales performance** across product categories and seasonal patterns in real time
2. **Detect anomalous transactions** that corrupt KPIs and lead to flawed business decisions
3. **Assess data pipeline quality** to ensure reliable reporting and trustworthy metrics
4. **Forecast future revenue** to support inventory planning, budgeting, and marketing strategy
> **This project solves all four problems** through an end-to-end analytics solution: a Python data pipeline for cleaning, anomaly detection, and forecasting — feeding a 4-page interactive Power BI dashboard that transforms raw transactional data into actionable business intelligence.
 
---
 
## 🎯 Objectives
 
| # | Objective | Delivered By |
|---|-----------|-------------|
| 1 | Identify top-performing categories and profitability drivers | Overview + Sales Analysis pages |
| 2 | Detect and classify transaction anomalies (IQR + Z-score + Percentile) | Data Quality page + Python pipeline |
| 3 | Score data pipeline health on a 0–100 scale | Data Quality Score (72.23/100) |
| 4 | Forecast next 3 months of sales using ARIMA and Linear Regression | Sales Forecast page |
| 5 | Enable interactive filtering by category and month | Sales Analysis slicers |
 
---
 
## 📊 Dashboard Pages
 
### Page 1 — Overview
<img width="830" height="514" alt="image" src="https://github.com/user-attachments/assets/2b09221e-3a22-4a5c-9b5d-fdcf6bf7bd0d" />
<img width="418" height="525" alt="image" src="https://github.com/user-attachments/assets/97de8fb2-7d49-4c57-bc17-b25fca6afeda" />



> *"What is the overall health of our e-commerce business?"*
 
| KPI | Value |
|-----|-------|
| Total Sales | 60.17M |
| Total Profit | 8.23M |
| Total Orders | 10K |
| Avg Customer Rating | 4.28 |
| Avg Profit Margin | 0.13 |
| Extreme Anomalies | 238 (2.38%) |
 
**Visuals:** KPI banner · Category performance table · Sales by Month line chart · Customer-by-Sales histogram · Outlier type distribution
 
---
 
### Page 2 — Sales Analysis
<img width="991" height="567" alt="image" src="https://github.com/user-attachments/assets/e56d4f91-89f9-4239-8bb1-abb7aee24ed3" />

> *"Which categories and months drive the most value, and how is the trend moving?"*
 
- Best category: **Electronics** (19.23M revenue)
- Best month: **November** (peak seasonal sales)
- Highest profit margin: **Sports** (0.14)
- Month-over-Month growth: **9.67%**
- Interactive slicers: Category × Month
**Visuals:** Profit margin by category bar chart · Sales by category bar chart · Monthly trend with 3-month moving average
 
---
 
### Page 3 — Data Quality
<img width="999" height="572" alt="image" src="https://github.com/user-attachments/assets/daa1a661-6963-4ead-bec7-f4108095b983" />

> *"Can we trust the data driving our decisions?"*
 
| Metric | Value |
|--------|-------|
| Clean Rows | 10K |
| Extreme Anomalies | 238 |
| Outlier % | 6.48% |
| Data Quality Score | 72.23 / 100 |
 
**Score formula:** `100 − outlier_penalty − data_risk_penalty`
 
**Visuals:** Category-by-outlier stacked bar · Data Risk Score by category · Data Quality Score by month · Pipeline Health gauge
 
---
 
### Page 4 — Sales Forecast
<img width="997" height="623" alt="image" src="https://github.com/user-attachments/assets/cb58c9f5-f082-45be-8807-09e45f5c5d33" />

> *"What will our sales look like over the next 3 months?"*
 
| Model | Jan 2026 | Feb 2026 | Mar 2026 | 3-Mo Total |
|-------|----------|----------|----------|------------|
| ARIMA | 5,439,410 | 5,408,338 | 5,415,491 | **16.26M** |
| Linear Regression | 4,449,420 | 4,389,217 | 4,329,014 | **13.17M** |
 
- Next month forecast: **5.44M**
- Growth status: **+12.2%**
- December actual: **5.30M**
**Visuals:** Sales trend + forecast line (Jan 2025 – Mar 2026) · 3-month detail table · ARIMA vs LR model comparison chart
 
---
 
## 🐍 Python Pipeline
 
### Architecture
 
```
Raw CSV (10K rows)
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Data Loading & Type Casting         │
│  2. Cleaning (duplicates, nulls, flags) │
│  3. Feature Engineering (margins, time) │
│  4. Outlier Detection (3-method union)  │
│  5. Category KPI Aggregation            │
│  6. Monthly Trend + Moving Average      │
│  7. Linear Regression (train/test)      │
│  8. ACF/PACF Analysis                   │
│  9. ARIMA / auto_arima Forecast         │
│ 10. Visualization Export (5 PNGs)       │
│ 11. CSV Export for Power BI (4 files)   │
└─────────────────────────────────────────┘
      │
      ▼
outputs/ (4 CSVs + 5 PNGs)  →  Power BI
```
 
### Outlier Detection Logic
 
Three methods combined via **union** — then priority-ranked:
 
```
Priority 1 (highest): Z-score > 3         → extreme_anomaly
Priority 2:           Price > P99, Qty=1  → data_issue  
Priority 3 (lowest):  Sales > P99, Qty≥3  → business_opportunity
Default:              All others           → normal
```
 
### Forecasting Models
 
**Linear Regression**
- Feature: time index `t`
- Split: 80% train / 20% test (`shuffle=False` — preserves time order)
- Evaluation: RMSE on held-out test set
**ARIMA (auto_arima)**
- Order selected automatically by AIC minimization
- `d=1` forced (trend differencing)
- 95% confidence interval on forecast
- Fallback to `ARIMA(1,1,1)` if `pmdarima` not installed
- Random walk check: warns if order is `(0,1,0)`
---
 
## 🗂️ Repository Structure
 
```
pakistan-ecommerce-powerbi/
│
├── 📁 data/
│   ├── raw/
│   │   └── pakistan_ecommerce_10000_rows.csv
│   └── processed/
│       └── cleaned_ecommerce.csv
│
├── 📁 notebooks/
│   └── analysis.py                  # Full pipeline (11 sections)
│
├── 📁 reports/
│   └── Pakistan_Ecommerce.pbix      # Power BI dashboard file
│
├── 📁 outputs/
│   ├── cleaned_ecommerce.csv
│   ├── category_kpi.csv
│   ├── monthly_sales.csv
│   ├── forecast_summary.csv
│   ├── linear_regression_metrics.csv
│   ├── acf_pacf.png
│   ├── monthly_trend.png
│   ├── arima_forecast.png
│   ├── category_kpi.png
│   └── outlier_distribution.png
│
├── 📁 screenshots/
│   ├── 01_overview.png
│   ├── 02_sales_analysis.png
│   ├── 03_data_quality.png
│   └── 04_sales_forecast.png
│
├── 📁 docs/
│   ├── dax_measures.md
│   └── data_dictionary.md
│
├── requirements.txt
└── README.md
```
 
---
 
## ⚙️ Setup & Installation
 
### Prerequisites
- Python 3.9+
- Power BI Desktop (for `.pbix` file)
### Install dependencies
 
```bash
git clone https://github.com/YOUR_USERNAME/pakistan-ecommerce-powerbi.git
cd pakistan-ecommerce-powerbi
pip install -r requirements.txt
```
 
### Run the pipeline
 
```bash
python notebooks/analysis.py
```
 
All outputs will be saved to the `outputs/` folder automatically.
 
### Open the dashboard
 
1. Open Power BI Desktop
2. File → Open → `reports/Pakistan_Ecommerce.pbix`
3. Update data source path to `outputs/cleaned_ecommerce.csv`
4. Click **Refresh**
---
 
## 📦 requirements.txt
 
```
pandas==2.1.0
numpy==1.24.0
matplotlib==3.7.0
seaborn==0.12.0
scipy==1.11.0
scikit-learn==1.3.0
statsmodels==0.14.0
pmdarima==2.0.3
```
 
---
 
## 🔢 Key DAX Measures
 
```dax
-- 3-Month Moving Average
Moving_Avg_3 =
AVERAGEX(
    DATESINPERIOD('Date'[Date], LASTDATE('Date'[Date]), -3, MONTH),
    [Total Sales]
)
 
-- Month-over-Month Growth %
MoM_Growth% =
DIVIDE(
    [Current Month Sales] - [Previous Month Sales],
    [Previous Month Sales]
)
 
-- Safe Profit Margin (handles zero sales)
Profit_Margin =
DIVIDE([Total Profit], [Total Sales], BLANK())
 
-- Data Quality Score
Data_Quality_Score =
100 - [Outlier_Penalty] - [Risk_Penalty]
```
 
---
 
## 📈 Results Summary
 
| Metric | Value |
|--------|-------|
| Total Sales | 60.17M |
| Total Profit | 8.23M |
| Avg Profit Margin | 13% |
| Total Orders | 10,000 |
| Avg Customer Rating | 4.28 / 5 |
| Extreme Anomalies Detected | 238 (2.38%) |
| Data Quality Score | 72.23 / 100 |
| Best Category (Revenue) | Electronics — 19.23M |
| Best Category (Margin) | Sports — 14% |
| Peak Sales Month | November |
| ARIMA 3-Month Forecast | 16.26M |
| LR 3-Month Forecast | 13.17M |
| MoM Growth Rate | 9.67% |
 
---
 
## 🧠 Skills Demonstrated
 
- **Data Engineering** — ETL pipeline, type casting, null handling, deduplication
- **Statistical Analysis** — IQR, Z-score, percentile-based anomaly detection
- **Machine Learning** — Linear Regression with train/test evaluation (RMSE)
- **Time Series Forecasting** — ARIMA / auto_arima with AIC model selection and confidence intervals
- **Business Intelligence** — Power BI data modeling, DAX measures, interactive slicers
- **Data Visualization** — Matplotlib, Seaborn, Power BI custom visuals
- **Data Quality Engineering** — Pipeline health scoring, outlier classification, risk scoring
---
 
## 👤 Author
 
**Laiba**
- 📧 laibaaslam383@gmail.com
- 💼 [LinkedIn](www.linkedin.com/in/laiba-aslam26)
- 🐙 [GitHub](https://github.com/laiba26-aslam)
---
 
## 📄 License
 
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
 
---
 
*Built as a data analytics portfolio project demonstrating end-to-end business intelligence development.*
