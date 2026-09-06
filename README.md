# 📊 AI Sales Forecasting & Power BI Dashboard

An end-to-end sales analytics and AI-based forecasting project built using Python, Machine Learning, and Microsoft Power BI.

## 🚀 Project Overview

This project analyzes historical sales data, performs data cleaning and exploratory analysis, engineers useful features, trains machine learning models, generates future sales forecasts, and presents the results through interactive Streamlit and Power BI dashboards.

The project combines **Machine Learning and Business Intelligence** to understand historical sales performance and forecast future sales.

### Key Components

- 🧹 Data Cleaning
- 🔎 Exploratory Data Analysis
- ⚙️ Feature Engineering
- 🤖 Machine Learning
- 📈 Model Evaluation
- 🔮 Future Sales Forecasting
- 🌐 Streamlit Dashboard
- 📊 Power BI Dashboard
- 💡 Business Insights

## 🎯 Project Objectives

The main objectives of this project are:

1. Analyze historical sales performance.
2. Identify trends and patterns in sales.
3. Analyze sales performance across categories and regions.
4. Engineer useful features for machine learning.
5. Train and evaluate machine learning models.
6. Forecast future sales.
7. Build interactive dashboards for business analysis.
8. Present actionable business insights using data visualization.

### 🔄 Project Workflow

Historical Sales Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Sales Forecasting
        ↓
Future Forecast
        ↓
Streamlit Dashboard
        ↓
Power BI Dashboard
        ↓
Business Insights

### 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and machine learning |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and model evaluation |
| XGBoost | Machine learning / forecasting |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualizations |
| Streamlit | Interactive web dashboard |
| Joblib | Model saving and loading |
| Power BI | Business intelligence and dashboards |
| Git & GitHub | Version control |

## 📂 Dataset

The project uses sales transaction data containing information such as:

- Order ID
- Order Date
- Customer ID
- Product
- Category
- Region
- Quantity
- Unit Price
- Discount
- Sales
- Profit

The dataset contains approximately 12,000 sales transactions.

## 📁 Project Structure

AI-Sales-Forecasting-PowerBI/
│
├── app.py
│
├── data_cleaning.py
├── eda.py
├── feature_engineering.py
├── train_model.py
├── model_evaluation.py
├── sales_forecasting.py
├── forecast_model.py
├── future_forecast.py
├── forecast_visualization.py
│
├── data/
│   ├── cleaned_sales.csv
│   ├── forecast_predictions.csv
│   ├── forecasting_sales_data.csv
│   ├── future_sales_forecast.csv
│   ├── ml_sales_data.csv
│   ├── model_results.csv
│   └── prediction_results.csv
│
├── models/
│   ├── sales_forecasting_model.pkl
│   └── sales_prediction_model.pkl
│
├── output/
│
├── PowerBI/
│   └── AI_Sales_Forecast_Final_v1.pbix
│
├── Screenshots/
│   ├── page 1.png
│   ├── page 2.png
│   ├── page 3.png
│   └── page 4.png
│
├── requirements.txt
├── .gitignore
└── README.md

## 📊 Power BI Dashboard

The dashboard provides an interactive view of sales performance, profitability, regional analysis, category analysis, AI forecasting, and business insights.

## Page 1 — Sales Overview

<img src="Screenshots/page%201.png" width="100%">

## Page 2 — Sales Analysis

<img src="Screenshots/page%202.png" width="100%">

## Page 3 — Business Performance

<img src="Screenshots/page%203.png" width="100%">

## Page 4 — Business Insights

<img src="Screenshots/page%204.png" width="100%">


## 🤖 Machine Learning Pipeline

The Python modules implement the following stages:

### 1. Data Cleaning

`data_cleaning.py`

Cleans and prepares the sales data for further analysis.

### 2. Exploratory Data Analysis

`eda.py`

Analyzes sales trends, distributions, categories, regions, and other patterns.

### 3. Feature Engineering

`feature_engineering.py`

Creates useful features from the sales dataset for machine learning.

### 4. Model Training

`train_model.py`

Trains the machine learning models using the prepared dataset.

### 5. Model Evaluation

`model_evaluation.py`

Evaluates the trained models using appropriate performance metrics.

### 6. Sales Forecasting

`sales_forecasting.py`

Generates sales predictions using the trained forecasting model.

### 7. Future Forecast

`future_forecast.py`

Generates future sales predictions.

### 8. Forecast Visualization

`forecast_visualization.py`

Visualizes historical and forecasted sales.

## 🌐 Streamlit Dashboard

The project includes an interactive Streamlit dashboard for analyzing historical sales and future forecasts.

Run the application using:

python -m streamlit run app.py

## 🚀 Live Dashboard

[Open the AI Sales Forecasting Dashboard](https://ai-sales-forecasting-powerbi.streamlit.app/)

## 💡 Business Insights

The project provides insights into:

- Category-wise sales performance
- Regional sales performance
- Monthly sales trends
- Profitability
- Future sales forecasts
- Forecast trends
- Model prediction performance

These insights can support data-driven business decisions related to sales planning, regional strategy, and future demand.

## 📈 Key Performance Indicators

The dashboard tracks the following major business KPIs:

| KPI | Description |
|---|---|
| Total Sales | Total revenue generated |
| Total Profit | Total profit generated |
| Profit Margin | Profit as a percentage of sales |
| Total Orders | Number of orders |
| Quantity Sold | Total quantity sold |
| Average Order Value | Average sales value per order |

## ⚙️ Installation

### 1. Clone the repository

git clone https://github.com/ritikraj77320/AI-Sales-Forecasting-PowerBI.git

cd AI-Sales-Forecasting-PowerBI

pip install -r requirements.txt

## ▶️ Running the Dashboard

Run the Streamlit application using:

python -m streamlit run app.py


## 🔮 Future Improvements

Possible future improvements include:

- 🌐 Online deployment of the Streamlit dashboard
- 🔄 Automated data updates
- 📊 Automated Power BI refresh
- 🤖 Experimentation with advanced forecasting algorithms
- 📦 Inventory demand forecasting
- 🚨 Sales anomaly detection
- ☁️ Cloud-based data integration
- 📈 Real-time forecasting
- 🗄️ Database integration

## 👨‍💻 Author

### Ritik Raj

AI/ML & Data Analytics Enthusiast

GitHub: https://github.com/ritikraj77320

---

⭐ If you found this project useful, consider giving the repository a star!