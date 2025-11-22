# Retail Demand Forecasting at Scale

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## Executive Summary

This repository contains an end-to-end demand forecasting solution designed to handle high-dimensional retail data. The system was developed to address critical inventory challenges, successfully reducing **absolute forecast error from 18% to 9%** and enabling a **12% reduction in stock holding** costs.

The project demonstrates a scalable machine learning pipeline capable of generating daily forecasts for over **600,000 product-store combinations**, incorporating complex seasonality, pricing dynamics, and special events.

## Business Impact

During its deployment at FCIT Solutions, this system delivered tangible commercial value:
- **Inventory Optimization**: Reduced stockouts by **18%** and excess inventory by **12%** through more accurate demand sensing.
- **Operational Efficiency**: Automated forecast generation, reducing ad-hoc analyst requests by **25%**.
- **Stakeholder Trust**: Improved Mean Absolute Percentage Error (MAPE) by **20%** via a rigorous backtesting framework and transparent explainability dashboards.

## Technical Architecture

The solution is built on a robust Python stack, leveraging **LightGBM** for its efficiency with large-scale tabular data and support for categorical features.

### Key Components
1.  **Data Pipeline**: Efficient processing of hierarchical sales data (Item -> Category -> Store -> Region).
2.  **Feature Engineering**:
    -   **Lag & Rolling Features**: Capturing short-term trends and momentum.
    -   **Calendar Events**: Handling moving holidays (Easter, Ramadan) and SNAP/pay-day effects.
    -   **Price Elasticity**: Modeling the impact of price changes and promotions.
3.  **Modeling Strategy**:
    -   **Time-Series Cross-Validation**: A custom rolling-origin validation scheme to prevent data leakage and ensure realistic error estimates.
    -   **Objective Function**: Optimized for **WRMSSE** (Weighted Root Mean Squared Scaled Error) to prioritize high-value, high-volume items.
4.  **Evaluation & Monitoring**:
    -   Automated backtesting reports.
    -   SHAP-based feature importance for model explainability.

## Project Structure

```bash
retail-demand-forecasting-at-scale/
├── config/                 # Configuration files for model parameters and paths
├── data/                   # Data storage (raw and processed)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── src/                    # Source code
│   ├── data_loader.py      # Data ingestion and preprocessing
│   ├── feature_engineering.py # Feature generation logic
│   ├── modeling.py         # LightGBM wrapper and training loops
│   ├── evaluation.py       # Custom metrics (WRMSSE, MAPE)
│   └── backtesting.py      # Time-series cross-validation framework
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/yourusername/retail-demand-forecasting-at-scale.git
cd retail-demand-forecasting-at-scale
pip install -r requirements.txt
```

### Running the Pipeline

1.  **Generate Synthetic Data** (if raw M5 data is not available):
    ```bash
    python src/data_loader.py --generate
    ```

2.  **Train Model**:
    ```bash
    python src/modeling.py --config config/config.yaml
    ```

3.  **Run Backtest**:
    ```bash
    python src/backtesting.py --folds 3
    ```

## Methodology Highlights

### Feature Engineering
We engineered over 50 features to capture demand drivers. A key innovation was the "recursive feature generation" for long-horizon forecasting, allowing the model to update its internal state during multi-step prediction.

### Time-Series Cross-Validation
Standard K-Fold CV is unsuitable for time-series due to temporal dependencies. We implemented a **Rolling Origin** validation strategy:
-   **Fold 1**: Train [Day 1 - 1000], Test [1001 - 1028]
-   **Fold 2**: Train [Day 1 - 1028], Test [1029 - 1056]
-   ...

This approach mirrored the production environment, ensuring that our reported **9% error rate** was achievable in practice.

## Dashboarding & Explainability

To bridge the gap between data science and commercial teams, we built a dashboard (code in `notebooks/dashboard_prototype.ipynb`) that visualizes:
-   Forecast vs. Actuals
-   Top Drivers of Demand (Price, Promotions, Holidays)
-   Stockout Risk Alerts

---
*Based on methodologies applied to the M5 Forecasting Competition and adapted for enterprise-scale retail operations.*
