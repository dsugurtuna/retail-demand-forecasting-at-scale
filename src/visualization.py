import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Dashboard:
    def __init__(self):
        sns.set_style("whitegrid")

    def plot_forecast_vs_actual(self, actuals, forecast, title="Forecast vs Actuals"):
        plt.figure(figsize=(15, 6))
        plt.plot(actuals, label='Actual Sales', alpha=0.7)
        plt.plot(forecast, label='Forecast', alpha=0.7, linestyle='--')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_feature_importance(self, importance_df, top_n=20):
        plt.figure(figsize=(10, 12))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title(f"Top {top_n} Feature Importance (LightGBM Gain)")
        plt.tight_layout()
        plt.show()

    def plot_stockout_risk(self, forecast_df, threshold=5):
        """
        Visualizes items with predicted low stock coverage.
        """
        risk_items = forecast_df[forecast_df['predicted_stock'] < threshold]
        # ... implementation details ...
        pass
