import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.config = {
            'feature_engineering': {
                'lags': [1, 2],
                'rolling_windows': [2],
                'categorical_features': []
            }
        }
        self.fe = FeatureEngineer(self.config)
        
        # Create dummy data
        dates = pd.date_range(start='2023-01-01', periods=10)
        self.df = pd.DataFrame({
            'id': ['item_1'] * 10,
            'date': dates,
            'sales': np.arange(10),
            'sell_price': [10] * 10
        })

    def test_lag_features(self):
        df = self.fe._create_lag_features(self.df)
        # Lag 1 of sales=1 (at index 1) should be 0 (from index 0)
        self.assertEqual(df.iloc[1]['sales_lag_1'], 0)
        
    def test_date_features(self):
        df = self.fe._create_date_features(self.df)
        self.assertIn('day_of_week', df.columns)
        self.assertEqual(df.iloc[0]['day_of_week'], 6) # 2023-01-01 is Sunday (6)

if __name__ == '__main__':
    unittest.main()
