import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.lags = config['feature_engineering']['lags']
        self.rolling_windows = config['feature_engineering']['rolling_windows']
        
    def generate_features(self, df):
        """
        Main method to generate all features.
        """
        print("Generating features...")
        df = self._create_date_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        df = self._create_price_features(df)
        
        # Drop rows with NaNs created by lags (optional, or handle in training)
        # df = df.dropna() 
        
        return df
    
    def _create_date_features(self, df):
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df

    def _create_lag_features(self, df):
        # Sort by id and date to ensure correct shifting
        df = df.sort_values(['id', 'date'])
        
        for lag in self.lags:
            df[f'sales_lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
            
        return df

    def _create_rolling_features(self, df):
        # Rolling features usually applied on a shifted series to avoid leakage
        # We use the first lag (e.g., 28 days for M5) as the base for rolling
        base_lag = 28 
        
        for window in self.rolling_windows:
            df[f'sales_roll_mean_{window}'] = df.groupby('id')['sales'].transform(
                lambda x: x.shift(base_lag).rolling(window).mean())
            
            df[f'sales_roll_std_{window}'] = df.groupby('id')['sales'].transform(
                lambda x: x.shift(base_lag).rolling(window).std())
                
        return df

    def _create_price_features(self, df):
        # Price momentum: current price / average price of last 4 weeks
        df['price_momentum'] = df['sell_price'] / df.groupby('id')['sell_price'].transform(
            lambda x: x.shift(1).rolling(4).mean())
            
        df['price_momentum'] = df['price_momentum'].fillna(1.0)
        return df

    def encode_categorical(self, df):
        # Simple Label Encoding for LightGBM
        cat_feats = self.config['feature_engineering']['categorical_features']
        for col in cat_feats:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df
