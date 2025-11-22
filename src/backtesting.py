import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer
from src.evaluation import Evaluator
import yaml
import os

class Backtester:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.loader = DataLoader(self.config)
        self.fe = FeatureEngineer(self.config)
        self.trainer = ModelTrainer(self.config)
        
    def run_backtest(self):
        # 1. Load Data
        calendar, sales, prices = self.loader.load_data()
        df = self.loader.melt_and_merge(calendar, sales, prices)
        
        # 2. Feature Engineering
        df = self.fe.generate_features(df)
        df = self.fe.encode_categorical(df)
        
        # 3. Time-Series CV Loop
        folds = self.config['training']['n_folds']
        horizon = self.config['data']['forecast_horizon']
        
        # Determine last available day
        # For synthetic data, we have d_1 to d_1941
        # We want to backtest on the last 'folds' * 'horizon' days
        
        # Convert 'd' to integer for easier slicing
        df['d_int'] = df['d'].apply(lambda x: int(x.split('_')[1]))
        max_d = df['d_int'].max()
        
        metrics_log = []
        
        for fold in range(folds):
            print(f"\n=== Running Backtest Fold {fold + 1}/{folds} ===")
            
            # Define split point
            # Fold 0: Test on [max_d - horizon + 1, max_d]
            # Fold 1: Test on [max_d - 2*horizon + 1, max_d - horizon]
            # ...
            
            test_end = max_d - (fold * horizon)
            test_start = test_end - horizon + 1
            train_end = test_start - 1
            
            print(f"Train range: d_1 to d_{train_end}")
            print(f"Test range: d_{test_start} to d_{test_end}")
            
            train_mask = df['d_int'] <= train_end
            valid_mask = (df['d_int'] >= test_start) & (df['d_int'] <= test_end)
            
            train_df = df[train_mask]
            valid_df = df[valid_mask]
            
            # Prepare X and y
            features = [c for c in df.columns if c not in ['id', 'd', 'sales', 'date', 'wm_yr_wk', 'd_int']]
            target = 'sales'
            
            X_train = train_df[features]
            y_train = train_df[target]
            X_valid = valid_df[features]
            y_valid = valid_df[target]
            
            # Train
            cat_feats = self.config['feature_engineering']['categorical_features']
            # Filter cat_feats to only those present in X_train
            cat_feats = [c for c in cat_feats if c in X_train.columns]
            
            self.trainer.train(X_train, y_train, X_valid, y_valid, categorical_features=cat_feats)
            
            # Predict
            preds = self.trainer.predict(X_valid)
            
            # Evaluate
            # Create dummy weights for now
            weights = pd.DataFrame({'id': valid_df['id'].unique(), 'weight': 1.0, 'scale': 1.0})
            
            evaluator = Evaluator(train_df, valid_df, weights)
            res = evaluator.generate_report(y_valid, preds)
            print(f"Fold {fold+1} Results: {res}")
            metrics_log.append(res)
            
        # Average metrics
        avg_rmse = np.mean([m['RMSE'] for m in metrics_log])
        print(f"\nAverage RMSE across {folds} folds: {avg_rmse:.4f}")

if __name__ == "__main__":
    backtester = Backtester('config/config.yaml')
    backtester.run_backtest()
