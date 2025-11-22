import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.params = config['model']['params']
        self.model = None
        
    def train(self, X_train, y_train, X_valid, y_valid, categorical_features=None):
        """
        Trains the LightGBM model.
        """
        print(f"Training LightGBM model with params: {self.params}")
        
        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
        
        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.params['early_stopping_rounds']),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        importance = pd.DataFrame({
            'feature': self.model.feature_name(),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return importance

    def save_model(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
