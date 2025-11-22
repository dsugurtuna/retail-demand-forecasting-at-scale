import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self, train_df, valid_df, weights=None):
        self.train_df = train_df
        self.valid_df = valid_df
        self.weights = weights # DataFrame with 'id' and 'weight' columns

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def wrmsse(self, y_true, y_pred, id_col='id'):
        """
        Calculates Weighted Root Mean Squared Scaled Error (WRMSSE).
        
        Note: This is a simplified implementation. The full M5 WRMSSE requires
        hierarchical aggregation and scaling factors based on training data.
        
        Here we assume 'weights' are pre-calculated for the bottom level.
        """
        if self.weights is None:
            raise ValueError("Weights must be provided for WRMSSE calculation.")
            
        # Align weights with validation data
        # Assuming y_true and y_pred are aligned with self.valid_df
        
        df = self.valid_df.copy()
        df['target'] = y_true
        df['prediction'] = y_pred
        
        # Merge weights
        df = df.merge(self.weights, on=id_col, how='left')
        
        # Calculate squared error
        df['se'] = (df['target'] - df['prediction'])**2
        
        # Calculate scaling factor (denominator of RMSSE)
        # In M5, this is the mean squared difference of consecutive training values
        # We assume this is pre-calculated and stored in 'scale' column of weights
        
        if 'scale' not in df.columns:
             # Fallback: use variance of target as proxy for scale if not provided
             df['scale'] = df.groupby(id_col)['target'].transform('var')
             df['scale'] = df['scale'].replace(0, 1) # Avoid div by zero

        df['rmsse'] = np.sqrt(df['se'] / df['scale'])
        
        # Weighted average
        wrmsse_score = np.average(df['rmsse'], weights=df['weight'])
        return wrmsse_score

    def generate_report(self, y_true, y_pred):
        res = {
            'RMSE': self.rmse(y_true, y_pred),
            'MAE': self.mae(y_true, y_pred)
        }
        if self.weights is not None:
            res['WRMSSE'] = self.wrmsse(y_true, y_pred)
        return res
