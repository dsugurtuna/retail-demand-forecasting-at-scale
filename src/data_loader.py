import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']
        self.processed_path = config['data']['processed_path']
        
    def load_data(self):
        """
        Loads M5 data from CSVs. If not found, generates synthetic data.
        """
        try:
            print("Attempting to load M5 data...")
            calendar = pd.read_csv(os.path.join(self.raw_path, 'calendar.csv'))
            sales = pd.read_csv(os.path.join(self.raw_path, 'sales_train_evaluation.csv'))
            prices = pd.read_csv(os.path.join(self.raw_path, 'sell_prices.csv'))
            print("Data loaded successfully.")
            return calendar, sales, prices
        except FileNotFoundError:
            print("Data not found. Generating synthetic M5-like data...")
            return self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """
        Generates synthetic data mimicking the M5 competition structure.
        """
        # 1. Calendar
        start_date = datetime.strptime(self.config['data']['start_date'], "%Y-%m-%d")
        date_range = [start_date + timedelta(days=x) for x in range(1941)] # M5 length
        
        calendar = pd.DataFrame({
            'date': date_range,
            'wm_yr_wk': [d.isocalendar()[0]*100 + d.isocalendar()[1] for d in date_range],
            'weekday': [d.strftime("%A") for d in date_range],
            'wday': [d.isoweekday() for d in date_range],
            'month': [d.month for d in date_range],
            'year': [d.year for d in date_range],
            'd': [f'd_{i+1}' for i in range(len(date_range))],
            'event_name_1': [np.random.choice(['SuperBowl', 'ValentinesDay', 'Easter', None], p=[0.01, 0.01, 0.01, 0.97]) for _ in range(len(date_range))],
            'event_type_1': [np.random.choice(['Sporting', 'Cultural', 'Religious', None], p=[0.01, 0.01, 0.01, 0.97]) for _ in range(len(date_range))],
            'snap_CA': np.random.randint(0, 2, len(date_range)),
            'snap_TX': np.random.randint(0, 2, len(date_range)),
            'snap_WI': np.random.randint(0, 2, len(date_range)),
        })
        
        # 2. Sales (Wide format)
        n_items = 100
        items = [f'item_{i}' for i in range(n_items)]
        stores = ['CA_1', 'CA_2', 'TX_1', 'WI_1']
        
        sales_data = {
            'id': [],
            'item_id': [],
            'dept_id': [],
            'cat_id': [],
            'store_id': [],
            'state_id': []
        }
        
        # Add d_1 to d_1941 columns
        for i in range(1, 1942):
            sales_data[f'd_{i}'] = []
            
        for store in stores:
            for item in items:
                sales_data['id'].append(f'{item}_{store}_evaluation')
                sales_data['item_id'].append(item)
                sales_data['dept_id'].append('FOODS_1')
                sales_data['cat_id'].append('FOODS')
                sales_data['store_id'].append(store)
                sales_data['state_id'].append(store.split('_')[0])
                
                # Generate random sales with some seasonality
                base_demand = np.random.poisson(5, 1941)
                sales_data_list = list(base_demand)
                
                for day_idx, val in enumerate(sales_data_list):
                    sales_data[f'd_{day_idx+1}'].append(val)

        sales = pd.DataFrame(sales_data)
        
        # 3. Prices
        prices_data = []
        for store in stores:
            for item in items:
                # Generate prices for weeks
                weeks = calendar['wm_yr_wk'].unique()
                base_price = np.random.uniform(2.0, 10.0)
                for wk in weeks:
                    prices_data.append({
                        'store_id': store,
                        'item_id': item,
                        'wm_yr_wk': wk,
                        'sell_price': base_price * np.random.uniform(0.9, 1.1)
                    })
        prices = pd.DataFrame(prices_data)
        
        print(f"Generated synthetic data: {sales.shape[0]} series.")
        return calendar, sales, prices

    def melt_and_merge(self, calendar, sales, prices):
        """
        Transforms wide sales data to long format and merges with calendar and prices.
        """
        print("Melting sales data...")
        # Melt sales
        id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sales_long = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='sales')
        
        print("Merging with calendar...")
        sales_long = sales_long.merge(calendar, on='d', how='left')
        
        print("Merging with prices...")
        sales_long = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        return sales_long

if __name__ == "__main__":
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    loader = DataLoader(config)
    cal, sales, prices = loader.load_data()
    df = loader.melt_and_merge(cal, sales, prices)
    print(df.head())
