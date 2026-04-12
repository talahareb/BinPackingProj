import os
import pandas as pd

class Instance():
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.df_items = pd.read_csv(
            os.path.join('.', 'datasets', self.name, 'items.csv'),
            index_col = 0,
            dtype={"allowedRotations": str}
        )
        self.df_vehicles = pd.read_csv(
            os.path.join('.', 'datasets', self.name, 'vehicles.csv'),
            index_col = 0
        )
    
    def __repr__(self):
        output = ''
        output += f'--- ITEMS ---\n{self.df_items}\n'
        output += f'--- VEHICLES ---\n{self.df_vehicles}\n'
        return output