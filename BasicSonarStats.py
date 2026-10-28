import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
data_folder = os.path.join(cwd, r'data\raw')
result_folder = os.path.join(cwd, r'data\processed')
sonar_folder = os.path.join(data_folder, 'Chandeleur_Islands_2023_MBES')
raw_files = os.listdir(os.path.join(data_folder, sonar_folder))

for file in raw_files:
    if file.endswith(".xyz"):
        df = pd.read_csv(os.path.join(sonar_folder, file))
        print(file)
        columns = df.columns
        print(len(df))
        h_name = columns[2]
        u_name = columns[3]
        print(df[h_name].mean(), df[h_name].min(), df[h_name].max())
        print(df[u_name].mean())
        print(' ')
