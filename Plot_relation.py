import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import pandas as pd

log_ratio_path = 'data/processed/blue_over_green.tif'
log_ratio = rio.open(log_ratio_path)

transform = log_ratio.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
im_data = log_ratio.read(1)

height_df = pd.read_csv('data/processed/per_pixel_heights_MainAOI.csv')
height_col = height_df['median_height']
x_col = height_df['true_x']
y_col = height_df['true_y']

x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
y_pix = np.array((y_col - offset_y) / scale_y).astype(int)
log_ratio_col = im_data[x_pix, y_pix]

plt.figure(figsize=(10,10))
plt.scatter(height_col, log_ratio_col,s = 0.2, color = 'black', alpha = 0.1)
plt.xlabel('NAVD88 Height (m)')
plt.ylabel('Log Ratio')
plt.show()
log_ratio.close()