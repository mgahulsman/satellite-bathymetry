import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import pandas as pd

log_ratio_path = 'data/processed/aero_over_green.tif'
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
log_ratio_col = im_data[y_pix, x_pix]

h_bins = np.linspace(np.min(height_col), np.max(height_col),201)
log_bins = np.linspace(np.percentile(log_ratio_col,0.01), np.percentile(log_ratio_col,99.9),201)

plt.figure(figsize=(10,10))
plt.hist2d(log_ratio_col,height_col,bins=[log_bins,h_bins],cmap='magma', cmin =5)
plt.colorbar()
plt.xlabel('Log Ratio')
plt.ylabel('NAVD88 Height (m)')
plt.show()
log_ratio.close()