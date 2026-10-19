import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a, b):
    return a * x + b

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
uncer_col = height_df['mean_uncertainty']
x_col = height_df['true_x']
y_col = height_df['true_y']

x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
y_pix = np.array((y_col - offset_y) / scale_y).astype(int)
log_ratio_col = im_data[y_pix, x_pix]

popt, pcov = curve_fit(lin, log_ratio_col, height_col, [0,0], uncer_col, absolute_sigma=True)
x = np.linspace(np.min(log_ratio_col), np.max(log_ratio_col), 100)

plt.figure(figsize=(10,10))
plt.scatter(log_ratio_col, height_col,s = 0.2, color = 'black', alpha = 0.1)
plt.plot(x, lin(x, *popt), 'r', zorder = 1)
plt.xlabel('Log Ratio')
plt.ylabel('NAVD88 Height (m)')
plt.show()
print(popt)

predicted_height = lin(log_ratio_col, *popt)
plt.figure(figsize=(10,10))
plt.scatter(height_col, predicted_height, s = 0.2, color = 'black', alpha = 0.1)
plt.xlabel('True Height')
plt.ylabel('Predicted Height')
plt.savefig('HeightvsPrediction.png')
plt.show()

residuals = height_col - predicted_height
plt.figure(figsize=(10,6))
plt.scatter(height_col, residuals, s = 0.2, color = 'black', alpha = 0.1)
plt.xlabel('NAVD88 Height (m)')
plt.ylabel('Residual (m)')
plt.savefig('residual.png')
plt.show()
print('Bias: ',np.mean(residuals))
print('RMSE: ',np.sqrt(np.mean(residuals ** 2)))

log_ratio.close()
