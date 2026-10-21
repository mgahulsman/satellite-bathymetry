import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a, b):
    return a * x + b

log_ratio_path = 'data/processed/aero_over_green_corr.tif'
log_ratio = rio.open(log_ratio_path)

ndwi_path = 'data/processed/ndwi.tif'
ndwi = rio.open(ndwi_path)

transform = log_ratio.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
im_data = log_ratio.read(1)

height_df = pd.read_csv('data/processed/per_pixel_heights_MainAOI.csv')
height_col = height_df['median_height']
height_mask = np.where(height_col > -8, True, False)
uncer_col = height_df['mean_uncertainty']
x_col = height_df['true_x']
y_col = height_df['true_y']

x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
y_pix = np.array((y_col - offset_y) / scale_y).astype(int)
log_ratio_col = im_data[y_pix, x_pix]

popt, pcov = curve_fit(lin,
                       log_ratio_col[height_mask],
                       height_col[height_mask],
                       [0,0],
                       uncer_col[height_mask],
                       absolute_sigma=True)
x_bins = np.linspace(np.min(log_ratio_col), np.percentile(log_ratio_col,99.9), 201)
y_bins = np.linspace(np.min(height_col),np.max(height_col),201)

plt.figure(figsize=(10,10))
plt.hist2d(log_ratio_col, height_col, bins = [x_bins,y_bins],cmap='jet',cmin=5)
plt.plot(x_bins, lin(x_bins, *popt), 'r', zorder = 1)
plt.xlim(np.min(log_ratio_col), np.percentile(log_ratio_col,99.9))
plt.xlabel('Log Ratio')
plt.ylabel('NAVD88 Height (m)')
plt.show()
print(popt)

predicted_height = lin(log_ratio_col, *popt)
plt.figure(figsize=(10,10))
plt.hist2d(height_col,predicted_height, bins=y_bins, cmap='jet',cmin=5)
plt.plot(y_bins, y_bins, color = 'black', linestyle = 'dashed')
plt.colorbar()
plt.xlabel('True Height')
plt.ylabel('Predicted Height')
plt.savefig('HeightvsPrediction.png')
plt.show()

residuals = height_col - predicted_height
res_bins = np.linspace(np.percentile(residuals,0.01),np.percentile(residuals,99.9),201)
plt.figure(figsize=(10,6))
plt.hist2d(height_col,residuals, bins=[y_bins,res_bins], cmap='jet',cmin=5)
plt.xlabel('NAVD88 Height (m)')
plt.ylabel('Residual (m)')
plt.savefig('residual.png')
plt.show()
print('Bias: ',np.mean(residuals))
print('RSME: ',np.sqrt(np.mean(residuals ** 2)))

log_ratio.close()
