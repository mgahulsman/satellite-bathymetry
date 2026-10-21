import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a1, a2, a3, a4, b):
    return a1 * x[0] + a2 * x[1] + a3 * x[2] + a4 * x[3] + b

data_dir = 'data/processed'
blue_over_green = rio.open(data_dir + '/blue_over_green_corr.tif')
blue_over_red = rio.open(data_dir + '/blue_over_red_corr.tif')
aero_over_green = rio.open(data_dir + '/aero_over_green_corr.tif')
aero_over_red = rio.open(data_dir + '/aero_over_red_corr.tif')

transform = blue_over_green.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]

im_bg = blue_over_green.read(1)
im_br = blue_over_red.read(1)
im_ag = aero_over_green.read(1)
im_ar = aero_over_red.read(1)

height_df = pd.read_csv('data/processed/per_pixel_heights_MainAOI.csv')
height_col = height_df['median_height']
height_mask = np.where(height_col > -8, True, False)
uncer_col = height_df['mean_uncertainty']
x_col = height_df['true_x']
y_col = height_df['true_y']

x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
y_pix = np.array((y_col - offset_y) / scale_y).astype(int)

vals_bg = im_bg[y_pix,x_pix]
vals_br = im_br[y_pix,x_pix]
vals_ag = im_ag[y_pix,x_pix]
vals_ar = im_ar[y_pix,x_pix]
x_array = np.array([vals_bg,vals_br,vals_ag,vals_ar])

popt, pcov = curve_fit(lin,
                       x_array[:,height_mask],
                       height_col[height_mask],
                       p0=[0,0,0,0,0],
                       sigma = uncer_col[height_mask],
                       absolute_sigma = True)
print(popt)
y_bins = np.linspace(np.min(height_col),np.max(height_col),201)

predicted_height = lin(x_array, *popt)
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
print('RMSE: ',np.sqrt(np.mean(residuals ** 2)))

blue_over_green.close()
blue_over_red.close()
aero_over_green.close()
aero_over_red.close()