import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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
height_mask = np.where(height_col > -12, True, False)
uncer_col = height_df['mean_uncertainty']
x_col = height_df['true_x']
y_col = height_df['true_y']

x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
y_pix = np.array((y_col - offset_y) / scale_y).astype(int)

vals_bg = im_bg[y_pix,x_pix]
mask_bg = np.where((vals_bg > np.percentile(vals_bg,0.1))&(vals_bg < np.percentile(vals_bg, 99.9)), True, False)
vals_br = im_br[y_pix,x_pix]
mask_br = np.where((vals_br > np.percentile(vals_br,0.1))&(vals_br < np.percentile(vals_br, 99.9)), True, False)
vals_ag = im_ag[y_pix,x_pix]
mask_ag = np.where((vals_ag > np.percentile(vals_ag,0.1))&(vals_ag < np.percentile(vals_ag, 99.9)), True, False)
vals_ar = im_ar[y_pix,x_pix]
mask_ar = np.where((vals_ar > np.percentile(vals_ar,0.1))&(vals_ar < np.percentile(vals_ar, 99.9)), True, False)
x_array = np.array([vals_bg,vals_br,vals_ag,vals_ar])
log_mask = mask_bg & mask_br & mask_ag & mask_ar
total_mask = np.logical_and(log_mask,height_mask)

x_array_masked = x_array.T[total_mask]
y_array_masked = height_col[total_mask]
w_array_masked = 1/(uncer_col[total_mask]**2)

regressor = RandomForestRegressor(n_estimators=150, random_state=0, oob_score=True,min_samples_split=30,min_samples_leaf=10)
regressor.fit(x_array_masked, y_array_masked, w_array_masked)
y_pred = regressor.predict(x_array.T)

y_bins = np.linspace(np.min(height_col),np.max(height_col),201)

plt.figure(figsize=(10,10))
plt.hist2d(height_col,y_pred, bins=y_bins, cmap='magma',cmin=5)
plt.plot(y_bins, y_bins, color = 'black', linestyle = 'dashed')
plt.colorbar()
plt.xlabel('True Height')
plt.ylabel('Predicted Height')
plt.savefig('HeightvsPrediction.png')
plt.show()

residuals = height_col - y_pred
res_bins = np.linspace(np.percentile(residuals,0.01),np.percentile(residuals,99.9),201)
plt.figure(figsize=(10,6))
plt.hist2d(height_col,residuals, bins=[y_bins,res_bins], cmap='jet',cmin=5)
plt.axhline(y=0, color='black', linestyle='dashed',zorder=1)
plt.xlabel('NAVD88 Height (m)')
plt.ylabel('Residual (m)')
plt.savefig('residual.png')
plt.show()

print('Bias: ',np.mean(residuals))
print('RMSE: ',np.sqrt(np.mean(residuals ** 2)))
print('MAE: ',np.mean(np.abs(residuals)))

big_bins = [0,-4,-6,-8,-12]
for i in range(len(big_bins)-1):
    bin_mask = np.where((height_col < big_bins[i]) & (height_col > big_bins[i+1]),True,False)
    bin_true = height_col[bin_mask]
    bin_pred = y_pred[bin_mask]
    bin_residual = bin_true - bin_pred
    print(f'For heights between {big_bins[i+1]} and {big_bins[i]}')
    print('Bias: ',np.mean(bin_residual))
    print('RMSE: ',np.sqrt(np.mean(bin_residual ** 2)))
    print('MAE: ',np.mean(np.abs(bin_residual)))

blue_over_green.close()
blue_over_red.close()
aero_over_green.close()
aero_over_red.close()