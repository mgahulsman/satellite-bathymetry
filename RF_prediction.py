import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os

# Create output directories
output_dir = 'data/output/linear/'
figs_dir = os.path.join(output_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

data_dir = 'data/processed'
blue_over_green = rio.open(data_dir + '/blue_over_green.tif')
blue_over_red = rio.open(data_dir + '/blue_over_red.tif')
aero_over_green = rio.open(data_dir + '/aero_over_green.tif')
aero_over_red = rio.open(data_dir + '/aero_over_red.tif')

transform = blue_over_green.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]

im_bg = blue_over_green.read(1)
im_br = blue_over_red.read(1)
im_ag = aero_over_green.read(1)
im_ar = aero_over_red.read(1)

train_df = pd.read_csv('data/processed/per_pixel_heights_MainAOI.csv')
train_h = train_df['median_height']
height_mask = np.where(train_h > -12, True, False)
uncer_col = train_df['mean_uncertainty']
x_col = train_df['true_x']
y_col = train_df['true_y']

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
y_array_masked = train_h[total_mask]
w_array_masked = 1/(uncer_col[total_mask]**2)

regressor = RandomForestRegressor(n_estimators=150, random_state=0, oob_score=True,min_samples_split=30, min_samples_leaf= 10)
regressor.fit(x_array_masked, y_array_masked, w_array_masked)

val_df = pd.read_csv('data/processed/per_pixel_heights_SouthernRecon.csv')
val_h = val_df['median_height']
val_x = val_df['true_x']
val_y = val_df['true_y']
val_ypix = np.array((val_y - offset_y) / scale_y).astype(int)
val_xpix = np.array((val_x - offset_x) / scale_x).astype(int)

val_bg = im_bg[val_ypix,val_xpix]
val_br = im_br[val_ypix,val_xpix]
val_ag = im_ag[val_ypix,val_xpix]
val_ar = im_ar[val_ypix,val_xpix]
val_x_array = np.array([val_bg,val_br,val_ag,val_ar])

def create_results(log_ratio_col, height_col, name="set"):
    y_bins = np.linspace(np.min(height_col), np.max(height_col), 201)

    # Plot 2: True vs Predicted Height
    predicted_height = regressor.predict(log_ratio_col.T)
    print(len(height_col))
    print(len(predicted_height))
    plt.figure(figsize=(10, 10))
    plt.hist2d(height_col, predicted_height, bins=y_bins, cmap='jet', cmin=5)
    plt.plot(y_bins, y_bins, color='black', linestyle='dashed')
    plt.colorbar()
    plt.xlabel('True Height')
    plt.ylabel('Predicted Height')
    plt.savefig(os.path.join(figs_dir, f'HeightvsPrediction_{name}.png'))
    plt.show()

    # Plot 3: Residuals
    residuals = height_col - predicted_height
    res_bins = np.linspace(np.percentile(residuals, 0.01), np.percentile(residuals, 99.9), 201)
    plt.figure(figsize=(10, 6))
    plt.hist2d(height_col, residuals, bins=[y_bins, res_bins], cmap='jet', cmin=5)
    plt.axhline(y=0, color='black', linestyle='dashed', zorder=1)
    plt.xlabel('NAVD88 Height (m)')
    plt.ylabel('Residual (m)')
    plt.savefig(os.path.join(figs_dir, f'residual_{name}.png'))
    plt.show()

    # Build evaluation metrics log
    lines = []
    lines.append(f'--- Results for {name} ---')
    lines.append(f'Bias: {np.mean(residuals)}')
    lines.append(f'RMSE: {np.sqrt(np.mean(residuals ** 2))}')
    lines.append(f'MAE:  {np.mean(np.abs(residuals))}')

    big_bins = [0, -4, -6, -8, -12]
    for i in range(len(big_bins) - 1):
        bin_mask = np.where((height_col < big_bins[i]) & (height_col > big_bins[i+1]), True, False)
        bin_true = height_col[bin_mask]
        bin_pred = predicted_height[bin_mask]
        bin_residual = bin_true - bin_pred
        lines.append(f'For heights between {big_bins[i+1]} and {big_bins[i]}')
        lines.append(f'Bias: {np.mean(bin_residual)}')
        lines.append(f'RMSE: {np.sqrt(np.mean(bin_residual ** 2))}')
        lines.append(f'MAE:  {np.mean(np.abs(bin_residual))}')

    # Print results to console
    output_text = "\n".join(lines)
    print(output_text)

    # Save metrics to text file
    metrics_file = os.path.join(output_dir, f'metrics_{name}.txt')
    with open(metrics_file, 'w') as f:
        f.write(output_text)

create_results(x_array, train_h, name="train_noglint_rf")
create_results(val_x_array, val_h, name="val_noglint_rf")

blue_over_green.close()
blue_over_red.close()
aero_over_green.close()
aero_over_red.close()