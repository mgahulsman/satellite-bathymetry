import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a, b):
    return a * x + b

# Data en tiff bestanden laden
log_ratio_path = 'data/processed/blue_over_green_corr.tif'
log_ratio_file = rio.open(log_ratio_path)

ndwi_path = 'data/processed/ndwi.tif'
ndwi = rio.open(ndwi_path)

transform = log_ratio_file.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
im_data = log_ratio_file.read(1)

def process_csv_data(csv_file: str):
    height_df = pd.read_csv(f'data/processed/{csv_file}')
    height_col = height_df['median_height'].values
    height_mask = np.where(height_col > -8, True, False)
    uncer_col = height_df['mean_uncertainty'].values
    numb_col = height_df['number_of_points'].values
    x_col = height_df['true_x'].values
    y_col = height_df['true_y'].values

    x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
    y_pix = np.array((y_col - offset_y) / scale_y).astype(int)
    log_ratio_col = im_data[y_pix, x_pix]
    outlier_mask = np.where((log_ratio_col > np.percentile(log_ratio_col, 0.1)) & 
                            (log_ratio_col < np.percentile(log_ratio_col, 99.9)), True, False)
    total_mask = np.logical_and(outlier_mask, height_mask)
    
    return log_ratio_col[total_mask], height_col[total_mask], uncer_col[total_mask]

# Load en fit train set
train_lr, train_h, train_uncer = process_csv_data("per_pixel_heights_MainAOI.csv")

popt, pcov = curve_fit(lin,
                       train_lr,
                       train_h,
                       [0, 0],
                       train_uncer,
                       absolute_sigma=True)
print('popt:', popt)

def create_results(log_ratio_col, height_col, name="set"):
    x_bins = np.linspace(np.min(log_ratio_col), np.percentile(log_ratio_col, 99.9), 201)
    y_bins = np.linspace(np.min(height_col), np.max(height_col), 201)

    plt.figure(figsize=(10, 10))
    plt.hist2d(log_ratio_col, height_col, bins=[x_bins, y_bins], cmap='jet', cmin=5)
    plt.plot(x_bins, lin(x_bins, *popt), 'r', zorder=1)
    plt.xlim(np.min(log_ratio_col), np.percentile(log_ratio_col, 99.9))
    plt.xlabel('Log Ratio')
    plt.ylabel('NAVD88 Height (m)')
    plt.show()

    predicted_height = lin(log_ratio_col, *popt)
    plt.figure(figsize=(10, 10))
    plt.hist2d(height_col, predicted_height, bins=y_bins, cmap='jet', cmin=5)
    plt.plot(y_bins, y_bins, color='black', linestyle='dashed')
    plt.colorbar()
    plt.xlabel('True Height')
    plt.ylabel('Predicted Height')
    plt.savefig(f'HeightvsPrediction_{name}.png')
    plt.show()

    residuals = height_col - predicted_height
    res_bins = np.linspace(np.percentile(residuals, 0.01), np.percentile(residuals, 99.9), 201)
    plt.figure(figsize=(10, 6))
    plt.hist2d(height_col, residuals, bins=[y_bins, res_bins], cmap='jet', cmin=5)
    plt.axhline(y=0, color='black', linestyle='dashed', zorder=1)
    plt.xlabel('NAVD88 Height (m)')
    plt.ylabel('Residual (m)')
    plt.savefig(f'residual_{name}.png')
    plt.show()

    print(f'--- Results for {name} ---')
    print('Bias: ', np.mean(residuals))
    print('RSME: ', np.sqrt(np.mean(residuals ** 2)))
    print('MAE: ', np.mean(np.abs(residuals)))

    big_bins = [0, -4, -6, -8, -12]
    for i in range(len(big_bins) - 1):
        bin_mask = np.where((height_col < big_bins[i]) & (height_col > big_bins[i+1]), True, False)
        bin_true = height_col[bin_mask]
        bin_pred = predicted_height[bin_mask]
        bin_residual = bin_true - bin_pred
        print(f'For heights between {big_bins[i+1]} and {big_bins[i]}')
        print('Bias: ', np.mean(bin_residual))
        print('RMSE: ', np.sqrt(np.mean(bin_residual ** 2)))
        print('MAE: ', np.mean(np.abs(bin_residual)))

# Run resultaten per dataset
create_results(train_lr, train_h, name="train")

# val_lr, val_h, _ = process_csv_data("val_heights.csv")
# create_results(val_lr, val_h, name="val")

# test_lr, test_h, _ = process_csv_data("test_heights.csv")
# create_results(test_lr, test_h, name="test")

log_ratio_file.close()
ndwi.close()