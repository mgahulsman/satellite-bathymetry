import os
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a, b, c1, c2):
    return a * (np.log10(10000*(x[0] + c1 * x[2]))/np.log10(10000*(x[1] + c2 * x[2]))) + b

# Create output directories
output_dir = 'data/output/linear/'
figs_dir = os.path.join(output_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

# Load raster datasets
i_path = 'data/processed/blue.tif'
j_path = 'data/processed/green.tif'
nir_path = 'data/processed/nir_smooth.tif'

i_file = rio.open(i_path)
j_file = rio.open(j_path)
nir_file = rio.open(nir_path)

transform = i_file.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
i_data = i_file.read(1)
j_data = j_file.read(1)
nir_data = nir_file.read(1)

def process_csv_data(csv_file: str, height_mask_h = -8):
    height_df = pd.read_csv(f'data/processed/{csv_file}')
    height_col = height_df['median_height'].values
    height_mask = np.where(height_col > height_mask_h, True, False)
    uncer_col = height_df['mean_uncertainty'].values
    numb_col = height_df['number_of_points'].values
    x_col = height_df['true_x'].values
    y_col = height_df['true_y'].values

    x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
    y_pix = np.array((y_col - offset_y) / scale_y).astype(int)

    i_col = i_data[y_pix, x_pix]
    j_col = j_data[y_pix, x_pix]
    nir_col = nir_data[y_pix, x_pix]
    x_arr = np.array([i_col, j_col, nir_col])

    total_mask = height_mask

    # Return un-filtered data alongside total_mask
    return x_arr, height_col, uncer_col, total_mask

# Load training set
train_lr, train_h, train_uncer, train_mask = process_csv_data("per_pixel_heights_MainAOI.csv")

# Perform curve fit using total_mask
popt, pcov = curve_fit(lin,
                       train_lr[:,train_mask],
                       train_h[train_mask],
                       [-40, 0, 0.1, 0.1],
                       train_uncer[train_mask],
                       bounds = ((-np.inf, -np.inf, 0, 0),(np.inf, np.inf, 10, 10)),
                       absolute_sigma=True)
print('popt:', popt)

def create_results(log_ratio_col, height_col, name="set"):
    #x_bins = np.linspace(np.min(log_ratio_col), np.percentile(log_ratio_col, 99.9), 201)
    y_bins = np.linspace(np.min(height_col), np.max(height_col), 201)

    # Plot 1: Linear fit
    '''
    plt.figure(figsize=(10, 10))
    plt.hist2d(log_ratio_col, height_col, bins=[x_bins, y_bins], cmap='jet', cmin=5)
    plt.plot(x_bins, lin(x_bins, *popt), 'r', zorder=1)
    plt.xlim(np.min(log_ratio_col), np.percentile(log_ratio_col, 99.9))
    plt.xlabel('Log Ratio')
    plt.ylabel('NAVD88 Height (m)')
    plt.savefig(os.path.join(figs_dir, f'Fit_{name}.png'))
    plt.show()
    '''

    # Plot 2: True vs Predicted Height
    predicted_height = lin(log_ratio_col, *popt)
    mask = ~np.isnan(predicted_height)
    print(predicted_height)
    plt.figure(figsize=(10, 10))
    plt.hist2d(height_col[mask], predicted_height[mask], bins=y_bins, cmap='jet', cmin=5)
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

# Execute results for dataset splits
all_lr, all_h, _, _ = process_csv_data("per_pixel_heights_MainAOI.csv")
create_results(train_lr, train_h, name="train_noglint_bg_corr")

val_lr, val_h, _, _ = process_csv_data("per_pixel_heights_SouthernRecon.csv")
create_results(val_lr, val_h, name="val_noglint_bg_corr")

# test_lr, test_h, _, _ = process_csv_data("test_heights.csv")
# create_results(test_lr, test_h, name="test")

i_file.close()
j_file.close()
nir_file.close()