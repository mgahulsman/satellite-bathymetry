import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

def lin(x, a1, a2, a3, a4, b):
    return a1 * x[0] + a2 * x[1] + a3 * x[2] + a4 * x[3] + b

output_dir = 'data/output/linear/'
figs_dir = os.path.join(output_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

data_dir = 'data/processed'
blue_over_green = rio.open(data_dir + '/blue_over_green.tif')
blue_over_red = rio.open(data_dir + '/blue_over_red.tif')
aero_over_green = rio.open(data_dir + '/aero_over_green.tif')
aero_over_red = rio.open(data_dir + '/aero_over_red.tif')
nir = rio.open(data_dir + '/nir_smooth.tif')

transform = blue_over_green.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]

im_bg = blue_over_green.read(1)
im_br = blue_over_red.read(1)
im_ag = aero_over_green.read(1)
im_ar = aero_over_red.read(1)
im_nir = nir.read(1)

height_df = pd.read_csv('data/processed/per_pixel_heights_MainAOI.csv')
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

    ag_col = im_ag[y_pix, x_pix]
    bg_col = im_bg[y_pix, x_pix]
    ar_col = im_ar[y_pix, x_pix]
    br_col = im_br[y_pix, x_pix]
    nir_col = im_nir[y_pix, x_pix]
    x_arr = np.array([ag_col, bg_col, ar_col, br_col, nir_col])
    total_mask = height_mask

    return x_arr, height_col, uncer_col, total_mask

train_lr, train_h, train_uncer, train_mask = process_csv_data("per_pixel_heights_MainAOI.csv")

popt, pcov = curve_fit(lin,
                       train_lr[:,train_mask],
                       train_h[train_mask],
                       p0=[0,0,0,0,0],
                       sigma = train_uncer[train_mask],
                       absolute_sigma = True)
print(popt)

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


all_lr, all_h, _, _ = process_csv_data("per_pixel_heights_MainAOI.csv")
create_results(train_lr, train_h, name="train_glint_full_nocorr")

val_lr, val_h, _, _ = process_csv_data("per_pixel_heights_SouthernRecon.csv")
create_results(val_lr, val_h, name="val_glint_full_nocorr")


blue_over_green.close()
blue_over_red.close()
aero_over_green.close()
aero_over_red.close()
nir.close()