import os
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def load_raster_bands(band_paths: dict):
    """
    Opens raster files and extracts 2D numpy arrays along with reference affine transform.
    """
    files = {}
    bands = {}
    transform = None

    for key, path in band_paths.items():
        src = rio.open(path)
        files[key] = src
        bands[key] = src.read(1)
        if transform is None:
            transform = src.transform

    return files, bands, transform

def extract_csv_pixels(csv_file: str, bands: dict, transform, height_mask_h=-8, apply_outlier_filter=False):
    """
    Loads coordinate data from CSV, samples raster values at pixel indices, and builds data masks.
    """
    df = pd.read_csv(f'data/processed/{csv_file}')
    height_col = df['median_height'].values
    uncer_col = df['mean_uncertainty'].values
    x_col = df['true_x'].values
    y_col = df['true_y'].values

    offset_x, offset_y = transform[2], transform[5]
    scale_x, scale_y = transform[0], transform[4]

    x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
    y_pix = np.array((y_col - offset_y) / scale_y).astype(int)

    # Sample pixel values for all loaded bands
    sampled_bands = [data[y_pix, x_pix] for data in bands.values()]
    x_arr = np.array(sampled_bands) if len(sampled_bands) > 1 else sampled_bands[0]

    # Masking logic
    height_mask = height_col > height_mask_h
    if apply_outlier_filter and x_arr.ndim == 1:
        outlier_mask = (x_arr > np.percentile(x_arr, 0.1)) & (x_arr < np.percentile(x_arr, 99.9))
        total_mask = outlier_mask & height_mask
    else:
        total_mask = height_mask

    return x_arr, height_col, uncer_col, total_mask

def evaluate_and_plot(model_func, popt, x_data, height_col, name="set", output_dir='data/output/linear/'):
    """
    Evaluates predictions, generates diagnostics plots, calculates metrics, and writes log files.
    """
    figs_dir = os.path.join(output_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    predicted_height = model_func(x_data, *popt)
    valid_mask = ~np.isnan(predicted_height)
    
    y_bins = np.linspace(np.min(height_col), np.max(height_col), 201)

    # Plot: True vs Predicted Height
    plt.figure(figsize=(10, 10))
    plt.hist2d(height_col[valid_mask], predicted_height[valid_mask], bins=y_bins, cmap='jet', cmin=5)
    plt.plot(y_bins, y_bins, color='black', linestyle='dashed')
    plt.colorbar()
    plt.xlabel('True Height')
    plt.ylabel('Predicted Height')
    plt.savefig(os.path.join(figs_dir, f'HeightvsPrediction_{name}.png'))
    plt.show()

    # Plot: Residuals
    residuals = height_col - predicted_height
    res_bins = np.linspace(np.percentile(residuals, 0.01), np.percentile(residuals, 99.9), 201)
    plt.figure(figsize=(10, 6))
    plt.hist2d(height_col, residuals, bins=[y_bins, res_bins], cmap='jet', cmin=5)
    plt.axhline(y=0, color='black', linestyle='dashed', zorder=1)
    plt.xlabel('NAVD88 Height (m)')
    plt.ylabel('Residual (m)')
    plt.savefig(os.path.join(figs_dir, f'residual_{name}.png'))
    plt.show()

    # Calculate metrics
    lines = [
        f'--- Results for {name} ---',
        f'Bias: {np.mean(residuals)}',
        f'RMSE: {np.sqrt(np.mean(residuals ** 2))}',
        f'MAE:  {np.mean(np.abs(residuals))}'
    ]

    big_bins = [0, -4, -6, -8, -12]
    for i in range(len(big_bins) - 1):
        bin_mask = (height_col < big_bins[i]) & (height_col > big_bins[i+1])
        bin_residual = height_col[bin_mask] - predicted_height[bin_mask]
        lines.append(f'For heights between {big_bins[i+1]} and {big_bins[i]}')
        lines.append(f'Bias: {np.mean(bin_residual)}')
        lines.append(f'RMSE: {np.sqrt(np.mean(bin_residual ** 2))}')
        lines.append(f'MAE:  {np.mean(np.abs(bin_residual))}')

    output_text = "\n".join(lines)
    print(output_text)

    metrics_file = os.path.join(output_dir, f'metrics_{name}.txt')
    with open(metrics_file, 'w') as f:
        f.write(output_text)

def run_fit_pipeline(band_paths, fit_func, p0, bounds=(-np.inf, np.inf), apply_outlier_filter=False, train_name="", val_name=""):
    """
    Main execution routine for training, fitting, and evaluating model pipelines.
    """
    files, bands, transform = load_raster_bands(band_paths)
    
    try:
        # Load datasets
        train_x, train_h, train_uncer, train_mask = extract_csv_pixels(
            "per_pixel_heights_MainAOI.csv", bands, transform, apply_outlier_filter=apply_outlier_filter
        )
        val_x, val_h, _, _ = extract_csv_pixels(
            "per_pixel_heights_SouthernRecon.csv", bands, transform, apply_outlier_filter=apply_outlier_filter
        )

        # Optimization
        fit_inputs = train_x[:, train_mask] if train_x.ndim > 1 else train_x[train_mask]
        popt, _ = curve_fit(
            fit_func,
            fit_inputs,
            train_h[train_mask],
            p0=p0,
            sigma=train_uncer[train_mask],
            bounds=bounds,
            absolute_sigma=True
        )
        print('popt:', popt)

        # Evaluation
        evaluate_and_plot(fit_func, popt, train_x, train_h, name=train_name)
        evaluate_and_plot(fit_func, popt, val_x, val_h, name=val_name)

    finally:
        for f in files.values():
            f.close()