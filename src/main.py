import os
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. PARAMETRIC MODEL DEFINITIONS
# ==========================================

def lin_simple(x, a, b):
    return a * x + b

def lin_full(x, a1, a2, a3, a4, b):
    return a1 * x[0] + a2 * x[1] + a3 * x[2] + a4 * x[3] + b

def lin_simple_corr(x, a, b, c1, c2):
    return a * (np.log10(10000 * (x[0] + c1 * x[2])) / np.log10(10000 * (x[1] + c2 * x[2]))) + b

def lin_full_corr(x, a1, a2, a3, a4, b, c1, c2, c3, c4):
    ag = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    bg = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    ar = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    br = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    return a1 * ag + a2 * bg + a3 * ar + a4 * br + b

# ==========================================
# 2. SHARED UTILITY FUNCTIONS
# ==========================================

def load_raster_bands(band_paths: dict):
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
    df = pd.read_csv(f'data/processed/{csv_file}')
    height_col = df['median_height'].values
    uncer_col = df['mean_uncertainty'].values
    x_col = df['true_x'].values
    y_col = df['true_y'].values

    offset_x, offset_y = transform[2], transform[5]
    scale_x, scale_y = transform[0], transform[4]

    x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
    y_pix = np.array((y_col - offset_y) / scale_y).astype(int)

    sampled_bands = [data[y_pix, x_pix] for data in bands.values()]
    x_arr = np.array(sampled_bands) if len(sampled_bands) > 1 else sampled_bands[0]

    height_mask = height_col > height_mask_h

    if apply_outlier_filter:
        if x_arr.ndim == 1:
            outlier_mask = (x_arr > np.percentile(x_arr, 0.1)) & (x_arr < np.percentile(x_arr, 99.9))
        else:
            outlier_masks = [
                (row > np.percentile(row, 0.1)) & (row < np.percentile(row, 99.9))
                for row in x_arr
            ]
            outlier_mask = np.logical_and.reduce(outlier_masks)
        total_mask = outlier_mask & height_mask
    else:
        total_mask = height_mask

    return x_arr, height_col, uncer_col, total_mask

def evaluate_and_plot(predict_fn, x_data, height_col, name="set", output_dir='data/output/linear/'):
    figs_dir = os.path.join(output_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    predicted_height = predict_fn(x_data)
    valid_mask = ~np.isnan(predicted_height)
    y_bins = np.linspace(np.min(height_col), np.max(height_col), 201)

    # Plot 1: True vs Predicted Height
    plt.figure(figsize=(10, 10))
    plt.hist2d(height_col[valid_mask], predicted_height[valid_mask], bins=y_bins, cmap='jet', cmin=5)
    plt.plot(y_bins, y_bins, color='black', linestyle='dashed')
    plt.colorbar()
    plt.xlabel('True Height')
    plt.ylabel('Predicted Height')
    plt.savefig(os.path.join(figs_dir, f'HeightvsPrediction_{name}.png'))
    plt.show()

    # Plot 2: Residuals
    residuals = height_col - predicted_height
    res_bins = np.linspace(np.percentile(residuals, 0.01), np.percentile(residuals, 99.9), 201)
    plt.figure(figsize=(10, 6))
    plt.hist2d(height_col, residuals, bins=[y_bins, res_bins], cmap='jet', cmin=5)
    plt.axhline(y=0, color='black', linestyle='dashed', zorder=1)
    plt.xlabel('NAVD88 Height (m)')
    plt.ylabel('Residual (m)')
    plt.savefig(os.path.join(figs_dir, f'residual_{name}.png'))
    plt.show()

    # Log metrics calculation
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
    files, bands, transform = load_raster_bands(band_paths)
    
    try:
        train_x, train_h, train_uncer, train_mask = extract_csv_pixels(
            "per_pixel_heights_MainAOI.csv", bands, transform, apply_outlier_filter=apply_outlier_filter
        )
        val_x, val_h, _, _ = extract_csv_pixels(
            "per_pixel_heights_SouthernRecon.csv", bands, transform, apply_outlier_filter=apply_outlier_filter
        )

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

        predict_fn = lambda x: fit_func(x, *popt)

        evaluate_and_plot(predict_fn, train_x, train_h, name=train_name)
        evaluate_and_plot(predict_fn, val_x, val_h, name=val_name)

    finally:
        for f in files.values():
            f.close()

# ==========================================
# 3. INDIVIDUAL EXPERIMENT RUNNERS
# ==========================================

def run_simple_fit():
    run_fit_pipeline(
        band_paths={'ratio': 'data/processed/aero_over_green.tif'},
        fit_func=lin_simple,
        p0=[0, 0],
        apply_outlier_filter=True,
        train_name="train_noglint_ag_nocorr",
        val_name="val_noglint_ag_nocorr"
    )

def run_full_fit():
    run_fit_pipeline(
        band_paths={
            'bg': 'data/processed/blue_over_green.tif',
            'br': 'data/processed/blue_over_red.tif',
            'ag': 'data/processed/aero_over_green.tif',
            'ar': 'data/processed/aero_over_red.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_full,
        p0=[0, 0, 0, 0, 0],
        train_name="train_glint_full_nocorr",
        val_name="val_glint_full_nocorr"
    )

def run_simple_fit_corr():
    run_fit_pipeline(
        band_paths={
            'i': 'data/processed/blue.tif',
            'j': 'data/processed/green.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_simple_corr,
        p0=[-40, 0, 0.1, 0.1],
        bounds=((-np.inf, -np.inf, 0, 0), (np.inf, np.inf, 10, 10)),
        train_name="train_noglint_bg_corr",
        val_name="val_noglint_bg_corr"
    )

def run_full_fit_corr():
    run_fit_pipeline(
        band_paths={
            'a': 'data/processed/aero.tif',
            'b': 'data/processed/blue.tif',
            'g': 'data/processed/green.tif',
            'r': 'data/processed/red.tif',
            'nir': 'data/processed/nir_smooth.tif'
        },
        fit_func=lin_full_corr,
        p0=[0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1],
        bounds=(
            (-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, 10, 10, 10, 10)
        ),
        train_name="train_glint_full_corr",
        val_name="val_glint_full_corr"
    )

def run_rf():
    band_paths = {
        'bg': 'data/processed/blue_over_green.tif',
        'br': 'data/processed/blue_over_red.tif',
        'ag': 'data/processed/aero_over_green.tif',
        'ar': 'data/processed/aero_over_red.tif'
    }
    files, bands, transform = load_raster_bands(band_paths)

    try:
        train_x, train_h, train_uncer, train_mask = extract_csv_pixels(
            "per_pixel_heights_MainAOI.csv", bands, transform, height_mask_h=-12, apply_outlier_filter=True
        )
        val_x, val_h, _, _ = extract_csv_pixels(
            "per_pixel_heights_SouthernRecon.csv", bands, transform, height_mask_h=-12, apply_outlier_filter=False
        )

        x_train_masked = train_x.T[train_mask]
        y_train_masked = train_h[train_mask]
        w_train_masked = 1 / (train_uncer[train_mask] ** 2)

        regressor = RandomForestRegressor(
            n_estimators=150, random_state=0, oob_score=True,
            min_samples_split=30, min_samples_leaf=10
        )
        regressor.fit(x_train_masked, y_train_masked, sample_weight=w_train_masked)

        predict_fn = lambda x: regressor.predict(x.T if x.ndim > 1 else x)

        evaluate_and_plot(predict_fn, train_x, train_h, name="train_noglint_rf")
        evaluate_and_plot(predict_fn, val_x, val_h, name="val_noglint_rf")

    finally:
        for f in files.values():
            f.close()

if __name__ == "__main__":
    run_simple_fit()
    run_full_fit()
    run_simple_fit_corr()
    run_full_fit_corr()
    run_rf()