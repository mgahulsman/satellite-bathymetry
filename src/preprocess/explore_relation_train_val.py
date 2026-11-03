import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import pandas as pd

# Create output directory for figures
output_dir = 'data/output/exploritory/'
figs_dir = os.path.join(output_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

# Load raster dataset
log_ratio_path = 'data/processed/aero_over_green.tif'
log_ratio = rio.open(log_ratio_path)

transform = log_ratio.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
im_data = log_ratio.read(1)

def process_csv_data(csv_file: str):
    height_df = pd.read_csv(f'data/processed/{csv_file}')
    height_col = height_df['median_height'].values
    x_col = height_df['true_x'].values
    y_col = height_df['true_y'].values

    x_pix = np.array((x_col - offset_x) / scale_x).astype(int)
    y_pix = np.array((y_col - offset_y) / scale_y).astype(int)
    log_ratio_col = im_data[y_pix, x_pix]

    return log_ratio_col, height_col

def plot_relation(log_ratio_col, height_col, name="set"):
    h_bins = np.linspace(np.min(height_col), np.max(height_col), 101)
    log_bins = np.linspace(np.percentile(log_ratio_col, 0.01), np.percentile(log_ratio_col, 99.9), 101)

    plt.figure(figsize=(10, 10))
    plt.hist2d(log_ratio_col, height_col, bins=[log_bins, h_bins], cmap='magma', cmin=3)
    plt.colorbar()
    plt.xlabel('Log Ratio / NDWI')
    plt.ylabel('NAVD88 Height (m)')
    plt.title(f'Relation - {name}')
    plt.savefig(os.path.join(figs_dir, f'Relation_{name}.png'))
    plt.show()

# Load train set
train_lr, train_h = process_csv_data("per_pixel_heights_MainAOI.csv")
plot_relation(train_lr, train_h, name="train")

val_lr, val_h = process_csv_data("per_pixel_heights_SouthernRecon.csv")
plot_relation(val_lr, val_h, name="val")

# test_lr, test_h = process_csv_data("test_heights.csv")
# plot_relation(test_lr, test_h, name="test")

log_ratio.close()