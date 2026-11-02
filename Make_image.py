import os
import numpy as np
import pandas as pd
import rasterio as rio
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lin(x, a1, a2, a3, a4 , b, c1, c2, c3, c4):
    ag = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    bg = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[2] + c3 * x[4]))
    ar = np.log10(10000 * (x[0] + c1 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    br = np.log10(10000 * (x[1] + c2 * x[4])) / np.log10(10000 * (x[3] + c4 * x[4]))
    return a1*ag + a2*bg + a3*ar + a4*br +b

# Create output directories
output_dir = 'data/output/linear/'
ims_dir = os.path.join(output_dir, 'ims')
os.makedirs(ims_dir, exist_ok=True)

# Path rasters
a_path = 'data/processed/aero.tif'
b_path = 'data/processed/blue.tif'
g_path = 'data/processed/green.tif'
r_path = 'data/processed/red.tif'
nir_path = 'data/processed/nir_smooth.tif'
ndwi_path = 'data/processed/ndwi.tif'

# Load Rasters
a_file = rio.open(a_path)
b_file = rio.open(b_path)
g_file = rio.open(g_path)
r_file = rio.open(r_path)
nir_file = rio.open(nir_path)
ndwi_file = rio.open(ndwi_path)

transform = a_file.transform
proj = a_file.crs
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]
a_data = a_file.read(1)
b_data = b_file.read(1)
g_data = g_file.read(1)
r_data = r_file.read(1)
nir_data = nir_file.read(1)
ndwi_data = ndwi_file.read(1)

x_arr = np.array([a_data,b_data,g_data,r_data,nir_data])

fit_popt = [ 2.19729426e+02, -2.48543073e+02, -2.42883501e+02,  1.44312990e+02, 1.28518440e+02,  2.52026072e-22,  1.00000000e+01 , 2.31025279e+00, 6.85030676e+00]

height_im = lin(x_arr, *fit_popt)
ndwi_mask = np.where(ndwi_data > 0 , True, False)
height_im = np.where(ndwi_mask, height_im, 9999)

save_name = 'chandeleur_im.tif'
save_path = os.path.join(ims_dir, save_name)
save_im = rio.open(
    save_path,
    'w',
    driver='GTiff',
    height=height_im.shape[0],
    width=height_im.shape[1],
    count=1,
    transform=a_file.transform,
    dtype=height_im.dtype,
    crs=a_file.crs,
    nodata=9999,
)
save_im.write(height_im,1)
save_im.close()