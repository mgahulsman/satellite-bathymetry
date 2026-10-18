import numpy as np
import rasterio as rio
import pandas as pd
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import ndimage

data_dir = r'.\data'
image_path = os.path.join(data_dir, 'S2A_MSI_2023_08_15_16_45_16_T16RCU_L2R.nc')
output_dir = os.path.join(data_dir, 'processed')

image_data = nc.Dataset(image_path)

width = image_data.data_dimensions[0]
height = image_data.data_dimensions[1]
proj_string = image_data.proj4_string
xmin = image_data.xrange[0]
xmax = image_data.xrange[1]
ymax = image_data.yrange[0]
ymin = image_data.yrange[1]
pixel = image_data.pixel_size
transform = rio.transform.Affine(pixel[0], 0, xmin, 0, pixel[1], ymax)

aero = image_data['rhos_443'][:]
aero = ndimage.median_filter(aero, size=3)
blue = image_data['rhos_492'][:]
blue = ndimage.median_filter(blue, size=3)
green = image_data['rhos_560'][:]
green = ndimage.median_filter(green, size=3)
red = image_data['rhos_665'][:]
red = ndimage.median_filter(red, size=3)
nir = image_data['rhos_833'][:]
nir = ndimage.median_filter(nir, size=3)


#The processed image contain a few negative values, don't work with log, should discuss
red_neg = np.where(red < 0, 1, 0)
aero_neg = np.where(aero < 0, 1, 0)
green_neg = np.where(green < 0, 1, 0)
blue_neg = np.where(blue < 0, 1, 0)
nir_neg = np.where(nir < 0, 1, 0)
some_neg = aero_neg + blue_neg + green_neg + red_neg + nir_neg
neg_mask = np.where(some_neg > 0, False, True)
print(np.sum(aero_neg), np.sum(blue_neg), np.sum(green_neg), np.sum(red_neg), np.sum(nir_neg))
print(np.sum(~neg_mask))
del aero_neg, blue_neg, green_neg, red_neg, nir_neg, some_neg #memory managing

ndwi = (green - nir)/(green + nir)
print('Calculated NDWI')
n = 10000

aero_over_green = np.log(n*aero) / np.log(n*green)
print('Calculated log band ratio 1/4')
blue_over_green = np.log(n*blue) / np.log(n*green)
print('Calculated log band ratio 2/4')
aero_over_red = np.log(n*aero) / np.log(n*red)
print('Calculated log band ratio 3/4')
blue_over_red = np.log(n*blue) / np.log(n*red)
print('Calculated log band ratio 4/4')

ndwi_path = os.path.join(output_dir, 'ndwi.tif')
ndwi_save = rio.open(ndwi_path,
                     'w',
                     driver='GTiff',
                     height=height,
                     width=width,
                     count=1,
                     dtype=ndwi.dtype,
                     crs=proj_string,
                     transform=transform,
                     )
ndwi_save.write(ndwi, 1)
ndwi_save.close()

aero_over_green_path = os.path.join(output_dir, 'aero_over_green.tif')
aero_over_green_save = rio.open(aero_over_green_path,
                     'w',
                     driver='GTiff',
                     height=height,
                     width=width,
                     count=1,
                     dtype=aero_over_green.dtype,
                     crs=proj_string,
                     transform=transform,
                     )
aero_over_green_save.write(aero_over_green, 1)
aero_over_green_save.close()

aero_over_red_path = os.path.join(output_dir, 'aero_over_red.tif')
aero_over_red_save = rio.open(aero_over_red_path,
                              "w",
                              driver='GTiff',
                              height=height,
                              width=width,
                              count=1,
                              dtype=aero_over_red.dtype,
                              crs=proj_string,
                              transform=transform,
                              )
aero_over_red_save.write(aero_over_red, 1)
aero_over_red_save.close()

blue_over_red_path = os.path.join(output_dir, 'blue_over_red.tif')
blue_over_red_save = rio.open(blue_over_red_path,
                              'w',
                              driver='GTiff',
                              height=height,
                              width=width,
                              count=1,
                              dtype=blue_over_red.dtype,
                              crs=proj_string,
                              transform=transform,
                              )
blue_over_red_save.write(blue_over_red, 1)
blue_over_red_save.close()

blue_over_green_path = os.path.join(output_dir, 'blue_over_green.tif')
blue_over_green_save = rio.open(blue_over_green_path,
                                'w',
                                driver='GTiff',
                                height=height,
                                width=width,
                                count=1,
                                dtype=blue_over_green.dtype,
                                crs=proj_string,
                                transform=transform,
                                )
blue_over_green_save.write(blue_over_green, 1)
blue_over_green_save.close()
