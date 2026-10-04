#Import the relevant modules
import numpy as np
import pandas as pd
import os
import rasterio

#Set up data paths
area = 'MainAOI'
cwd = os.getcwd()
base_folder = os.path.join(cwd, r"data\raw")
image_folder = r"S2A_L1C_15_8_2023\GRANULE\L1C_T16RCU_A042551_20230815T164413\IMG_DATA"
img_path = os.path.join(base_folder, image_folder)
blue_filename = "T16RCU_20230815T162901_B02.jp2"
blue_path = os.path.join(img_path, blue_filename)
bbox_file = r'data\processed\2023_325_Chandeleurs_MBES_' + area + '_1m_WGS84_UTM16N_bbox.txt'
wgs84_path = os.path.join(base_folder, r'Chandeleur_Islands_2023_MBES\2023_325_Chandeleurs_MBES_'+area+'_1m_WGS84_UTM16N.xyz')
navd88_path = os.path.join(base_folder, r'Chandeleur_Islands_2023_MBES\2023_325_Chandeleurs_MBES_'+area+'_1m_NAD83_NAVD88_G18.xyz')
out_path = r'data\processed\per_pixel_heights_'+area+'.csv'

#Obtain sentinel-2 image grid parameters
blue = rasterio.open(blue_path)
transform = blue.transform
offset_x = transform[2]
offset_y = transform[5]
scale_x = transform[0]
scale_y = transform[4]

#Use the calculated bounding box to set up a grid where we have data
df = pd.read_csv(bbox_file, sep = ',', dtype = np.float64)
pixel_xmin = ((df['minx'] - offset_x) / scale_x).iloc[0] - 2
pixel_xmax = ((df['maxx'] - offset_x) / scale_x).iloc[0] + 2
pixel_ymin = ((df['miny'] - offset_y) / scale_y).iloc[0] + 2
pixel_ymax = ((df['maxy'] - offset_y) / scale_y).iloc[0] - 2
print(pixel_xmin, pixel_xmax, pixel_ymin, pixel_ymax)
x = np.arange(np.round(pixel_xmin),np.round(pixel_xmax), 1)
y = np.arange(np.round(pixel_ymax),np.round(pixel_ymin), 1)
grid = np.meshgrid(x, y)
#Set up a DataFrame with pixel x and y, WGS84 UTM-16N x and y and empty lists of observations and uncertainties
coords = np.reshape(grid, (2,-1)).T
coords_df = pd.DataFrame(coords, columns = ['pixel_x', 'pixel_y'])
coords_df['true_x'] = coords_df['pixel_x'] * scale_x + offset_x
coords_df['true_y'] = coords_df['pixel_y'] * scale_y + offset_y
coords_df['observations'] = np.empty((len(coords_df),0)).tolist()
coords_df['uncertainty'] = np.empty((len(coords_df),0)).tolist()

#Set up a Space-filling curve index for the dataframe
min_true_x = np.min(coords_df['true_x'])
delta_x = np.max(coords_df['true_x']) - min_true_x
min_true_y = np.min(coords_df['true_y'])
coords_df['SFC'] = coords_df['true_x'] - min_true_x + delta_x * (coords_df['true_y'] - min_true_y)
coords_df.set_index('SFC', inplace=True)

print('done setting up dataframe')

#open the SONAR data files and skip first row
wgs84_coords_file = open(wgs84_path, 'r')
line_wgs84 =wgs84_coords_file.readline()
navd88_heights_file = open(navd88_path, 'r')
line_navd88 = navd88_heights_file.readline()

i = 0 #counter for updates
while line_wgs84 != "":
    #Determine which pixel the sonar value should be attached to.
    line_wgs84 = wgs84_coords_file.readline()
    line_wgs84 = line_wgs84.rstrip("\n")
    wgs84_parts = line_wgs84.split(',')

    if wgs84_parts[0] == '': #for last line I believe
        continue
    #the coords are on the topleft of the pixel
    x_wgs = np.float64(wgs84_parts[0][:-5]) * 10 #Effectively floors wgs84 coord to nearest 10
    y_wgs = np.float64(wgs84_parts[1][:-5]) * 10 + 10 #Effectively ceils wgs84 coord to nearest 10
    sfc_value = x_wgs - min_true_x + delta_x * (y_wgs - min_true_y)

    #Add to the list of the relevant pixel
    line_navd88 = navd88_heights_file.readline()
    line_navd88 = line_navd88.rstrip("\n")
    navd88_parts = line_navd88.split(',')

    height_val = float(navd88_parts[2])
    height_uncertainty = float(navd88_parts[3])

    coords_df.loc[sfc_value,'observations'].append(height_val)
    coords_df.loc[sfc_value,'uncertainty'].append(height_uncertainty)
    i += 1
    if i % 1000000 == 0:
        print(f'Processed {i} lines')

print('done reading in the observations')

#Determine one height per pixel two options median or uncertainty weighted mean
median_heights = []
weighted_mean_heights = []
mean_uncertainty = []
number_of_points_in_cell = []
i=0
for obs, uncer in zip(coords_df['observations'],coords_df['uncertainty']):
    if len(obs) == 0: #Lots of pixels within bbox with no observations
        median_heights.append(np.nan)
        weighted_mean_heights.append(np.nan)
        mean_uncertainty.append(np.nan)
        number_of_points_in_cell.append(np.nan)
    else:
        median_heights.append(np.median(obs))
        weighted_mean_heights.append(np.average(obs, weights= (1/np.power(uncer,2))))
        mean_uncertainty.append(np.mean(uncer)) #Differences in order of 0.1 cm with 4cm uncertainty so heavily correlated
        number_of_points_in_cell.append(len(obs))
    i += 1
    if i % 100000 == 0:
        print(f'Processed {i} sites')
coords_df['median_height'] = median_heights
coords_df['weighted_mean_height'] = weighted_mean_heights
coords_df['mean_uncertainty'] = mean_uncertainty
coords_df['number_of_points'] = number_of_points_in_cell

print('done calculating pixel heights')
coords_df.dropna().to_csv(out_path, index = False, columns = ['true_x', 'true_y', 'median_height','weighted_mean_height','mean_uncertainty','number_of_points'])

#Close files
wgs84_coords_file.close()
navd88_heights_file.close()
blue.close()

