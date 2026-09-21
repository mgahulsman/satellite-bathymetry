import os
import numpy as np

cwd = os.getcwd()
data_folder = os.path.join(cwd, r'data\raw')
result_folder = os.path.join(cwd, r'data\processed')
sonar_folder = os.path.join(data_folder, 'Chandeleur_Islands_2023_MBES')
raw_files = os.listdir(os.path.join(data_folder, sonar_folder))

for file in raw_files:
    if file.endswith(".xyz"):
        f = open(os.path.join(sonar_folder, file), 'r')
        line = f.readline()
        minx = np.inf
        maxx = -np.inf
        miny = np.inf
        maxy = -np.inf

        while line != '':
            line = f.readline()
            line = line.strip()
            split = line.split(',')
            if split[0] == '' or split[1] == '':
                continue
            if np.float64(split[0]) < minx:
                minx = np.float64(split[0])
            if np.float64(split[0]) > maxx:
                maxx = np.float64(split[0])
            if np.float64(split[1]) < miny:
                miny = np.float64(split[1])
            if np.float64(split[1]) > maxy:
                maxy = np.float64(split[1])
        f.close()
        file_raw = file.split('.')[0]
        result_file = os.path.join(result_folder, file_raw + '_bbox.txt')
        result_file = open(result_file, 'w')
        result_file.write(f'minx,miny,maxx,maxy\n')
        result_file.write(f'{minx},{miny},{maxx},{maxy}\n')
        print(f'Determined BBox for {file}')
        result_file.close()

