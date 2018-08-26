#!/usr/bin/env python

import json
import rasterio
from rasterio.mask import mask
from os.path import join, isfile, isdir
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_processing import crop_windows, crop_windows_2


def norm(x):
    """
    Normalizes a matrix
    :param x: matrix
    :return:
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


path = r"D:\Projects\Building segmentation\ml"
bad_examples = [8, 46, 88, 98, 108, 119, 132,
                194, 233, 244, 208, 159, 206, 272, 470, 505, 6]
HEIGHT, WIDTH = 650, 650
CHANNELS = 3
# images_save_folder = join(path, 'images_npy')
# masks_save_folder = join(path, 'masks_npy')
images_save_folder = 'D:\\'
masks_save_folder = 'D:\\'

# Reads .tif and geojson (converted in binary matrix)
numbers_to_use = []
for i in range(506):
    geojson = r'buildings\buildings_AOI_3_Paris_img{}.geojson'.format(i)
    tif = r'RGB-PanSharpen_AOI_3_Paris_img{}.tif'.format(i)
    condition = (i in bad_examples) or not (isfile(join(path, geojson))) or not (isfile(join(path, tif)))
    if condition:
        continue
    else:
        numbers_to_use.append(i)

x_data_raw = np.empty(shape=(len(numbers_to_use), HEIGHT, WIDTH, CHANNELS))
y_data_raw = np.empty(shape=(len(numbers_to_use), HEIGHT, WIDTH, 1))

cntr = 0
print('Processing data...')
for i, n in enumerate(numbers_to_use):
    n_image = n

    geojson = r'buildings\buildings_AOI_3_Paris_img{}.geojson'.format(n_image)
    tif = r'RGB-PanSharpen_AOI_3_Paris_img{}.tif'.format(n_image)

    if not isdir(images_save_folder):
        mkdir(images_save_folder)
    if not isdir(masks_save_folder):
        mkdir(masks_save_folder)

    with open(join(path, geojson)) as data_file:
        geoms = json.loads(data_file.read())
    geoms = [feature["geometry"] for feature in geoms["features"]]

    # load the raster, mask it by the polygon and crop it
    with rasterio.open(join(path, tif)) as src:
        temp = np.empty((HEIGHT, WIDTH, CHANNELS))
        for j in range(3):
            temp[:, :, j] = norm(src.read(j + 1))
        x_data_raw[i, :, :, :] = temp
        # np.save(join(images_save_folder, str(i)), temp)
        # plt.imsave(join(images_save_folder, str(i) + '.png'), temp)

        if not len(geoms) == 0:
            out_image, out_transform = mask(src, geoms)

    if len(geoms) == 0:
        y_data_raw[i, :, :, 0] = np.zeros(shape=(HEIGHT, WIDTH))
        cntr += 1
        # plt.imsave(join(masks_save_folder, str(i) + '.png'), np.zeros(shape=(HEIGHT, WIDTH)))
    else:
        mtrx = out_image[2, :, :]
        mtrx[mtrx > 0] = 1
        y_data_raw[i, :, :, 0] = mtrx
        # np.save(join(masks_save_folder, str(i)), mtrx)
        # plt.imsave(join(masks_save_folder, str(i) + '.png'), mtrx)

x_data, x_test, y_data, y_test = train_test_split(x_data_raw, y_data_raw, test_size=0.2, random_state=42)
del x_data_raw, y_data_raw

x_data, y_data = crop_windows_2(x_data, y_data, (256, 256), 3, 0.05, 0.15)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
np.save(join(images_save_folder, 'x_train'), x_train)
np.save(join(images_save_folder, 'x_val'), x_val)
np.save(join(images_save_folder, 'x_test'), x_test)
np.save(join(masks_save_folder, 'y_train'), y_train)
np.save(join(masks_save_folder, 'y_val'), y_val)
np.save(join(masks_save_folder, 'y_test'), y_test)
print('x_data, y_data saved\nGot {} images with masks'.format(len(numbers_to_use)))

# # CODE TO MASK AS TIF
# out_meta = src.meta.copy()

# print(i, out_image.shape)
# # save the resulting raster
# out_meta.update({"driver": "GTiff",
#     "height": out_image.shape[1],
#     "width": out_image.shape[2],
#     "transform": out_transform})

# save_name = 'masks/{}clipped.tif'.format(n_image)

# with rasterio.open(join(path, save_name), "w", **out_meta) as dest:
#     dest.write(out_image)
