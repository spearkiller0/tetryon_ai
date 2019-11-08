'''
Copyright (c) 2019 Kedion Inc. (Sean McClure)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import os
import sys
import pkg_resources
import re
import shutil
import json
import glob
import zipfile
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import cv2
from sklearn import linear_model

DATA_PATH = pkg_resources.resource_filename('tetryonai', 'data/')
IMG_PATH = pkg_resources.resource_filename('tetryonai', 'img/')

def check_dependencies(lib_list):
    for lib in lib_list:
        if lib not in sys.modules:
            print(lib + ' not imported. Importing now...')
            os.system("pip install " + lib)

def check_versions(lib_list):
    for lib in lib_list:
        res = os.popen("pip list | grep -F " + lib).read()
        print(res)

def csv_to_dataframe(path_to_csv):
    res = pd.read_csv(path_to_csv)
    return(res)

def example_datasets(target_directory, type):
    if (type == 'iris'):
        if (os.path.exists(target_directory)):
            copy_files(**{
                "file_paths": [DATA_PATH + 'iris.csv'],
                "target_directory": target_directory
            })
        else:
            directory(**{
                "choice": "make",
                "directory_path": target_directory
            })
            copy_files(**{
                "file_paths": [DATA_PATH + 'iris.csv'],
                "target_directory": target_directory
            })
        return ('Dataset copied to your target directory.')


    if(data == 'iris'):
        res = csv_to_dataframe(**{
            "path_to_csv": DATA_PATH + "iris.csv"
        })
    return(res)

def example_images(target_directory, type):
    if(type == 'defects'):
        if(os.path.exists(target_directory)):
            copy_files(**{
                "file_paths" : [IMG_PATH + 'template.jpg', IMG_PATH + 'test.jpg'],
                "target_directory" : target_directory
            })
        else:
            directory(**{
                "choice": "make",
                "directory_path": target_directory
            })
            copy_files(**{
                "file_paths": [IMG_PATH + 'template.jpg', IMG_PATH + 'test.jpg'],
                "target_directory": target_directory
            })
        return('Images copied to your target directory.')

def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

def remove_features(original_dataframe, features_to_remove):
    res = original_dataframe.drop(features_to_remove, 1)
    return(res)

def filter_observations(original_dataframe, column, filter_choice, filter_value):
    if(filter_choice == 'remove'):
        res = original_dataframe[original_dataframe[column] != filter_value]
    if (filter_choice == 'keep'):
        res = original_dataframe[original_dataframe[column] == filter_value]
    return(res)

def sum_frame_by_column(original_dataframe, new_column_name, columns_to_sum):
    res_cp = original_dataframe.copy()
    res_cp[new_column_name] = res_cp[columns_to_sum].astype(float).sum(axis=1)
    return(res_cp)

def scrape_website(url, element):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    res = soup.findAll(element)
    return(res)

def get_everything_between(string, first_character, second_character):
    res = re.search(first_character + "(.+?)" + second_character, string).group(0).replace(first_character, '').replace(second_character, '')
    print(res)

def directory(choice, directory_path):
    if(choice == 'make'):
        os.mkdir(directory_path)
    if (choice == 'remove'):
        os.rmdir(directory_path)

def copy_files(file_paths, target_directory):
    for file in file_paths:
        shutil.copy(file, target_directory)

def subtract_images(image_path_1, image_path_2, write_path):
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)
    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]
    cv2.imwrite(write_path, image1)

def read_files_by_pattern(folder, pattern):
    res = glob.glob(folder + '/*' + pattern)
    return(res)

def find_in_list(pass_list, find):
    res = next((x for x in pass_list if find in x), None)
    return(res)

def apply_to_list(pass_list, function):
    res = list(map(function, pass_list))
    return(res)

def extract_contours_from_image(image_path, write_path, hsv_lower, hsv_upper):
    image = cv2.imread(image_path)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(hsv_lower)
    hsv_upper = np.array(hsv_upper)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 20
    ROI_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (36, 255, 12), 2)
        ROI = original[y - offset:y + h + offset, x - offset:x + w + offset]
        cv2.imwrite(write_path + 'contour_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

def write_dict_as_json(write_path, pass_dict):
    with open(write_path, 'w') as outfile:
        json.dump(pass_dict, outfile)

def zip_files_in_directory(directory, zip_filename):
    zf = zipfile.ZipFile(zip_filename, "w")
    for dirname, subdirs, files in os.walk(directory):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

# MACHINE LEARNING

## MODEL TRAINING

def linear_regression(data):
    reg = linear_model.LinearRegression()
    reg.fit(data)
    model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    return(model)





