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
import scipy.stats as st
from scipy.stats import uniform, norm, gamma, expon, poisson, binom, bernoulli
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

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

def get_dataset_description(type):
    if(type == 'iris'):
        res = "The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant."
    if(type == 'boston'):
        res = "Contains information about different houses in Boston."
    if(type == 'people'):
        res = "Demographic data collected on individuals."
    if(type == 'cardio'):
        res = "The dataset consists of measurements of fetal heart rate (FHR) and uterine contraction (UC) features on cardiotocograms classified by expert obstetricians."

def example_datasets(target_directory, type):
    if (type == 'iris'):
        use_data = 'iris.csv'
    if (type == 'boston'):
        use_data = 'boston_housing.csv'
    if (type == 'people'):
        use_data = 'people.csv'
    if (type == 'cardio'):
        use_data = 'cardio.csv'
    if (os.path.exists(target_directory)):
        copy_files(**{
            "file_paths": [DATA_PATH + use_data],
            "target_directory": target_directory
        })
    else:
        directory(**{
            "choice": "make",
            "directory_path": target_directory
        })
        copy_files(**{
            "file_paths": [DATA_PATH + use_data],
            "target_directory": target_directory
        })
    return ('Dataset copied to your target directory.')

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

def get_feature_counts(data_frame, feature):
    counts = data_frame[feature].value_counts()
    all_features = list(data_frame[feature].value_counts().index.values)
    res = {}
    for index, ff in enumerate(all_features):
        res[ff] = counts[index]
    return(res)

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

def get_everything_before(string, character):
    res = string.split(character)[0]
    print(res)

def get_everything_after(string, character):
    res = string.split(character, 1)[1]
    print(res)

def get_everything_between(string, first_character, second_character):
    res = re.search(first_character + "(.+?)" + second_character, string).group(0).replace(first_character, '').replace(second_character, '')
    print(res)

def directory(choice, directory_path, force=False):
    if choice == 'make':
        os.mkdir(directory_path)
    if choice == 'remove':
        if force:
            shutil.rmtree(directory_path)
        else:
            os.rmdir(directory_path)

def copy_files(file_paths, target_directory):
    for file in file_paths:
        shutil.copy(file, target_directory)

def move_files(file_paths, target_directory):
    for file in file_paths:
        shutil.move(file, target_directory)

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
        try:
            cv2.imwrite(write_path + 'contour_{}.png'.format(ROI_number), ROI)
        except:
            print("skipping image " + image_path)
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

def remove_nas(original_dataframe, type, threshold, columns):
    if(type == 'all_columns'):
        print("Removing all columns where at least one element is missing.")
        res = original_dataframe.dropna(axis='columns')
    if(type == 'all_rows'):
        print("Removing all rows where all elements are missing.")
        res = original_dataframe.dropna(how='all')
    if(type == 'threshold'):
        print("Keeping only rows with at least " + str(threshold) + " non-NA values.")
        res = original_dataframe.dropna(thresh=threshold)
    if(type == 'by_column'):
        print("Removing all specified columns that contain missing values.")
        res = original_dataframe.dropna(subset=columns)
    return(res)

def convert_to_numbers(original_dataframe, column):
    res = pd.to_numeric(original_dataframe[column], errors='coerce')
    return(res)

# DATA PROFILING
def create_distribution(type, pass_size, pass_location, pass_scale):
    if(type == 'uniform'):
        res = uniform.rvs(size=pass_size, loc=pass_location, scale=pass_scale)
    if (type == 'normal'):
        res = norm.rvs(size=pass_size, loc=pass_location, scale=pass_scale)
    if (type == 'gamma'):
        res = gamma.rvs(a=5, size=pass_size)
    if (type == 'exponential'):
        res = expon.rvs(scale=pass_scale, loc=pass_location, size=pass_size)
    if (type == 'poisson'):
        res = poisson.rvs(mu=3, size=pass_size)
    if (type == 'binomial'):
        res = binom.rvs(n=10, p=0.8, size=pass_size)
    if (type == 'bernoulli'):
        res = bernoulli.rvs(size=pass_size, p=0.6)
    return(res)

def detect_distribution(data):
    res = {}
    dist_names = ["uniform", "norm", "gamma", "expon", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))
    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    res["best_fit_distribution"] = str(best_dist)
    res["best fit p value"] = str(best_p)
    res["best fit parameters"] = str(params[best_dist])
    return(res)

# MODEL TRAINING

## model training

def split_data(original_dataframe, target_feature, split_fraction):
    datasets = {}
    train = original_dataframe.sample(frac=split_fraction, random_state=42)
    test = original_dataframe.drop(train.index)
    train_X = remove_features(**{
        "original_dataframe": train,
        "features_to_remove": [target_feature]
    })
    headers_tr = list(train.columns)
    headers_tr.remove(target_feature)
    train_y = remove_features(**{
        "original_dataframe": train,
        "features_to_remove": headers_tr
    })
    test_X = remove_features(**{
        "original_dataframe": test,
        "features_to_remove": [target_feature]
    })
    headers_te = list(test.columns)
    headers_te.remove(target_feature)
    test_y = remove_features(**{
        "original_dataframe": test,
        "features_to_remove": headers_te
    })
    datasets['train'] = train
    datasets['test'] = test
    datasets['train_X'] = train_X
    datasets['train_y'] = train_y
    datasets['test_X'] = test_X
    datasets['test_y'] = test_y
    return(datasets)

def linear_regression(train_X, train_y, test_X, test_y):
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_y)
    model_results = {}
    predictions = regr.predict(test_X)
    model_results['coefficients'] = regr.coef_
    model_results['mean_squared_error'] = mean_squared_error(test_y, predictions)
    model_results['variance_score'] = r2_score(test_y, predictions)
    model_results['predictions'] = predictions
    return(model_results)


def logistic_regression(train_X, train_y, test_X, test_y):
    logreg = LogisticRegression()
    logreg.fit(train_X, train_y)
    model_results = {}
    predictions = logreg.predict(test_X)
    cnf_matrix = confusion_matrix(test_y, predictions)
    model_results['confusion_matrix'] = cnf_matrix
    model_results['accuracy'] = accuracy_score(test_y, predictions)
    model_results['precision'] = precision_score(test_y, predictions)
    model_results['recall'] = recall_score(test_y, predictions)
    y_pred_prob = logreg.predict_proba(test_X)[::, 1]
    model_results['roc'] = roc_curve(test_y, y_pred_prob)
    model_results['auc'] = roc_auc_score(test_y, y_pred_prob)
    return(model_results)