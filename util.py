import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import os
import json

import settings


def trim_json_files():
    """
    this function prepares all json files and removes all training data which ahs no topic classification
    :return: 
    """
    # List directory for training data files
    file_list = os.listdir(settings.PATH_TRAINING_DATA)
    file_names = []
    for file in file_list:
        file_name = file.split('.')[0]
        file_names.append(file_name)

    for idx, file in enumerate(file_list):
        # Import json file
        with open(settings.PATH_TRAINING_DATA + file_list[idx], 'r') as json_file:
            data = json.load(json_file)
            json_file.close()
            # Trim training set for items with no classification
            for article in list(data['TrainingData']):
                if data['TrainingData'][article]['topics'] == []:
                    data['TrainingData'].pop(article)

        # Save trimmed training data
        with open('E:\Python\MultiLabel\data\TrimmedTrainingData\{}.json'.format(file_names[idx]), 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)
            outfile.close()


if __name__ == '__main__':
    trim_json_files()
