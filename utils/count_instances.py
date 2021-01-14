from PIL import Image
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import random
import argparse
import logging
import json
import cv2
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)

damage_intensity_encoding = defaultdict(lambda: 0)
damage_intensity_encoding['destroyed'] = 3
damage_intensity_encoding['major-damage'] = 2
damage_intensity_encoding['minor-damage'] = 1
damage_intensity_encoding['no-damage'] = 0


def process_data(input_path):
    """Process Raw Data into

        Args:
            dir_path (path): Path to the xBD dataset.
            data_type (string): String to indicate whether to process
                                train, test, or holdout data.

        Returns:
            x_data: A list of numpy arrays representing the images for training
            y_data: A list of labels for damage represented in matrix form

    """
    x_data = []
    y_data = []

    disasters = [folder for folder in os.listdir(input_path) if not folder.startswith('.')]
    #disaster_paths = ([input_path + "/" +  d + "/images" for d in disasters])
    disaster_paths = ([input_path + "/images"])
    image_paths = []
    image_paths.extend([(disaster_path + "/" + pic) for pic in os.listdir(disaster_path)] for disaster_path in disaster_paths)
    img_paths = np.concatenate(image_paths)

    for img_path in tqdm(img_paths):

        #Get corresponding label for the current image
        label_path = img_path.replace('png', 'json').replace('images', 'labels')
        label_file = open(label_path)
        label_data = json.load(label_file)

        for feat in label_data['features']['xy']:

            # only images post-disaster will have damage type
            try:
                damage_type = feat['properties']['subtype']
            except: # pre-disaster damage is default no-damage
                damage_type = "no-damage"
                continue

            poly_uuid = feat['properties']['uid'] + ".png"

            y_data.append(damage_intensity_encoding[damage_type])
            x_data.append(poly_uuid)

    print("##########################")
    print("0", y_data.count(0))
    print("1", y_data.count(1))
    print("2", y_data.count(2))
    print("3", y_data.count(3))


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--input_dir',
                        required=True,
                        metavar="/path/to/xBD_input",
                        help="Full path to the parent dataset directory")

    args = parser.parse_args()

    logging.info("Started Processing for Data")
    process_data(args.input_dir)
    logging.info("Finished Processing Data")


if __name__ == '__main__':
    main()