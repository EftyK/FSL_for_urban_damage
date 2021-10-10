"""
Code adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/model/damage_inference.py

xview2-baseline Copyright 2019 Carnegie Mellon University. BSD-3

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. 
Please see Copyright notice for non-US Government use and distribution.
"""
import sys
sys.path.append('../protonet') # import protonet modules

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
from sys import exit
import datetime

import shapely.wkt
import shapely
from shapely.geometry import Polygon
from collections import defaultdict

import tensorflow as tf
import keras
from keras.models import load_model

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense

from data_generator import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from util.tensor_op import *
from util.loss import *

damage_intensity_encoding = dict() 
damage_intensity_encoding[3] = 'destroyed' 
damage_intensity_encoding[2] = 'major-damage'
damage_intensity_encoding[1] = 'minor-damage'
damage_intensity_encoding[0] = 'no-damage'


import json
import h5py

def fix_layer0(filename, batch_input_shape, dtype):
    with h5py.File(filename, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
        layer0 = model_config['config'][0]['config']
        layer0['batch_input_shape'] = batch_input_shape
        layer0['dtype'] = dtype
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')


###
# Creates data generator for validation set
###
def create_generator(test_df, test_dir, output_json_path):

    gen = keras.preprocessing.image.ImageDataGenerator(
                             rescale=1.4)

    try:
        gen_flow = gen.flow_from_dataframe(dataframe=test_df,
                                   directory=test_dir,
                                   x_col='uuid',
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   seed=RANDOM_SEED,
                                   class_mode=None,
                                   target_size=(128, 128))
    except:
        # No polys detected so write out a blank json
        blank = {}
        with open(output_json_path , 'w') as outfile:
            json.dump(blank, outfile)
        exit(0)


    return gen_flow

# Runs inference on given test data and pretrained model
def run_inference(test_data, test_csv, model_weights, output_json_path):

    model = load_model(model_weights, custom_objects={'tf': tf})

    df = pd.read_csv(test_csv)
    samples = df["uuid"].count()

    test_gen = DataGenerator(csv_file=test_csv, data_dir=test_data, way=4, query=100, shot=50, num_batch=50)

    predictions = model.predict(test_gen)
    predicted_indices = np.argmax(predictions, axis=1)
    print("predicted_indices", len(predicted_indices))
    predictions_json = dict()

    for i, row in df.iterrows():
        filename_raw = df["uuid"][i]
        filename = filename_raw.split(".")[0]
        try:
            predictions_json[filename] = damage_intensity_encoding[predicted_indices[i]]
        except:
            continue
    with open(output_json_path , 'w') as outfile:
        json.dump(predictions_json, outfile)
        print(outfile)


def main():

    parser = argparse.ArgumentParser(description='Run Building Damage Classification Training & Evaluation')
    parser.add_argument('--test_data',
                        required=True,
                        metavar="/path/to/xBD_test_dir",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--test_csv',
                        required=True,
                        metavar="/path/to/xBD_test_csv",
                        help="Full path to the parent dataset directory")
    parser.add_argument('--model_weights',
                        default=None,
                        metavar='/path/to/input_model_weights',
                        help="Path to input weights")
    parser.add_argument('--output_json',
                        required=True,
                        metavar="/path/to/output_json")

    args = parser.parse_args()

    run_inference(args.test_data, args.test_csv, args.model_weights, args.output_json)


if __name__ == '__main__':
    main()
