import argparse
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from data_generator import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from util.tensor_op import *
from util.loss import *

val_way = 4
shot = 50

RANDOM_SEED = 123

damage_intensity_encoding = dict()
damage_intensity_encoding[3] = '3'
damage_intensity_encoding[2] = '2' 
damage_intensity_encoding[1] = '1' 
damage_intensity_encoding[0] = '0' 



parser = argparse.ArgumentParser(description='Test a model')
parser.add_argument('--path', metavar='/path/to/input_model', required=True, help='full path to saved model')
parser.add_argument('--test_dir', metavar='/path/to/test_dir', required=True, help='full path to the directory of test images')
parser.add_argument('--test_csv', metavar='/path/to/test.csv', required=True, help='full path to the test.csv file')


args = parser.parse_args()
print("###########################", args.path)

model = load_model(args.path, custom_objects={'tf': tf})

df = pd.read_csv(args.test_csv)
df = df.replace({"labels" : damage_intensity_encoding })

test_datagen = DataGenerator(csv_file=args.test_csv , data_dir=args.test_dir , way=4, query=470, shot=50, num_batch=1)

predictions = model.predict(test_datagen)

val_trues = test_datagen.produced_classes
val_pred = np.argmax(predictions, axis=-1)

VAL_PRED = np.array(val_pred).tolist()
print("###################### classification_report #######################")
print(classification_report(val_trues, VAL_PRED))
print("###################### confusion_matrix #######################")
print(confusion_matrix(val_trues, VAL_PRED))