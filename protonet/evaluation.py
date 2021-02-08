import argparse
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
#import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

from data_generator import DataGenerator
from model_proto import conv_net, hinge_loss, l2_distance, acc, l1_distance
from util.tensor_op import *
from util.loss import *

val_way = 4
shot = 50

BATCH_SIZE = 32 # TODO: is a big batch size good? - research it in google
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
# model = load_model(args.path)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

df = pd.read_csv(args.test_csv)
df = df.replace({"labels" : damage_intensity_encoding })

# test_datagen1 = datagen.flow_from_dataframe(dataframe=df, 
#                                         directory=args.test_dir, 
#                                         x_col='uuid', 
#                                         y_col='labels', 
#                                         # batch_size=BATCH_SIZE,
#                                         shuffle=False,
#                                         seed=RANDOM_SEED,
#                                         class_mode="categorical",
#                                         target_size=(128, 128))

# test_datagen2 = datagen.flow_from_dataframe(dataframe=df, 
#                                         directory=args.test_dir, 
#                                         x_col='uuid', 
#                                         y_col='labels', 
#                                         # batch_size=BATCH_SIZE,
#                                         shuffle=False,
#                                         seed=RANDOM_SEED,
#                                         class_mode="categorical",
#                                         target_size=(128, 128))


test_datagen1 = DataGenerator(data_type='train', way=4, query=470, shot=50, num_batch=1)
# test_datagen2 = DataGenerator(data_type='validation', way=4, shot=50)

# for x,y in test_datagen:
#     score= model.evaluate_generator(x, y)
#     print(score)

# score = model.evaluate_generator(test_datagen)
# print(score)

# #Evalulate f1 weighted scores on test set
# test_datagen = DataGenerator(data_type='test',way=val_way, shot=shot)

# (x,y), z = test_datagen[0]

# print("############################# shapes", test_datagen1.shape, test_datagen2.shape)

# zipped_input = zip(test_datagen1, test_datagen2)

# (x,y), z = test_datagen1[0]


# predictions = model.predict(test_datagen1)
predictions = model.predict(test_datagen1)
# print("######################### predictions")
# print(len(predictions), predictions.shape)
# print(predictions)


# val_trues = list(test_datagen1.class_data.values())
# val_trues = [int(i) for i in val_trues]
val_trues = test_datagen1.produced_classes
val_pred = np.argmax(predictions, axis=-1)
# print("######################### val_pred")
# print(val_pred)
# print("######################### val_trues")
# print(val_trues)



# print("#########################", len(val_trues), len(val_pred))
# print("#########################", val_trues.shape)
# print("#########################", val_pred.shape)

# # f1_weighted = f1_score(val_trues, val_pred, average='weighted')
# # print(f1_weighted)

VAL_PRED = np.array(val_pred).tolist()
print("###################### classification_report #######################")
print(classification_report(val_trues, VAL_PRED))
print("###################### confusion_matrix #######################")
print(confusion_matrix(val_trues, VAL_PRED))


# layer0 = model.get_layer(index=0)
# layer1 = model.get_layer(index=1)
# layer2 = model.get_layer(index=2)
# layer3 = model.get_layer(index=3)

# print("Layerssssssssssssssssss", layer0, layer1, layer2, layer3)