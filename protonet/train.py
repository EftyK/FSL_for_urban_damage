"""
Code adapted from https://github.com/barnrang/Prototypical-network-keras-reimplementation
"""

import argparse
import os
import datetime 

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest='gpu', type=int, default=0)
    parser.add_argument('--data_dir', required=True, metavar='/path/to/data/dir', help="Data directory")
    parser.add_argument('--train_csv', required=True, metavar='/path/to/train/csv', help="Train .csv in the same dir as the data")
    parser.add_argument('--val_csv', required=True, metavar='/path/to/val/csv', help="Test .csv in the same dir as the data")
    parser.add_argument('--model_in', default=None, metavar='/path/to/input_model', help="Path to a saved model")
    parser.add_argument('--model_out', required=True, metavar='/path/to/output_model', help="Path to save model")

    return parser.parse_args()

args = parser()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import tensorflow as tf

from tensorflow.keras import callbacks as cb
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers as rg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import backend as K
from keras.utils.vis_utils import plot_model

import numpy.random as rng
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
from data_generator import DataGenerator
from model import conv_net, hinge_loss, l2_distance, acc, l1_distance
from util.tensor_op import *
from util.loss import *

input_shape = (None,128,128,3)
batch_size = 20
train_way = 4
train_query = 50
val_way = 4
shot = 50
lr = 0.002
LOG_DIR = '/tmp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def scheduler(epoch):
    global lr
    if epoch % 100 == 0:
        lr /= 2
    return lr

# class SaveConv(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch % 50 == 0:
#             save_model(conv, f"model/omniglot_conv_{epoch}_{shot}_{val_way}")

if __name__ == "__main__":
    conv = conv_net()
    conv_5d = TimeDistributed(conv)
    sample = Input(input_shape)
    out_feature = conv_5d(sample)
    out_feature = Lambda(reduce_tensor)(out_feature)
    inp = Input(input_shape)
    map_feature = conv_5d(inp)
    map_feature = Lambda(reshape_query)(map_feature)
    pred = Lambda(proto_dist)([out_feature, map_feature]) #negative distance
    combine = Model([sample, inp], pred)

    # Add model weights if provided by user
    if args.model_in is not None:
        combine.load_weights(args.model_in)

    optimizer = Adam(0.001)

    #Set up tensorboard logging
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                        batch_size=batch_size)

    #Filepath to save model weights
    filepath = args.model_out + "-saved-model-{epoch:02d}-{categorical_accuracy:.2f}.hdf5"
    checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor=['loss', 'categorical_accuracy'],
                                                    verbose=1,
                                                    save_best_only=False,
                                                    mode='max')

    combine.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    train_loader = DataGenerator(csv_file=args.train_csv, data_dir=args.data_dir, way=train_way, query=train_query, shot=shot, num_batch=64)
    val_loader = DataGenerator(csv_file=args.val_csv, data_dir=args.data_dir, way=val_way, shot=shot)

    print(combine.summary())

    reduce_lr = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=2, min_lr=1e-8)
    lr_sched = cb.LearningRateScheduler(scheduler)
    tensorboard = cb.TensorBoard()


    #Training begins
    combine.fit_generator(generator=train_loader,
                        validation_data=val_loader,
                        epochs=50,
                        workers=4,
                        use_multiprocessing=True,
                        callbacks=[tensorboard_callbacks, lr_sched, checkpoints],
                        verbose=1)


    print("END OF TRAINING")