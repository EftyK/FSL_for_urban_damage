"""
Code adapted from https://github.com/barnrang/Prototypical-network-keras-reimplementation
"""
import numpy as np
from keras.utils import np_utils
import tensorflow
import keras
import random
import os
import cv2 

# function to crop np array image
def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, csv_file, data_dir, dim=(128,128), n_channels=3,
                way=4, shot=50, query=50, num_batch=64):
        self.csv_file = csv_file
        self.dataset_path = data_dir
        self.dim = dim
        self.n_channels = n_channels
        self.num_per_class = 50
        self.num_batch = num_batch
        self.build_data(self.csv_file)
        self.on_epoch_end()
        self.way = way
        self.shot = shot
        self.query = query


    def build_data(self, csv_file):
        with open(csv_file, 'r') as csv_file:
            instances = [line.rstrip() for line in csv_file.readlines()]
            label_dict = {}
            for instance in instances[1:]:
                train_inst_id, filename, instance_class = instance.split(',')
                label_dict[filename] = instance_class
            self.class_data = label_dict
        self.n_classes = 4 # TODO: Load dynamically

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_batch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        X_sample, X_query, label = self.__data_generation()
        return [X_sample, X_query], label

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        pass


    def __data_generation(self):
        'Generates data containing batch_size samples'
        # Initialization
        X_sample = np.empty((self.way, self.shot, *self.dim, self.n_channels))
        X_query = np.empty((self.way, self.query, *self.dim, self.n_channels))
        label = np.empty(self.way * self.query)
        for i in range(self.way):
            temp_list = [k for k,v in self.class_data.items() if int(v) == int(i)]
            sample_idx = random.sample(temp_list, self.shot + self.query)
            j = 0
            for img_file in sample_idx[:self.shot]:
                img_path = os.path.join(self.dataset_path, img_file)
                img_array = cv2.imread(img_path)
                img_resized = cv2.resize(img_array, self.dim) 
                X_sample[i][j] = img_resized/255.
                j += 1
            m = 0
            for img_file in sample_idx[self.shot:self.shot + self.query]:
                img_path = os.path.join(self.dataset_path, img_file)
                img_array = cv2.imread(img_path)
                img_resized = cv2.resize(img_array, self.dim) 
                X_query[i][m] = img_resized/255.
                m += 1
            label[i * self.query: (i+1) * self.query] = i
        self.produced_classes = label
        return X_sample, X_query, np_utils.to_categorical(label)