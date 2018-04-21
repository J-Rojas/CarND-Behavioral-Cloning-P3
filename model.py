import os
import sys
import cv2
import tensorflow as tf
import math
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import keras
import keras.models as ksm
import keras.layers as ksl
import keras.layers.pooling as kslp
import keras.layers.convolutional as kslc
import keras.layers.core as ksc
import keras.optimizers as ksopt
import keras.preprocessing.image as kimg
import sklearn
from keras import backend as K
import csv

#
# CONSTANTS
#

yuv_scale = np.array([[0.299, 0.587, 0.114],
                [-0.14713, -0.28886, 0.436],
                [0.615, -0.51499, -0.10001]])

rgb2yuv = np.array([[0.299, 0.587, 0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]])

gray_scale = np.array([[1.0, 0, 0]])

#
# HYPER-PARAMETERS
#

INPUT_SHAPE=(80,320,1)
BATCH_SIZE=128
VAL_SPLIT=0.0
EPOCHS=10
VAL_SPLIT=0.0
PREV_STEERING_TIME=525
LEARNING_RATE=0.0001
LEARNING_RATE_DECAY=0.00001
STEERING_CORRECTION=0.05
FILE_SAVE='sdc_weights.hdf5'
FILE_LOAD='sdc_weights.hdf5'

K.set_floatx('float32')
K.set_image_dim_ordering('tf')

#
# KERAS UTILITIES
#
class ScaleLayer(ksl.Layer):
    def __init__(self, output_dim, scale, **kwargs):
        self.output_dim = output_dim
        self.scale = scale
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = K.variable(self.scale)
        super(ScaleLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        config['scale'] = self.scale.tolist()
        config['output_dim'] = self.output_dim
        return config

    @classmethod
    def from_config(cls, config):

        scale = None
        output_dim = None
        if 'scale' in config:
            scale = config['scale']
            config.pop('scale', None)
        if 'output_dim' in config:
            output_dim = config['output_dim']
            config.pop('output_dim', None)

        return cls(output_dim, scale, **config)

# Custom Keras History

class LossHistory(keras.callbacks.Callback):

    def __init__(self):
        self.losses = []
        self.accum = {}

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accum = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.accum = {}

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()
def metric_min(y, y_pred):
    if 'metric_min' in history.accum:
        val = history.accum['metric_min']= K.min(history.accum['metric_min'], K.abs(y-y_pred))
    else:
        val = history.accum['metric_min']= K.abs(y-y_pred)
    print(K.shape(val))
    return val

def metric_max(y, y_pred):
    if 'metric_max' in history.accum:
        val = K.max(history.accum['metric_max'], K.abs(y-y_pred))
    else:
        val = K.abs(y-y_pred)
    history.accum['metric_max'] = val
    print(K.eval(val))
    return val

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

ksl.ScaleLayer = ScaleLayer

# Load CSV File data

def find_data_samples_from_file(csvfile, dir, filter=None):
    samples = []
    with open(csvfile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if filter == None or filter(row):
                row['center'] = (dir + row['center']).replace(' ', '')
                row['left'] = (dir + row['left']).replace(' ', '')
                row['right'] = (dir + row['right']).replace(' ', '')
                samples.append(row)

    samples = sorted(samples, key=lambda k: k['center'])

    prev_row = None
    prev_time = 0
    for row in samples:
        time = row['center'].replace('.jpg', '').split('_')
        timestamp = int(time[-2]) * 1000 + int(time[-1])
        if prev_row:
            prev_row['steering_ahead'] = float(row['steering']) if timestamp - prev_time < PREV_STEERING_TIME else 0
            prev_row['steering_time'] = timestamp - prev_time
            #print(timestamp, prev_time, timestamp - prev_time, prev_row['steering_ahead'], prev_row['steering_time'])
        prev_steering = float(row['steering'])
        prev_row = row
        prev_time = timestamp

    #print(samples)

    return samples

#
# Input preprocessing utilities
#

def normalize_image(input_shape):
    input_shape -= K.mean(input_shape)
    input_shape /= (K.std(input_shape) + 0.0001) * 1.5
    return input_shape

def convert_scale(input_shape, scale=None):
    return K.dot(input_shape, scale)

def convert_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#
# Perform image filtering (GaussianBlur) and normalization.
#

def filter_road_pixels(img):

    img = img.astype('uint8')

    img = cv2.GaussianBlur(img, (5, 5), 2.0)
    img = img.reshape(INPUT_SHAPE).astype('float32')

    img -= np.mean(img)
    img /= (np.std(img) + 0.0001)

    return img

# Perform YUV conversion, grayscaling and clipping

def processImage(pixels, trans=0):

    pixels = np.dot(pixels, rgb2yuv.T)
    pixels = np.dot(pixels, gray_scale.T)

    #crop rows from 60 to 140 - we do this here for better contrast normalization
    pixels = pixels[60+trans:140+trans,:]

    return filter_road_pixels(pixels)

def loadImage(img, steering, X, y, bFlip, trans):
    loaded_pixels = mpimg.imread(img)
    pixels = processImage(loaded_pixels)
    #pixels = mpimg.imread(img)
    X.append(pixels)
    y.append(steering)
    if bFlip:
        X.append(np.fliplr(pixels))
        y.append(-steering)
    if trans:
        pixels = processImage(loaded_pixels, trans)
        X.append(pixels)
        y.append(steering)
        if bFlip:
            X.append(np.fliplr(pixels))
            y.append(-steering)
    return (X, y)

def showImages(images):
    imgs_row = 8
    imgs_col = 4
    fig, axes = plt.subplots(imgs_row, imgs_col, figsize=(12, 12),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax, image in zip(axes.flat, images):
        ax.imshow(image.squeeze(), cmap='gray')

    plt.suptitle('Input Data')
    plt.show()
    plt.close()

#
# Image loading generator - loads source images and performs data augmentation (flips, translations, etc)
#

def loadDataGenerator(data, train_opts=None):
    y = []
    X = []

    flip_images = train_opts['flip_images'] if train_opts and 'flip_images' in train_opts else False
    use_left_right = train_opts['use_left_right'] if train_opts and 'use_left_right' in train_opts else False
    use_trans = train_opts['use_trans'] if train_opts and 'use_trans' in train_opts else 0
    shuffle = train_opts['shuffle'] if train_opts and 'shuffle' in train_opts else True

    while 1:

        if shuffle:
            np.random.shuffle(data)

        for row in data:
            steering = float(row['steering'])
            loadImage(row['center'], steering, X, y, flip_images, use_trans)
            if use_left_right:
                loadImage(row['left'], steering + STEERING_CORRECTION, X, y, flip_images, 0)
                loadImage(row['right'], steering - STEERING_CORRECTION, X, y, flip_images, 0)

            if len(X) == BATCH_SIZE:
                # shuffle the batch to prevent any bias based on the order of augmented data
                arrays = sklearn.utils.shuffle(X, y)
                yield (np.array(arrays[0]), arrays[1])
                y = []
                X = []

        if len(X) > 0:
            arrays = sklearn.utils.shuffle(X, y)
            yield (np.array(arrays[0]), arrays[1])
            y = []
            X = []

#
# Model Definition
#
# The model is based on NVIDIA's drive architecture. The model has been slightly tweaked to reduce the
# number of parameters within the model and decrease training time.
#

def sdc_model(train_opts=None):
    model = ksm.Sequential()

    activation1 = 'relu'
    activation2 = 'sigmoid'
    initializer = 'he_normal'
    regularizer = None #keras.regularizers.l2(0.0)

    model.add(kslp.AveragePooling2D(pool_size=(1, 2), strides=None, padding='valid', input_shape=INPUT_SHAPE))
    model.add(kslc.Convolution2D(12, kernel_size=(5, 5), activation=activation1, strides=(2,2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(kslc.Convolution2D(24, kernel_size=(5, 5), activation=activation1, strides=(2,2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(kslc.Convolution2D(36, kernel_size=(5, 5), activation=activation1, padding='valid', kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(kslp.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(kslc.Convolution2D(40, kernel_size=(3, 3), activation=activation1, padding='valid', kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(kslp.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(ksc.Flatten())
    model.add(ksc.Dropout(train_opts['dropout_rate'] if train_opts and train_opts['dropout_rate'] else 0))
    model.add(ksc.Dense(100, activation=activation2, kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(ksc.Dropout(train_opts['dropout_rate'] if train_opts and train_opts['dropout_rate'] else 0))
    model.add(ksc.Dense(50, activation=activation2, kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(ksc.Dropout(train_opts['dropout_rate'] if train_opts and train_opts['dropout_rate'] else 0))
    model.add(ksc.Dense(10, activation=activation2, kernel_initializer='he_normal', kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=True))
    model.add(ksc.Dense(1, init=initializer, bias=True))

    optimizer = ksopt.Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_absolute_error'])

    return model

#
# Training and Validation
#
# These functions will take a model and a list of training data (images, labels) and a list
# of validation data to performing training and validation using Keras' 'fit_generator'.
#
# The images are loaded via a generator function and
#

def trainAndValidate(model, weights_file, data_train, data_validation=None, train_opts=None):
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    show_generator = loadDataGenerator(data_train, train_opts)
    train_generator = loadDataGenerator(data_train, train_opts)

    #it's unnecessary to flip validation images
    val_opts = train_opts.copy() if train_opts else dict()
    val_opts['flip_images'] = False

    val_generator = loadDataGenerator(data_validation, val_opts)

    #showImages(next(show_generator)[0])

    steps = math.ceil(len(data_train) / BATCH_SIZE)
    val_steps = math.ceil(len(data_validation) / BATCH_SIZE)

    model.fit_generator(train_generator, \
        steps_per_epoch=steps, validation_steps=val_steps, \
        epochs=train_opts['epochs'] if train_opts and train_opts['epochs'] else EPOCHS, \
        verbose=1, \
        callbacks=[history], \
        validation_data=val_generator if data_validation else None)

    model.save_weights(weights_file)

#
# Load model and predict (no training)
#

def predictModel(model, weights_file, data_test):
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    predict_generator = loadDataGenerator(data_test, {'shuffle': False})

    return model.predict_generator(predict_generator, val_samples=len(data_test))

#
# Wrapper functions for loading, training and prediction
#

def loadAndTrain(csv_file_train, csv_file_test, weights_file, dir_train, dir_validate = None, train_opts=None):

    train_data = find_data_samples_from_file(csv_file_train, dir_train)
    val_data = None
    if dir_validate:
        val_data = find_data_samples_from_file(csv_file_test, dir_validate)

    model = sdc_model(train_opts)
    model.summary()

    trainAndValidate(model, weights_file, train_data, val_data, train_opts)

    if train_opts and train_opts['save_file']:
        model.save(train_opts['save_file'])

def predict(csv_file, weights_file, dir_test):

    test_data = find_data_samples_from_file(csv_file, dir_test)
    model = sdc_model()

    predictions = predictModel(model, weights_file, test_data)

    predictions = predictions.flatten()

    print(predictions)

    steering = np.array([x['steering'] for x in test_data]).flatten().astype(np.float32)

    print(steering)

    print(predictions - steering)
