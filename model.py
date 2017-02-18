import os
import sys
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import keras.models as ksm
import keras.layers as ksl
import keras.layers.pooling as kslp
import keras.layers.convolutional as kslc
import keras.layers.core as ksc
import keras.optimizers as ksopt
import sklearn
from keras import backend as K
import csv

INPUT_SHAPE=(160,320,1)
BATCH_SIZE=36
VAL_SPLIT=0.0
EPOCHS=10
LEARNING_RATE=0.0001
LEARNING_RATE_DECAY=0.00001
STEERING_CORRECTION=0.1
FILE_SAVE='sdc_weights.hdf5'
FILE_LOAD='sdc_weights.hdf5'

K.set_floatx('float32')
K.set_image_dim_ordering('tf')

def find_data_samples_from_file(csvfile, dir, filter=None):
    samples = []
    with open(csvfile) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if filter == None or filter(row):
                row['center'] = dir + row['center']
                row['left'] = dir + row['left']
                row['right'] = dir + row['right']
                samples.append(row)
    return samples

def normalize_image(input_shape):
    return input_shape / 255. - 0.5

def convert_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def filter_road_pixels(img):

    h = img[:,:,0]
    s = img[:,:,1]
    v = img[:,:,2]

    #filter out the areas with a hue/saturation outside of the valid range
    h_filter_lb = h >= 80
    h_filter_ub = h <= 100

    s_filter_lb = s >= 20
    s_filter_ub = s <= 55

    v_filter_lb = v >= 0
    v_filter_ub = v <= 180

    h_filter = np.logical_and(h_filter_lb, h_filter_ub)
    s_filter = np.logical_and(s_filter_lb, s_filter_ub)
    v_filter = np.logical_and(v_filter_lb, v_filter_ub)

    hs_filter = np.logical_and(h_filter, s_filter)
    img_filter = np.logical_and(hs_filter, v_filter)

    #keep only the value pixels for the valid areas, color isn't very important
    #img = v * img_filter.astype('uint8')

    img = v

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    img = img.reshape(INPUT_SHAPE)

    return img

def processImage(pixels):
    pixels = convert_hsv(pixels)
    return filter_road_pixels(pixels)
    #return pixels

def loadImage(img, steering, X, y, bFlip):
    pixels = processImage(mpimg.imread(img))
    X.append(pixels)
    y.append(steering)
    if bFlip:
        X.append(np.fliplr(pixels))
        y.append(-steering)
    return (X, y)


def loadDataGenerator(data, train_opts=None):
    y = []
    X = []

    flip_images = train_opts['flip_images'] if train_opts else False
    use_left_right = train_opts['use_left_right'] if train_opts else False

    while 1:

        np.random.shuffle(data)

        for row in data:
            steering = float(row['steering'])
            loadImage(row['center'], steering, X, y, flip_images)
            if use_left_right:
                loadImage(row['left'], steering + STEERING_CORRECTION, X, y, flip_images)
                loadImage(row['right'], steering - STEERING_CORRECTION, X, y, flip_images)

            if len(X) == BATCH_SIZE:
                yield (np.array(X),y)
                y = []
                X = []

        if len(X) > 0:
            yield (np.array(X), y)
            y = []
            X = []
    ##
    ##imgs_row = 2
    #imgs_col = 4
    #fig, axes = plt.subplots(imgs_row, imgs_col, figsize=(12, 12),
    #                         subplot_kw={'xticks': [], 'yticks': []})

    #fig.subplots_adjust(hspace=0.1, wspace=0.1)

    #for ax, image in zip(axes.flat, X):
    #    ax.imshow(image.reshape(160,320), cmap='gray')

    #plt.suptitle('Input Data')
    #plt.show()
    #plt.close()

def sdc_model(train_opts=None):
    model = ksm.Sequential()

    activation1 = 'relu'
    activation2 = 'sigmoid'

    model.add(ksl.Cropping2D(cropping=((60,20), (0,0)), input_shape=INPUT_SHAPE))
    model.add(ksc.Lambda(normalize_image))
    model.add(kslp.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid'))
    model.add(kslc.Convolution2D(24, 5, 5, activation=activation1, subsample=(2,2), border_mode='valid', init='he_normal', bias=True))
    model.add(kslc.Convolution2D(36, 5, 5, activation=activation1, subsample=(2,2), border_mode='valid', init='he_normal', bias=True))
    model.add(kslc.Convolution2D(48, 3, 5, activation=activation1, subsample=(2,2), border_mode='valid', init='he_normal', bias=True))
    model.add(kslc.Convolution2D(64, 3, 5, activation=activation1, subsample=(2,2), border_mode='valid', init='he_normal', bias=True))
    model.add(kslc.Convolution2D(86, 1, 5, activation=activation1, border_mode='valid', init='he_normal', bias=True))

    model.add(ksc.Flatten())
    model.add(ksc.Dropout(train_opts['dropout_rate'] if train_opts and train_opts['dropout_rate'] else 0))
    model.add(ksc.Dense(100, activation=activation2, init='he_normal', bias=True))
    model.add(ksc.Dense(50, activation=activation2, init='he_normal', bias=True))
    model.add(ksc.Dropout(train_opts['dropout_rate'] if train_opts and train_opts['dropout_rate'] else 0))
    model.add(ksc.Dense(10, activation=activation2, init='he_normal', bias=True))
    model.add(ksc.Dense(1, init='he_normal', bias=True))

    optimizer = ksopt.Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

    return model

def trainAndValidate(model, weights_file, data_train, data_validation=None, train_opts=None):
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    np.random.shuffle(data_train)
    np.random.shuffle(data_validation)

    train_generator = loadDataGenerator(data_train, train_opts)
    val_generator = loadDataGenerator(data_validation, train_opts)

    flip_images = train_opts['flip_images'] if train_opts else False
    use_left_right = train_opts['use_left_right'] if train_opts else False
    mult = 1
    if use_left_right:
        mult += 2
    if flip_images:
        mult *= 2

    model.fit_generator(train_generator, \
        samples_per_epoch=len(data_train * mult), nb_val_samples=len(data_validation * mult), \
        nb_epoch=EPOCHS, \
        verbose=1, \
        validation_data=val_generator if data_validation else None)

    model.save_weights(weights_file)

def predictModel(model, weights_file, data_test):
    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    predict_generator = loadDataGenerator(data_test)

    return model.predict_generator(predict_generator, val_samples=len(data_test))

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

    print(predictions)

    print([x['steering'] for x in test_data])
