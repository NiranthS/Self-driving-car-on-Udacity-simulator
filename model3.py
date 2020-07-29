import pandas as pd # data analysis toolkit - create, read, update, delete datasets
import numpy as np #matrix math
from sklearn.model_selection import train_test_split #to split out training and testing data 
#keras is a high level wrapper on top of tensorflow (machine learning library)
#The Sequential container is a linear stack of layers
from keras.models import Sequential
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam
#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization
#helper class to define input shape and generate training images given image paths & steering angles
from utils_new import INPUT_SHAPE, batch_generator2

from keras.regularizers import l2
#for command line arguments
import argparse
#for reading files
import os

from sklearn.utils import shuffle

import math

import random

#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)
steering_adjustment = 0.25


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    # data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    
    # #yay dataframes, we can select rows and columns by their names
    # #we'll store the camera images as our input data
    # X = data_df[['center', 'left', 'right']].values
    # X=X[1:]
    # #and our steering commands as our output data
    # y = data_df['steering'].values
    # y=y[1:]
    # #now we can split the data into a training (80), testing(20), and validation set
    # #thanks scikit learn
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
    colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), skiprows=[0], names=colnames)
    center = data.center.tolist()
    center_recover = data.center.tolist() 
    left = data.left.tolist()
    right = data.right.tolist()
    steering = data.steering.tolist()
    steering_recover = data.steering.tolist()
    print("12223224444443",len(left))
    ## SPLIT TRAIN AND VALID ##
    #  Shuffle center and steering. Use 10% of central images and steering angles for validation.
    center, steering = shuffle(center, steering)
    center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 

    ## FILTER STRAIGHT, LEFT, RIGHT TURNS ## 
    #  (d_### is list of images name, a_### is list of angles going with list)
    d_straight, d_left, d_right = [], [], []
    a_straight, a_left, a_right = [], [], []
    for i in steering:
      #Positive angle is turning from Left -> Right. Negative is turning from Right -> Left#
      index = steering.index(i)
      if i > 0.15:
        d_right.append(center[index])
        a_right.append(i)
      if i < -0.15:
        d_left.append(center[index])
        a_left.append(i)
      else:
        d_straight.append(center[index])
        a_straight.append(i)

    ## ADD RECOVERY ##
    #  Find the amount of sample differences between driving straight & driving left, driving straight & driving right #
    ds_size, dl_size, dr_size = len(d_straight), len(d_left), len(d_right)
    main_size = math.ceil(len(center_recover))
    l_xtra = ds_size - dl_size
    r_xtra = ds_size - dr_size
    # Generate random list of indices for left and right recovery images
    indice_L = random.sample(range(main_size), l_xtra)
    indice_R = random.sample(range(main_size), r_xtra)

    # Filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
    # while(len(d_left)<(len(d_straight)-1000)):
    for i in indice_L:
        if steering_recover[i] < -0.15:
            d_left.append(right[i])
            a_left.append(steering_recover[i] - steering_adjustment)

    # Filter angle more than 0.15 and add left camera images into driving right list, add an adjustment angle #  
    # while(len(d_right)<(len(d_straight)-1000)):
    for i in indice_R:
        if steering_recover[i] > 0.15:
            d_right.append(left[i])
            a_right.append(steering_recover[i] + steering_adjustment)
    # d_lleft=[]
    # d_lright=[]
    # a_lleft=[]
    # a_lright=[]
    # for i in steering:
    #     index = steering.index(i)
    #     if i > 0.15:
    #         d_lleft.append(left[index])
    #         a_lleft.append(i+)
    #     if i < -0.15:
    #         d_left.append(center[index])
    #         a_left.append(i)



    ## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##  
    X_train = d_straight + d_left + d_right
    y_train = np.float32(a_straight + a_left + a_right)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation:RELU
    Fully connected: neurons: 50, activation: RELU
    Fully connected: neurons: 10, activation: RELU
    Fully connected: neurons: 1 (output)

    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
   
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0-0.5, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2),W_regularizer = l2(0.001)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2),W_regularizer = l2(0.001)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2),W_regularizer = l2(0.001)))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    model.add(Conv2D(64, 3, 3, activation='relu',W_regularizer = l2(0.001)))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    model.add(Conv2D(64, 3, 3, activation='relu',W_regularizer = l2(0.001)))
    # model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

    # model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(1164,activation='relu',W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu',W_regularizer = l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu',W_regularizer = l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='relu',W_regularizer = l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.summary()
    model.load_weights("model-ghosts.h5")
    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('all_models/model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(batch_generator2(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator2(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data/new_data5')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=5)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, *data)
    model.save('model_ghosts_2.h5')


if __name__ == '__main__':
    main()

