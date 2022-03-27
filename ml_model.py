import argparse
import pathlib
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from constants import *
from tensorflow.keras import layers
# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
parser.add_argument("--save", "-s", action="store_true", help="Save the resulting analysis")
parser.add_argument("--graph", "-g", action="store_true", help="Graph the resulting analysis")
args = parser.parse_args()

training = pd.read_csv(PATH / args.path / "ml" / "testing.csv")
testing = pd.read_csv(PATH / args.path / "ml" / "testing.csv")

feature_titles = []
sensor_titles = []
for column_name in training:
    if len(column_name) > 15:
        sensor_titles.append(column_name)
    else:
        feature_titles.append(column_name)

for sensor_id in sensor_titles:    
    target_col = sensor_id
    # print(training[list(set(feature_titles)-set(["weather_desc"]))])
    training[list(set(feature_titles)-set(["weather_desc"]))] = \
        training[list(set(feature_titles)-set(["weather_desc"]))]/ \
        training[list(set(feature_titles)-set(["weather_desc"]))].max()

    training.drop("weather_desc", axis=1, inplace=True)
    
    training2 = training[training[sensor_id] != -159]

    X = training2[list(set(feature_titles)-set(["weather_desc"]))].values
    y = training2[target_col].values
    
    features = feature_titles[0:-1] + [sensor_id]
    # training2.to_csv(PATH / args.path / "ml" / "data" / f"{sensor_id}_processed.csv", index=False, columns=features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    np.save(PATH / args.path / "ml" / "data" / "x_test", X_test)
    np.save(PATH / args.path / "ml" / "data" / "y_test", y_test)
    np.save(PATH / args.path / "ml" / "data" / "x_train", X_train)
    np.save(PATH / args.path / "ml" / "data" / "y_train", y_train)
    np.save(PATH / args.path / "ml" / "data" / "x_total", X)
    np.save(PATH / args.path / "ml" / "data" / "y_total", y)
    
    model = Sequential()
    model.add(Dense(500, input_dim=9, activation="relu"))
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    model.fit(X_train, y_train, epochs=50)
    
    model.save(PATH / args.path / "ml" / "models" / f"{sensor_id}.keras")
    
    pred_train = model.predict(X_train)
    print(np.sqrt(mean_squared_error(y_train,pred_train)))

    pred= model.predict(X_test)
    # print(y_test)
    # print(pred)
    print(np.sqrt(mean_squared_error(y_test,pred)))

    sys.exit(0)
    