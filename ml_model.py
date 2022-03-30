import argparse
from hashlib import new
import pathlib
import sys
from constants import *
tensorflow_shutup()
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
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

EPOCHS = 70

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
parser.add_argument("--save", "-s", action="store_true", help="Save the resulting analysis")
parser.add_argument("--graph", "-g", action="store_true", help="Graph the resulting analysis")
args = parser.parse_args()

data = pd.read_csv(PATH / args.path / "ml" / "all.csv")

sensor_titles = []
for column_name in data:
    if len(column_name) > 15:
        sensor_titles.append(column_name)

for sensor_id in sensor_titles:  
        
        target_col = sensor_id

        df = data[data[sensor_id]!=-196.6]
        df = df[df[sensor_id]!=-159.0]
        
        X = df[FEATURES].values
        y = df[sensor_id].values
        
        X1 = df[FEATURES].values[0:int(len(X)/2)]
        y1 = df[sensor_id].values[0:int(len(y)/2)]
        
        X2 = df[FEATURES].values[int(len(X)/2):]
        y2 = df[sensor_id].values[int(len(y)/2):]
        
        X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

        # np.save(PATH / args.path / "ml" / "data" / f"{sensor_id}_x", X)
        # np.save(PATH / args.path / "ml" / "data" / f"{sensor_id}_y", y)
        
        model = Sequential()
        model.add(Dense(500, input_dim=9, activation="relu"))
        model.add(Dense(100, activation= "relu"))
        model.add(Dense(50, activation= "relu"))
        model.add(Dense(1))
        
        model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
        model.fit(X_train, y_train, epochs=EPOCHS)
        
        
        pred_train = model.predict(X_train)
        train_err = (np.sqrt(mean_squared_error(y_train,pred_train)))

        pred = model.predict(X_test)
        test_err = (np.sqrt(mean_squared_error(y_test,pred)))
        
        pred_new = model.predict(X2)
        new_err = (np.sqrt(mean_squared_error(y2,pred_new)))
        
        print(f"XXXXXXXXX {new_err} XXXXXXXXX")
        
        if args.save:
            model.save(PATH / args.path / "ml" / "models" / f"{sensor_id}_newmodel.keras")

        
        # with open("results_2.txt", "a+") as file:
        #     file.write(f"{epochs},{train_err},{test_err},{new_err}\n")
        
    