import argparse
import ast
import pathlib
import os
from constants import *

tensorflow_shutup()

from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os


parser = argparse.ArgumentParser()
parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
parser.add_argument("--sensor", "-s", required=True, type=int, help="The path of the folder containing data to analyze")
parser.add_argument("--index", "-i", required=False, default=-1, type=int, help="The path of the folder containing data to analyze")

args = parser.parse_args()

path = pathlib.Path(args.path)

model = keras.models.load_model(path / "ml" / "models" / f"{args.sensor}.keras")

# x = np.load(path / "ml" / "data" / "x_test.npy", "r")
# y = np.load(path / "ml" / "data" / "y_test.npy", "r")

# x = np.load(path / "ml" / "data" / "x_train.npy", "r")
# y = np.load(path / "ml" / "data" / "y_train.npy", "r")

# x = np.load(path / "ml" / "data" / f"{args.sensor}_x.npy", "r")
# y = np.load(path / "ml" / "data" / f"{args.sensor}_y.npy", "r")

data = pd.read_csv(PATH / args.path / "ml" / "all.csv")

sensor_titles = []
for column_name in data:
    if len(column_name) > 15:
        sensor_titles.append(column_name)


df = data[data[str(args.sensor)]!=-196.6]
df = df[df[str(args.sensor)]!=-159.0]
X = df[FEATURES].values
y = df[str(args.sensor)].values


index = int(args.index)

if index == -1:
    pass
    # pred = model.predict(x)
    # print(f"RMSE: {np.sqrt(mean_squared_error(y,pred))}")
    # print(pd.DataFrame(y-(pred.flatten())).describe())
else:
    actual = y[index:index+1][0]
    # a = [[0.64681867,0.82608696,0.89010989,0.98933075,0.56,0.95305082,0.94,0.2926045,0.43611111]]
    # pred = model.predict(x[index:index+1])[0][0]
    print(args.sensor)
    pred = model.predict(X[index:index+1])[0][0]
    print(f"Actual: {actual}")
    print(f"Prediction: {pred}")
    print(f"Diff: {actual-pred}")



