import argparse
import ast
import pathlib
import sys
from tensorflow import keras
import pandas as pd
from constants import *
import numpy as np
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
parser.add_argument("--sensor", "-s", required=True, type=int, help="The path of the folder containing data to analyze")
args = parser.parse_args()

path = pathlib.Path(args.path)

model = keras.models.load_model(path / "ml" / "models" / f"{args.sensor}.keras")

x_test = np.load(path / "ml" / "data" / "x_test.npy", "r")
y_test = np.load(path / "ml" / "data" / "y_test.npy", "r")
x_train = np.load(path / "ml" / "data" / "x_train.npy", "r")
y_train = np.load(path / "ml" / "data" / "y_train.npy", "r")

# print(x_test[0])
# print(x_test[0:1])
index = 0
print(y_train[index:index+1][0])
# data = pd.read_csv(path / "ml" / "data" / f"{args.sensor}_processed.csv")

# feature_columns = data.columns.tolist()
# feature_columns.remove(str(args.sensor))
# target_column = str(args.sensor)

# X = data[feature_columns].values
# y = data[target_column].values

# print(X)

pred = model.predict(x_train[index:index+1])[0][0]
print(pred)



