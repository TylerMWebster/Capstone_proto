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
parser.add_argument("--index", "-i", required=False, default=0, type=int, help="The path of the folder containing data to analyze")

args = parser.parse_args()

path = pathlib.Path(args.path)

model = keras.models.load_model(path / "ml" / "models" / f"{args.sensor}.keras")

# x = np.load(path / "ml" / "data" / "x_test.npy", "r")
# y = np.load(path / "ml" / "data" / "y_test.npy", "r")

# x = np.load(path / "ml" / "data" / "x_train.npy", "r")
# y = np.load(path / "ml" / "data" / "y_train.npy", "r")

x = np.load(path / "ml" / "data" / "x_total.npy", "r")
y = np.load(path / "ml" / "data" / "y_total.npy", "r")

index = args.index

print(f"Actual: {y[index:index+1][0]}")
pred = model.predict(x[index:index+1])[0][0]
print(f"Prediction: {pred}")



