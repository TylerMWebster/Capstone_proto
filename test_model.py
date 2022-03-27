import pathlib
import sys
from tensorflow import keras
import pandas as pd


PATH = pathlib.Path(__file__).parent.resolve()

model = keras.models.load_model('model.keras')
training = pd.read_csv(PATH / "data/bluestone/predictions/training.csv")
testing = pd.read_csv(PATH / "data/bluestone/predictions/testing.csv")

feature_titles = []
sensor_titles = []
for i in training:
    if len(i) > 15:
        sensor_titles.append(i)
    else:
        feature_titles.append(i)

for i in sensor_titles:    
    target_col = i
    training[list(set(feature_titles)-set(["weather_desc"]))] = training[list(set(feature_titles)-set(["weather_desc"]))] / training[list(set(feature_titles)-set(["weather_desc"]))].max()
    training.drop("weather_desc", axis=1, inplace=True)

    X = training[list(set(feature_titles)-set(["weather_desc"]))].values
    y = training[target_col].values
    
    pred=model.predict(X)
    print(pred)
    
    sys.exit(0)
