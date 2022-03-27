import pathlib
import sys
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

PATH = pathlib.Path(__file__).parent.resolve()

training = pd.read_csv(PATH / "data/bluestone/predictions/testing.csv")
testing = pd.read_csv(PATH / "data/bluestone/predictions/testing.csv")

feature_titles = []
sensor_titles = []
for i in training:
    if len(i) > 15:
        sensor_titles.append(i)
    else:
        feature_titles.append(i)
   
'''        
# def dataframe_to_dataset(dataframe, target):
#     dataframe = dataframe.copy()
#     labels = dataframe.pop(target)
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     ds = ds.shuffle(buffer_size=len(dataframe))
#     return ds
    
# from tensorflow.keras.layers import IntegerLookup
# from tensorflow.keras.layers import Normalization
# from tensorflow.keras.layers import StringLookup

# def encode_numerical_feature(feature, name, dataset):
#     # Create a Normalization layer for our feature
#     normalizer = Normalization()

#     # Prepare a Dataset that only yields our feature
#     feature_ds = dataset.map(lambda x, y: x[name])
#     feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

#     # Learn the statistics of the data
#     normalizer.adapt(feature_ds)

#     # Normalize the input feature
#     encoded_feature = normalizer(feature)
#     return encoded_feature

# def encode_categorical_feature(feature, name, dataset, is_string):
#     lookup_class = StringLookup if is_string else IntegerLookup
#     # Create a lookup layer which will turn strings into integer indices
#     lookup = lookup_class(output_mode="binary")

#     # Prepare a Dataset that only yields our feature
#     feature_ds = dataset.map(lambda x, y: x[name])
#     feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

#     # Learn the set of possible string values and assign them a fixed integer index
#     lookup.adapt(feature_ds)

#     # Turn the string input into integer indices
#     encoded_feature = lookup(feature)
#     return encoded_feature
'''

# descriptors = training[feature_titles]
# descriptors.drop("season")

for i in sensor_titles:    
    target_col = i
    # print(training[list(set(feature_titles)-set(["weather_desc"]))])
    training[list(set(feature_titles)-set(["weather_desc"]))] = training[list(set(feature_titles)-set(["weather_desc"]))] / training[list(set(feature_titles)-set(["weather_desc"]))].max()

    training.drop("weather_desc", axis=1, inplace=True)
    
    training2 = training[training[i] != -159]

    X = training2[list(set(feature_titles)-set(["weather_desc"]))].values
    y = training2[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
    # print(X_train.shape); print(X_test.shape)
    
    model = Sequential()
    model.add(Dense(500, input_dim=9, activation="relu"))
    model.add(Dense(100, activation= "relu"))
    model.add(Dense(50, activation= "relu"))
    model.add(Dense(1))
    
    model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    model.fit(X_train, y_train, epochs=50)
    
    model.save("model.keras")
    
    pred_train = model.predict(X_train)
    # print(np.sqrt(mean_squared_error(y_train,pred_train)))

    pred= model.predict(X_test)
    # print(y_test)
    # print(pred)
    print(np.sqrt(mean_squared_error(y_test,pred)))
    
    
    
    sys.exit(0)
    
    # feature_titles[-1] = i
    # a = training[feature_titles]
    # b = testing[feature_titles]
    
    # train_ds = dataframe_to_dataset(a, i)
    # test_ds = dataframe_to_dataset(b, i)
    
    # train_ds = train_ds.batch(32)
    # test_ds = test_ds.batch(32)
    
    # # Categorical features encoded as integers
    # tod = keras.Input(shape=(1,), name="tod", dtype="int64")
    # season = keras.Input(shape=(1,), name="season", dtype="int64")
    # pressure = keras.Input(shape=(1,), name="pressure", dtype="int64")
    # humidity = keras.Input(shape=(1,), name="humidity", dtype="int64")
    # clouds = keras.Input(shape=(1,), name="clouds", dtype="int64")
    # wind_deg = keras.Input(shape=(1,), name="wind_deg", dtype="int64")

    # # Categorical feature encoded as string
    # weather_desc = keras.Input(shape=(1,), name="weather_desc", dtype="string")

    # # Numerical features
    # temp = keras.Input(shape=(1,), name="temp")
    # dew_pt = keras.Input(shape=(1,), name="dew_pt")
    # wind_speed = keras.Input(shape=(1,), name="wind_speed")

    # all_inputs = [
    #     tod,
    #     season,
    #     pressure,
    #     humidity,
    #     clouds,
    #     wind_deg,
    #     weather_desc,
    #     temp,
    #     dew_pt,
    #     wind_speed,
    # ]

    # # Integer categorical features
    # tod_encoded = encode_categorical_feature(tod, "tod", train_ds, False)
    # season_encoded = encode_categorical_feature(season, "season", train_ds, False)
    # pressure_encoded = encode_categorical_feature(pressure, "pressure", train_ds, False)
    # humidity_encoded = encode_categorical_feature(humidity, "humidity", train_ds, False)
    # clouds_encoded = encode_categorical_feature(clouds, "clouds", train_ds, False)
    # wind_deg_encoded = encode_categorical_feature(wind_deg, "wind_deg", train_ds, False)

    # # String categorical features
    # weather_desc_encoded = encode_categorical_feature(weather_desc, "weather_desc", train_ds, True)

    # # Numerical features
    # temp_encoded = encode_numerical_feature(temp, "temp", train_ds)
    # dew_pt_encoded = encode_numerical_feature(dew_pt, "dew_pt", train_ds)
    # wind_speed_encoded = encode_numerical_feature(wind_speed, "wind_speed", train_ds)
        
    # all_features = layers.concatenate(
    # [
    #     tod_encoded,
    #     season_encoded,
    #     pressure_encoded,
    #     humidity_encoded,
    #     clouds_encoded,
    #     wind_deg_encoded,
    #     weather_desc_encoded,
    #     temp_encoded,
    #     dew_pt_encoded,
    #     wind_speed_encoded,
    # ])
    
    # x = layers.Dense(32, activation="relu")(all_features)
    # x = layers.Dropout(0.5)(x)
    # output = layers.Dense(1, activation="sigmoid")(x)
    # model = keras.Model(all_inputs, output)
    # model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

    # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
    
    # model.fit(train_ds, epochs=50, validation_data=test_ds)
