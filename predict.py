import json
import pandas as pd
import numpy as np
import sys
import os

def k_to_f(temp):
    return (temp - 273.15) * 1.8 + 32

def binary_search(array, name):
    low = 0
    high = len(array)
    mid = len(array)//2
    count = 0

    while abs(array[mid]-name) > 30:
        count += 1
        point = array[mid]
        if count > 100:
            break
        elif point > name:
            high = mid
        elif name > point:
            low = mid
        mid = (high + low) // 2

    return mid

for i in [5]:#,5,6,7,8,9]:
    file = open(f"test_{i}.csv", "w")

    file.write("time_measured,sensor_temp,forecast_temp,estimation_time,time_diff_seconds,predicted_temp_f,actual_temp_f,diff_f\n")

    all_filenames = os.listdir("./api_temp_data")

    for filename in all_filenames:
        if filename.endswith(".json"):
            name = int(filename[:-5])

            # Gets temp of sensor at time of prediction file
            sensor_file = pd.read_csv('11_17_2021_2.csv', header=None, skiprows=2)
            index = binary_search(list(sensor_file[0]), name)
            sensor_temp = sensor_file[i][index]
            # print(f"Sensor Temp: {sensor_temp} F")

            # Gets forcast temp at time of prediction file
            forecast_file = pd.read_csv('api_temp_data/temperature_file.csv', header=None)
            index = binary_search(list(forecast_file[0]), name)
            forecast_temp = forecast_file[1][index]
            # print(f"Forecast Temp: {forecast_temp} F\n")

            # Gets one of the prediction files
            prediction_file = open(f'api_temp_data/{name}.json','r')
            prediction_dict = json.load(prediction_file)
            for j in range(1,5):
                pred_time = prediction_dict['hourly'][j]['dt']
                measured_temp = round(k_to_f(prediction_dict['hourly'][j]['temp']) + (sensor_temp - forecast_temp), 3)
                predicted_temp = sensor_file[1][binary_search(list(sensor_file[0]), pred_time)]
                if not np.isnan(measured_temp) and not np.isnan(predicted_temp) and sensor_temp > -190 and predicted_temp > -190:
                    file.write(f"{name},{pred_time},{pred_time-name},{sensor_temp},{forecast_temp},{measured_temp},{predicted_temp},{round(measured_temp-predicted_temp,3)}\n")

# Predicts future temperature based on current temp, forecast temp, and prediction temp


# Compare predicted future temperature to sensor mesured future temp
