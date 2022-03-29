import argparse
import ast
import json
import os
import pathlib
import sys
import progressbar
import pandas as pd
import matplotlib.pyplot as plt
from constants import *
tensorflow_shutup()
from tensorflow import keras


class DataAnalyzer:


    def __init__(self, **kwargs) -> None:
        self.raw_data: pd.DataFrame = None
        self.api_temp: pd.DataFrame = None
        self.path = kwargs["path"]
        self.data_path = PATH / self.path
        self.graph = kwargs["graph"]
        self.save: bool = kwargs["save"]
        self.results_name = kwargs["results_name"]
        self.results = {}
        self.forecast_temps = {}
        self.forecast_files = []
        self.sensors_data: dict[int, pd.DataFrame] = {}
        self.ml_df: pd.DataFrame = None
        self.bar = None
        self.ml_result = {}
    
    def start(self):
        # Only reload data if specified. Otherwise use pre-save data
        # if self.save:
        #     self.read_data()
        #     self.clean_data()
        #     self.predict_ml()
        #     self.save_data()
        # else:
        #     self.load_data()
            
        self.load_data()
        self.predict_ml()
                
        self.graph_data()
        
    def read_data(self):  
        print("Reading data...")      
        # Raw data from all sensors
        self.raw_data = pd.read_csv(self.data_path / SENSOR_TEMP_FILENAME, dtype="float64")
        
        # Make seperate sensor arrays
        for i in self.raw_data:
            if i != "time":
                self.sensors_data[str(i).strip()] = self.raw_data[["time", i]]
        
        # Get data from forecast json files
        self.forecast_files = os.listdir(str(self.data_path / "predictions"))
        # self.forecast_files.sort()
        for file in self.forecast_files:
            if file.endswith(".json"):
                name = int(file[:-5])
                self.forecast_temps[name] = (self.get_forecast_temps(name))
        self.api_temp = pd.read_csv(self.data_path / API_TEMP_FILENAME, dtype="float64")

        # Get forecast data
        forecast_times = list(self.forecast_temps.keys())
        forecast_times.sort()
        
        forecast = {}
        forecast["time"] = []
        forecast["time_diff"] = []

        
        for i in FEATURES:
            forecast[i] = []

        for sensor_id in self.sensors_data:
            forecast[str(sensor_id).strip()] = []
            
        bar = progressbar.ProgressBar(maxval=len(forecast_times)).start()
        count = 0
        for forecast_time in forecast_times:
            for i in range(0,48):
                forecast["time"].append(self.forecast_temps[forecast_time][i][0]) # Get the first prediction (index 0) time (index 0)
                forecast["time_diff"].append(self.forecast_temps[forecast_time][i][0]-self.forecast_temps[forecast_time][0][0]) # Get the first prediction (index 0) time (index 0)
                forecast["temp"].append(self.forecast_temps[forecast_time][i][1]) # Get the first prediction (index 0) temperature (index 1)
                forecast["tod"].append(self.forecast_temps[forecast_time][i][2]) # Get the first prediction (index 0) temperature (index 1)
                forecast["season"].append(self.forecast_temps[forecast_time][i][3]) # Get the first prediction (index 0) temperature (index 1)
                forecast["pressure"].append(self.forecast_temps[forecast_time][i][4]) # Get the first prediction (index 0) temperature (index 1)
                forecast["humidity"].append(self.forecast_temps[forecast_time][i][5]) # Get the first prediction (index 0) temperature (index 1)
                forecast["dew_pt"].append(self.forecast_temps[forecast_time][i][6]) # Get the first prediction (index 0) temperature (index 1)
                forecast["clouds"].append(self.forecast_temps[forecast_time][i][7]) # Get the first prediction (index 0) temperature (index 1)
                forecast["wind_speed"].append(self.forecast_temps[forecast_time][i][8]) # Get the first prediction (index 0) temperature (index 1)
                forecast["wind_deg"].append(self.forecast_temps[forecast_time][i][9]) # Get the first prediction (index 0) temperature (index 1)
                for sensor_id in self.sensors_data:
                    sensor_temp, success = self.get_temp_at_time(self.forecast_temps[forecast_time][i][0], self.sensors_data[sensor_id])
                    if success: 
                        forecast[str(sensor_id).strip()].append(sensor_temp)
                    else:
                        forecast[str(sensor_id).strip()].append(-159)
            count += 1
            bar.update(count)
        self.ml_df = pd.DataFrame(forecast)
        self.ml_df[FEATURES] = self.ml_df[FEATURES]/self.ml_df[FEATURES].max()
                   
    def clean_data(self):
        for i in self.sensors_data:
            # Drop NaNs and -196's
            self.sensors_data[i] = self.sensors_data[i][self.sensors_data[i] > -100].dropna()
        # print(self.sensors_data)
            
    def save_data(self):
        print("Saving data...")
        # Save results if specified to
        
        # Create training file and save it
        self.ml_df.to_csv(self.data_path / "ml" / "all.csv", index=False)
        
        with open(self.data_path / "forecast_temps", "w+") as f:f.write(repr(self.forecast_temps))
        
        for i in self.sensors_data:
            self.sensors_data[i].to_pickle(self.data_path / "sensors" / str(i).strip())
            
        results_df = pd.DataFrame(self.results)
        results_df.to_pickle(self.data_path / "results" / self.results_name)

    def load_data(self):
        print("Loading data...")
        # Raw data from all sensors
        self.raw_data = pd.read_csv(self.data_path / SENSOR_TEMP_FILENAME, dtype="float64")
        self.api_temp = pd.read_csv(self.data_path / API_TEMP_FILENAME, dtype="float64")
        self.ml_df = pd.read_csv(self.data_path / "ml" / "all.csv", dtype="float64")
        with open(self.data_path / "forecast_temps", "r") as f: self.forecast_temps = ast.literal_eval(f.read())
        for file in os.listdir(str(self.data_path / "sensors")):
            self.sensors_data[file] = pd.read_pickle(str(self.data_path / "sensors" / file))
        self.results = pd.read_pickle(self.data_path / "results" / self.results_name)
        
    def predict_nodelta(self):
        print("Predicting...")
        self.results["x"] = []
        self.results["y"] = []
        forecast_times = list(self.forecast_temps.keys())
        forecast_times.sort()
        # print(len(forecast_times))
        self.bar = progressbar.ProgressBar(maxval=len(forecast_times)).start()
        count = 0
        for forecast_time in forecast_times:
            start_time = (self.forecast_temps[forecast_time][0][0])
            api_temp, success = self.get_temp_at_time(start_time, self.api_temp)
            # Go through each sensor
            # print(len(self.sensors_data))
            for sensor_id in self.sensors_data:
                # print(sensor_temp)
                # delta, success = self.get_delta(start_time, self.sensors_data[sensor_id], self.api_temp)
                if success:
                    # print(delta)
                    # print(len(self.forecast_temps[forecast_time]))
                    for forecast in self.forecast_temps[forecast_time]:
                        # print(forecast)
                        a = (forecast[0]-start_time)/60
                        # print(a)
                        b = round(forecast[1], 3)
                        # b = round(forecast[1]+delta, 3)
                        # print(b)
                        # sys.exit(0)
                        c,d = self.get_temp_at_time(forecast[0], self.sensors_data[sensor_id])
                        # print(c)
                        # sys.exit(0)
                        if d:
                            self.results["x"].append(a)
                            self.results["y"].append(b-c)
            count += 1
            self.bar.update(count)
            
    def predict_delta(self):
        print("Predicting...")
        self.results["x"] = []
        self.results["y"] = []
        forecast_times = list(self.forecast_temps.keys())
        forecast_times.sort()
        # print(len(forecast_times))
        self.bar = progressbar.ProgressBar(maxval=len(forecast_times)).start()
        count = 0
        for forecast_time in forecast_times:
            start_time = (self.forecast_temps[forecast_time][0][0])
            api_temp, success = self.get_temp_at_time(start_time, self.api_temp)
            # Go through each sensor
            # print(len(self.sensors_data))
            for sensor_id in self.sensors_data:
                sensor_temp, success_2 = self.get_temp_at_time(start_time, self.sensors_data[sensor_id])
                # print(sensor_temp)
                # delta, success = self.get_delta(start_time, self.sensors_data[sensor_id], self.api_temp)
                if success and success_2:
                    delta = sensor_temp - api_temp
                    # print(delta)
                    # print(len(self.forecast_temps[forecast_time]))
                    for forecast in self.forecast_temps[forecast_time]:
                        # print(forecast)
                        a = (forecast[0]-start_time)/60
                        # print(a)
                        # b = round(forecast[1], 3)
                        b = round(forecast[1]+delta, 3)
                        # print(b)
                        # sys.exit(0)
                        c,d = self.get_temp_at_time(forecast[0], self.sensors_data[sensor_id])
                        # print(c)
                        # sys.exit(0)
                        if d:
                            self.results["x"].append(a)
                            self.results["y"].append(b-c)
            count += 1
            self.bar.update(count)
       
    def predict_ml(self):
        print("Predicting...")
        for sensor_id in self.sensors_data:
            df = self.ml_df[self.ml_df[sensor_id]!=-196.6]
            df = df[df[sensor_id]!=-159.0]
            a = df[FEATURES].values
            b = df[sensor_id].values
            model = keras.models.load_model(self.data_path / "ml" / "models" / f"{str(sensor_id)}.keras")
            # print(sensor_id)
            pred = model.predict(a)
            # print(pred)
            # print(b[0:1])
            self.ml_result["time"] = df["time"]
            self.ml_result["temp"] = b
            self.ml_result["pred_temp"] = pred.flatten()
            
            df2 = pd.DataFrame(self.ml_result)
            
            plt.scatter(df2["time"], df2["temp"])
            plt.scatter(df2["time"], df2["pred_temp"])
            plt.show()
            # print(df2)
            
                    
    def get_delta(self, time: int, array_1: pd.DataFrame, array_2: pd.DataFrame) -> tuple:
        temp_1 = array_1.iloc[(array_1['time']-time).abs().argsort()[:1]]
        temp_2 = array_2.iloc[(array_2['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1[temp_1.columns[1]]) - int(temp_2[temp_2.columns[1]]))
        if time_diff > 60:
            return None, False
        diff = round(float(temp_1[temp_1.columns[1]]) - float(temp_2[temp_2.columns[1]]), 3)
        return diff, True
    
    def get_temp_at_time(self, time: int, array: pd.DataFrame) -> tuple:
        temp_1 = array.iloc[(array['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1["time"])-time)
        if time_diff > 60:
            return None, False
        return float(temp_1[temp_1.columns[1]]), True
    
    def get_forecast_temps(self, time: int) -> list:
        forecast_temps: list[tuple[int,float]] = []
        prediction_file = open(self.data_path / "predictions" / f"{time}.json",'r')
        prediction_dict = json.load(prediction_file)
        for j in range(0,48):
            pred_time = int(prediction_dict['hourly'][j]['dt'])
            pred_temp = round(self.k_to_f(prediction_dict['hourly'][j]['temp']), 3)
            pred_tod = int(prediction_dict['hourly'][j]['dt'])//60%1440 # minutes into the day
            pred_season = int(pred_time/86400%365) # days into the year
            pred_pressure = prediction_dict['hourly'][j]['pressure']
            pred_humidity = prediction_dict['hourly'][j]['humidity']
            pred_dew_pt = prediction_dict['hourly'][j]['dew_point']
            pred_clouds = prediction_dict['hourly'][j]['clouds']
            pred_wind_speed = prediction_dict['hourly'][j]['wind_speed']
            pred_wind_deg = prediction_dict['hourly'][j]['wind_deg']
            pred_weather_desc = prediction_dict['hourly'][j]['weather'][0]["description"]
            forecast_temps.append([pred_time, pred_temp, pred_tod, pred_season, pred_pressure,
                                   pred_humidity, pred_dew_pt, pred_clouds, pred_wind_speed,
                                   pred_wind_deg, pred_weather_desc])
        return forecast_temps
        
    def graph_data(self):
        print("Graphing...")
        # Graphs results if specified to
        if self.graph:
            
            # # Plot sensors vs api temp
            # for i in self.sensors_data:
            #     plt.plot(self.sensors_data[i]["time"], self.sensors_data[i][i], color="green")
            # plt.plot(self.api_temp["time"], self.api_temp["temp"], color="orange")
            
            # # Plot results
            # plt.scatter(self.results_x, self.results_y)
            
            # Plot and print stats
            data = {}
            data["time_diff"] = self.results["x"]
            data["temp_diff"] = self.results["y"]
            df = pd.DataFrame(data)
            df = df[df["time_diff"] > 20]
            # df = df[df["time_diff"] < 100]
            # print(df)
            # df1 = df[df["time"] ]
            plt.xlim(-30, 30)
            # plt.ylim(0, 600)
            plt.xlabel("Error (F)", fontsize=15)
            plt.ylabel("Frequency", fontsize=15)
            plt.title("Error Distribution for Unaltered Prediction", fontsize=20)
            plt.hist(df["temp_diff"], bins=100)
            print(df["temp_diff"].describe())
            
            # # Plot api accuracy
            # for name in self.forecast_temps:
            #     a = []
            #     b = []
            #     for forecast in self.forecast_temps[name]:
            #         a.append(forecast[0])
            #         b.append(forecast[1])
            #     plt.plot(a,b)
            plt.show()
                        
    def k_to_f(self, temp):
        return (temp - 273.15) * 1.8 + 32




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-f", required=True, help="The path of the folder containing data to analyze")
    parser.add_argument("--save", "-s", action="store_true", help="Save the resulting analysis")
    parser.add_argument("--graph", "-g", action="store_true", help="Graph the resulting analysis")
    parser.add_argument("--num_sensors", "-n", type=int, required=True, help="The number or sensor columns in the specified file")
    parser.add_argument("--results_name", "-r", type=str, required=True, default="results", help="The name to store the results file with")
    args = parser.parse_args()
    
    da = DataAnalyzer(**vars(args))
    da.start()