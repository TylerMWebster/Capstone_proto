import argparse
import ast
import json
import os
import pathlib
import sys

import pandas as pd
import matplotlib.pyplot as plt


class DataAnalyzer:

    PATH = pathlib.Path(__file__).parent.resolve()
    API_TEMP_FILENAME = "api_temp_file_2.csv"
    SENSOR_TEMP_FILENAME = "sensor_temp.csv"

    def __init__(self, **kwargs) -> None:
        self.raw_data: pd.DataFrame = None
        self.api_temp: pd.DataFrame = None
        self.path = kwargs["path"]
        self.data_path = self.PATH / self.path
        self.graph = kwargs["graph"]
        self.save: bool = kwargs["save"]
        self.results_x = []
        self.results_y = []
        self.forecast_temps = {}
        self.forecast_files = []
        self.sensors_data: dict[int, pd.DataFrame] = {}
    
    def start(self):
        # Only reload data if specified. Otherwise use pre-save data
        if self.save:
            self.read_data()
            self.clean_data()
            self.save_data()
        else:
            self.load_data()
        
        # Start loop for each prediction
        self.predict()
        
        self.graph_data()
        
    def read_data(self):
        # try:
        #     for file in os.listdir(str(self.data_path)):
        #         if file.endswith(".csv"):
        #             csv_file = file
        # except Exception as e:
        #     print(e.message)
        #     print("invalid directory. Directory must contain a csv file of sensor data and a folder named \"predictions\"")
        #     sys.exit(1)
                
        # for file in os.listdir(str(self.data_path / "predictions")):
        #     if file.endswith(".csv"):
        #         api_temp_file = file
        
        # Raw data from all sensors
        self.raw_data = pd.read_csv(self.data_path / self.SENSOR_TEMP_FILENAME, dtype="float64")
        
        # Minute-by-minute data from api
        self.api_temp = pd.read_csv(self.data_path / "predictions" / self.API_TEMP_FILENAME, dtype="float64")
        
        # # Start time at 0
        # self.raw_data["time"] = self.raw_data["time"] - self.raw_data["time"][0]
        # self.api_temp["time"] = self.api_temp["time"] - self.api_temp["time"][0]
        
        # Make seperate sensor arrays
        for i in self.raw_data:
            if i != "time":
                self.sensors_data[i] = self.raw_data[["time", i]]
        
        # Get data from forecast json files
        self.forecast_files = os.listdir(str(self.data_path / "predictions"))
        # self.forecast_files.sort()
        for file in self.forecast_files:
            if file.endswith(".json"):
                name = int(file[:-5])
                self.forecast_temps[name] = (self.get_forecast_temps(name))
                        
    def clean_data(self):
        for i in self.sensors_data:
            # Drop NaNs and -196's
            self.sensors_data[i] = self.sensors_data[i][self.sensors_data[i] > -100].dropna()
        # print(self.sensors_data)
            
    def save_data(self):
        # Save results if specified to
        forecast_times = list(self.forecast_temps.keys())
        forecast_times.sort()
        a = {}
        a["time"] = []
        a["temp"] = []
        for forecast_time in forecast_times:
            # print(self.forecast_temps[forecast_time][0][1]) # Get the temperature (index 1) of the first prediction (index 0)
            a["time"].append(self.forecast_temps[forecast_time][0][0]) # Get the first prediction (index 0) time (index 0)
            a["temp"].append(self.forecast_temps[forecast_time][0][1]) # Get the first prediction (index 0) temperature (index 1)
            
        df = pd.DataFrame(a)
        df.to_csv(self.data_path / "predictions" / "temperature_file_2.csv", index=False)
        
        with open(self.data_path / "forecast_temps", "w+") as f:f.write(repr(self.forecast_temps))
        
        for i in self.sensors_data:
            self.sensors_data[i].to_pickle(self.data_path / "sensors" / str(i))

    def load_data(self):
        # Raw data from all sensors
        self.raw_data = pd.read_csv(self.data_path / self.SENSOR_TEMP_FILENAME, dtype="float64")
        self.api_temp = pd.read_csv(self.data_path / "predictions" / self.API_TEMP_FILENAME, dtype="float64")
        with open(self.data_path / "forecast_temps", "r") as f: self.forecast_temps = ast.literal_eval(f.read())
        for file in os.listdir(str(self.data_path / "sensors")):
            self.sensors_data[file] = pd.read_pickle(str(self.data_path / "sensors" / file))
        
    def predict(self):
        forecast_times = list(self.forecast_temps.keys())
        forecast_times.sort()
        # print(self.forecast_temps)
        for forecast_time in forecast_times:
            start_time = (self.forecast_temps[forecast_time][0][0])
            api_temp, success = self.get_temp_at_time(start_time, self.api_temp)
            # Go through each sensor
            for sensor_id in self.sensors_data:
                sensor_temp, success_2 = self.get_temp_at_time(start_time, self.sensors_data[sensor_id])
                # print(sensor_temp)
                # delta, success = self.get_delta(start_time, self.sensors_data[sensor_id], self.api_temp)
                if success and success_2:
                    delta = sensor_temp - api_temp
                    # print(delta)
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
                            self.results_x.append(a)
                            self.results_y.append(b-c)
                    
    def get_delta(self, time: int, array_1: pd.DataFrame, array_2: pd.DataFrame) -> tuple[float, bool]:
        temp_1 = array_1.iloc[(array_1['time']-time).abs().argsort()[:1]]
        temp_2 = array_2.iloc[(array_2['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1[temp_1.columns[1]]) - int(temp_2[temp_2.columns[1]]))
        if time_diff > 60:
            return None, False
        diff = round(float(temp_1[temp_1.columns[1]]) - float(temp_2[temp_2.columns[1]]), 3)
        return diff, True
    
    def get_temp_at_time(self, time: int, array: pd.DataFrame) -> tuple[float, bool]:
        temp_1 = array.iloc[(array['time']-time).abs().argsort()[:1]]
        time_diff = abs(int(temp_1["time"])-time)
        if time_diff > 60:
            return None, False
        return float(temp_1[temp_1.columns[1]]), True
    
    def get_forecast_temps(self, time: int) -> list[tuple[int,float]]:
        forecast_temps: list[tuple[int,float]] = []
        prediction_file = open(self.data_path / "predictions" / f"{time}.json",'r')
        prediction_dict = json.load(prediction_file)
        for j in range(0,48):
            pred_time = int(prediction_dict['hourly'][j]['dt'])
            pred_temp = round(self.k_to_f(prediction_dict['hourly'][j]['temp']), 3)
            forecast_temps.append((pred_time, pred_temp))
        return forecast_temps
        
    def graph_data(self):
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
            data["time_diff"] = self.results_x
            data["temp_diff"] = self.results_y
            df = pd.DataFrame(data)
            df = df[df["time_diff"] > 20]
            # df = df[df["time_diff"] < 100]
            # print(df)
            # df1 = df[df["time"] ]
            plt.xlim(-25, 25)
            plt.ylim(0, 600)
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
    args = parser.parse_args()
    
    da = DataAnalyzer(**vars(args))
    da.start()