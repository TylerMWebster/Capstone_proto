import numpy as np
import pyqtgraph as pg
import pandas as pd
import json

FILE_NAME = '11_17_2021_2.csv'

def k_to_f(temp):
    return (temp - 273.15) * 1.8 + 32

data = pd.read_csv(FILE_NAME, header=None, skiprows=2, dtype=np.float64)
# data2 = pd.read_csv('api_temp_data/temperature_file.csv', header=None)

for i in range(1,10):
    for j in range(0, len(data[i])):
        try:
            if int(data[i][j]) < 0 or int(data[i][j]) > 150:
                data[i][j] = data[i][j-1]
        except:
            True

# prediction_file = open('api_temp_data/1637775769.json','r')
# prediction_dict = json.load(prediction_file)
# prediction_times = []
# prediction_temps = []
# for i in range(0,24):
#     prediction_times.append(prediction_dict['hourly'][i]['dt'])
#     prediction_temps.append(k_to_f(prediction_dict['hourly'][i]['temp']))    

plotWidget = pg.plot(title="Three plot curves")
for i in range(1,10):
    plotWidget.plot(data[0], data[i], pen=pg.mkPen('y', width=2))
# plotWidget.plot(data2[0], data2[1], pen=pg.mkPen('b', width=2))

# # Create a scatter of the average of the lines
# scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0))
# scatter.addPoints(prediction_times, prediction_temps)
# plotWidget.addItem(scatter)

#plotWidget.plot(new_X, avg_1, pen=pg.mkPen('b', width=2))
#plotWidget.plot(X, avg, pen=pg.mkPen('b', width=2))
#plotWidget.plot(new_X, avg_1, pen=pg.mkPen('g', width=2))

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
