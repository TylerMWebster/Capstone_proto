import numpy as np
import pyqtgraph as pg
import pandas as pd
from utils import *

def fit_data(x, y, d, pw, c):
    fit = np.polyfit(x, y, d)
    #print(f'Polynomial representation of day:\n {fit}')
    array_to_csv(NEW_FILE_NAME, fit)

    fit = np.poly1d(fit)
    fitted = fit(x)
    pw.plot(x, fitted, pen=pg.mkPen(c, width=2))
    print(f'Max error: {np.max(abs(y - fitted))}')
    return fitted 

def find_max_diff(arr1, arr2):
    diff = abs(arr1 - arr2)
    return diff


FILE_NAME = '11_17_2021_2.csv'

NEW_FILE_NAME = 'Test_0'
NEW_FILE_NAME = next_log(NEW_FILE_NAME, '.csv')
data = pd.read_csv(FILE_NAME, header=None, skiprows=1, dtype=np.float64)

X_1 = data[0].to_numpy()
Y1_1 = data[1].to_numpy()
Y2_1 = data[2].to_numpy()
Y3_1 = data[3].to_numpy()
Y4_1 = data[4].to_numpy()
Y5_1 = data[5].to_numpy()
Y6_1 = data[6].to_numpy()
Y7_1 = data[7].to_numpy()
Y8_1 = data[8].to_numpy()
Y9_1 = data[9].to_numpy()


days = int(np.floor(len(X_1) / 1440))
print(f'Days: {days}')
deg = 10 * days
if deg > 30:
    deg = 30

new_X = np.arange(0,len(X_1), dtype=int)

X = np.array([])
Y1 = np.array([])
Y2 = np.array([])
Y3 = np.array([])
Y4 = np.array([])
Y5 = np.array([])
Y6 = np.array([])
Y7 = np.array([])
Y8 = np.array([])
Y9 = np.array([])

for i in range(0, len(new_X), 1):
    X = np.append(X, new_X[i])
    if (Y1_1[i] > -100):
        Y1 = np.append(Y1, Y1_1[i])
    else:
        Y1 = np.append(Y1, Y1_1[-1])
    Y2 = np.append(Y2, Y2_1[i])
    Y3 = np.append(Y3, Y3_1[i])
    Y4 = np.append(Y4, Y4_1[i])
    Y5 = np.append(Y5, Y5_1[i])
    Y6 = np.append(Y6, Y6_1[i])
    Y7 = np.append(Y7, Y7_1[i])
    Y8 = np.append(Y8, Y8_1[i])
    Y9 = np.append(Y9, Y9_1[i])




print(len(new_X), len(X))
avg_1 = (Y1_1 + Y2_1 + Y3_1) / 9 
avg = (Y1 + Y5 + Y6 + Y7 + Y8 + Y9) / 6

poly_csv_title = ['x^' + str(i) for i in range(deg, -1, -1)]

array_to_csv(NEW_FILE_NAME, poly_csv_title)

plotWidget = pg.plot(title="Three plot curves")
#Y1_fit = fit_data(X, Y1, deg, plotWidget, 'y')
#Y2_fit = fit_data(X, Y2, deg, plotWidget, 'r')
#Y3_fit = fit_data(X, Y3, deg, plotWidget, 'm')
#avg_fit = fit_data(X, avg, deg, plotWidget, 'w')

#Y1_Y2 = find_max_diff(Y1_fit, Y2_fit)
#Y1_Y3 = find_max_diff(Y1_fit, Y3_fit)
#Y2_Y3 = find_max_diff(Y2_fit, Y3_fit)

#max_diffs = [np.max(Y1_Y2), np.max(Y1_Y3), np.max(Y2_Y3)]

#max_dif = np.max(max_diffs)
#print(f'Maximum Difference: {max_dif}')


print(f'Minimum T Sensor 1: {np.min(Y1)}')
print(f'Maximum T Sensor 1: {np.max(Y1)}\n')
print(f'Minimum T Sensor 2: {np.min(Y2)}')
print(f'Maximum T Sensor 2: {np.max(Y2)}\n')
print(f'Minimum T Sensor 3: {np.min(Y3)}')
print(f'Maximum T Sensor 3: {np.max(Y3)}\n')


plotWidget.plot(X, Y1, pen=pg.mkPen('y', width=2))
#plotWidget.plot(X, Y2, pen=pg.mkPen('g', width=2))
#plotWidget.plot(X, Y3, pen=pg.mkPen('r', width=2))
#plotWidget.plot(X, Y4, pen=pg.mkPen('y', width=2))
plotWidget.plot(X, Y5, pen=pg.mkPen('r', width=2))
plotWidget.plot(X, Y6, pen=pg.mkPen('g', width=2))
plotWidget.plot(X, Y7, pen=pg.mkPen('b', width=2))
plotWidget.plot(X, Y8, pen=pg.mkPen('w', width=2))
plotWidget.plot(X, Y9, pen=pg.mkPen('c', width=2))
scatter = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(255, 255, 255, 120))
scatter.addPoints(X, avg)
plotWidget.addItem(scatter)

#plotWidget.plot(new_X, avg_1, pen=pg.mkPen('b', width=2))
#plotWidget.plot(X, avg, pen=pg.mkPen('b', width=2))
#plotWidget.plot(new_X, avg_1, pen=pg.mkPen('g', width=2))

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
