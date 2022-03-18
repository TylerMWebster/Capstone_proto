import pandas as pd


file = pd.read_csv("11_17_2021_2.csv")

file.drop("4010224513677320196")
file.drop("4025185137773201155")
file.drop("4067173146773201181")

for i in range(0, len(file)):
    True