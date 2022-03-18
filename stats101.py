import pandas as pd
import matplotlib.pyplot as plt


file = pd.read_csv("test_1.csv")

fifteen_min = file[file["time_diff_seconds"] < 900] # 15 min

thirty_min_tmp = file[file["time_diff_seconds"] < 1800] # 30 min
thirty_min = thirty_min_tmp[thirty_min_tmp["time_diff_seconds"] > 900] # 30 min

sixty_min_tmp = file[file["time_diff_seconds"] < 3600] # 60 min
sixty_min = sixty_min_tmp[sixty_min_tmp["time_diff_seconds"] > 1800] # 60 min

temp_diff_15 = fifteen_min["diff_f"]
temp_diff_30 = thirty_min["diff_f"]
temp_diff_60 = sixty_min["diff_f"]

print(f"Fifteen Minutes:\n{temp_diff_15.describe()}")
print(f"Thirty Minutes:\n{temp_diff_30.describe()}")
print(f"Sixty Minutes:\n{temp_diff_60.describe()}")

#plt.scatter(file["time_diff_seconds"], file["diff_f"])

kwargs1 = dict(alpha=1.0, bins=25)
kwargs2 = dict(alpha=1.0, bins=100)
kwargs3 = dict(alpha=1.0, bins=100)

plt.subplot(1,3,3)
plt.hist(sixty_min["diff_f"], color="green", **kwargs3)
plt.xlim(-10,10)
plt.ylim(0,20)
plt.xlabel("Prediction Error (F)")
plt.ylabel("Frequency")
plt.legend(["Predictions 30-60 min in the future"])

plt.subplot(1,3,2)
plt.hist(thirty_min["diff_f"], color="orange", **kwargs2)
plt.xlim(-10,10)
plt.ylim(0,20)
plt.xlabel("Prediction Error (F)")
plt.ylabel("Frequency")
plt.legend(["Predictions 15-30 min in the future"])

plt.subplot(1,3,1)
plt.hist(fifteen_min["diff_f"], **kwargs1)
plt.xlim(-10,10)
plt.ylim(0,20)
plt.xlabel("Prediction Error (F)")
plt.ylabel("Frequency")
plt.legend(["Predictions 0-15 min in the future"])

plt.show()

