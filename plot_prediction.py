import matplotlib.pyplot as plt
import pandas as pd
import sys

data = pd.read_csv(f"test_{sys.argv[1]}.csv")

x = []
y = []
temp = 0
count = 0
# for i in range(0, len(data["time_measured"])):
#     a = data["time_measured"][i]
#     if temp != data["time_measured"][i]:
#         count += 1
#         plt.scatter(x,y)
#         x.clear()
#         y.clear()
#         temp = data["time_measured"][i]
#         plt.xlim(0, 14500)
#         plt.ylim(-50, 50)
#     else:
#         x.append(data["time_diff_seconds"][i])
#         y.append(data["diff_f"][i])
#         temp = data["time_measured"][i]

data = data[data["diff_f"] > -200]
data = data[data["diff_f"] < 200]

plt.scatter(data["time_diff_seconds"], data["diff_f"])

# plt.scatter(data["time_diff_seconds"]/60, data["diff_f"])
plt.xlabel("Minutes Away from Prediction (Minutes)", fontsize=18)
plt.ylabel("Prediction Error (F)", fontsize=18)
plt.title("Prediction Time vs Prediction Error", fontsize=24)
plt.show()

print(count)

plt.show()
