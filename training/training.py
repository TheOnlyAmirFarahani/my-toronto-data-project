import matplotlib
import matplotlib.pyplot as plt
import csv

X_MIN, X_MAX = -8870000, -8800000
Y_MIN, Y_MAX = 5400000, 5450000

xs, ys = [], []

with open("data/assault_data.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x = float(row["x"])
        y = float(row["y"])

        if X_MIN < x < X_MAX and Y_MIN < y < Y_MAX:
            xs.append(x)
            ys.append(y)


plt.figure(figsize=(12, 10))
plt.scatter(xs, ys, color="green", s=0.2, alpha=0.1)
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.title("Toronto Assault Data")
plt.xlabel("X")
plt.ylabel("X")
plt.grid(True, linestyle="--", alpha=0.3)

plt.savefig("/app/output/my_plot.png", dpi=300)
print("Image saved to output/my_plot.png")
