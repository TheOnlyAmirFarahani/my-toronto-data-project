import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)
np.random.seed(42)

df = pd.read_csv("data/assault_data.csv")
df.columns = df.columns.str.strip()
df = df[
    (df["x"] > -8870000) & (df["x"] < -8800000) &
    (df["y"] >  5400000) & (df["y"] <  5450000)
].copy().reset_index(drop=True)

df["OCC_HOUR"]   = pd.to_numeric(df["OCC_HOUR"],   errors="coerce").fillna(0)
df["OCC_MONTH"]  = df["OCC_MONTH"].str.strip()
df["OCC_YEAR"]   = pd.to_numeric(df["OCC_YEAR"],   errors="coerce").fillna(2014)
df["LAT"]        = pd.to_numeric(df["LAT_WGS84"],  errors="coerce")
df["LON"]        = pd.to_numeric(df["LONG_WGS84"], errors="coerce")
df["UCR_CODE"]   = pd.to_numeric(df["UCR_CODE"],   errors="coerce")
df["weapon"]     = (df["UCR_CODE"] != 1430).astype(int)
df["is_night"]   = ((df["OCC_HOUR"] >= 22) | (df["OCC_HOUR"] <= 4)).astype(int)
df["is_weekend"] = df["OCC_DOW"].str.strip().isin(["Saturday", "Sunday"]).astype(int)


# Dashboard

month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
dow_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Toronto Assault Data — Overview", fontsize=15, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

hour_counts = df.groupby("OCC_HOUR").size()
ax1.bar(hour_counts.index, hour_counts.values,
        color=["#F44336" if (h >= 22 or h <= 4) else "#2196F3" for h in hour_counts.index],
        edgecolor="white")
ax1.set_title("Incidents by Hour of Day\n(red = nighttime)")
ax1.set_xlabel("Hour (0–23)")
ax1.set_ylabel("Number of Incidents")
ax1.grid(axis="y", alpha=0.3)

month_counts = df.groupby("OCC_MONTH").size().reindex(month_order).fillna(0)
ax2.bar(range(12), month_counts.values, color="#2196F3", edgecolor="white")
ax2.set_xticks(range(12))
ax2.set_xticklabels([m[:3] for m in month_order], rotation=45)
ax2.set_title("Incidents by Month")
ax2.set_ylabel("Number of Incidents")
ax2.grid(axis="y", alpha=0.3)

dow_counts = df["OCC_DOW"].str.strip().value_counts().reindex(dow_order).fillna(0)
ax3.bar(range(7), dow_counts.values,
        color=["#F44336" if d in ["Saturday","Sunday"] else "#2196F3" for d in dow_order],
        edgecolor="white")
ax3.set_xticks(range(7))
ax3.set_xticklabels([d[:3] for d in dow_order])
ax3.set_title("Incidents by Day of Week\n(red = weekend)")
ax3.set_ylabel("Number of Incidents")
ax3.grid(axis="y", alpha=0.3)

prem_counts = df["PREMISES_TYPE"].value_counts().head(6)
ax4.barh(range(len(prem_counts)), prem_counts.values, color="#2196F3", edgecolor="white")
ax4.set_yticks(range(len(prem_counts)))
ax4.set_yticklabels(prem_counts.index, fontsize=8)
ax4.set_title("Top 6 Location Types")
ax4.set_xlabel("Number of Incidents")
ax4.grid(axis="x", alpha=0.3)

year_counts = df.groupby("OCC_YEAR").size()
ax5.plot(year_counts.index, year_counts.values, color="#2196F3", marker="o", linewidth=2)
ax5.fill_between(year_counts.index, year_counts.values, alpha=0.15, color="#2196F3")
ax5.set_title("Incidents Per Year")
ax5.set_xlabel("Year")
ax5.set_ylabel("Number of Incidents")
ax5.grid(alpha=0.3)

night_n = int(df["is_night"].sum())
day_n   = len(df) - night_n
ax6.pie([day_n, night_n],
        labels=[f"Daytime\n{day_n:,}", f"Nighttime\n{night_n:,}"],
        colors=["#2196F3", "#F44336"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5})
ax6.set_title("Day vs Night\n(Night = 10 pm – 4 am)")

plt.savefig("output/chart_1_overview.png", dpi=150, bbox_inches="tight")
plt.close()


# Yearly trend

yearly = df.groupby("OCC_YEAR").size().reset_index(name="count")
max_year = yearly["OCC_YEAR"].max()
yearly = yearly[(yearly["OCC_YEAR"] >= 2014) & (yearly["OCC_YEAR"] <= max_year - 1)]

X = yearly["OCC_YEAR"].values.reshape(-1, 1)
y = yearly["count"].values

model = LinearRegression()
model.fit(X, y)

y_hat = model.predict(X)

future_years = np.array([max_year, max_year + 1, max_year + 2]).reshape(-1, 1)
fc = model.predict(future_years)

r2 = model.score(X, y)
rmse = np.sqrt(np.mean((y - y_hat)**2))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(yearly["OCC_YEAR"].astype(int), y, color="#2196F3", alpha=0.7, edgecolor="white", label="Actual count")
ax.plot(yearly["OCC_YEAR"].astype(int), y_hat, color="#F44336", linewidth=2.5, marker="o", label="Trend line")
ax.plot(future_years.flatten().astype(int), fc, color="#F44336", linewidth=2, linestyle="--",
        marker="o", markerfacecolor="white", label="Forecast")
ax.axvspan(future_years[0][0] - 0.5, future_years[-1][0] + 0.5, alpha=0.07, color="#F44336")

for yr, val in zip(future_years.flatten().astype(int), fc):
    ax.annotate(f"{val:,.0f}", (yr, val), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color="#F44336")

ax.set_xlabel("Year")
ax.set_ylabel("Total Assaults")
ax.set_title(f"Yearly Assault Count with Forecast\nR² = {r2:.3f}   RMSE = {rmse:.0f}")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("output/chart_2_yearly_trend.png", dpi=150, bbox_inches="tight")
plt.close()

# Weapon rate

data = df[["LAT", "LON", "weapon"]].dropna()

BINS = 40
lat_edges = np.linspace(data["LAT"].min(), data["LAT"].max(), BINS + 1)
lon_edges = np.linspace(data["LON"].min(), data["LON"].max(), BINS + 1)

total_counts, _, _ = np.histogram2d(data["LAT"], data["LON"],
                                    bins=[lat_edges, lon_edges])
weapon_data = data[data["weapon"] == 1]
weapon_counts, _, _ = np.histogram2d(weapon_data["LAT"], weapon_data["LON"],
                                     bins=[lat_edges, lon_edges])

with np.errstate(invalid="ignore", divide="ignore"):
    weapon_rate = np.where(total_counts >= 5, weapon_counts / total_counts, np.nan)

lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Weapon Involvement by Location in Toronto", fontsize=13, fontweight="bold")

im1 = axes[0].pcolormesh(lon_centers, lat_centers, weapon_rate,
                          cmap="YlOrRd", shading="auto", vmin=0, vmax=1)
plt.colorbar(im1, ax=axes[0], label="Weapon Involvement Rate (0 = never, 1 = always)")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Weapon Rate per Map Cell\n(cells with fewer than 5 incidents are hidden)")
axes[0].set_facecolor("#eeeeee")

im2 = axes[1].pcolormesh(lon_centers, lat_centers, total_counts,
                          cmap="Blues", shading="auto")
plt.colorbar(im2, ax=axes[1], label="Number of Incidents")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Total Incidents per Map Cell\n(shows where data is concentrated)")
axes[1].set_facecolor("#eeeeee")

plt.tight_layout()
plt.savefig("output/chart_3_weapon_map.png", dpi=150, bbox_inches="tight")
plt.close()

# Classification 

data = df[["LAT", "LON", "weapon"]].dropna().copy()

X = data[["LAT", "LON"]].values
y = data["weapon"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle("Logistic Regression Classification of Weapon Involvement", fontsize=13, fontweight="bold")

ax.scatter(
    data["LON"][y_pred == 0],
    data["LAT"][y_pred == 0],
    s=5, alpha=0.5, label="Predicted: No Weapon"
)

ax.scatter(
    data["LON"][y_pred == 1],
    data["LAT"][y_pred == 1],
    s=5, alpha=0.7, label="Predicted: Weapon"
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Model Predictions Based on Location")
ax.legend()
ax.set_facecolor("#eeeeee")

plt.tight_layout()
plt.savefig("output/chart_4_weapon_classification.png", dpi=150, bbox_inches="tight")
plt.close()