import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib.colors import TwoSlopeNorm

# ── Setup ─────────────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

if not os.path.exists("data/assault_data.csv"):
    print("ERROR: Put your assault_data.csv inside a folder called 'data'")
    exit()

# ── 1. Load & Filter ──────────────────────────────────────────────────────────
df = pd.read_csv("data/assault_data.csv")
print(f"Loaded {len(df)} rows")

df = df[
    (df["x"] > -8870000) & (df["x"] < -8800000) &
    (df["y"] >  5400000) & (df["y"] <  5450000)
].copy()
print(f"After location filter: {len(df)} rows")

# ── 2. Create Target: NIGHTTIME (1) vs DAYTIME (0) ───────────────────────────
# Nighttime = 10pm (22:00) to 4am (04:00) — hours where assault patterns
# differ meaningfully based on criminology research
# This is a genuine, causal label: the time of day directly affects
# the circumstances and nature of an assault
df["IS_NIGHT"] = df["OCC_HOUR"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)

night_count = df["IS_NIGHT"].sum()
day_count   = (df["IS_NIGHT"] == 0).sum()
print(f"\nNighttime assaults (10pm–4am): {night_count} ({night_count/len(df)*100:.1f}%)")
print(f"Daytime assaults  (4am–10pm):  {day_count}  ({day_count/len(df)*100:.1f}%)")

# ── 3. Encode Categorical Features ───────────────────────────────────────────
le_dow = LabelEncoder()
df["OCC_DOW"] = le_dow.fit_transform(df["OCC_DOW"].astype(str))

le_month = LabelEncoder()
df["OCC_MONTH"] = le_month.fit_transform(df["OCC_MONTH"].astype(str))

le_premises = LabelEncoder()
df["PREMISES_TYPE"] = le_premises.fit_transform(df["PREMISES_TYPE"].astype(str))

# ── 4. Select Features ────────────────────────────────────────────────────────
# We deliberately EXCLUDE OCC_HOUR from features — otherwise the model
# would just memorise the hour directly instead of learning real patterns.
# Instead it must learn from location, premises type, month, and day of week.
FEATURES = ["x", "y", "OCC_MONTH", "OCC_DOW", "PREMISES_TYPE", "HOOD_158"]
TARGET   = "IS_NIGHT"

df = df[FEATURES + [TARGET]].dropna()
df["HOOD_158"] = pd.to_numeric(df["HOOD_158"], errors="coerce")
df = df.dropna()
print(f"After dropping nulls: {len(df)} rows\n")

X = df[FEATURES].values
y = df[TARGET].values

# ── 5. Standardize ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 6. Train Logistic Regression ─────────────────────────────────────────────
print("Training model...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_scaled, y)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]  # P(nighttime)

acc = accuracy_score(y, y_pred)
print(f"Training Accuracy: {acc * 100:.2f}%")

# ── 8. Sample Predictions ─────────────────────────────────────────────────────
# Reload original data to show readable values alongside predictions
df_orig = pd.read_csv("data/assault_data.csv")
df_orig = df_orig[
    (df_orig["x"] > -8870000) & (df_orig["x"] < -8800000) &
    (df_orig["y"] >  5400000) & (df_orig["y"] <  5450000)
].copy().reset_index(drop=True)
df_orig["IS_NIGHT"] = df_orig["OCC_HOUR"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)
df_orig = df_orig.dropna(subset=FEATURES + [TARGET]).head(10)

print("\nSample predictions (first 10 incidents):")
print(f"{'Hour':<6} {'Day':<12} {'Premises':<14} {'Actual':<12} {'Predicted':<12} {'P(Night)%'}")
print("-" * 68)
for i in range(min(10, len(df))):
    row    = df_orig.iloc[i]
    actual = "Nighttime" if y[i] == 1 else "Daytime"
    pred   = "Nighttime" if y_pred[i] == 1 else "Daytime"
    print(f"{int(row['OCC_HOUR']):<6} {str(row['OCC_DOW']):<12} {str(row['PREMISES_TYPE']):<14} "
          f"{actual:<12} {pred:<12} {y_proba[i]*100:.1f}%")

# ── 9. Plot 1: Confusion Matrix ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y, y_pred,
    display_labels=["Daytime", "Nighttime"],
    ax=ax, colorbar=False, cmap="Blues"
)
plt.title(f"Nighttime Assault Prediction\nConfusion Matrix  —  Accuracy: {acc * 100:.2f}%")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=200)
plt.close()
print("\nSaved: confusion_matrix.png")

# ── 10. Plot 2: Geographic Heatmap of Nighttime Probability ──────────────────
# Scale colour to actual data range (not 0-1) so differences are visible.
# Use percentile clipping to prevent outliers washing out the colour range.
p5, p95 = np.percentile(y_proba, 5), np.percentile(y_proba, 95)
vcenter  = (p5 + p95) / 2  # midpoint of actual data range

# Sort points so highest-probability dots render on top
sort_idx  = np.argsort(y_proba)
x_sorted  = df["x"].values[sort_idx]
y_sorted  = df["y"].values[sort_idx]
c_sorted  = y_proba[sort_idx]

fig, ax = plt.subplots(figsize=(14, 12))
norm = TwoSlopeNorm(vmin=p5, vcenter=vcenter, vmax=p95)
sc   = ax.scatter(x_sorted, y_sorted, c=c_sorted,
                  cmap="RdYlBu_r", norm=norm, s=1.2, alpha=0.6)

cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Predicted Probability of Nighttime Assault", fontsize=12)
# Label the actual low/mid/high values from the data
cbar.set_ticks([p5, vcenter, p95])
cbar.set_ticklabels([
    f"Low ({p5*100:.0f}%)",
    f"Mid ({vcenter*100:.0f}%)",
    f"High ({p95*100:.0f}%)"
], fontsize=10)

# Add annotation explaining the colour scale
ax.annotate("Red areas = model predicts higher chance\nassault occurred at night (10pm–4am)",
            xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=10, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_title("Toronto Assault Map — Predicted Nighttime Probability\n"
             "Colour shows relative likelihood of nighttime assault by location", fontsize=13)
ax.set_xlabel("X Coordinate (Web Mercator)", fontsize=11)
ax.set_ylabel("Y Coordinate (Web Mercator)", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.15)
plt.tight_layout()
plt.savefig("output/risk_map.png", dpi=300)
plt.close()
print("Saved: risk_map.png")

# ── 11. Plot 3: Actual Assault Count by Hour (what really happens) ────────────
df_orig_full = pd.read_csv("data/assault_data.csv")
df_orig_full = df_orig_full[
    (df_orig_full["x"] > -8870000) & (df_orig_full["x"] < -8800000) &
    (df_orig_full["y"] >  5400000) & (df_orig_full["y"] <  5450000)
].copy()

hour_counts = df_orig_full["OCC_HOUR"].value_counts().sort_index()
night_hours = [22, 23, 0, 1, 2, 3, 4]
bar_colors  = ["#F44336" if h in night_hours else "#2196F3" for h in hour_counts.index]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(hour_counts.index, hour_counts.values, color=bar_colors, edgecolor="black")
ax.set_title("Actual Assault Incidents by Hour of Day\n(Red = Nighttime window the model predicts)", fontsize=13)
ax.set_xlabel("Hour of Day (0 = Midnight, 12 = Noon)")
ax.set_ylabel("Number of Incidents")
ax.set_xticks(range(0, 24))
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#F44336", label="Nighttime (10pm–4am)"),
                   Patch(facecolor="#2196F3", label="Daytime (4am–10pm)")]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("output/assaults_by_hour.png", dpi=200)
plt.close()
print("Saved: assaults_by_hour.png")

# ── 12. Plot 4: Nighttime vs Daytime Assault Count by Premises Type ─────────
# This shows the ACTUAL split of night vs day assaults per premises type
# so you can directly see which locations are more dangerous at night
df_time = df_orig_full.copy()
df_time["IS_NIGHT"] = df_time["OCC_HOUR"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)
df_time = df_time.dropna(subset=["PREMISES_TYPE"])

# Count nighttime and daytime incidents per premises type
night_counts = df_time[df_time["IS_NIGHT"] == 1]["PREMISES_TYPE"].value_counts()
day_counts   = df_time[df_time["IS_NIGHT"] == 0]["PREMISES_TYPE"].value_counts()

# Align both series to the same premises types, sorted by night count
all_premises = night_counts.add(day_counts, fill_value=0).sort_values(ascending=False).index
night_vals   = [night_counts.get(p, 0) for p in all_premises]
day_vals     = [day_counts.get(p, 0)   for p in all_premises]

x     = np.arange(len(all_premises))
width = 0.4

fig, ax = plt.subplots(figsize=(12, 6))
bars_night = ax.bar(x - width/2, night_vals, width, label="Nighttime (10pm–4am)",
                    color="#F44336", edgecolor="black", alpha=0.9)
bars_day   = ax.bar(x + width/2, day_vals,   width, label="Daytime (4am–10pm)",
                    color="#2196F3", edgecolor="black", alpha=0.9)

# Add count labels on top of each bar
for bar in bars_night:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
for bar in bars_day:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

ax.set_title("Nighttime vs Daytime Assault Count by Premises Type", fontsize=13)
ax.set_xlabel("Premises Type")
ax.set_ylabel("Number of Incidents")
ax.set_xticks(x)
ax.set_xticklabels(all_premises, rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("output/risk_by_premises.png", dpi=200)
plt.close()
print("Saved: risk_by_premises.png")

# ── 13. Plot 5: Nighttime vs Daytime by Hour — actual distribution ────────────
hour_night = df_time[df_time["IS_NIGHT"] == 1]["OCC_HOUR"].value_counts().sort_index()
hour_day   = df_time[df_time["IS_NIGHT"] == 0]["OCC_HOUR"].value_counts().sort_index()

all_hours  = sorted(df_time["OCC_HOUR"].dropna().unique())
night_h    = [hour_night.get(h, 0) for h in all_hours]
day_h      = [hour_day.get(h, 0)   for h in all_hours]

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(all_hours, night_h, label="Nighttime window", color="#F44336", edgecolor="black", alpha=0.9)
ax.bar(all_hours, day_h,   label="Daytime window",   color="#2196F3", edgecolor="black",
       alpha=0.7, bottom=night_h)
ax.set_title("Assault Incidents by Hour of Day\n(Stacked: Red = Nighttime window, Blue = Daytime window)",
             fontsize=13)
ax.set_xlabel("Hour of Day (0 = Midnight, 12 = Noon)")
ax.set_ylabel("Number of Incidents")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("output/assaults_by_hour.png", dpi=200)
plt.close()
print("Saved: assaults_by_hour.png")

# ── 13b. Plot 5b: Assault Count by Month (night vs day split) ────────────────
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
month_night = df_time[df_time["IS_NIGHT"] == 1]["OCC_MONTH"].value_counts()
month_day   = df_time[df_time["IS_NIGHT"] == 0]["OCC_MONTH"].value_counts()
month_night = month_night.reindex([m for m in month_order if m in month_night.index]).fillna(0)
month_day   = month_day.reindex([m for m in month_order if m in month_day.index]).fillna(0)

x2    = np.arange(len(month_order))
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x2, [month_night.get(m, 0) for m in month_order],
       label="Nighttime", color="#F44336", edgecolor="black", alpha=0.9)
ax.bar(x2, [month_day.get(m, 0) for m in month_order],
       label="Daytime",   color="#2196F3", edgecolor="black", alpha=0.7,
       bottom=[month_night.get(m, 0) for m in month_order])
ax.set_title("Assault Incidents by Month (Nighttime vs Daytime)", fontsize=13)
ax.set_xlabel("Month")
ax.set_ylabel("Number of Incidents")
ax.set_xticks(x2)
ax.set_xticklabels(month_order, rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig("output/assaults_by_month.png", dpi=200)
plt.close()
print("Saved: assaults_by_month.png")

# ── 14. Neighbourhood Risk Table ─────────────────────────────────────────────
df_hoods = pd.read_csv("data/assault_data.csv")
df_hoods = df_hoods[
    (df_hoods["x"] > -8870000) & (df_hoods["x"] < -8800000) &
    (df_hoods["y"] >  5400000) & (df_hoods["y"] <  5450000)
][["HOOD_158", "NEIGHBOURHOOD_158"]].dropna().reset_index(drop=True)

# Align indices with df
shared_idx = df.index.intersection(df_hoods.index)
df_hoods   = df_hoods.loc[shared_idx].reset_index(drop=True)
y_proba_aligned = y_proba[:len(df_hoods)]
y_pred_aligned  = y_pred[:len(df_hoods)]

hood_table = pd.DataFrame({
    "HOOD_158":          df_hoods["HOOD_158"].values,
    "NEIGHBOURHOOD_158": df_hoods["NEIGHBOURHOOD_158"].values,
    "P(Nighttime)":      y_proba_aligned,
})
hood_table = hood_table.groupby(["HOOD_158", "NEIGHBOURHOOD_158"]).agg(
    Total_Incidents=("P(Nighttime)", "count"),
    Avg_Night_Prob  =("P(Nighttime)", "mean"),
).reset_index()

hood_table["Night Assault Risk"]   = hood_table["Avg_Night_Prob"].apply(
    lambda p: "HIGH" if p >= 0.5 else "LOW"
)
hood_table["Avg_Night_Prob %"] = (hood_table["Avg_Night_Prob"] * 100).round(1)
hood_table = hood_table.drop(columns=["Avg_Night_Prob"])
hood_table = hood_table.sort_values("Avg_Night_Prob %", ascending=False).reset_index(drop=True)
hood_table.index += 1

hood_table.to_csv("output/neighbourhood_risk_table.csv")
print("\nSaved: neighbourhood_risk_table.csv")
print("\nNeighbourhood Night-Assault Risk Rankings:")
print(hood_table.to_string())

# ── 15. Plot 6: Top 20 Neighbourhoods ────────────────────────────────────────
# Show top 20 vs bottom 20 side by side so differences are stark and clear
top20    = hood_table.head(20).copy()
bottom20 = hood_table.tail(20).copy()

# Colour each bar by its actual value using the full table range
all_vals  = hood_table["Avg_Night_Prob %"].values
norm_hood = plt.Normalize(all_vals.min(), all_vals.max())
cmap_hood = plt.cm.RdYlBu_r

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

# ── Left: Top 20 highest nighttime probability ──
colors_top = [cmap_hood(norm_hood(v)) for v in top20["Avg_Night_Prob %"]]
bars = axes[0].bar(range(len(top20)), top20["Avg_Night_Prob %"],
                   color=colors_top, edgecolor="black")
axes[0].set_ylim(top20["Avg_Night_Prob %"].min() - 1,
                 top20["Avg_Night_Prob %"].max() + 1.5)
for i, (bar, val) in enumerate(zip(bars, top20["Avg_Night_Prob %"])):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
axes[0].set_xticks(range(len(top20)))
axes[0].set_xticklabels(top20["NEIGHBOURHOOD_158"], rotation=45, ha="right", fontsize=8)
axes[0].set_title("Top 20 Neighbourhoods\n(Highest Nighttime Assault Probability)", fontsize=11)
axes[0].set_ylabel("Avg P(Nighttime Assault) %", fontsize=10)
axes[0].axhline(top20["Avg_Night_Prob %"].mean(), color="black",
                linestyle="--", linewidth=1, label="Group avg")
axes[0].legend(fontsize=9)

# ── Right: Bottom 20 lowest nighttime probability ──
colors_bot = [cmap_hood(norm_hood(v)) for v in bottom20["Avg_Night_Prob %"]]
bars2 = axes[1].bar(range(len(bottom20)), bottom20["Avg_Night_Prob %"],
                    color=colors_bot, edgecolor="black")
axes[1].set_ylim(bottom20["Avg_Night_Prob %"].min() - 1,
                 bottom20["Avg_Night_Prob %"].max() + 1.5)
for bar, val in zip(bars2, bottom20["Avg_Night_Prob %"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
axes[1].set_xticks(range(len(bottom20)))
axes[1].set_xticklabels(bottom20["NEIGHBOURHOOD_158"], rotation=45, ha="right", fontsize=8)
axes[1].set_title("Bottom 20 Neighbourhoods\n(Lowest Nighttime Assault Probability)", fontsize=11)
axes[1].set_ylabel("Avg P(Nighttime Assault) %", fontsize=10)
axes[1].axhline(bottom20["Avg_Night_Prob %"].mean(), color="black",
                linestyle="--", linewidth=1, label="Group avg")
axes[1].legend(fontsize=9)

fig.suptitle("Neighbourhood Nighttime Assault Probability — Highest vs Lowest",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("output/top20_neighbourhood_risk.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: top20_neighbourhood_risk.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✓ Done! All outputs saved to output/")
print(f"\nSummary:")
print(f"  Model:    Logistic Regression (baseline)")
print(f"  Task:     Predict whether an assault occurred at NIGHTTIME (10pm–4am)")
print(f"  Accuracy: {acc * 100:.2f}%")
print(f"  Features: location (x,y), month, day of week, premises type, neighbourhood")