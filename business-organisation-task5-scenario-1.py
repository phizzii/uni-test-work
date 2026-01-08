# example graph for jon's assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

CSV_PATH = "sales_data_sample.csv"
CATEGORY = "Classic Cars"

HOLDOUT_MONTHS = 8  # forecast will be this many months long

df = pd.read_csv(CSV_PATH, encoding="latin1")
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
df = df.dropna(subset=["ORDERDATE"])

df_cat = df[df["PRODUCTLINE"] == CATEGORY].copy()
if df_cat.empty:
    raise ValueError(f"No data for PRODUCTLINE='{CATEGORY}'")

df_cat["month_start"] = df_cat["ORDERDATE"].dt.to_period("M").dt.to_timestamp()
ts = df_cat.groupby("month_start", as_index=False)["SALES"].sum().sort_values("month_start")

full_idx = pd.date_range(ts["month_start"].min(), ts["month_start"].max(), freq="MS")
ts = ts.set_index("month_start").reindex(full_idx).fillna({"SALES": 0.0}).reset_index()
ts = ts.rename(columns={"index": "month_start"})

# features
ts["t"] = np.arange(len(ts))
ts["month"] = ts["month_start"].dt.month
ts["sin_m"] = np.sin(2 * np.pi * ts["month"] / 12)
ts["cos_m"] = np.cos(2 * np.pi * ts["month"] / 12)
feature_cols = ["t", "sin_m", "cos_m"]

if len(ts) <= HOLDOUT_MONTHS + 3:
    raise ValueError("Not enough months for the requested holdout window. Reduce HOLDOUT_MONTHS.")

# holdout = last N months
holdout = ts.iloc[-HOLDOUT_MONTHS:].copy()
train = ts.iloc[:-HOLDOUT_MONTHS].copy()

# train
model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
model.fit(train[feature_cols], train["SALES"].astype(float))

# predict
train["fitted"] = model.predict(train[feature_cols])
holdout["forecast"] = model.predict(holdout[feature_cols])

# plot
plt.figure(figsize=(12, 6))
plt.plot(ts["month_start"], ts["SALES"], label="Actual sales")
plt.plot(train["month_start"], train["fitted"], linestyle="--", label="Model fit (trained period)")
plt.plot(holdout["month_start"], holdout["SALES"], linewidth=2.5, label="Actual sales (held out)")
plt.plot(holdout["month_start"], holdout["forecast"], linestyle=":", linewidth=2.5, label="Forecast (holdout period)")

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)

plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

holdout_start_label = holdout["month_start"].min().strftime("%b %Y")
plt.title(
    f"Monthly Sales — Actual vs Forecast on Same Months (Holdout)\n"
    f"Category: {CATEGORY} | Forecast begins: {holdout_start_label}"
)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()