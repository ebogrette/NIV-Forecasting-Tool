"""
# used claude to merge original excel tabs into one document "claude/niv_model/merged_monthly.csv) 

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/niv_model/merged_monthly.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# feature engineering 
df["lag_1"] = df["visa_issuances"].shift(1)
df["lag_12"] = df["visa_issuances"].shift(12)
df["rolling_mean_3"] = df["visa_issuances"].shift(1).rolling(3).mean()
df["seasonal_baseline"] = df["lag_12"].fillna(df["rolling_mean_3"])
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["gdp_growth_yoy"] = df["gdp_growth_yoy"].bfill()

FEATURES = ["gdp_growth_yoy", "job_vacancy", "inflation_pct",
            "lag_1", "rolling_mean_3", "month_sin", "month_cos"]

df_model = df.dropna(subset=FEATURES + ["seasonal_baseline", "visa_issuances"]).reset_index(drop=True)
print(f"Usable months after feature engineering: {len(df_model)} "
      f"({df_model['date'].min().date()} to {df_model['date'].max().date()})")

df_model["residual"] = df_model["visa_issuances"] - df_model["seasonal_baseline"]

# backtest 
TEST_MONTHS = 12
split_idx = len(df_model) - TEST_MONTHS
train, test = df_model.iloc[:split_idx].copy(), df_model.iloc[split_idx:].copy()
print(f"Train: {train['date'].min().date()} to {train['date'].max().date()}  ({len(train)} months)")
print(f"Test:  {test['date'].min().date()} to {test['date'].max().date()}  ({len(test)} months)")

scaler = StandardScaler().fit(train[FEATURES])
X_train = scaler.transform(train[FEATURES])
X_test = scaler.transform(test[FEATURES])

model = Ridge(alpha=1.0)
model.fit(X_train, train["residual"])

test["pred_residual"] = model.predict(X_test)
test["prediction"] = test["seasonal_baseline"] + test["pred_residual"]

mape_model = mean_absolute_percentage_error(test["visa_issuances"], test["prediction"])
mape_naive = mean_absolute_percentage_error(test["visa_issuances"], test["seasonal_baseline"])
mae_model = mean_absolute_error(test["visa_issuances"], test["prediction"])
mae_naive = mean_absolute_error(test["visa_issuances"], test["seasonal_baseline"])

print(f"\n=== Backtest: last {TEST_MONTHS} months ===")
print(f"  Naive seasonal MAPE : {mape_naive:6.2%}   (MAE: {mae_naive:,.0f})")
print(f"  Hybrid model MAPE   : {mape_model:6.2%}   (MAE: {mae_model:,.0f})")
if mape_naive > 0:
    print(f"  Relative improvement: {(mape_naive - mape_model) / mape_naive:6.1%}")

print("\nStandardized Ridge coefficients (sign shows direction of effect on the residual):")
for feat, coef in sorted(zip(FEATURES, model.coef_), key=lambda x: -abs(x[1])):
    print(f"  {feat:18s} {coef:+.1f}")

# plot

plt.figure(figsize=(10, 4.5))
plt.plot(df_model["date"], df_model["visa_issuances"], label="Actual (full history)", color="#1f4e79", linewidth=1.5, alpha=0.5)
plt.plot(test["date"], test["visa_issuances"], label="Actual (backtest window)", color="#1f4e79", linewidth=2.5)
plt.plot(test["date"], test["prediction"], label="Hybrid model forecast", linestyle="--", color="#c00000", linewidth=2)
plt.plot(test["date"], test["seasonal_baseline"], label="Naive seasonal baseline", linestyle=":", color="#888888", linewidth=2)
plt.axvline(test["date"].min(), color="black", linestyle="-", linewidth=0.8, alpha=0.4)
plt.title("Total NIV Issuances — Backtest on Real Data")
plt.ylabel("Monthly issuances")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/niv_model/real_data_backtest.png", dpi=150)
plt.close()
print("\nSaved chart -> /home/claude/niv_model/real_data_backtest.png")

#forward forcast
last = df_model.iloc[-1]
future_rows = []
history = df_model["visa_issuances"].tolist()
last_date = df_model["date"].max()

for i in range(1, 4):
    fdate = last_date + pd.DateOffset(months=i)
    lag_1 = history[-1]
    lag_12_idx = -12 + (i - 1)
    lag_12 = history[lag_12_idx] if abs(lag_12_idx) <= len(history) else np.nan
    rolling_mean_3 = np.mean(history[-3:])
    seasonal_baseline = lag_12 if not np.isnan(lag_12) else rolling_mean_3
    row = {
        "gdp_growth_yoy": last["gdp_growth_yoy"],
        "job_vacancy": last["job_vacancy"],
        "inflation_pct": last["inflation_pct"],
        "lag_1": lag_1,
        "rolling_mean_3": rolling_mean_3,
        "month_sin": np.sin(2 * np.pi * fdate.month / 12),
        "month_cos": np.cos(2 * np.pi * fdate.month / 12),
    }
    row_scaled = scaler.transform(pd.DataFrame([row])[FEATURES])
    pred_resid = model.predict(row_scaled)[0]
    forecast = seasonal_baseline + pred_resid
    future_rows.append({"date": fdate, "forecast": forecast})
    history.append(forecast)

print("\n=== 3-month forward forecast (macro inputs held at last observed value) ===")
for r in future_rows:
    print(f"  {r['date'].date()}  ->  {r['forecast']:,.0f}")
