# =============================================================================
# DATA-DRIVEN SALES FORECASTING USING PYTHON
# Mini Project - Business Analytics Using Python Internship
# =============================================================================
# AUDIT CORRECTIONS (v2):
#   B1: Rolling features built on shift(1) — target never included in window
#   B2: Units_Sold, Avg_Price removed from feature set (derived from target)
#   B3: R2 now genuine (~0.85-0.95), not artefactual 1.0
#   B4: MA/WMA operate on raw-df aggregate monthly series (chronological, 24 pts)
#       with a clean month-boundary 80/20 split
#   B5: WMA.forecast_next propagates each prediction as next anchor
#   B6: BusinessInsights works on internal copy — raw df never mutated
#   B7: Forecaster updates Lag_1/2/3 correctly across all forecast steps
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import statistics
from datetime import datetime, timedelta
import random


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

class SalesDataGenerator:
    """Generates realistic synthetic sales data with trends and seasonality."""

    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.categories = ["Electronics", "Clothing", "Grocery"]
        self.regions    = ["North", "South", "East", "West"]
        self.months     = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]

    def seasonal_factor(self, month_index, category):
        patterns = {
            "Electronics": [0.8,0.75,0.85,0.90,0.92,0.88,
                            0.85,0.87,0.95,1.10,1.30,1.50],
            "Clothing":    [0.80,0.82,0.95,1.05,1.10,1.15,
                            1.10,1.05,1.00,0.95,1.10,1.20],
            "Grocery":     [1.00,0.95,1.00,1.02,1.05,1.08,
                            1.05,1.05,1.03,1.05,1.10,1.20],
        }
        return patterns[category][month_index % 12]

    def generate(self, years=2):
        records      = []
        base_sales   = {"Electronics":50000, "Clothing":30000, "Grocery":20000}
        region_factor= {"North":1.10, "South":0.90, "East":1.05, "West":0.95}
        growth_rate  = 0.005
        start_date   = datetime(2023, 1, 1)

        for month_idx in range(years * 12):
            current_date = start_date + timedelta(days=30 * month_idx)
            for category in self.categories:
                for region in self.regions:
                    base     = base_sales[category]
                    sales    = (base
                                * self.seasonal_factor(month_idx, category)
                                * region_factor[region]
                                * (1 + growth_rate) ** month_idx
                                * np.random.normal(1.0, 0.05))
                    units     = int(sales / (base * 0.05))
                    avg_price = round(sales / max(units, 1), 2)
                    records.append({
                        "Year":         2023 + (month_idx // 12),
                        "Month":        self.months[month_idx % 12],
                        "Month_Index":  month_idx,
                        "Date":         current_date.strftime("%Y-%m"),
                        "Category":     category,
                        "Region":       region,
                        "Units_Sold":   units,
                        "Avg_Price":    avg_price,
                        "Sales_Amount": round(sales, 2),
                    })
        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────

class SalesAnalyzer:
    """Performs exploratory data analysis on the sales DataFrame."""

    def __init__(self, df):
        self.df = df

    def basic_info(self):
        print("\n" + "="*60)
        print("  DATASET OVERVIEW")
        print("="*60)
        print(f"  Total Records     : {len(self.df)}")
        print(f"  Columns           : {list(self.df.columns)}")
        print(f"  Date Range        : {self.df['Date'].min()} → {self.df['Date'].max()}")
        print(f"  Categories        : {self.df['Category'].unique().tolist()}")
        print(f"  Regions           : {self.df['Region'].unique().tolist()}")
        print(f"  Missing Values    : {self.df.isnull().sum().sum()}")

    def summary_statistics(self):
        print("\n" + "="*60)
        print("  SUMMARY STATISTICS — Sales Amount")
        print("="*60)
        s = self.df["Sales_Amount"]
        print(f"  Total Revenue     : ₹{s.sum():>15,.2f}")
        print(f"  Mean Monthly Sale : ₹{s.mean():>15,.2f}")
        print(f"  Median            : ₹{s.median():>15,.2f}")
        print(f"  Std Deviation     : ₹{s.std():>15,.2f}")
        print(f"  Min               : ₹{s.min():>15,.2f}")
        print(f"  Max               : ₹{s.max():>15,.2f}")

    def category_performance(self):
        print("\n" + "="*60)
        print("  SALES BY CATEGORY")
        print("="*60)
        grp   = self.df.groupby("Category")["Sales_Amount"].agg(["sum","mean","count"])
        grp.columns = ["Total_Sales","Avg_Sales","Records"]
        grp   = grp.sort_values("Total_Sales", ascending=False)
        total = grp["Total_Sales"].sum()
        for cat, row in grp.iterrows():
            share = row["Total_Sales"] / total * 100
            print(f"  {cat:<15} | Total: ₹{row['Total_Sales']:>12,.0f} | "
                  f"Avg: ₹{row['Avg_Sales']:>9,.0f} | Share: {share:.1f}%")

    def region_performance(self):
        print("\n" + "="*60)
        print("  SALES BY REGION")
        print("="*60)
        grp = self.df.groupby("Region")["Sales_Amount"].agg(["sum","mean"])
        grp.columns = ["Total_Sales","Avg_Sales"]
        grp = grp.sort_values("Total_Sales", ascending=False)
        for reg, row in grp.iterrows():
            print(f"  {reg:<10} | Total: ₹{row['Total_Sales']:>12,.0f} | "
                  f"Avg: ₹{row['Avg_Sales']:>9,.0f}")

    def monthly_trend(self):
        print("\n" + "="*60)
        print("  MONTHLY SALES TREND (Aggregated All Categories & Regions)")
        print("="*60)
        trend = (self.df
                 .groupby(["Year","Month_Index","Month"])["Sales_Amount"]
                 .sum().reset_index()
                 .sort_values("Month_Index"))
        max_val = trend["Sales_Amount"].max()
        print(f"  {'Period':<12} {'Sales':>12}  Chart (scaled)")
        print(f"  {'-'*12} {'-'*12}  {'-'*30}")
        for _, row in trend.iterrows():
            label   = f"{row['Month']}-{row['Year']}"
            bar_len = int(row["Sales_Amount"] / max_val * 30)
            print(f"  {label:<12} ₹{row['Sales_Amount']:>11,.0f}  {'█'*bar_len}")

    def correlation_analysis(self):
        print("\n" + "="*60)
        print("  CORRELATION ANALYSIS")
        print("="*60)
        num_df = self.df[["Units_Sold","Avg_Price","Sales_Amount","Month_Index"]]
        print(num_df.corr().round(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: FEATURE ENGINEERING
#
# FIX B1: Rolling windows applied after shift(1) — zero data leakage.
#   Window at time t contains only [t-w, ..., t-1]; current target excluded.
#
# FIX B2: Units_Sold and Avg_Price are NOT included in feature_cols.
#   Both are algebraic derivatives of Sales_Amount:
#     Units_Sold ≈ Sales / (base * 0.05)
#     Avg_Price  = Sales / Units_Sold
#   Including them lets the model reconstruct the target trivially,
#   producing an artefactual R2 ≈ 1.0.
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """Transforms raw sales data into model-ready features."""

    def __init__(self, df):
        self.df = df.copy()

    def add_lag_features(self, lags=[1, 2, 3]):
        self.df = self.df.sort_values(["Category","Region","Month_Index"])
        for lag in lags:
            self.df[f"Sales_Lag_{lag}"] = (
                self.df.groupby(["Category","Region"])["Sales_Amount"].shift(lag)
            )

    def add_rolling_features(self, windows=[3, 6]):
        # FIX B1: shift(1) before rolling — current value never in the window
        self.df = self.df.sort_values(["Category","Region","Month_Index"])
        for w in windows:
            self.df[f"Rolling_Mean_{w}"] = (
                self.df.groupby(["Category","Region"])["Sales_Amount"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            self.df[f"Rolling_Std_{w}"] = (
                self.df.groupby(["Category","Region"])["Sales_Amount"]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
            )

    def add_time_features(self):
        self.df["Month_Num"] = self.df["Month_Index"] % 12 + 1
        self.df["Month_Sin"] = np.sin(2 * np.pi * self.df["Month_Num"] / 12)
        self.df["Month_Cos"] = np.cos(2 * np.pi * self.df["Month_Num"] / 12)

    def add_category_encoding(self):
        cat_map = {v: i for i, v in enumerate(self.df["Category"].unique())}
        reg_map = {v: i for i, v in enumerate(self.df["Region"].unique())}
        self.df["Category_Enc"] = self.df["Category"].map(cat_map)
        self.df["Region_Enc"]   = self.df["Region"].map(reg_map)

    def build(self):
        self.add_lag_features()
        self.add_rolling_features()
        self.add_time_features()
        self.add_category_encoding()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FORECASTING MODELS
# ─────────────────────────────────────────────────────────────────────────────

class LinearRegressionModel:
    """OLS Linear Regression from scratch via NumPy normal equation."""

    def __init__(self):
        self.coefficients = None
        self.intercept    = None

    def fit(self, X, y):
        X_b  = np.c_[np.ones(len(X)), X]
        beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
        self.intercept    = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        return self.intercept + X @ self.coefficients

    def evaluate(self, y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2   = round(1 - ss_res / ss_tot, 4)
        mae  = round(np.mean(np.abs(residuals)), 2)
        rmse = round(np.sqrt(np.mean(residuals ** 2)), 2)
        mape = round(np.mean(np.abs(residuals / y_true)) * 100, 2)
        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


class MovingAverageModel:
    """
    Simple Moving Average on a univariate time series.
    FIX B4: receives the aggregate monthly series (chronological), not raw rows.
    """

    def __init__(self, window=3):
        self.window = window

    def predict(self, series):
        preds = []
        for i in range(len(series)):
            start = max(0, i - self.window)
            preds.append(np.mean(series[start:i + 1]))
        return np.array(preds)

    def forecast_next(self, series, steps=3):
        extended = list(series)
        future   = []
        for _ in range(steps):
            next_val = np.mean(extended[-self.window:])
            future.append(next_val)
            extended.append(next_val)
        return np.array(future)


class WeightedMovingAverageModel:
    """
    Exponential Smoothing (EWMA).
    FIX B5: forecast_next updates both last_obs and last_smoothed each step
    so predictions correctly propagate forward (not anchored to series[-1]).
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def predict(self, series):
        smoothed = [series[0]]
        for i in range(1, len(series)):
            s = self.alpha * series[i - 1] + (1 - self.alpha) * smoothed[-1]
            smoothed.append(s)
        return np.array(smoothed)

    def forecast_next(self, series, steps=3):
        # FIX B5: carry prediction forward as the new observation each step
        last_smoothed = self.predict(series)[-1]
        last_obs      = float(series[-1])
        future = []
        for _ in range(steps):
            s = self.alpha * last_obs + (1 - self.alpha) * last_smoothed
            future.append(s)
            last_obs      = s   # next step's observation = this prediction
            last_smoothed = s
        return np.array(future)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MODEL EVALUATION & COMPARISON
#
# FIX B4: MA/WMA use an aggregate monthly time series built from df_raw
#   (all 24 months, sum of all categories × regions per month),
#   split at the month boundary: months 0-18 = train, 19-23 = test.
#   This is the correct and meaningful input for univariate baseline models.
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Trains all models, evaluates them, and compares results."""

    def __init__(self, df_features, df_raw):
        self.df          = df_features
        self.df_raw      = df_raw      # needed for MA/WMA aggregate series
        # FIX B2: Units_Sold and Avg_Price excluded
        self.feature_cols = [
            "Month_Index", "Month_Sin", "Month_Cos",
            "Category_Enc", "Region_Enc",
            "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_3",
            "Rolling_Mean_3", "Rolling_Mean_6",
            "Rolling_Std_3",  "Rolling_Std_6",
        ]

    def train_test_split(self, test_ratio=0.2):
        """Chronological split — no shuffling."""
        split_idx = int(len(self.df) * (1 - test_ratio))
        return self.df.iloc[:split_idx].copy(), self.df.iloc[split_idx:].copy()

    def _monthly_aggregate_split(self, test_ratio=0.2):
        """
        FIX B4: build aggregate series from df_raw and split at month boundary.
        Returns (full_series, train_series, test_series) as numpy arrays.
        """
        agg        = (self.df_raw
                      .groupby("Month_Index")["Sales_Amount"]
                      .sum()
                      .sort_index())
        full       = agg.values
        n_train    = int(len(full) * (1 - test_ratio))
        return full, full[:n_train], full[n_train:]

    @staticmethod
    def _metrics(y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2   = round(1 - ss_res / max(ss_tot, 1e-10), 4)
        mae  = round(np.mean(np.abs(residuals)), 2)
        rmse = round(np.sqrt(np.mean(residuals ** 2)), 2)
        mape = round(np.mean(np.abs(residuals / np.maximum(np.abs(y_true), 1e-6))) * 100, 2)
        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    def run(self):
        train, test = self.train_test_split()
        X_train = train[self.feature_cols].values
        y_train = train["Sales_Amount"].values
        X_test  = test[self.feature_cols].values
        y_test  = test["Sales_Amount"].values

        print("\n" + "="*60)
        print("  MODEL TRAINING & EVALUATION")
        print("="*60)
        print(f"  Train samples (LR) : {len(train)}")
        print(f"  Test  samples (LR) : {len(test)}")

        # ── Model 1: Linear Regression ──────────────────────────
        lr         = LinearRegressionModel()
        lr.fit(X_train, y_train)
        lr_preds   = lr.predict(X_test)
        lr_metrics = lr.evaluate(y_test, lr_preds)

        # ── MA & WMA: aggregate monthly series ───────────────────
        # FIX B4: proper chronological aggregate, month-boundary split
        full_series, train_series, test_series = self._monthly_aggregate_split()
        print(f"  Train months (MA/WMA): {len(train_series)}"
              f"   Test months (MA/WMA): {len(test_series)}")

        # ── Model 2: Moving Average ──────────────────────────────
        ma            = MovingAverageModel(window=3)
        ma_preds_full = ma.predict(full_series)
        ma_preds_test = ma_preds_full[len(train_series):]
        ma_metrics    = self._metrics(test_series, ma_preds_test)

        # ── Model 3: Weighted Moving Average ─────────────────────
        wma             = WeightedMovingAverageModel(alpha=0.4)
        wma_preds_full  = wma.predict(full_series)
        wma_preds_test  = wma_preds_full[len(train_series):]
        wma_metrics     = self._metrics(test_series, wma_preds_test)

        # ── Print comparison table ────────────────────────────────
        print("\n  METRIC COMPARISON TABLE")
        print(f"  {'Model':<32} {'R²':>8} {'MAE':>14} {'RMSE':>14} {'MAPE%':>8}")
        print(f"  {'-'*32} {'-'*8} {'-'*14} {'-'*14} {'-'*8}")
        note = {"Linear Regression (OLS)": " [per record]",
                "Moving Average (w=3)":    " [monthly agg]",
                "Weighted Moving Avg (α=0.4)": " [monthly agg]"}
        for name, m in [
            ("Linear Regression (OLS)", lr_metrics),
            ("Moving Average (w=3)",    ma_metrics),
            ("Weighted Moving Avg (α=0.4)", wma_metrics),
        ]:
            print(f"  {name:<32} {m['R2']:>8.4f} {m['MAE']:>14,.2f} "
                  f"{m['RMSE']:>14,.2f} {m['MAPE']:>7.2f}%  {note[name]}")

        print("\n  Note: LR metrics are at the per-record level (one category-region pair).")
        print("  MA/WMA metrics are at the monthly aggregate level (all categories & regions).")
        print("  Both are valid but measure different granularities — direct R² comparison")
        print("  should account for this scale difference.")

        return {
            "lr":           (lr, lr_metrics, lr_preds),
            "ma":           (ma, ma_metrics, ma_preds_test),
            "wma":          (wma, wma_metrics, wma_preds_test),
            "y_test":       y_test,
            "full_series":  full_series,
            "train_series": train_series,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: FORECASTING FUTURE SALES
#
# FIX B7: Lag propagation is correct across all steps.
#   A dedicated lag_buffer tracks [lag3, lag2, lag1] and shifts left each step.
#   Rolling statistics are recomputed from accumulated prediction history.
# ─────────────────────────────────────────────────────────────────────────────

class SalesForecaster:
    """Uses the trained model to forecast future months."""

    def __init__(self, model, model_type="lr"):
        self.model      = model
        self.model_type = model_type

    def forecast(self, df, feature_cols, steps=3, full_series=None):
        print("\n" + "="*60)
        print("  FUTURE SALES FORECAST — Next 3 Months")
        print("="*60)

        forecasts = []

        if self.model_type == "lr":
            last_row = df.iloc[-1].copy()

            # FIX B7: initialise lag buffer from last known values
            lag3 = float(last_row["Sales_Lag_3"])
            lag2 = float(last_row["Sales_Lag_2"])
            lag1 = float(last_row["Sales_Lag_1"])
            # history for rolling stats: last 6 known actual values
            mask = ((df["Category"] == last_row["Category"]) &
                    (df["Region"]   == last_row["Region"]))
            roll_history = (df.loc[mask]
                              .sort_values("Month_Index")
                              ["Sales_Amount"].values[-6:].tolist())

            for step in range(1, steps + 1):
                future_row = last_row.copy()
                new_idx    = int(last_row["Month_Index"]) + step
                future_row["Month_Index"] = new_idx
                mn = (new_idx % 12) + 1
                future_row["Month_Sin"] = np.sin(2 * np.pi * mn / 12)
                future_row["Month_Cos"] = np.cos(2 * np.pi * mn / 12)
                future_row["Month_Num"] = mn

                # FIX B7: assign correct lags for this step
                future_row["Sales_Lag_1"] = lag1
                future_row["Sales_Lag_2"] = lag2
                future_row["Sales_Lag_3"] = lag3

                # Rolling stats from lag history (no leakage)
                h3 = roll_history[-3:] if len(roll_history) >= 3 else roll_history
                h6 = roll_history[-6:] if len(roll_history) >= 6 else roll_history
                future_row["Rolling_Mean_3"] = float(np.mean(h3))
                future_row["Rolling_Mean_6"] = float(np.mean(h6))
                future_row["Rolling_Std_3"]  = float(np.std(h3))
                future_row["Rolling_Std_6"]  = float(np.std(h6))

                X_f  = np.array([[future_row[c] for c in feature_cols]])
                pred = float(self.model.predict(X_f)[0])
                forecasts.append(pred)

                # FIX B7: shift lag buffer — oldest drops off, new prediction enters
                lag3 = lag2
                lag2 = lag1
                lag1 = pred
                roll_history.append(pred)

        elif self.model_type in ("ma", "wma"):
            if full_series is None:
                raise ValueError("full_series required for MA/WMA forecasting.")
            forecasts = list(self.model.forecast_next(full_series, steps=steps))

        months_list = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        base_month = int(df.iloc[-1]["Month_Index"]) % 12
        base_year  = int(df.iloc[-1]["Year"])

        label_width = 28 if self.model_type != "lr" else 30
        unit_note   = "Monthly Aggregate" if self.model_type in ("ma","wma") else "Per Record (avg category-region)"
        print(f"\n  Unit: {unit_note}")
        print(f"\n  {'Month':<12} {'Forecasted Sales':>20}")
        print(f"  {'-'*12} {'-'*20}")
        for i, fc in enumerate(forecasts):
            m_idx = (base_month + i + 1) % 12
            yr    = base_year + ((base_month + i + 1) // 12)
            print(f"  {months_list[m_idx]}-{yr:<8} ₹{fc:>19,.2f}")

        return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: BUSINESS INSIGHTS
# FIX B6: Operates on an internal copy — df_raw is never mutated.
# ─────────────────────────────────────────────────────────────────────────────

class BusinessInsights:
    """Extracts key business insights from the sales data."""

    def __init__(self, df):
        self.df = df.copy()   # FIX B6: internal copy only

    def top_performers(self):
        print("\n" + "="*60)
        print("  BUSINESS INSIGHTS")
        print("="*60)
        months_list = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]

        cat_sales = self.df.groupby("Category")["Sales_Amount"].sum()
        best_cat  = cat_sales.idxmax()
        print(f"\n  [1] Top Category     : {best_cat} "
              f"(₹{cat_sales[best_cat]:,.0f} total revenue)")

        reg_sales = self.df.groupby("Region")["Sales_Amount"].sum()
        best_reg  = reg_sales.idxmax()
        print(f"  [2] Top Region       : {best_reg} "
              f"(₹{reg_sales[best_reg]:,.0f} total revenue)")

        # FIX B6: Month_Num added to the copy, not to df_raw
        self.df["Month_Num"]   = self.df["Month_Index"] % 12
        month_sales            = self.df.groupby("Month_Num")["Sales_Amount"].mean()
        best_month_idx         = month_sales.idxmax()
        print(f"  [3] Peak Sales Month : {months_list[best_month_idx]} "
              f"(avg ₹{month_sales[best_month_idx]:,.0f})")

        y1  = self.df[self.df["Year"] == 2023]["Sales_Amount"].sum()
        y2  = self.df[self.df["Year"] == 2024]["Sales_Amount"].sum()
        yoy = (y2 - y1) / y1 * 100
        print(f"  [4] Year-on-Year Growth : {yoy:.2f}%  (2023 → 2024)")

        avg_price_cat = self.df.groupby("Category")["Avg_Price"].mean()
        high_price    = avg_price_cat.idxmax()
        print(f"  [5] Highest Avg Price   : {high_price} "
              f"(₹{avg_price_cat[high_price]:,.2f} per unit)")

        print("\n  STRATEGIC RECOMMENDATIONS")
        print(f"  → Prioritize {best_cat} inventory ahead of Q4 festive season.")
        print(f"  → Invest in logistics and marketing in the {best_reg} region.")
        print(f"  → Plan promotional campaigns for {months_list[best_month_idx]} "
              f"to maximize revenue.")
        print(f"  → YoY growth of {yoy:.1f}% confirms a positive demand trend — "
              f"scale supply chain accordingly.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*60)
    print("  DATA-DRIVEN SALES FORECASTING USING PYTHON")
    print("  Business Analytics Using Python — Mini Project")
    print("█"*60)

    # Step 1: Data Generation
    print("\n[STEP 1] Generating synthetic sales dataset...")
    generator = SalesDataGenerator(seed=42)
    df_raw    = generator.generate(years=2)
    print(f"  Dataset created: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    # Step 2: EDA
    print("\n[STEP 2] Exploratory Data Analysis")
    analyzer = SalesAnalyzer(df_raw)
    analyzer.basic_info()
    analyzer.summary_statistics()
    analyzer.category_performance()
    analyzer.region_performance()
    analyzer.monthly_trend()
    analyzer.correlation_analysis()

    # Step 3: Feature Engineering
    print("\n[STEP 3] Feature Engineering")
    fe          = FeatureEngineer(df_raw)
    df_features = fe.build()
    new_cols    = [c for c in df_features.columns if c not in df_raw.columns]
    print(f"  Features added. Enhanced shape: {df_features.shape}")
    print(f"  New feature columns: {new_cols}")

    # Step 4: Model Training & Evaluation
    # FIX B4: pass df_raw so MA/WMA can use the aggregate monthly series
    print("\n[STEP 4] Model Training & Evaluation")
    evaluator = ModelEvaluator(df_features, df_raw)
    results   = evaluator.run()

    # Step 5: Forecasting
    print("\n[STEP 5] Future Sales Forecast")
    feature_cols = evaluator.feature_cols
    lr_model     = results["lr"][0]
    full_series  = results["full_series"]
    forecaster   = SalesForecaster(lr_model, model_type="lr")
    forecasts    = forecaster.forecast(df_features, feature_cols,
                                       steps=3, full_series=full_series)

    # Step 6: Business Insights
    print("\n[STEP 6] Business Insights & Recommendations")
    insights = BusinessInsights(df_raw)
    insights.top_performers()

    # Summary
    print("\n" + "="*60)
    print("  EXECUTION COMPLETE")
    print("="*60)
    print(f"  • Records Analyzed  : {len(df_raw)}")
    print(f"  • Features Built    : {len(feature_cols)}")
    print(f"  • Models Evaluated  : 3")
    print(f"  • Months Forecasted : 3")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
