# AI assistance disclosure (per CS50 final project policy):
# Design guidance and specifications from Claude (Anthropic) and ChatGPT (OpenAI).
# Code written by Carlo Bam.

import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

DB_PATH = "retail.db"
FEATURES = ["store_id", "product_id", "dayofweek", "month", "dayofmonth",
            "promo", "lag_7", "lag_14", "rolling_7"]


def load_data(conn):
    df = pd.read_sql("SELECT store_id, product_id, sale_date, units_sold, promo FROM sales", conn)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["store_id", "product_id", "sale_date"])
    return df


def add_features(df):
    df["dayofweek"] = df["sale_date"].dt.dayofweek
    df["month"] = df["sale_date"].dt.month
    df["dayofmonth"] = df["sale_date"].dt.day

    g = df.groupby(["store_id", "product_id"])["units_sold"]
    df["lag_7"] = g.shift(7)
    df["lag_14"] = g.shift(14)
    df["rolling_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())

    return df.dropna()


def split_data(df):
    # Chronological split. A random split would leak future information
    # into training, which inflates metrics on time series data.
    df = df.sort_values("sale_date")
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def train_model(train, val):
    # Candidates are selected on validation MAE. The test set is only
    # touched once, by the champion, in evaluate().

    # Naive benchmark: predict last week's same-day sales.
    naive_mae = np.mean(np.abs(val["units_sold"] - val["lag_7"]))
    print(f"naive lag_7 benchmark: validation MAE {naive_mae:.2f}")

    candidates = {
        "ridge_linear": Ridge(alpha=1.0),
        "gbm_default": HistGradientBoostingRegressor(random_state=42),
        "gbm_deeper": HistGradientBoostingRegressor(
            random_state=42, max_iter=300, max_depth=6
        ),
    }

    best_name = None
    best_model = None
    best_mae = float("inf")

    for name, model in candidates.items():
        model.fit(train[FEATURES], train["units_sold"])
        val_preds = model.predict(val[FEATURES])
        val_mae = np.mean(np.abs(val["units_sold"] - val_preds))
        print(f"{name}: validation MAE {val_mae:.2f}")
        if val_mae < best_mae:
            best_name = name
            best_model = model
            best_mae = val_mae

    print(f"champion: {best_name}")
    return best_model


def evaluate(model, test):
    preds = model.predict(test[FEATURES])
    errors = test["units_sold"] - preds
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return round(float(mae), 2), round(float(rmse), 2)


def build_future_features(df):
    last_date = df["sale_date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    rows = []
    history = df.set_index(["store_id", "product_id", "sale_date"])["units_sold"]

    for store_id in df["store_id"].unique():
        for product_id in df["product_id"].unique():
            pair = df[(df["store_id"] == store_id) & (df["product_id"] == product_id)]
            recent = pair.tail(7)["units_sold"].mean()
            for date in future_dates:
                lag7 = history.get((store_id, product_id, date - pd.Timedelta(days=7)), recent)
                lag14 = history.get((store_id, product_id, date - pd.Timedelta(days=14)), recent)
                rows.append({
                    "store_id": store_id, "product_id": product_id,
                    "forecast_date": date, "dayofweek": date.dayofweek,
                    "month": date.month, "dayofmonth": date.day,
                    "promo": 0, "lag_7": lag7, "lag_14": lag14, "rolling_7": recent,
                })

    return pd.DataFrame(rows)


def write_results(conn, future, preds, mae, rmse):
    future = future.copy()
    future["predicted_units"] = np.round(preds).astype(int)

    stores = pd.read_sql("SELECT id, name FROM stores", conn).set_index("id")["name"]
    products = pd.read_sql("SELECT id, name FROM products", conn).set_index("id")["name"]
    future["store_name"] = future["store_id"].map(stores)
    future["product_name"] = future["product_id"].map(products)
    future["forecast_date"] = future["forecast_date"].dt.strftime("%Y-%m-%d")

    out = future[["store_name", "product_name", "forecast_date", "predicted_units"]]
    out.to_sql("forecasts", conn, if_exists="replace", index=False)

    pd.DataFrame([{"mae": mae, "rmse": rmse}]).to_sql("metrics", conn, if_exists="replace", index=False)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = add_features(load_data(conn))
    train, val, test = split_data(df)
    model = train_model(train, val)
    mae, rmse = evaluate(model, test)
    print(f"Test MAE: {mae}, Test RMSE: {rmse}")
    future = build_future_features(df)
    preds = model.predict(future[FEATURES])
    write_results(conn, future, preds, mae, rmse)
    print(f"Wrote {len(future)} forecast rows.")
    conn.close()