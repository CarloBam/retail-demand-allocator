# AI assistance disclosure (per CS50 final project policy):
# Design guidance and specifications from Claude (Anthropic) and ChatGPT (OpenAI).
# Code written by Carlo Bam.

import sqlite3
import numpy as np
import pandas as pd
import pulp

DB_PATH = "retail.db"
SHELF_CAPACITY = 4500  # max total units a store can receive across all products


def load_inputs(conn):
    # Weekly demand per store and product, summed over the 7 forecast days
    demand = pd.read_sql(
        """
        SELECT s.id AS store_id, p.id AS product_id,
               f.store_name, f.product_name,
               SUM(f.predicted_units) AS demand
        FROM forecasts f
        JOIN stores s ON s.name = f.store_name
        JOIN products p ON p.name = f.product_name
        GROUP BY f.store_name, f.product_name
        """,
        conn,
    )
    stock = pd.read_sql("SELECT id AS product_id, warehouse_stock FROM products", conn)
    return demand, stock


def solve_allocation(demand, stock):
    stores = demand["store_id"].unique()
    products = demand["product_id"].unique()
    D = demand.set_index(["store_id", "product_id"])["demand"].to_dict()
    S = stock.set_index("product_id")["warehouse_stock"].to_dict()

    prob = pulp.LpProblem("stock_allocation", pulp.LpMaximize)

    # x[(s, p)] = units of product p sent to store s
    x = pulp.LpVariable.dicts("alloc", [(s, p) for s in stores for p in products], lowBound=0, cat="Integer")

    # f[(s, p)] = units of demand actually met, linearisation of min(allocation, demand)
    f = pulp.LpVariable.dicts("fulfilled", [(s, p) for s in stores for p in products], lowBound=0, cat="Continuous")

    # Objective: maximise total fulfilled demand
    prob += pulp.lpSum(f[(s, p)] for s in stores for p in products)

    # Linearisation: fulfilled cannot exceed the allocation or the demand
    for s in stores:
        for p in products:
            prob += f[(s, p)] <= x[(s, p)]
            prob += f[(s, p)] <= D[(s, p)]

    # Warehouse stock limit per product
    for p in products:
        prob += pulp.lpSum(x[(s, p)] for s in stores) <= S[p]

    # Shelf capacity limit per store
    for s in stores:
        prob += pulp.lpSum(x[(s, p)] for p in products) <= SHELF_CAPACITY

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print(f"solver status: {pulp.LpStatus[prob.status]}")

    demand = demand.copy()
    demand["optimal_units"] = [int(x[(r.store_id, r.product_id)].value()) for r in demand.itertuples()]
    return demand


def naive_baseline(demand, stock):
    # Head office rule of thumb: split each product's stock evenly across stores
    demand = demand.copy()
    S = stock.set_index("product_id")["warehouse_stock"].to_dict()
    n_stores = demand["store_id"].nunique()
    demand["naive_units"] = (demand["product_id"].map(S) // n_stores).astype(int)
    return demand


def write_results(conn, result):
    result["optimal_fulfilled"] = np.minimum(result["optimal_units"], result["demand"])
    result["naive_fulfilled"] = np.minimum(result["naive_units"], result["demand"])

    out = result[["store_name", "product_name", "optimal_units", "naive_units"]]
    out.to_sql("allocations", conn, if_exists="replace", index=False)

    summary = pd.DataFrame([{
        "optimal_fulfilled": int(result["optimal_fulfilled"].sum()),
        "naive_fulfilled": int(result["naive_fulfilled"].sum()),
    }])
    summary.to_sql("summary", conn, if_exists="replace", index=False)

    print(f"optimal fulfilled: {summary.optimal_fulfilled[0]:,}")
    print(f"naive fulfilled:   {summary.naive_fulfilled[0]:,}")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    demand, stock = load_inputs(conn)
    result = solve_allocation(demand, stock)
    result = naive_baseline(result, stock)
    write_results(conn, result)
    conn.close()