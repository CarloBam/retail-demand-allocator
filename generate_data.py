# AI assistance disclosure (per CS50 final project policy):
# Design guidance and specifications from Claude (Anthropic) and ChatGPT (OpenAI).
# Code written by Carlo Bam.

import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "retail.db"

STORES = ["Canal Walk", "Promenade", "Tygervalley", "Gardens", "Somerset Mall"]
PRODUCTS = [
    ("Ground Coffee 250g", "Beverages"),
    ("Rooibos Tea 40s", "Beverages"),
    ("Sparkling Water 1L", "Beverages"),
    ("Orange Juice 2L", "Beverages"),
    ("White Bread 700g", "Food"),
    ("Long Grain Rice 2kg", "Food"),
    ("Pasta Shells 500g", "Food"),
    ("Canned Tomatoes 410g", "Food"),
    ("Breakfast Cereal 750g", "Food"),
    ("Peanut Butter 400g", "Food"),
    ("Dishwashing Liquid 750ml", "Household"),
    ("Laundry Powder 2kg", "Household"),
    ("Paper Towels 2pk", "Household"),
    ("Refuse Bags 20pk", "Household"),
    ("Toothpaste 100ml", "Toiletries"),
    ("Shampoo 400ml", "Toiletries"),
    ("Bath Soap 175g", "Toiletries"),
    ("Potato Chips 125g", "Snacks"),
    ("Chocolate Slab 80g", "Snacks"),
    ("Mixed Nuts 200g", "Snacks"),
]

DATES = pd.date_range("2024-07-01", "2026-06-30")
np.random.seed(42)


def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS stores;

        CREATE TABLE stores (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            warehouse_stock INTEGER NOT NULL
        );

        CREATE TABLE sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            sale_date TEXT NOT NULL,
            units_sold INTEGER NOT NULL,
            promo INTEGER NOT NULL
        );
    """)

    for store_id, name in enumerate(STORES, start=1):
        conn.execute("INSERT INTO stores (id, name) VALUES (?, ?)", (store_id, name))

    for product_id, (name, category) in enumerate(PRODUCTS, start=1):
        stock = int(np.random.randint(500, 1500))
        conn.execute("INSERT INTO products (id, name, category, warehouse_stock) VALUES (?, ?, ?, ?)", (product_id, name, category, stock))

    conn.commit()


def build_sales_rows():
    weekday_factor = {0: 0.85, 1: 0.95, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.4, 6: 1.15}
    rows = []

    for store_id in range(1, len(STORES) + 1):
        for product_id in range(1, len(PRODUCTS) + 1):
            base = np.random.uniform(5, 40)

            for date in DATES:
                expected = base * weekday_factor[date.dayofweek]

                if date.day >= 25 or date.day == 1:
                    expected *= 1.3

                if date.month == 12:
                    expected *= 1.5
                elif date.month == 1:
                    expected *= 0.85

                promo = np.random.random() < 0.05

                if promo:
                    expected *= 1.6

                units = np.random.poisson(expected)

                rows.append((store_id, product_id, date.strftime("%Y-%m-%d"), int(units), int(promo)))

    return rows


def insert_data(conn, rows):
    conn.executemany("INSERT INTO sales (store_id, product_id, sale_date, units_sold, promo) VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()


def verify(conn):
    df = pd.read_sql("SELECT sale_date, units_sold FROM sales", conn)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["weekday"] = df["sale_date"].dt.day_name()
    print(f"Total rows: {len(df):,}")
    print(df.groupby("weekday")["units_sold"].mean().round(2))


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    rows = build_sales_rows()
    insert_data(conn, rows)
    verify(conn)
    conn.close()