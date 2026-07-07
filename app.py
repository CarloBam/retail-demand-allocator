# AI assistance disclosure (per CS50 final project policy):
# Boilerplate scaffolding (routes, templates, CSS) provided by Claude (Anthropic).
# Design guidance and debugging support from Claude and ChatGPT (OpenAI).
# Forecasting, allocation and data generation logic written by Carlo Bam.

import sqlite3
from flask import Flask, render_template

app = Flask(__name__)

DB_PATH = "retail.db"


def query_db(sql, params=()):
    """Run a read query and return rows as dictionaries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return [dict(row) for row in rows]


@app.route("/")
def dashboard():
    # Headline numbers for the landing page
    totals = query_db(
        """
        SELECT COUNT(*) AS sale_rows,
               SUM(units_sold) AS total_units,
               MIN(sale_date) AS first_date,
               MAX(sale_date) AS last_date
        FROM sales
        """
    )[0]

    by_weekday = query_db(
        """
        SELECT strftime('%w', sale_date) AS weekday,
               ROUND(AVG(units_sold), 2) AS avg_units
        FROM sales
        GROUP BY weekday
        ORDER BY weekday
        """
    )

    return render_template("dashboard.html", totals=totals, by_weekday=by_weekday)


@app.route("/forecasts")
def forecasts():
    # forecast.py writes its output to the forecasts table.
    # Columns expected: store_name, product_name, forecast_date, predicted_units
    rows = query_db(
        """
        SELECT store_name, product_name, forecast_date, predicted_units
        FROM forecasts
        ORDER BY store_name, product_name, forecast_date
        """
    )

    # metrics table holds one row with the test set results from forecast.py.
    # Columns expected: mae, rmse
    metrics = query_db("SELECT mae, rmse FROM metrics")
    metrics = metrics[0] if metrics else None

    return render_template("forecasts.html", rows=rows, metrics=metrics)


@app.route("/allocation")
def allocation():
    # allocate.py writes its output to the allocations table.
    # Columns expected: store_name, product_name, optimal_units, naive_units
    rows = query_db(
        """
        SELECT store_name, product_name, optimal_units, naive_units
        FROM allocations
        ORDER BY store_name, product_name
        """
    )

    # summary table holds one row comparing the optimiser to the naive split.
    # Columns expected: optimal_fulfilled, naive_fulfilled
    summary = query_db("SELECT optimal_fulfilled, naive_fulfilled FROM summary")
    summary = summary[0] if summary else None

    return render_template("allocation.html", rows=rows, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
