# Retail Demand Forecasting and Stock Allocation Simulator
#### Video Demo: <URL will be pased here after I record it>
#### Description:

This project is a small decision support tool for a fictional retail chain. It generates two years of synthetic daily sales data for 5 stores and 20 products, trains a machine learning model to forecast demand for the next week, and then uses linear programming to decide how a limited warehouse stock should be allocated across the stores. A Flask web application presents the results in three pages, a dashboard with summary statistics, a forecasts page showing the model predictions and test metrics, and an allocation page comparing the optimiser against a simple rule of thumb.

I work in people analytics and I am moving towards data science, so I wanted a final project that combined forecasting and optimisation in one pipeline rather than another to do list or blog. All data is synthetic. No real company data is used anywhere in the project.

## Files

generate_data.py builds the dataset. Each store and product pair gets a base daily demand drawn once, then every day that base is adjusted by multipliers for weekday, month end paydays, December and January seasonality, and random promotions. The final sales figure is drawn from a Poisson distribution with the adjusted expectation as its rate. I chose Poisson rather than adding normal noise because sales are non negative integer counts, and a count distribution gives realistic day to day variation without producing negative or fractional values. Everything is written to a SQLite database, retail.db, with a fixed random seed so the dataset is reproducible.

forecast.py trains the demand model. It builds calendar features, the promotion flag, lagged sales at 7 and 14 days, and a 7 day rolling mean, then splits the data chronologically into 70 percent training, 15 percent validation and 15 percent test. A random split would leak future information into training, which is the classic mistake with time series data, so the split follows the calendar. Three candidate models are compared on validation MAE, a ridge regression as a linear floor, and two gradient boosted tree configurations. There is also a naive benchmark that simply predicts last week's same day sales, because any model that cannot beat that has no reason to exist. The champion is selected on validation only and then evaluated once on the test set. The final model, the deeper gradient boosted configuration, reached a test MAE of 4.17 and RMSE of 5.44 against average daily demand of roughly 25 units. Forecasts for the next 7 days are written back to the database. Future promotions are unknown at forecast time, so predictions represent a no promotion baseline.

allocate.py takes the weekly forecast per store and product as demand, reads the available warehouse stock, and solves an integer linear programme with PuLP. The objective maximises expected fulfilled demand. Because min(allocation, demand) is not a linear expression, I introduced a helper variable for fulfilled units that the objective pushes upward while two constraints cap it at the allocation and at the demand, so at the optimum it lands on the smaller of the two. Constraints cover warehouse stock per product and a shelf capacity limit per store. The optimiser is compared against the heuristic a head office might actually use, splitting each product's stock evenly across the five stores. In the final run the optimiser fulfilled 14,049 units against 13,067 for the even split, a gain of about 7.5 percent.

app.py is the Flask application. It reads the tables the three scripts produce and renders them through Jinja templates in the templates folder, with styling in static/styles.css. If a stage has not run yet, the relevant page shows a notice instead of crashing.

## Design decisions

The decision I learned the most from was calibrating stock levels. My first attempt made warehouse stock abundant, and the optimiser could not beat the even split because there was enough stock for everyone. My second attempt made stock uniformly scarce, and again the gap collapsed, because every store could absorb its share and nothing was wasted. Optimisation only earns its keep in the middle regime, where total stock is close to total demand and demand differs enough between stores that an even split wastes units at quiet stores while busy stores run short. Watching the value of the optimiser appear and disappear as a function of one parameter taught me more about when optimisation matters than the formulation itself did.

I also deliberately kept the test set untouched during model selection. Models were compared on validation data and the champion met the test set exactly once. The metrics on the forecasts page are from that single evaluation.

## Running the project

Create a virtual environment, install the packages in requirements.txt, then run the scripts in order, python generate_data.py, python forecast.py, python allocate.py, and start the site with flask run. Rerunning generate_data.py invalidates everything downstream, so the whole chain must be rerun after it.

## Future improvements

A revenue weighted objective would let the optimiser also beat a demand proportional split, which is provably optimal for the unweighted case. Other candidates are hierarchical forecasting across product categories, retraining the champion on train plus validation before final evaluation, and stochastic demand in the allocation stage instead of treating forecasts as certain.

## AI assistance

Per the CS50 final project policy, design guidance, specifications, debugging help and the boilerplate scaffolding (Flask routes, templates and CSS) came from Claude (Anthropic) and ChatGPT (OpenAI). The data generation, forecasting and allocation logic was written by me, and each source file carries a disclosure comment.