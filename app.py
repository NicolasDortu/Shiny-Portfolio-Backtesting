import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.express as px

from shinywidgets import render_plotly
from shiny import reactive
from shiny.express import input, ui

from pathlib import Path


tickerspath = Path(__file__).parent / "tickers" / "tickers.csv"
tickersList = pd.read_csv(tickerspath)

# UI Elements
ui.page_opts(title="Backtesting Portfolio Compositions")
# Parameters
with ui.sidebar(title="Parameters"):
    ui.input_numeric(
        "days", "Days of historical data", 100, min=1
    )  # Number of days of historical data to be used
    ui.input_numeric(
        "simulations", "Number of portfolio compositions to test", 10, min=1
    )  # Number of portfolio compositions to test
    ui.input_numeric(
        "initialPortfolio", "Initial value of the portfolio ($)", 10000, min=1
    )  # Initial value of the portfolio ($)
    ui.input_selectize(
        "tickers", "Select tickers", tickersList["Symbol"].tolist(), multiple=True
    )
    ui.input_action_button("validate", "Validate")


def getData(stocks, start, end):
    try:
        stockData = yf.download(stocks, start=start, end=end)
        if stockData.empty:
            raise ValueError(
                f"Data for ticker(s) {', '.join(stocks)} could not be downloaded."
            )

        stockData = stockData["Adj Close"]
        stockData = (
            stockData.ffill().bfill()
        )  # Forward fill, then backward fill (in case value is missing)
        returns = stockData.pct_change().dropna()  # Drop NaN values from returns
        return returns, stockData

    except Exception as e:
        ui.notification_show(f"Error: {str(e)}", type="error", duration=5000)
        return None, None


@render_plotly
@reactive.event(input.validate)
def fig():
    with ui.Progress() as p:
        p.set(
            message="Plotting data in progress", detail="This may take a few seconds..."
        )

        # Fetch the user inputs
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=input.days())
        tickers = input.tickers()  # Get the selected tickers as a list

        # Get data for the selected tickers
        returns, prices = getData(tickers, start=start_date, end=end_date)

        # Align prices with returns data
        returns = returns.dropna()
        prices = prices.loc[returns.index]

        # DataFrame to store backtesting results
        backtesting_df = pd.DataFrame()

        # Generate multiple portfolio compositions and backtest
        for m in range(input.simulations()):
            # Generate new random weights for each portfolio composition
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)  # Normalize the weights

            # Calculate portfolio returns based on historical data
            portfolio_returns = returns.dot(weights)

            # Calculate the cumulative portfolio value over time
            portfolio_values = (
                portfolio_returns + 1
            ).cumprod() * input.initialPortfolio()

            # Create a DataFrame to hold the daily portfolio composition with gains/losses
            temp_df = pd.DataFrame(
                {
                    "Date": portfolio_values.index,
                    "Portfolio Value": portfolio_values.values,
                }
            )
            temp_df["Simulation"] = f"Composition {m+1}"

            # Calculate daily gains/losses for each stock
            daily_gains_losses = (prices.div(prices.iloc[0]) - 1) * (
                input.initialPortfolio() * weights
            )

            # Add the gains/losses information to the portfolio composition
            for i, date in enumerate(temp_df["Date"]):
                portfolio_composition = "<br>" + "<br>".join(
                    [
                        f"{stock}: {weight:.2%} ({'+' if gain >= 0 else ''}{gain:.2f}$)"
                        for stock, weight, gain in zip(
                            tickers, weights, daily_gains_losses.iloc[i]
                        )
                    ]
                )
                temp_df.loc[temp_df.index[i], "Portfolio Composition"] = (
                    portfolio_composition
                )

            # Combine with the overall backtesting DataFrame
            backtesting_df = pd.concat([backtesting_df, temp_df], ignore_index=True)

        # Plot the results using Plotly Express
        fig = px.line(
            backtesting_df,
            x="Date",
            y="Portfolio Value",
            color="Simulation",
            title="Backtesting Different Portfolio Compositions",
            labels={"Portfolio Value": "Portfolio Value ($)", "Date": "Date"},
            hover_data={"Portfolio Composition": True},
        )
    return fig
