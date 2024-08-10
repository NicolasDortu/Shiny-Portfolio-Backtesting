import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.express as px

# Parameters
days = 300  # Number of days of historical data to be used
simulations = 10  # Number of portfolio compositions to test
initialPortfolio = 10000  # Initial value of the portfolio ($)


# Import data
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData["Adj Close"]
    stockData = stockData.ffill().bfill()  # Forward fill, then backward fill
    returns = stockData.pct_change().dropna()  # Drop NaN values from returns
    return returns, stockData


stockList = ["AAPL", "MSFT", "PLTR"]

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=days)

returns, prices = getData(stockList, start=startDate, end=endDate)
returns = returns.dropna()
prices = prices.loc[returns.index]  # Align prices with returns data

# DataFrame to store backtesting results
backtesting_df = pd.DataFrame()

# Generate multiple portfolio compositions and backtest
for m in range(0, simulations):
    # Generate new random weights for each portfolio composition
    weights = np.random.random(len(stockList))
    weights /= np.sum(weights)  # Normalize the weights

    # Calculate portfolio returns based on historical data
    portfolio_returns = returns.dot(weights)

    # Calculate the cumulative portfolio value over time
    portfolio_values = (portfolio_returns + 1).cumprod() * initialPortfolio

    # Create a DataFrame to hold the daily portfolio composition with gains/losses
    temp_df = pd.DataFrame(
        {"Date": portfolio_values.index, "Portfolio Value": portfolio_values.values}
    )
    temp_df["Simulation"] = f"Composition {m+1}"

    # Calculate daily gains/losses for each stock
    daily_gains_losses = (prices.div(prices.iloc[0]) - 1) * (initialPortfolio * weights)

    # Add the gains/losses information to the portfolio composition
    for i, date in enumerate(temp_df["Date"]):
        portfolio_composition = "<br>" + "<br>".join(
            [
                f"{stock}: {weight:.2%} ({'+' if gain >= 0 else ''}{gain:.2f}$)"
                for stock, weight, gain in zip(
                    stockList, weights, daily_gains_losses.iloc[i]
                )
            ]
        )
        temp_df.loc[temp_df.index[i], "Portfolio Composition"] = portfolio_composition

    # Combine with the overall backtesting DataFrame
    backtesting_df = pd.concat([backtesting_df, temp_df], ignore_index=True)

# Plot using Plotly Express
fig = px.line(
    backtesting_df,
    x="Date",
    y="Portfolio Value",
    color="Simulation",
    title="Backtesting Different Portfolio Compositions",
    labels={"Portfolio Value": "Portfolio Value ($)", "Date": "Date"},
    hover_data={"Portfolio Composition": True},
)

fig.show()
