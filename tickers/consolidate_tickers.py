import pandas as pd

# Source : https://github.com/jonfriesen/symbol-list-data
df = pd.read_csv("./2022-10-11.csv")
df2 = pd.read_csv("./2022-10-12-crypto.csv")

# get the first two columns of each dataframe and concatenate them into one :
df_global = pd.concat([df.iloc[:, :2], df2.iloc[:, :2]], axis=0)

df_global.to_csv("./tickers.csv", index=False, sep=",", encoding="utf-8-sig")
