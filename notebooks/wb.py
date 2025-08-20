import polars as pl
import missingno as msno
import pandas as pd
from src.data.data_pull import DataClean
import matplotlib.pyplot as plt

dc = DataClean()


# Initialice raw data
df = dc.pull_wb()
data = pl.read_csv("data/external/countries.csv")
df = df.join(data, on="country", how="inner", validate="m:1")
msno.matrix(df.to_pandas())
plt.show()
