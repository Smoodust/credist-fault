import polars as pl

train = pl.read_csv("data/raw/bureau.csv")
train = train.group_by("SK_ID_CURR").agg(
    pl.col("SK_ID_BUREAU").len().alias("Count of bureau"),
    (pl.col("CREDIT_ACTIVE") == "ACTIVE").sum().alias("Count of active bureau"),
    pl.col("CNT_CREDIT_PROLONG").sum().alias("sum_overdue"),
    pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("AMT_CREDIT_SUM_DEBT_SUM"),
    pl.col("AMT_CREDIT_SUM_DEBT").mean().alias("AMT_CREDIT_SUM_DEBT_MEAN"),
)
train.write_csv("data/processed/bureau.csv")