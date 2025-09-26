import os
import polars as pl

os.makedirs("data/features/", exist_ok=True) 

train = pl.read_csv("data/processed/application_train.csv")
test = pl.read_csv("data/processed/application_test.csv")

bureau = pl.read_csv("data/processed/bureau.csv")
train = train.join(bureau, on="SK_ID_CURR", how="left")
test = test.join(bureau, on="SK_ID_CURR", how="left")

train.write_csv("data/features/train_features.csv")
test.write_csv("data/features/test_features.csv")