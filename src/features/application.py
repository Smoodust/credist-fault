import os
import gc
import polars as pl
from typing import cast
from skrub import TableVectorizer

os.makedirs("data/processed/", exist_ok=True) 

transformer = TableVectorizer()

train = pl.read_csv(f"data/raw/application_train.csv")
new_train = train.drop(["SK_ID_CURR", "TARGET"])
new_train = cast(pl.DataFrame, transformer.fit_transform(new_train))
new_train = new_train.with_columns(train["TARGET"], train["SK_ID_CURR"])
new_train.write_csv("data/processed/application_train.csv")

del train
gc.collect()

test = pl.read_csv(f"data/raw/application_test.csv")
new_test = test.drop("SK_ID_CURR")
new_test = cast(pl.DataFrame, transformer.transform(new_test))
new_test = new_test.with_columns(test["SK_ID_CURR"])
test.write_csv("data/processed/application_test.csv")