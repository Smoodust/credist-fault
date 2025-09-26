import polars as pl
import pickle
from tqdm import tqdm
import json

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from skrub import tabular_pipeline

N_FOLDS = 5

train = pl.read_csv(f"data/raw/application_train.csv")
train = train.drop("SK_ID_CURR")

X = train.drop("TARGET")
y = train["TARGET"]

scores = []
pipelines = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=True)
for train_index, val_index in tqdm(skf.split(X, y), total=N_FOLDS):
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]

    pipeline: Pipeline = tabular_pipeline('classifier')
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    scores.append(roc_auc_score(y_val, val_pred))
    pipelines.append(pipeline)

print(scores)
metric = {f"auc_roc_fold_{i}":scores[i] for i in range(N_FOLDS)}
metric["roc_auc_avg"] = sum(scores)/N_FOLDS

with open('models/pipelines_baseline.pickle', 'wb') as f:
    pickle.dump(pipelines, f)

with open('metric.json', "w") as f:
    json.dump(metric, f)