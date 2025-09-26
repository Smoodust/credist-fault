import polars as pl
import pickle
from tqdm import tqdm
import json
import os
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_FOLDS = 5

os.makedirs("models/", exist_ok=True) 

train = pl.read_csv(f"data/features/train_features.csv")

X = train.drop("TARGET")
y = train["TARGET"]

scores = []
pipelines = []

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for train_index, val_index in tqdm(skf.split(X, y), total=N_FOLDS):
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]

    pipeline = HistGradientBoostingClassifier(random_state=42)
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