import warnings
import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    LeaveOneOut,
    train_test_split,
    GridSearchCV,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_process import load_data
from xgboost import XGBClassifier
from constants import SELECTED_FEATURES_XG


class XGBM:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        self.data = self.data.loc[:, self.data.columns.isin(SELECTED_FEATURES_XG)]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.labels,
            test_size=0.2,
            shuffle=True,
            stratify=labels,
            random_state=777,
        )

        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
        )

        self.param_grid = {
            "booster": ["gbtree"],
            "max_depth": [8, 9, 10],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 1, 2, 3],
            "nthread": [4],
            "colsample_bytree": [0.8, 0.9],
            "colsample_bylevel": [0.9],
            "n_estimators": [400, 500, 600],
            "objective": ["binary:logistic"],
            "random_state": [777],
        }

        self.evals = [(self.x_test, self.y_test)]

        self.cv = LeaveOneOut()

    def train(self):
        grid = GridSearchCV(
            self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring="accuracy",
            n_jobs=4,
        )

        grid.fit(
            self.x_train.values,
            self.y_train.values.ravel(),
            eval_metric="logloss",
            eval_set=self.evals,
        )

        print("final params", grid.best_params_)
        print("best score", grid.best_score_)

        with open("model_XGBM.txt", "wb") as f:
            pickle.dump({"model": grid}, f)

    def validate(self):
        with open("model_XGBM.txt", "rb") as f:
            pretrained = pickle.load(f)

        grid = pretrained.get("model")

        model = grid.best_estimator_

        print(f"Train Score : {model.score(self.x_train, self.y_train)}")
        print(f"Test  Score : {model.score(self.x_test, self.y_test)}")
        print(
            f"CV Score    : {cross_val_score(model, self.data, self.labels, cv=self.cv).mean()}"
        )

        pred = model.predict(self.data)
        acc = accuracy_score(self.labels, pred)
        print(f"Whole ACC   : {acc}")

        con_mat = confusion_matrix(self.labels, pred)
        report = classification_report(self.labels, pred)

        print(con_mat)
        print(report)


if __name__ == "__main__":
    data, labels = load_data()
    xgb = XGBM(data, labels)
    # print(xgb.x_train.index)
    xgb.train()
    # xgb.validate()
    # print(xgb.y_train)
