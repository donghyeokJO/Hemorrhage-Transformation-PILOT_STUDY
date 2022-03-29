import pickle
import warnings

import pandas as pd

from sklearn.model_selection import (
    LeaveOneOut,
    train_test_split,
)
from sklearn.metrics import accuracy_score
from data_process import load_data
from xgboost import XGBClassifier
from constants import SELECTED_FEATURES


class XGBM:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        self.data = self.data.loc[:, self.data.columns.isin(SELECTED_FEATURES)]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.labels,
            test_size=0.2,
            shuffle=True,
            stratify=labels,
            random_state=777,
        )

        self.model = XGBClassifier(
            n_estimators=400,
            learning_rate=0.2,
            # max_depth=100,
            # objective="binary:logistic",
        )

        self.evals = [(self.x_test, self.y_test)]

    def train(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            early_stopping_rounds=100,
            eval_metric="logloss",
            eval_set=self.evals,
            verbose=True,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, labels = load_data()
    xgb = XGBM(data, labels)
    xgb.train()
