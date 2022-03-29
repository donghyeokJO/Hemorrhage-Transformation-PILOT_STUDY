import pickle
import warnings

import pandas as pd

from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_process import load_data
from constants import SELECTED_FEATURES


class GBM:
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

        self.model = GradientBoostingClassifier(
            random_state=777, max_depth=10, learning_rate=0.1
        )

        self.cv = LeaveOneOut()

    def train(self):
        self.model.fit(self.x_train, self.y_train)

        print(f"Train Score : {self.model.score(self.x_train, self.y_train)}")
        print(f"Test  Score : {self.model.score(self.x_test, self.y_test)}")
        print(
            f"CV Score    : {cross_val_score(self.model, self.data, self.labels, cv = self.cv).mean()}"
        )

        pred = self.model.predict(self.data)
        acc = accuracy_score(self.labels, pred)
        print(f"Whole ACC   : {acc}")

        con_mat = confusion_matrix(self.labels, pred)
        report = classification_report(self.labels, pred)

        print(con_mat)
        print(report)

        with open("model_pureGBM.txt", "wb") as f:
            pickle.dump({"model": self.model}, f)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, labels = load_data()
    gbm = GBM(data, labels)
    gbm.train()
