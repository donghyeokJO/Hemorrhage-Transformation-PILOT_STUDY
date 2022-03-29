import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier
from data_process import load_data
from xgboost import XGBClassifier
from scipy.stats import uniform, randint


class puerGBM:
    def __init__(self, data, labels):
        self.model = GradientBoostingClassifier()
        # self.init_param = {
        #     "learning_rate": uniform(),
        #     "subsample": uniform(),
        #     "n_estimators": randint(100, 1000),
        #     "max_depth": randint(4, 10),
        # }

        self.current_best = {
            "learning_rate": np.arange(0.74419, 1.0, 0.0001),
            "subsample": np.arange(0.05475, 0.1, 0.0001),
            "n_estimators": randint(600, 1000),
            "max_depth": randint(5, 10),
        }

        self.cv = LeaveOneOut()

        self.data = data
        self.labels = labels

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
        )

        self.best_estimator = None
        self.best_params = None
        self.best_score = 0

    def find_param(self):
        counter = 0
        while True:
            print(f"Results from Random Search - {counter} ")
            rscv = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.current_best,
                cv=self.cv,
                n_iter=10,
                n_jobs=1,
            )
            rscv.fit(self.x_train, self.y_train.values.ravel())

            print(
                "The best estimator across ALL searched params:", rscv.best_estimator_
            )
            print("The best score across ALL searched params: ", rscv.best_score_)
            print("The best parameters across ALL searched params: ", rscv.best_params_)

            if rscv.best_score_ > self.best_score:
                self.best_estimator = rscv.best_estimator_
                self.best_params = rscv.best_params_
                self.current_best = {
                    "learning_rate": np.arange(
                        rscv.best_params_.get("learning_rate"), 1.0, 0.0001
                    ),
                    "subsample": np.arange(
                        rscv.best_params_.get("subsample"), 0.1, 0.0001
                    ),
                    "n_estimators": randint(600, 1000),
                    "max_depth": randint(5, 10),
                }

            counter += 1

            if rscv.best_score_ >= 0.9:
                break

        with open("model_pureGBM.txt", "wb") as f:
            pickle.dump(
                {
                    "best_params": self.best_params,
                    "best_estimator": self.best_estimator,
                },
                f,
            )

    def load_model(self):
        with open("model_puerGBM.txt", "rb") as f:
            data = pickle.load(f)

        best_params = data.get("best_params")
        best_estimator = data.get("best_estimator")

        print(best_estimator.score(self.x_train, self.y_train))
        print(best_params)

    def test(self):
        tm = GradientBoostingClassifier(
            learning_rate=0.999899999999718,
            max_depth=8,
            n_estimators=729,
            subsample=0.099500000000013,
        )

        tm.fit(self.x_train, self.y_train)

        print(tm.score(self.x_train, self.y_train))
        # print(tm.score(self.x_test, self.y_test))
        print(cross_val_score(tm, self.data, self.labels))


if __name__ == "__main__":
    data, labels = load_data()

    gbm = puerGBM(data, labels)
    # gbm.find_param()
    gbm.test()
