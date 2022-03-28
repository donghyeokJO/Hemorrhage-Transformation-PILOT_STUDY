import pickle

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier
from data_process import load_data
from xgboost import XGBClassifier
from scipy.stats import uniform, randint


class puerGBM:
    def __init__(self, data, labels):
        self.model = GradientBoostingClassifier()
        self.init_param = {
            "learning_rate": uniform(),
            "subsample": uniform(),
            "n_estimators": randint(100, 1000),
            "max_depth": randint(4, 10),
        }
        self.cv = LeaveOneOut()

        self.data = data
        self.labels = labels

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
        )

        self.best_estimator = None
        self.best_params = None
        self.rank_test_score = None
        self.best_score = 0

    def find_param(self):
        counter = 0
        while True:
            print(f"Results from Random Search - {counter} ")
            rscv = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.init_param,
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

            counter += 1

            if rscv.best_score_ >= 0.9:
                break
        # self.rank_test_score = pd.DataFrame(rscv.cv_results_).sort_values(
        #     "rank_test_score"
        # )

        # print(self.rank_test_score)

        with open("model_pureGBM.txt", "wb") as f:
            pickle.dump(
                {
                    "best_params": self.best_params,
                    "best_estimator": self.best_estimator,
                },
                f,
            )

    def find_param_grid(self):
        current_best = {
            "learning_rate": 0.744,
            "n_estimator": 795,
            "subsample": 0.054,
            "max_depth": 5,
        }

        param_grid = {"max_depth": np.array}

        grdcv = GridSearchCV(
            estimator=self.model,
        )


if __name__ == "__main__":
    data, labels = load_data()

    gbm = puerGBM(data, labels)
    gbm.find_param()
