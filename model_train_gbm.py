import warnings
import random

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from data_process import load_data
from constants import SELECTED_FEATURES
from plot_roc import plot_roc_curve


class GBM:
    def __init__(
        self, data: pd.DataFrame, labels: pd.DataFrame, random_state: int = 4771
    ):
        self.data = data
        self.labels = labels
        self.random_state = random_state

        a = ["PI4", "HU_1", "rtpa", "wbc", "hct", "ldl"]
        # self.data = self.data.loc[:, self.data.columns.isin(SELECTED_FEATURES)]
        self.data = self.data.loc[:, self.data.columns.isin(a)]

        self.model = GradientBoostingClassifier(
            random_state=self.random_state,
            # random_state=4771,
            learning_rate=0.04585,
            n_estimators=800,
            min_samples_split=8,
            # min_impurity_decrease=0.1,
            # loss="exponential",
            # subsample=0.68,
        )

        self.loo = LeaveOneOut()

        self.train_score = list()
        self.test_result = list()
        self.pred_proba_HTf = list()

    def train_loo(self):
        for train_index, test_index in self.loo.split(self.data):
            x_train, x_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = (
                self.labels.iloc[train_index],
                self.labels.iloc[test_index],
            )

            self.model.fit(x_train, y_train)

            train_score = self.model.score(x_train, y_train)
            test_result = self.model.score(x_test, y_test)
            pred_proba_HTf = self.model.predict_proba(x_test)[:, 1]

            print(
                f"""
                Trial : {test_index}
                Train Score: {train_score}
                Test Result: {"Success" if test_result >= 1.0 else "fail"}
                """
            )

            self.train_score.append(train_score)
            self.test_result.append(test_result)
            self.pred_proba_HTf.append(pred_proba_HTf)

        print(
            f"""
            Total Result
            Train Score : {np.mean(self.train_score)}
            Test Score  : {np.mean(self.test_result)}
            """
        )

    def plotting(self) -> float:
        fper, tper, thresholds = roc_curve(
            self.labels.values.ravel(), self.pred_proba_HTf
        )

        score = roc_auc_score(self.labels.values.ravel(), self.pred_proba_HTf)

        J = tper - fper
        idx = np.argmax(J)
        best_threshold = thresholds[idx]

        plot_roc_curve(fper, tper, score, best_threshold, idx, "GBM")

        return score

    def grid_search(self):
        model = GradientBoostingClassifier()
        param_grid = {
            "random_state": [x for x in range(1, 101)],
            "learning_rate": [0.1, 1, 10],
            "n_estimators": [400, 500, 600],
        }
        grid = GridSearchCV(
            model,
            param_grid,
            cv=self.loo,
            n_jobs=4,
        )

        grid.fit(
            self.data,
            self.labels.values.ravel(),
        )

        print("Best CV Score", grid.best_score_)
        print("Best Params", grid.best_params_)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, labels = load_data()
    gbm = GBM(data, labels)
    gbm.train_loo()
    score = gbm.plotting()
    # gbm.grid_search()
