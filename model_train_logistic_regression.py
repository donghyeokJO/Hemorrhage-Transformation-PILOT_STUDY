import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from data_process import load_data
from constants import SELECTED_FEATURES_LOGISTIC_REGRESSION
from plot_roc import plot_roc_curve


class LOGIT:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        a = ["PI4", "HU_3", "rtpa", "wbc", "ldl", "crp"]
        # self.data = self.data.loc[
        #     :, self.data.columns.isin(SELECTED_FEATURES_LOGISTIC_REGRESSION)
        # ]
        self.data = self.data.loc[:, self.data.columns.isin(a)]

        # 19980811
        self.model = LogisticRegression(
            random_state=4771,
            max_iter=600,
            # n_jobs=4,
            C=10,
            fit_intercept=False
            # penalty="none",
            # solver="sag"
            # penalty="l1",
            # solver="liblinear",
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

    def plotting(self):
        fper, tper, thresholds = roc_curve(
            self.labels.values.ravel(), self.pred_proba_HTf
        )

        score = roc_auc_score(self.labels.values.ravel(), self.pred_proba_HTf)

        J = tper - fper
        idx = np.argmax(J)
        best_threshold = thresholds[idx]

        plot_roc_curve(fper, tper, score, best_threshold, idx, "LOGIT")


if __name__ == "__main__":
    data, labels = load_data()
    logit = LOGIT(data, labels)
    logit.train_loo()
    logit.plotting()
