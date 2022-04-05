import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, roc_auc_score
from xgboost import XGBClassifier
from data_process import load_data
from plot_roc import plot_roc_curve


class XGBM:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        xgboost_columns = ["PI4", "HU_1", "rtpa", "wbc", "hct", "ldl", "ha1c"]
        self.data = self.data.loc[:, self.data.columns.isin(xgboost_columns)]

        self.model = XGBClassifier(
            n_estimators=500,
            learning_rate=1.7,
            gamma=0.1,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.01,
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

            self.model.fit(x_train, y_train, verbose=False, eval_metric="error")

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

        plot_roc_curve(fper, tper, score, best_threshold, idx, "XGBoost")


if __name__ == "__main__":
    data, labels = load_data()
    xgb = XGBM(data, labels)
    xgb.train_loo()
    xgb.plotting()
    # xgb.grid_search()
