import warnings

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from data_process import load_data
from constants import SELECTED_FEATURES


class GBM:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        self.data = self.data.loc[:, self.data.columns.isin(SELECTED_FEATURES)]

        self.model = GradientBoostingClassifier(
            random_state=47717110, learning_rate=0.01
        )

        self.loo = LeaveOneOut()

        self.train_score = list()
        self.test_result = list()

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

            print(
                f"""
                Trial : {test_index}
                Train Score: {train_score}
                Test Result: {"Success" if test_result >= 1.0 else "fail"}
                """
            )

            self.train_score.append(train_score)
            self.test_result.append(test_result)

        print(
            f"""
            Total Result 
            
            Train Score : {np.mean(self.train_score)}
            Test Score  : {np.mean(self.test_result)}
            """
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, labels = load_data()
    gbm = GBM(data, labels)
    gbm.train_loo()
