import warnings

import pandas as pd

from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from data_process import load_data


class FeatureSelection:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        self.model = XGBClassifier(random_state=777)

        self.rfe = RFE(self.model, n_features_to_select=20)

        self.rfe.fit(self.data, self.labels)

        # print(self.rfe.support_)

        selected_columns = [
            col_nm
            for idx, col_nm in enumerate(self.data.columns)
            if self.rfe.support_[idx]
        ]

        print(selected_columns)
        # print(self.rfe.get_feature_names_out())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, labels = load_data()
    fs = FeatureSelection(data, labels)
