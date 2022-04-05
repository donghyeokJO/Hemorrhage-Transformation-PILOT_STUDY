import warnings

import pandas as pd

from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
from data_process import load_data


class FeatureSelection:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

        self.model = LGBMClassifier(min_child_samples=5)

        self.rfe = RFE(self.model, n_features_to_select=6)

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

    """
    ['wbc', 'hb', 'hct', 'plt', 'tc', 'tg', 'hdl', 'ldl', 'HU33', 'HU59', 'HU66', 'HU67', 'HU68'
    , 'HU69', 'HU70', 'HU71', 'HU72', 'HU73', 'Mu', 'Skewness']
    
    same with pure gradient boosting
    """
