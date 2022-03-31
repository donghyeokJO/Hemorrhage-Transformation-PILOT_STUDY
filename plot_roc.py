import pickle

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from data_process import load_data
from constants import SELECTED_FEATURES, SELECTED_FEATURES_XG


def plot_roc_curve(fper, tper, score):
    plt.plot(fper, tper, color="red", label="ROC " + str(score))
    plt.plot([0, 1], [0, 1], color="green", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve")
    plt.legend(loc="lower right")
    plt.show()


class ROC:
    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.labels = labels

    def xgbm_plot(self):
        data = self.data.loc[:, self.data.columns.isin(SELECTED_FEATURES_XG)]
        with open("model_XGBM.txt", "rb") as f:
            pretrained = pickle.load(f)

        model = pretrained.get("model")

        prob = model.predict_proba(data)
        prob = prob[:, 1]
        fper, tper, thresholds = roc_curve(self.labels, prob)

        score = roc_auc_score(self.labels, prob)
        plot_roc_curve(fper, tper, score)

        print("XGBM : ", score)


if __name__ == "__main__":
    data, labels = load_data()
    plotter = ROC(data, labels)
    plotter.xgbm_plot()
