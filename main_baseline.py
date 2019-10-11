import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer  # noqa
# noinspection PyUnresolvedReferences
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import make_scorer
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import Perceptron  #, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from AMS import AMS
import matplotlib.pyplot as plt

seed = 0

class Shift_log(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.shift = None

    # noinspection PyUnusedLocal
    def fit(self, x, y=None):
        self.shift = 1 - np.nanmin(x, axis=0)

        return self

    def transform(self, x):
        return np.log(x + self.shift)


def compute_AMS(y_true, y_pred, weights):
    # true positive
    s = weights[np.where((y_pred == True) & (y_true == True))].sum()

    # false positive
    b = weights[np.where((y_pred == True) & (y_true == False))].sum()

    br = 10

    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    return np.sqrt(radicand)

def grid_search(data, weights):
    X, y = data
    C_values = np.logspace(-2, .5, num=5)

    for C in C_values:
        for kernel in ['poly', 'rbf']:
            clf = SVC(gamma="auto", max_iter=100000, C=C, kernel=kernel)
            kf = KFold(n_splits=3)
            scores_param = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = compute_AMS(y_test, y_pred, weights[test_index])
                scores_param.append(score)

            print("C={}, kernel={}, mean AMS={}".format(C, kernel, np.mean(scores_param)))


def plot_score(scores, interval, scoring_function):
    ax = plt.gca()
    ax.plot(interval, scores)
    ax.set_xscale('log')
    plt.xlabel('C')
    if scoring_function == 'accuracy':
        plt.ylabel('Accuracy')
    elif scoring_function == 'f1':
        plt.ylabel('Score F1')
    else:
        plt.ylabel('Score AMS')
    plt.axis('tight')
    plt.show()

def main():
    n_train = 100
    n_test = 100
    RFnan = True
    t0 = time()
    data = shuffle(pd.read_csv('data.csv'), random_state=seed)[:n_train + n_test]
    y = data['Label']
    y = np.where(y == 's', 1, 0)
    x = data.drop(columns=['Label', "KaggleSet", "KaggleWeight", "EventId"])
    weights = data['Weight'].values
    x = x.drop(columns=['Weight'])
    x = x.replace(-999, np.nan)

    # preprocess
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
                "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]
    x = make_column_transformer((Shift_log(), cols_log), remainder="passthrough").fit_transform(x)
    if RFnan:
        x = StandardScaler().fit_transform(x)
        x = SimpleImputer(missing_values=np.nan, fill_value=-999999.0).fit_transform(x)
    else:
        x = IterativeImputer(max_iter=int(1e2)).fit_transform(x)
        x = StandardScaler().fit_transform(x)
        # x = PCA(15).fit_transform(x)
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(x, y, weights, random_state=seed,
                                                                                     test_size=n_test)

    # eval
    grid_search((X_train, y_train), weights_train)


if __name__ == '__main__':
    main()
