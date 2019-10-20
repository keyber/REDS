import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer  # noqa
# noinspection PyUnresolvedReferences
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import Perceptron  #, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
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


def main():
    eval_mode = False
    pca = False
    n_train = 10000
    n_test = 1000
    t0 = time()

    # LOAD DATA
    data = shuffle(pd.read_csv('data.csv'), random_state=seed)[:n_train + n_test]
    y = data['Label']
    y = np.where(y == 's', 1, 0)
    x = data.drop(columns=['Label', "KaggleSet", "KaggleWeight", "EventId"])
    weights = data['Weight'].values
    x = x.drop(columns=['Weight'])
    x = x.replace(-999, np.nan)

    # SPLIT
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(x, y, weights, random_state=seed,
                                                                                     test_size=n_test)

    # PREPROCESS
    transformers = []
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
                "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]
    transformers.append(make_column_transformer((Shift_log(), cols_log), remainder="passthrough"))
    transformers.append(IterativeImputer(max_iter=int(1e2)))
    transformers.append(StandardScaler())

    for trans in transformers:
        X_train = trans.fit_transform(X_train)
        X_test = trans.transform(X_test)

    start = time()

    # Fitting the PCA algorithm
    pca = PCA().fit(X_train)
    # Plotting the Cumulative Summation of the Explained Variance
    print(np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.98))
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')  # for each component
    plt.title('Explained Variance')
    plt.show()

    print("Total time : {}".format(time() - start))

if __name__ == '__main__':
    main()
