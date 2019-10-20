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


def compute_AMS(y_true, y_pred, weights):
    # true positive
    s = weights[np.where((y_pred == True) & (y_true == True))].sum()

    # false positive
    b = weights[np.where((y_pred == True) & (y_true == False))].sum()

    br = 10

    radicand = 2 * ((s + b + br) * np.log(1.0 + s / (b + br)) - s)
    return np.sqrt(radicand)

def grid_search(data, weights):
    n_estimators_values = [None, 500, 1000, 2000]
    X, y = data
    results_bagging = []
    results_boosting = []

    for n_estimators in n_estimators_values:
        if n_estimators:
            bagging = BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5,
                                        n_estimators=n_estimators)
            boosting = AdaBoostClassifier(n_estimators=n_estimators)
        else:
            bagging = BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5)
            boosting = AdaBoostClassifier()

        kf = KFold(n_splits=3)
        scores_bagging = []
        scores_boosting = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            bagging.fit(X_train, y_train)
            y_pred = bagging.predict(X_test)
            score = compute_AMS(y_test, y_pred, weights[test_index])
            scores_bagging.append(score)


            boosting.fit(X_train, y_train)
            y_pred = boosting.predict(X_test)
            score = compute_AMS(y_test, y_pred, weights[test_index])
            scores_boosting.append(score)

        results_bagging.append([n_estimators, np.mean(scores_bagging), np.std(scores_bagging)])
        results_boosting.append([n_estimators, np.mean(scores_boosting), np.std(scores_boosting)])

    return pd.DataFrame.from_records(results_bagging, columns=['n_estimators', 'average', 'std']),\
           pd.DataFrame.from_records(results_boosting, columns=['n_estimators', 'average', 'std'])

def grid_search_rf(data, weights):
    n_estimators_values = [None, 500, 1000, 2000]
    max_depth_values = [20, 50, None]
    X, y = data
    results = []

    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            if n_estimators:
                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            else:
                rf = RandomForestClassifier(max_depth=max_depth)

            kf = KFold(n_splits=3)
            scores_param = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                score = compute_AMS(y_test, y_pred, weights[test_index])
                scores_param.append(score)

            results.append([n_estimators, max_depth, np.mean(scores_param), np.std(scores_param)])

    return pd.DataFrame.from_records(results, columns=['n_estimators', 'max_depth', 'average', 'std'])

def eval_best(data, weights, clf):
    X, y = data

    kf = KFold(n_splits=3)
    scores_param = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = compute_AMS(y_test, y_pred, weights[test_index])
        scores_param.append(score)

    return np.mean(scores_param), np.std(scores_param)

def main():
    eval_mode = True
    n_train = 1000
    n_test = 1000
    RFnan = False
    pca = False
    
    start = time()
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
    if RFnan:
        transformers.append(StandardScaler())
        transformers.append(SimpleImputer(missing_values=np.nan, fill_value=-999999.0))
    else:
        transformers.append(IterativeImputer(max_iter=int(1e2)))
        transformers.append(StandardScaler())
        if pca:
            print("Using PCA")
            transformers.append(PCA(20))

    for trans in transformers:
        X_train = trans.fit_transform(X_train)
        X_test = trans.transform(X_test)

    if not eval_mode:
        if RFnan:
            results = grid_search_rf((X_train, y_train), weights_train)
            print("RF nan : \n\t{}".format(results))
            print("\tBest results : \n\t{}".format(results.ix[results['average'].idxmax()]))
        else:
            results_bagging, results_boosting = grid_search((X_train, y_train), weights_train)
            results_rf = grid_search_rf((X_train, y_train), weights_train)
            print("Bagging : \n\t{}".format(results_bagging.ix[results_bagging['average'].idxmax()]))
            print("Boosting : \n\t{}".format(results_boosting.ix[results_boosting['average'].idxmax()]))
            print("RF : \n\t{}".format(results_rf.ix[results_rf['average'].idxmax()]))
    else:
        if RFnan:
            rf_nan = RandomForestClassifier(n_estimators=2000, max_depth=None)
            average, std = eval_best((X_test, y_test), weights_test, rf_nan)
            print("RFNan : %.4f +/- %.4f" % (average, std))
        else:
            clfs = [
                RandomForestClassifier(n_estimators=2000, max_depth=50),
                BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5, n_estimators=500),
                AdaBoostClassifier(n_estimators=50),
            ]
            
            for clf in clfs:
                average, std = eval_best((X_test, y_test), weights_test, clf)
                print(clf)
                print("%.4f +/- %.4f" % (average, std))
    print("Total time : {}".format(time() - start))


if __name__ == '__main__':
    main()

