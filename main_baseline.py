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
    X, y = data
    C_values = np.logspace(-1, 1, num=100)
    results = []

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

            results.append([C, kernel, np.mean(scores_param), np.std(scores_param)])

    return pd.DataFrame.from_records(results, columns=['C', 'kernel', 'average', 'std'])

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

def plot_score(C_values, average, std):
    #ax = plt.gca()
    plt.fill_between(C_values, average - std, average + std, alpha=0.1)
    plt.plot(C_values, average)
    plt.xscale('log')
    #ax.set_xscale('log')
    plt.xlabel('C')
    plt.ylabel('AMS')
    plt.axis('tight')
    plt.show()

def main():
    eval_mode = True
    pca = True
    n_train = 1000
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
    if pca:
        print("Using PCA")
        transformers.append(PCA(20))

    for trans in transformers:
        X_train = trans.fit_transform(X_train)
        X_test = trans.transform(X_test)


    start = time()
    if not eval_mode:
        # TRAIN
        results = grid_search((X_train, y_train), weights_train)
        id_best_result = results['average'].values.argmax()
        C_values = np.unique(results['C'].values)
        grouped_results = results.groupby(results['C']).mean() # average over C values

        print("Best parameters : C = {}, kernel = {}".format(results.iloc[id_best_result]['C'],
                                                             results.iloc[id_best_result]['kernel']))
        print("Best train score : %.2f +- %.2f " % (results['average'].max() * 100,
                                                    results.iloc[results['average'].values.argmax()]['std'] * 100))

        # PLOT SCORES AGAINST C VALUES
        avg_ams = grouped_results['average'].values
        std_ams = grouped_results['std'].values
        plot_score(C_values, avg_ams, std_ams)
    else:
        # COMPUTE TEST SCORE
        clf = SVC(gamma="auto", max_iter=100000, C=1.71, kernel='rbf')
        print(eval_best((X_test, y_test), weights_test, clf))
    print("Total time : {}".format(time() - start))


if __name__ == '__main__':
    main()
