import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer  # noqa
# noinspection PyUnresolvedReferences
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.linear_model import Perceptron  #, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from AMS import AMS
from plot_learning_curve import plot_learning_curve
import matplotlib.pyplot as plt
import tempfile
from joblib import Memory

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

def grid_search(param, X_train, y_train, n_splits):
    pipe = Pipeline([('clf', None)])
    
    for p in param:
        t0 = time()
        gri = GridSearchCV(pipe, param_grid=p, scoring=AMS, cv=n_splits, iid=True, refit=False,
                          return_train_score=False)
        
        gri.fit(X_train, y_train)
        
        std_best = gri.cv_results_["std_test_score"][gri.best_index_] / np.sqrt(n_splits)
        std_param = np.std(gri.cv_results_["mean_test_score"]) / gri.best_score_ * 100
        print("gridsearch best score %.2f +/-%.1f  (var selon param %.2f%%)" % (gri.best_score_, std_best, std_param))
        print("gridsearch time %.1f" % (time() - t0))
        
        print("Best parameters set:")
        best_parameters = gri.best_params_#.get_params()
        if type(p) is list:
            params_defined_by_hand = set().union(*[list(p.keys()) for p in gri.param_grid])
        else:
            params_defined_by_hand = set(p.keys())
        
        for param_name in best_parameters:
            if param_name in params_defined_by_hand:
                val = best_parameters[param_name]
                if param_name == "clf":
                    val = type(val)
                print("\t%s: %r" % (param_name, val))
        
        print("\n\n")


def main_get_best_hyperparam(x, y, n_splits):
    param_grid = [
        {
            'clf': (SVC(gamma="auto", max_iter=100000),),
            'clf__kernel': ("poly", "rbf"),
            'clf__C': np.logspace(-2, .5, num=3),
        },
        {
            'clf': (BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5),),
            'clf__n_estimators': (1000,),
        },
        {
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': (1000,),
            'clf__max_depth': (None, 50),
        },
        {
            'clf': (AdaBoostClassifier(),),
            'clf__n_estimators': (1000,),
        },
    ]
    
    grid_search(param_grid, x, y, n_splits)


def main_get_best_hyperparam_nan(x, y, n_splits):
    param_grid = [
        {
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': (1000,),
            'clf__max_depth': (None, 50),
        },
    ]
    
    grid_search(param_grid, x, y, n_splits)


def main_eval_model(x, y, n_splits):
    clfs = [
        ("svc", SVC(C=3.16, kernel="rbf", max_iter=100000, gamma="auto",)),
        ("bag", BaggingClassifier(Perceptron(max_iter=1000), n_estimators=500, max_samples=0.5, max_features=0.5)),
        ("rfc", RandomForestClassifier(n_estimators=500, max_depth=50)),
        ("ada", AdaBoostClassifier(n_estimators=1000)),
    ]
    
    for name, clf in clfs:
        t0 = time()
        res = cross_val_score(clf, x, y, cv=n_splits, scoring='accuracy')
        print(name, "%.2f, +/- %.2f (%.0fs)" % (np.mean(res) * 100, np.std(res) / np.sqrt(n_splits) * 100, time() - t0))


def main_eval_model_nan(x, y, n_splits):
    clf = RandomForestClassifier(n_estimators=500, max_depth=None)
    t0 = time()
    res = cross_val_score(clf, x, y, cv=n_splits, scoring='accuracy')
    print("rfnan : %.2f, +/- %.2f (%.0fs)" % (np.mean(res) * 100, np.std(res) / np.sqrt(n_splits) * 100, time() - t0))

def main_learning_curve(x, y):
    title = "Learning Curves (100)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None)
    plot_learning_curve(clf, title, x, y, cv=cv, train_sizes=np.logspace(-3, 0, 4), log_x=True, n_jobs=-1)

    title = "Learning Curves (1000)"
    clf = RandomForestClassifier(n_estimators=100, max_depth=None)
    plot_learning_curve(clf, title, x, y, cv=cv, train_sizes=np.logspace(-3, 0, 4), log_x=True, n_jobs=-1)

    plt.show()

def main():
    n_train = 1000
    n_test = 1000
    RFnan = True
    
    # READ
    t0 = time()
    data = shuffle(pd.read_csv('data.csv'), random_state=seed)[:n_train + n_test]
    y = data['Label']
    y = np.where(y == 's', 1, 0)
    x = data.drop(columns=['Label', "KaggleSet", "Weight", "KaggleWeight", "EventId"])
    x = x.replace(-999, np.nan)
    
    
    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=n_test)
    
    
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
        # transformers.append(PCA(20))
    
    for trans in transformers:
        X_train = trans.fit_transform(X_train)
        X_test = trans.transform(X_test)
    
    
    # GRID SEARCH
    if RFnan:
        main_get_best_hyperparam_nan(X_train, y_train, n_splits=3)
    else:
        main_get_best_hyperparam(X_train, y_train, n_splits=3)
    
    # CROSS VAL
    if RFnan:
        main_eval_model_nan(X_train, y_train, n_splits=3)
    else:
        main_eval_model(X_train, y_train, n_splits=3)
    
    
    # LEARNING CURVE
    main_learning_curve(X_test, y_test)
    
    
    print("temps total", time() - t0)


if __name__ == '__main__':
    # import warnings
    # import sys
    # import os
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    #     os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
    main()
