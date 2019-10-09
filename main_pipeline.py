import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# noinspection PyUnresolvedReferences
from sklearn.experimental import enable_iterative_imputer  # noqa
# noinspection PyUnresolvedReferences
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron#, BayesianRidge
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from AMS import AMS
import matplotlib.pyplot as plt

import tempfile
from joblib import Memory


class Shift_log(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.shift = None

    # noinspection PyUnusedLocal
    def fit(self, x, y=None):
        self.shift = 1 - np.nanmin(x, axis=0)
        
        return self
    
    def transform(self, x):
        return np.log(x + self.shift)

def eval_pipe(pipe, pipe_param, X_train, X_test, y_train, y_test):
    for p in pipe_param:
        gri = pipe.named_steps['gri']
        gri.param_grid = p
        
        t0 = time()
        pipe.fit(X_train, y_train)
        # print(pipe_imputed_fast.named_steps['gri'])
        std = np.std(gri.cv_results_["mean_test_score"]) / gri.best_score_ * 100
        print("gridsearch best score %.2f (+/-%.2f%%)" % (gri.best_score_, std))
        print("gridsearch time %.1f" % (time() - t0))

        test_score = pipe.score(X_test, y_test)
        print("test_score %.2f" % test_score)
        
        print("Best parameters set:")
        best_parameters = gri.best_estimator_.get_params()
        
        if type(p) is list :
            params_defined_by_hand = set().union(*[list(p.keys()) for p in gri.param_grid])
        else:
            params_defined_by_hand = set(p.keys())
        
        for param_name in best_parameters:
            if param_name in params_defined_by_hand:
                val = best_parameters[param_name]
                if param_name=="clf":
                    val = type(val)
                print("\t%s: %r" % (param_name, val))
        
        # r = gri.cv_results_
        # plt.errorbar(range(len(r["mean_train_score"])), r["mean_train_score"], yerr=2*r["std_train_score"])
        # plt.errorbar(range(len(r["mean_test_score"])), r["mean_test_score"], yerr=2*r["std_test_score"])
        # plt.show()
        
        print("\n\n")


def make_pipe_fast():
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
            "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]

    mem = Memory(location=tempfile.mkdtemp(), verbose=0)
    
    pipe_imputed_fast = Pipeline([
        ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
        ('imp', IterativeImputer(max_iter=int(1e2))),
        ('sca', StandardScaler()),
        # ('pca', PCA(15)),
        ('gri', GridSearchCV(Pipeline([#('pca', None),
                                       ('clf', None)]),
            scoring=AMS, refit=True, cv=3, iid=True, return_train_score=False, param_grid={})),
    ], memory=mem, verbose=0)

    param_grid = [
        {
            # 'pca': (None, PCA(15)),
            'clf': (SVC(gamma="auto", max_iter=100000),),
            'clf__kernel': ("poly", "rbf"),
            'clf__C': np.logspace(-2, .5, num=3),
        },
        {
            'clf': (BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5),),
            'clf__n_estimators': (500,),
        },
        {
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': (500,),
            'clf__max_depth': (None, 10),
        },
        {
            'clf': (AdaBoostClassifier(),),
            'clf__n_estimators': (500,),
        },
    ]

    # pipe_nan = Pipeline([
    #     ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
    #     ('sca', StandardScaler()),
    #     ('imp', SimpleImputer(missing_values=np.nan, fill_value=-999999.0)),
    #     ('rfc', RandomForestClassifier()),
    # ])
    # param_nan = [{
    #     'rfc__n_estimators': (100,),
    #     'rfc__max_depth':    (10, None),
    # }]

    return pipe_imputed_fast, param_grid


def main():
    t0 = time()
    data = pd.read_csv('data.csv')[:1000]

    # seriesObj = data.apply(lambda x_: -999.0 in list(x_), axis=1)
    # data = data[seriesObj == False][:1000]
    # assert len(data)==1000
    
    y = data['Label']
    y = np.where(y == 's', 1, 0)
    
    x = data.drop(columns=['Label', "KaggleSet", "Weight", "KaggleWeight", "EventId"])
    
    x = x.replace(-999, np.nan)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3)
    
    pipe, pipe_param = make_pipe_fast()
    eval_pipe(pipe, pipe_param, X_train, X_test, y_train, y_test)
    

    print("temps total", time() - t0)
    

if __name__ == '__main__':
    # import warnings
    # import sys
    # import os
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    #     os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
    main()
