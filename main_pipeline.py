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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron#, BayesianRidge
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from AMS import AMS

import tempfile
from joblib import Memory


class Shift_log(BaseEstimator, TransformerMixin):
    cpt_ini = 0
    cpt_fit = 0
    def __init__(self):
        self.shift = None
        
        Shift_log.cpt_ini+=1

    # noinspection PyUnusedLocal
    def fit(self, x, y=None):
        Shift_log.cpt_fit+=1
        
        self.shift = 1 - np.nanmin(x, axis=0)
        
        return self
    
    def transform(self, x):
        return np.log(x + self.shift)

def main():
    train = pd.read_csv('data.csv')[:100]
    
    y = train['Label']
    y = np.where(y == 's', 1, 0)
    
    x = train.drop(columns=['Label', "KaggleSet", "Weight", "KaggleWeight", "EventId"])

    x = x.replace(-999, np.nan)
    
    
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
            "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]

    mem = Memory(location=tempfile.mkdtemp(), verbose=0)

    pipe_imputed = Pipeline([
        ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
        ('imp', IterativeImputer()),
        ('sca', StandardScaler()),
        ('pca', None),
        ('clf', None),
    ], memory=mem, verbose=0)
    
    param_imputed = [{
        # 'imp__estimator': (BayesianRidge(),),
        'imp__max_iter':  (1,),
        'pca':            (None, PCA(15)),
        'clf':            (SVC(),),
        'clf__kernel':    ("poly", "rbf"),
        'clf__max_iter':  (10,),
        'clf__C':         (.01, .1),
    },
    {
        'imp__max_iter': (1,),
        'pca': (None, PCA(15)),
        'clf': (BaggingClassifier(Perceptron(max_iter=1000), n_estimators=10, max_samples=0.5, max_features=0.5),
                RandomForestClassifier(n_estimators=10),
                AdaBoostClassifier(n_estimators=100)),
    },
    ]

    pipe_nan = Pipeline([
        ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
        ('sca', StandardScaler()),
        ('imp', SimpleImputer(missing_values=np.nan, fill_value=-999999.0)),
        ('rfc', RandomForestClassifier()),
    ])
    param_nan = [{
        'rfc__n_estimators': (100,),
        'rfc__max_depth':    (10, None),
    }]
    
    
    # for scoring in ["accuracy", AMS]:
    for scoring in [AMS]:
        print("scoring:", scoring)
        # for pipe, params in [(pipe_imputed, param_imputed), (pipe_nan, param_nan)]:
        for pipe, params in [(pipe_imputed, param_imputed)]:
            grid_search = GridSearchCV(pipe, params, scoring= scoring, cv=3
                                       , iid=True, n_jobs=1, return_train_score=True, verbose=0)
            t0 = time()
            grid_search.fit(x, y)
            print("done in %0.3fs" % (time() - t0))

            std = np.std(grid_search.cv_results_["mean_test_score"]) / grid_search.best_score_ * 100
            print("Best score         : %0.3f (+/-%.1f%%)" % (grid_search.best_score_, std))
            print("Best score (train) : %0.3f " % (grid_search.cv_results_["mean_train_score"][grid_search.best_index_]))
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(set().union(*[list(p.keys()) for p in params])):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            print()
    

if __name__ == '__main__':
    import warnings
    import sys
    import os
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
    main()