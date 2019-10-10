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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Perceptron#, BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from AMS import AMS
import matplotlib.pyplot as plt
import tempfile
from joblib import Memory

seed = 3

class Shift_log(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.shift = None

    # noinspection PyUnusedLocal
    def fit(self, x, y=None):
        self.shift = 1 - np.nanmin(x, axis=0)
        
        return self
    
    def transform(self, x):
        return np.log(x + self.shift)

def eval_pipe(pipe, pipe_param, X_train, X_test, y_train, y_test, n_splits):
    for p in pipe_param:
        t0 = time()
        gri = pipe.named_steps['gri']
        gri.param_grid = p
        
        pipe.fit(X_train, y_train)
        
        std_best = gri.cv_results_["std_test_score"][gri.best_index_] / np.sqrt(n_splits)
        std_param = np.std(gri.cv_results_["mean_test_score"]) / gri.best_score_ * 100
        print("gridsearch best score %.2f +/-%.2f  (var selon param %.2f%%)" % (gri.best_score_ * 100, std_best * 100,
                                                                                std_param))
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
        
        # plt.errorbar(range(len(r["mean_train_score"])), r["mean_train_score"], yerr=2*r["std_train_score"])
        # plt.errorbar(range(len(r["mean_test_score"])), r["mean_test_score"], yerr=2*r["std_test_score"])
        # plt.show()
        
        print("\n\n")


def make_pipe(n_splits):
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
                "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]
    
    mem = Memory(location=tempfile.mkdtemp(), verbose=0)
    
    pipe_imputed_fast = Pipeline([
        ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
        ('imp', IterativeImputer(max_iter=int(1e2))),
        ('sca', StandardScaler()),
        # ('pca', PCA(15)),
        ('gri', GridSearchCV(Pipeline([  #('pca', None),
            ('clf', None)]),
            scoring='accuracy', refit=True, cv=n_splits, iid=True, return_train_score=False, param_grid={})),
    ], memory=mem, verbose=0)
    
    param_grid = [
        {
            # 'pca': (None, PCA(15)),
            'clf': (SVC(gamma="auto", max_iter=100000),),
            'clf__kernel': ("poly", "rbf"),
            'clf__C': np.logspace(-2, .5, num=5),
        },
        {
            'clf': (BaggingClassifier(Perceptron(max_iter=1000), max_samples=0.5, max_features=0.5),),
            'clf__n_estimators': (500, 1000, 2000,),
        },
        {
            'clf': (RandomForestClassifier(),),
            'clf__n_estimators': (500, 1000, 2000, ),
            'clf__max_depth': (None, 20, 50),
        },
        {
            'clf': (AdaBoostClassifier(),),
            'clf__n_estimators': (500, 1000, 2000,),
        },
    ]
    
    return pipe_imputed_fast, param_grid


def make_pipe_nan(n_splits):
    cols_log = ["DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h", "DER_pt_ratio_lep_tau",
                "DER_pt_tot", "DER_sum_pt", "PRI_jet_all_pt", "PRI_lep_pt", "PRI_met", "PRI_met_sumet", "PRI_tau_pt"]
    
    mem = Memory(location=tempfile.mkdtemp(), verbose=0)
    
    pipe_nan = Pipeline([
        ('col', make_column_transformer((Shift_log(), cols_log), remainder="passthrough")),
        ('sca', StandardScaler()),
        ('imp', SimpleImputer(missing_values=np.nan, fill_value=-999999.0)),
        ('gri', GridSearchCV(Pipeline([('clf', RandomForestClassifier())]),
                             scoring='accuracy', refit=True, cv=n_splits, iid=True, return_train_score=False, param_grid={})),
    ], memory=mem, verbose=0)
    
    param_nan = [{
        'clf__n_estimators': (500, 1000, 2000,),
        'clf__max_depth': (20, 50, None),
    }]
    
    return pipe_nan, param_nan


def main():
    n_train = 1000
    n_test = 1000
    t0 = time()
    data = shuffle(pd.read_csv('data.csv'), random_state=seed)[:n_train + n_test]
    
    # seriesObj = data.apply(lambda x_: -999.0 in list(x_), axis=1)
    # data = data[seriesObj == False][:1000]
    # assert len(data)==1000
    
    y = data['Label']
    y = np.where(y == 's', 1, 0)
    
    x = data.drop(columns=['Label', "KaggleSet", "Weight", "KaggleWeight", "EventId"])
    
    # x = x.replace(-999, np.nan)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=n_test)
    
    n_splits = 3
    pipe, pipe_param = make_pipe(n_splits)
    # pipe, pipe_param = make_pipe_nan(n_splits)
    eval_pipe(pipe, pipe_param, X_train, X_test, y_train, y_test, n_splits=n_splits)
    

    print("temps total", time() - t0)
    

if __name__ == '__main__':
    # import warnings
    # import sys
    # import os
    # if not sys.warnoptions:
    #     warnings.simplefilter("ignore")
    #     os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
    main()
