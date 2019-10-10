from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Perceptron
from preprocessing import preprocessing
from AMS import AMS, compute_AMS
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from time import time
from sklearn.utils import shuffle
import numpy as np

def test_score(X_train, y_train, X_test, y_test, scoring_function):
    classifiers = [
        BaggingClassifier(Perceptron(max_iter=1000), n_estimators=10, max_samples=0.5, max_features=0.5),
        RandomForestClassifier(n_estimators=10),
        AdaBoostClassifier(n_estimators=100)
    ]

    for clf in classifiers:
        t0 = time()
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        print(clf)
        if scoring_function == 'accuracy':
            print("\tAccuracy on training data : {}".format(clf.score(X_train, y_train)))
            print("\tAccuracy on test data : {}".format(clf.score(X_test, y_test)))
        elif scoring_function == 'f1':
            print("\tF1 score on training data :   {}".format(f1_score(y_train, y_pred_train, pos_label=1)))
            print("\tF1 score on test data :   {}".format(f1_score(y_test, y_pred_test, pos_label=1)))
        else:
            print("\tAMS score on training data :   {}".format(compute_AMS(y_train, y_pred_train)))
            print("\tAMS score on test data :   {}".format(compute_AMS(y_test, y_pred_test)))
        print("\ttime", time() - t0)

def randomized_grid_search(classifier, grid, X_train, y_train, X_test, y_test,
                           scoring_function):

    assert classifier is not None and grid is not None

    t0 = time()
    print("Randomized grid search ...")
    gs = RandomizedSearchCV(classifier, grid)
    gs.fit(X_train, y_train)

    print("Mean train scores : {}".format(gs.cv_results_['mean_train_score']))
    print("Mean test scores : {}".format(gs.cv_results_['mean_test_score']))
    print("Best parameters : {}".format(gs.best_params_))

    best_clf = gs.best_estimator_

    y_pred_train = best_clf.predict(X_train)
    y_pred_test = best_clf.predict(X_test)
    if scoring_function == 'accuracy':
        print("\tAccuracy on training data : {}".format(
            best_clf.score(X_train, y_train)))
        print("\tAccuracy on test data : {}".format(
            best_clf.score(X_test, y_test)))
    elif scoring_function == 'f1':
        print("\tF1 score on training data :   {}".format(
            f1_score(y_train, y_pred_train, pos_label=1)))
        print("\tF1 score on test data :   {}".format(
            f1_score(y_test, y_pred_test, pos_label=1)))
    else:
        print("\tAMS score on training data :   {}".format(
            compute_AMS(y_train, y_pred_train)))
        print("\tAMS score on test data :   {}".format(
            compute_AMS(y_test, y_pred_test)))
    print("\ttime", time() - t0)

def main(clf_name, scoring_function):
    classifier = None
    random_grid = None
    nelement = 10000
    assert clf_name == 'RandomForest' or clf_name == 'AdaBoost'
    assert scoring_function == 'accuracy' or scoring_function == 'f1' or scoring_function == 'AMS'

    if clf_name == 'RandomForest':
        print("Using random forest")
        classifier = RandomForestClassifier()

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    else:
        print("Using AdaBoost")
        classifier = AdaBoostClassifier()

        random_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        }

    print("Preprocessing data... ({} samples)".format(nelement))
    X, y = preprocessing("data.csv", nelement=nelement)

    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3,random_state=0,stratify=y)
    
    randomized_grid_search(classifier, random_grid,X_train, y_train, X_test, y_test,
                           scoring_function)


if __name__ == '__main__':
    scoring_function = 'AMS'
    clf_name = 'AdaBoost'
    main(clf_name, scoring_function)