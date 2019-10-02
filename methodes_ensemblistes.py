from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from preprocessing import preprocessing
from AMS import AMS, compute_AMS
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from time import time
from sklearn.utils import shuffle


def cross_val(classifiers, X, y):
    for clf in classifiers:
        print(clf)
        print(np.mean(cross_val_score(clf, X, y, cv=5, scoring=AMS)))

def test_score(classifiers, X_train, y_train, X_test, y_test, scoring_function):
    for clf in classifiers:
        t0 = time()
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        print(clf)
        if scoring_function == 'accuracy':
            print("Accuracy on training data : {}".format(clf.score(X_train, y_train)))
            print("Accuracy on test data : {}".format(clf.score(X_test, y_test)))
        elif scoring_function == 'f1':
            print("F1 score on training data :   {}".format(f1_score(y_train, y_pred_train, pos_label=1)))
            print("F1 score on test data :   {}".format(f1_score(y_test, y_pred_test, pos_label=1)))
        else:
            print("AMS score on training data :   {}".format(compute_AMS(y_train, y_pred_train)))
            print("AMS score on test data :   {}".format(compute_AMS(y_test, y_pred_test)))
        print("time", time() - t0)
    

def main():
    scoring_function = 'AMS'
    X, y = preprocessing("data.csv", nelement=10000)

    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
        y, test_size=0.3,random_state=0,stratify=y)
    
    classifiers = [
        BaggingClassifier(Perceptron(max_iter=1000), n_estimators=10, max_samples=0.5, max_features=0.5),
        RandomForestClassifier(n_estimators=10),
        AdaBoostClassifier(n_estimators=100)
    ]
    print(classifiers[0].__class__)
    
    # cross_val(classifiers, X, y)
    test_score(classifiers, X_train, y_train, X_test, y_test, scoring_function)


if __name__ == '__main__':
    main()