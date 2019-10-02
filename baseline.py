from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from AMS import AMS, compute_AMS

def optimize_SVM(train, scoring_function, interval):
    X_train, y_train = train
    mean_scores = []

    for C in interval:
        clf = LinearSVC(C=C)

        if scoring_function == 'accuracy':
            scores = cross_val_score(clf, X_train, y_train, cv=4)
        elif scoring_function == 'f1':
            scores = cross_val_score(clf, X_train, y_train, cv=4,scoring='f1')
        else:
            scores = cross_val_score(clf, X_train, y_train, cv=4,scoring=AMS)
        mean_scores.append(np.mean(scores))

    C_opt = interval[np.argmax(mean_scores)]
    print("Optimal value for C : {}".format(C_opt))
    clf = LinearSVC(C=C_opt)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_train)

    if scoring_function == 'accuracy':
        print("Accuracy on training data : {}".format(clf.score(X_train, y_train)))
    elif scoring_function == 'f1':
        print("F1 score on training data : {}".format(f1_score(y_train, y_pred, pos_label=1)))
    else:
        print("AMS on training data : {}".format(compute_AMS(y_train, y_pred)))

    return clf, mean_scores

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

def grid_search(scoring_function):
    nelement = 100000
    print("Preprocessing data...")
    print("Working on {} elements".format(nelement))
    X, y = preprocessing.preprocessing('data.csv', nelement=nelement)
    print("Data shape : {}, labels shape : {}".format(X.shape, y.shape))
    print(np.count_nonzero(y == 1))

    X, y = shuffle(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, test_size=0.3, random_state=0, stratify=y)

    interval = [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2, 10 ** 3]

    print("Training...")
    clf, mean_scores = optimize_SVM((X_train, y_train), scoring_function, interval)
    y_pred = clf.predict(X_test)

    if scoring_function == 'accuracy':
        print("Accuracy on test data : {}".format(clf.score(X_test, y_test)))
    elif scoring_function == 'f1':
        print("F1 score on test data : {}".format(f1_score(y_test, y_pred, pos_label=1)))
    else:
        print("AMS on test data : {}".format(compute_AMS(y_test, y_pred)))

    plot_score(mean_scores, interval, scoring_function)



def main():
    scoring_function = "AMS"
    grid_search(scoring_function)

if __name__ == '__main__':
    main()