import numpy as np
import pandas as pd

def preprocessing(filename, threshold=0.4, nelement=None, verbose=False):
    data = pd.read_csv(filename)
    if verbose:
        print("Original data shape : {}".format(data.shape))

    data['Label'] = np.where(data['Label']=='s', 1, 0)
    data = data.drop(columns=["KaggleSet","Weight","KaggleWeight", "EventId"])

    columns_to_remove = []
    for column in data.columns:
        n_invalid = np.count_nonzero(data[column] == -999.0)
        rate_invalid = n_invalid / data.shape[0]
        if verbose:
            print("{} - percentage of missing values : {} %".format(column, 100*rate_invalid))
        if rate_invalid >= threshold:
            if verbose:
                print("\tRemoving {}".format(column))
            columns_to_remove.append(column)

    data = data.drop(columns=columns_to_remove)

    seriesObj = data.apply(lambda x: -999.0 in list(x), axis=1)
    numOfRows = len(seriesObj[seriesObj == False].index)
    if verbose:
        print("{} % of valid rows".format(numOfRows/data.shape[0]))

    data = data[seriesObj == False]
    if nelement:
        data = data[:nelement]

    y = np.array(data['Label'])
    data = data.drop(columns=['Label'])
    x = data.values

    if verbose:
        print("Preprocessed data shape : {}".format(data.shape))

    return x, y

def main():
    data = preprocessing('data.csv')

if __name__ == '__main__':
    main()