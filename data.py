import numpy as np
import pandas as pd


class DataSource():
    def get_training_data(self):

        column_names = ['MPG', 'Cylinders', 'Displacement',
                        'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
        raw_dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                                  names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

        dataset = raw_dataset.copy()
        dataset = dataset.dropna()

        origin = dataset.pop('Origin')
        dataset['USA'] = (origin == 1)*1.0
        dataset['Europe'] = (origin == 2)*1.0
        dataset['Japan'] = (origin == 3)*1.0

        desc = dataset.describe()

        train_stats = dataset.describe()
        train_stats.pop("MPG")
        train_stats = train_stats.transpose()

        labels = dataset.pop('MPG')

        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        normed_data = norm(dataset)

        X_raw = normed_data.to_numpy().T

        Y_raw = np.reshape(labels.to_numpy(), (1, len(labels)))

        X_train = X_raw[:, :-10]
        Y_train = Y_raw[:, :-10]
        X_test = X_raw[:, -10:]
        Y_test = Y_raw[:, -10:]

        return X_train, Y_train, X_test, Y_test, desc
