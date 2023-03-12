import pandas as pd
from typing import Dict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class UrbanSounds(Dataset):
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.DataFrame,
                 class_to_idx: Dict[str, int],
                 mean: np.ndarray = None,
                 std: np.ndarray = None):
        self.X = X.to_numpy()

        self.y = y
        self.class_to_idx = class_to_idx
        self.y = self.y.replace(self.class_to_idx).to_numpy()

        self.mean = mean
        self.std = std
        self.class_weights = self.compute_class_weights()

        if self.mean is None and self.std is None:
            self.mean, self.std = np.mean(self.X, axis=0), np.std(self.X, axis=0)

        # Normalization
        self.X = self.X - self.mean / self.std

    def compute_class_weights(self):
        (_, nb_samples_per_class) = np.unique(self.y, return_counts=True)
        return list(map(lambda frq: len(self.y) / frq, nb_samples_per_class))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx, :])
        X = X.type(torch.FloatTensor)
        y = self.y[idx].item()

        return X, y


def filter_data(df):
    df = df.loc[(df['class'] == 'jackhammer') | (df['class'] == 'engine_idling')]
    return df


def get_datasets(df_path: str):
    df = pd.read_csv(df_path)
    # df = filter_data(df)

    X = df.iloc[:, 4:]
    y = df.iloc[:, 1]

    class_to_idx = {class_name: idx for idx, class_name in enumerate(y.unique())}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1,
                                                      stratify=y_train)

    train_dataset = UrbanSounds(X_train, y_train, class_to_idx)
    val_dataset = UrbanSounds(X_val, y_val, class_to_idx, train_dataset.mean, train_dataset.std)
    test_dataset = UrbanSounds(X_test, y_test, class_to_idx, train_dataset.mean, train_dataset.std)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    train, val, test = get_datasets(df_path="statistics.csv")

    for i in range(len(train)):
        _, _ = train[i]
        break
