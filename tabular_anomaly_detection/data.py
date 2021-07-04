import numpy as np
import torch
from pyarrow.compute import min_max
from torch import optim
from torch.utils.data import DataLoader


class TabularFeatureExtractor:
    def __init__(self, categorical_columns, numeric_columns, label_column=None):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.label_column = label_column
        self.feature_columns = categorical_columns + numeric_columns + [label_column]

    def fit(self, ds):
        self.id_to_col = {}
        self.col_to_id = {}
        for column_name in self.categorical_columns:

            self.id_to_col[column_name] = {}
            self.col_to_id[column_name] = {}

            for i, value in enumerate(ds.unique(column_name)):
                self.id_to_col[column_name][i] = value
                self.col_to_id[column_name][value] = i

        min_max_map = {}
        min_max_map_scaled = {}
        for column_name in self.numeric_columns:
            min, max = tuple(min_max(ds[column_name]).as_py().values())
            min_max_map[column_name] = (min, max)
            min_max_map_scaled[column_name] = (np.log(min + 1e-4), np.log(max + 1e-4))
        self.min_max_map = min_max_map
        self.min_max_map_scaled = min_max_map_scaled

        num_cat_feats_per_col = [len(_map) for col, _map in self.col_to_id.items()]
        num_cat_feats_compounding = []
        for i, n in enumerate(num_cat_feats_per_col):
            num_cat_feats_compounding.append(n + sum(num_cat_feats_per_col[:i]))
        self.num_cat_feats_per_col = num_cat_feats_per_col
        self.total_cat_feats = sum(num_cat_feats_per_col)
        self.num_cat_feats_compounding = num_cat_feats_compounding

        self.feature_dim = self.total_cat_feats + len(self.numeric_columns)

        if self.label_column is not None:
            self.labels = sorted(ds.unique(self.label_column))
            self.id_to_label = dict(enumerate(self.labels))
            self.label_to_id = {v: k for k, v in self.id_to_label.items()}

        return self

    def __call__(self, example):

        for col, (min, max) in self.min_max_map_scaled.items():
            example[col] = [(np.log(x + 1e-4) - min) / (max - min) for x in example[col]]

        for col, _map in self.col_to_id.items():
            example[col] = [_map[x] for x in example[col]]

        encodings = {}
        encodings["categorical_features"] = list(zip(*(example[c] for c in self.categorical_columns)))
        encodings["numeric_features"] = list(zip(*(example[c] for c in self.numeric_columns)))

        if self.label_column is not None:
            encodings["labels"] = [self.label_to_id[x] for x in example[self.label_column]]

        return encodings


class TabularCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):

        # Categorical IDs to Multi-One-Hot
        cat_features = torch.LongTensor([ex["categorical_features"] for ex in batch])
        cat_features[:, 1:] += torch.tensor(self.feature_extractor.num_cat_feats_compounding[:-1])

        # TODO - There's probably a more efficient way to do this...
        cat_features = torch.nn.functional.one_hot(cat_features, num_classes=self.feature_extractor.total_cat_feats)
        cat_features = cat_features.sum(1)

        numeric_features = torch.tensor([ex["numeric_features"] for ex in batch])

        # HACK - This never returns
        if self.feature_extractor.label_column is not None:
            labels = torch.tensor([ex["labels"] for ex in batch])

        return cat_features.float(), numeric_features
