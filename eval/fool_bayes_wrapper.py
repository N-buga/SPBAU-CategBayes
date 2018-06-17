import numpy as np
from sklearn.utils import shuffle

from eval.wrapper import BaseWrapper
import pickle
import pandas as pd

from eval.gradient_descent import _expect_score


class FoolBayesWrapper(BaseWrapper):
    WRAPPER_NAME = 'fool_bayes'

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(FoolBayesWrapper, self).save_model(fname, format, export_parameters)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'wb') as file_to:
            pickle.dump((self.cat_nums, self.prior, self.groupped), file_to)

    def load_model(self, fname, format='catboost'):
        super(FoolBayesWrapper, self).load_model(fname, format)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'rb') as file_from:
            self.cat_nums, self.prior, self.groupped = pickle.load(file_from)
        self.fun_prior = [self._prior]

    def handle_test_matrix(self, X, label, is_shuffle=False):
        X = np.nan_to_num(X)

        return self._change_test_dataset(X, label, is_shuffle)

    def handle_learn_matrix(self, X, label, is_shuffle=True):
        X = np.nan_to_num(X)

        if is_shuffle:
            inds = shuffle(list(range(X.shape[0])), random_state=42)
            X = X[inds]
            label = np.array(label)[inds]
        else:
            label = np.array(label)

        self._proceed_dataset(X, label)
        return self._change_learn_dataset(X, label)

    def _proceed_dataset(self, X, label):
        mean_theta = np.array(label).mean()
        self.prior = mean_theta, 1 - mean_theta

        target_col_name = 'target'

        self.groupped = {}
        self.score = {}
        self.fun_prior = [self._prior]
        for num in self.cat_nums:
            local_dataset = pd.DataFrame({num: X[:, num], target_col_name: label})

            self.groupped[num] = local_dataset.groupby(num, sort=False)
            lens = self.groupped[num].count()
            cnt_1 = self.groupped[num].sum()

            self.score[num] = _expect_score(self.prior[0], self.prior[1], lens, cnt_1)

    def _prior(self, col_num):
        return self.prior


class FoolBayesTimeWrapper(FoolBayesWrapper):
    WRAPPER_NAME = 'fool_bayes_time'

    def handle_learn_matrix(self, X, label, is_shuffle=True):
        X = np.nan_to_num(X)

        if is_shuffle:
            inds = shuffle(list(range(X.shape[0])), random_state=42)
            X = X[inds]
            label = np.array(label)[inds]
        else:
            label = np.array(label)

        self._proceed_dataset(X, label)
        return self._change_dataset_learn_time(X, label)
