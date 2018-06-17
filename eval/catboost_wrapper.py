from eval.wrapper import BaseWrapper
import numpy as np
import pandas as pd
import pickle
from functools import partial
from eval.gradient_descent import _expect_score

from sklearn.utils import shuffle


class CatboostWrapper(BaseWrapper):
    WRAPPER_NAME = 'catboost'
    prior = [(0, 1), (0.5, 0.5), (1, 0)]

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(BaseWrapper, self).save_model(fname, format, export_parameters)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'wb') as file_to:
            pickle.dump((self.cat_nums, self.groupped), file_to)

    def load_model(self, fname, format='catboost'):
        super(BaseWrapper, self).load_model(fname, format)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'rb') as file_from:
            self.cat_nums, self.groupped = pickle.load(file_from)
        self.fun_prior = [partial(self._prior, num=0), partial(self._prior, num=1), partial(self._prior, num=2)]

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
        target_col_name = 'target'

        self.groupped = {}
        self.score = {}
        self.fun_prior = [partial(self._prior, num=0), partial(self._prior, num=1), partial(self._prior, num=2)]
        for num in self.cat_nums:
            print("#{}".format(num))

            local_dataset = pd.DataFrame({num: X[:, num], target_col_name: label})

            self.groupped[num] = local_dataset.groupby(num, sort=False)
            lens = self.groupped[num].count()
            cnt_1 = self.groupped[num].sum()

            self.score[num] = {}
            for prior in self.prior:
                self.score[num][prior] = _expect_score(prior[0], prior[1], lens, cnt_1)
            print(self.score[num])
            #
            # del local_dataset
            # h = hpy()
            # print h.heap()

    def _prior(self, num, col_num):
        return self.prior[num]


class CatboostTimeWrapper(CatboostWrapper):
    WRAPPER_NAME = 'catboost_time'

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
