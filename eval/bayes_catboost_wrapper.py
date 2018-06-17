from eval.wrapper import BaseWrapper
import numpy as np
import pandas as pd
import pickle
from functools import partial
from eval.gradient_descent import _expect_score, get_approx_newton
# from guppy import hpy

from sklearn.utils import shuffle


class BayesCatboostWrapper(BaseWrapper):
    WRAPPER_NAME = 'catboost&bayes'
    catboost_prior = [(0, 1), (0.5, 0.5), (1, 0)]

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(BaseWrapper, self).save_model(fname, format, export_parameters)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'wb') as file_to:
            pickle.dump((self.cat_nums, self.groupped, self.prior), file_to)

    def load_model(self, fname, format='catboost'):
        super(BaseWrapper, self).load_model(fname, format)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'rb') as file_from:
            self.cat_nums, self.groupped, self.prior = pickle.load(file_from)
        self.fun_prior = [partial(self._prior, kind='bayes')]
        for i in range(len(self.catboost_prior)):
            self.fun_prior.append(partial(self._prior, kind='catboost', num=i))

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
        # return self._change_test_dataset(X, label)
        return self._change_learn_dataset(X, label)
    #
    # def _change_test_dataset(self, X, label, is_shuffle=True):
    #     if is_shuffle:
    #         inds = shuffle(list(range(X.shape[0])), random_state=42)
    #         X = X[inds]
    #         label = np.array(label)[inds]
    #     else:
    #         label = np.array(label)
    #
    #     cat_dataset = np.zeros((X.shape[0], len(self.cat_nums)*4))
    #     for num, col_num in enumerate(self.cat_nums):
    #         cat_dataset[:, num] = self._change_col(col_num, X[:, col_num],
    #                                                self.prior['bayes'][col_num][0], self.prior['bayes'][col_num][1])
    #     col_num = len(self.cat_nums)
    #     for prior in self.prior['catboost']:
    #         for num in self.cat_nums:
    #             cat_dataset[:, col_num] = self._change_col(num, X[:, num], prior[0], prior[1])
    #             col_num += 1
    #     X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
    #     return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label

    def _proceed_dataset(self, X, label):
        target_col_name = 'target'

        self.prior = {'catboost': self.catboost_prior, 'bayes': {}}
        self.groupped = {}
        self.score = {}
        self.fun_prior = [partial(self._prior, kind='bayes')]
        for i in range(len(self.catboost_prior)):
            self.fun_prior.append(partial(self._prior, kind='catboost', num=i))

        for num in self.cat_nums:
            local_dataset = pd.DataFrame({num: X[:, num], target_col_name: label})
            # print('Value counts in learn dataset: {}'.format(local_dataset.iloc[:, 0].value_counts()))
            # thetas = local_dataset.groupby(cat_col_name).mean()
            # print(thetas)
            # print('#{}'.format(num))
            # print('count of 1: {}'.format((thetas == 1).sum()[0]))
            # print()

            self.groupped[num] = local_dataset.groupby(num, sort=False)
            lens = self.groupped[num].count()
            cnt_1 = self.groupped[num].sum()

            # cnt = thetas.shape[0]
            # mean_thetas = thetas.mean()
            # var_thetas = cnt * np.var(thetas) / (cnt - 1)

            # alpha = min(50, (mean_thetas * (mean_thetas * (1 - mean_thetas) / (var_thetas * var_thetas) - 1))[0])
            # beta = min(50,
            #            ((1 - mean_thetas) * (mean_thetas * (1 - mean_thetas) / (var_thetas * var_thetas) - 1))[0])

            alpha_new, beta_new, p_new = get_approx_newton(lens.iloc[:, 0], cnt_1.iloc[:, 0], 0.5, 0.5) #, alpha, beta)
            print(alpha_new, beta_new, p_new)
            self.prior['bayes'][num] = (alpha_new, beta_new)
            self.score[num] = {}
            for prior in self.prior['catboost']:
                self.score[num][prior] = _expect_score(prior[0], prior[1], lens, cnt_1)
            self.score[num][self.prior['bayes'][num]] = p_new

    def _prior(self, kind, col_num, num=None):
        if kind == 'catboost':
            if num is None:
                raise AttributeError('If the kind is catboost then num should be provided')
            return self.prior[kind][num]
        else:
            return self.prior[kind][col_num]


class BayesCatboostTimeWrapper(BayesCatboostWrapper):
    WRAPPER_NAME = 'catboost&bayes_time'

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
