import pandas as pd

from eval.gradient_descent import get_approx_newton
from eval.wrapper import BaseWrapper
import numpy as np
import pickle
from sklearn.utils import shuffle


class BayesWrapper(BaseWrapper):
    WRAPPER_NAME = "bayes"

    def __init__(self, params=None, model_file=None):
        super(BayesWrapper, self).__init__(params, model_file)

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(BayesWrapper, self).save_model(fname, format, export_parameters)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'wb') as file_to:
            pickle.dump((self.cat_nums, self.prior, self.groupped), file_to)

    def load_model(self, fname, format='catboost'):
        super(BayesWrapper, self).load_model(fname, format)
        wrapper_name = self.create_wrapper_name(fname, self.WRAPPER_NAME)
        with open(wrapper_name, 'rb') as file_from:
            self.cat_nums, self.prior, self.groupped = pickle.load(file_from)
        self.fun_prior = [self._prior]

    def handle_test_matrix(self, X, label, is_shuffle=False):
        X = np.nan_to_num(X)

        return self._change_test_dataset(X, label, is_shuffle)

    def handle_learn_matrix(self, X, label, min_border=0.5, max_border=150, is_shuffle=True):
        if min_border is None:
            min_border = 0.5
        if max_border is None:
            max_border = 150

        X = np.nan_to_num(X)

        if is_shuffle:
            inds = shuffle(list(range(X.shape[0])), random_state=42)
            X = X[inds]
            label = np.array(label)[inds]
        else:
            label = np.array(label)

        self._proceed_dataset(X, label, min_border, max_border)
        return self._change_learn_dataset(X, label)

    def _prior(self, col_num):
        return self.prior[col_num]

    def _proceed_dataset(self, X, label, min_border, max_border):
        target_col_name = 'target'

        self.prior = {}
        self.fun_prior = [self._prior]
        self.groupped = {}
        self.score = {}

        for num in self.cat_nums:
            local_dataset = pd.DataFrame({num: X[:, num], target_col_name: label})
            # print('Value counts in learn dataset: {}'.format(local_dataset.iloc[:, 0].value_counts()))
            thetas = local_dataset.groupby(num, sort=False).mean()
            # print(thetas)
            print('#{}'.format(num))
            print('count of 1: {}'.format((thetas == 1).sum()[0]))
            print()

            self.groupped[num] = local_dataset.groupby(num, sort=False)

            lens = self.groupped[num].count()
            cnt_1 = self.groupped[num].sum()

            cnt = thetas.shape[0]
            mean_thetas = thetas.mean()
            var_thetas = cnt * np.var(thetas) / (cnt - 1)

            alpha = min(50, (mean_thetas * (mean_thetas * (1 - mean_thetas) / (var_thetas * var_thetas) - 1))[0])
            beta = min(50,
                       ((1 - mean_thetas) * (mean_thetas * (1 - mean_thetas) / (var_thetas * var_thetas) - 1))[0])

            alpha_new, beta_new, p_new = get_approx_newton(lens.iloc[:, 0], cnt_1.iloc[:, 0], 0.5, 0.5, min_border, max_border) #, alpha, beta)
            print(alpha_new, beta_new, p_new)
            self.prior[num] = (alpha_new, beta_new)
            self.score[num] = p_new


class BayesTimeWrapper(BayesWrapper):
    WRAPPER_NAME = 'bayes_time'

    def handle_learn_matrix(self, X, label, min_border=0.5, max_border=150, is_shuffle=True):
        if min_border is None:
            min_border = 0.5
        if max_border is None:
            max_border = 150

        X = np.nan_to_num(X)

        if is_shuffle:
            inds = shuffle(list(range(X.shape[0])), random_state=42)
            X = X[inds]
            label = np.array(label)[inds]
        else:
            label = np.array(label)

        self._proceed_dataset(X, label, min_border, max_border)
        return self._change_dataset_learn_time(X, label)
