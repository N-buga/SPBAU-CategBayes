from eval.bayes_wrapper import BayesWrapper, BayesTimeWrapper
import numpy as np
import pandas as pd


class BayesRandomWrapper(BayesWrapper):

    def _replace_fun(self, alpha, beta, size=None):
        return np.random.beta(alpha, beta, size)

    def _change_learn_col(self, num, col, col_size, alpha, beta):
        lens = self.groupped[num].count()
        cnt_1 = self.groupped[num].sum()

        new_col = np.zeros((col_size, ))
        for i, val in enumerate(col):
            if val != '?':
                if val in lens.index:
                    cur_alpha = alpha + cnt_1.loc[val].iloc[0]
                    cur_beta = alpha + beta + lens.loc[val].iloc[0] - cur_alpha
                else:
                    cur_alpha = alpha
                    cur_beta = beta
            else:
                cur_alpha = alpha
                cur_beta = beta

            new_col[i] = self._replace_fun(cur_alpha, cur_beta)
        return new_col


class BayesRandomTimeWrapper(BayesTimeWrapper):
    def _replace_fun(self, alpha, beta, size=None):
        return np.random.beta(alpha, beta, size)

    def _change_learn_col_time(self, col_num, col, label, alpha, beta):

        ones = pd.DataFrame(columns=['target'])
        lens = pd.DataFrame(columns=['target'])

        new_col = np.zeros((label.shape[0],))
        for i, val in enumerate(col):
            if val != '?':
                if val in ones.index:
                    cur_alpha = (alpha + ones.loc[val, 'target'])
                    cur_beta = (alpha + beta + lens.loc[val, 'target']) - cur_alpha
                    ones.loc[val, 'target'] += label[i]
                    lens.loc[val, 'target'] += 1
                else:
                    cur_alpha = alpha
                    cur_beta = beta
                    ones.set_value(val, 'target', label[i])
                    lens.set_value(val, 'target', 1)
            else:
                cur_alpha = alpha
                cur_beta = beta
            new_col[i] = self._replace_fun(cur_alpha, cur_beta)
        return new_col


