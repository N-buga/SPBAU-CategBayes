from catboost import CatBoost, Pool
import numpy as np
import pandas as pd
from functools import partial
from sklearn.utils import shuffle


class MetricCalcerWrapper:
    def __init__(self, metric_calcer, catboost_wrapper):
        self.metric_calcer = metric_calcer
        self.catboost_wrapper = catboost_wrapper

    def add(self, pool):
        pool = self.catboost_wrapper.handle_test_pool(pool)
        self.metric_calcer.add(pool)

    def metric_descriptions(self):
        return self.metric_calcer.metric_descriptions()

    def eval_metrics(self):
        return self.metric_calcer.eval_metrics()


class BaseWrapper(CatBoost):
    def create_wrapper_name(self, fname, wrapper_name):
        return '{}.{}'.format(fname, wrapper_name)

    def save_model(self, fname, format="cbm", export_parameters=None):
        super(BaseWrapper, self).save_model(fname, format, export_parameters)

    def load_model(self, fname, format='catboost'):
        super(BaseWrapper, self).load_model(fname, format)

    def handle_test_pool(self, X, is_shuffle=False):
        data = np.array(X.get_features())
        label = X.get_label()
        if self.cat_nums is None:
            self.cat_nums = X.get_cat_feature_indices()
        if self.cat_nums is None:
            self.cat_nums = []
        del X
        new_data, new_label = self.handle_test_matrix(data, label, is_shuffle)
        return Pool(new_data, new_label)

    def handle_test_matrix(self, X, label, is_shuffle):
        raise NotImplementedError()

    def handle_learn_pool(self, X, is_shuffle=True):
        self.cat_nums = X.get_cat_feature_indices()
        if self.cat_nums is None:
            self.cat_nums = []
        data = X.get_features()
        label = X.get_label()
        new_data, new_label = self.handle_learn_matrix(np.array(data), label, is_shuffle)
        return Pool(new_data, new_label)

    def handle_learn_matrix(self, X, label, is_shuffle):
        raise NotImplementedError()

    def fit(self, X, y=None, cat_features=None, pairs=None, sample_weight=None, group_id=None, subgroup_id=None,
            pairs_weight=None,
            baseline=None, use_best_model=None, eval_set=None, verbose=None, logging_level=None, plot=False,
            column_description=None, verbose_eval=None):

        if isinstance(X, Pool):
            X = self.handle_learn_pool(X)
        else:
            X, y = self.handle_learn_matrix(X, y)

        super(BaseWrapper, self).fit(X, y, None, pairs, sample_weight, group_id, subgroup_id, pairs_weight,
                                     baseline, use_best_model, eval_set, verbose, logging_level, plot,
                                     column_description, verbose_eval)

    def create_metric_calcer(self, metrics, ntree_start=0, ntree_end=0, eval_period=1, thread_count=-1, tmp_dir=None):
        metric_calcer = super(BaseWrapper, self).create_metric_calcer(metrics, ntree_start, ntree_end, eval_period,
                                                                      thread_count, tmp_dir)
        metric_calcer_wrapper = MetricCalcerWrapper(metric_calcer, self)
        return metric_calcer_wrapper

    # def apply_row(self, row, all_cnt1, lens):
    #     new_row = np.zeros((len(self.cat_nums)*len(self.fun_prior)))
    #     for cat_num, col_num in enumerate(self.cat_nums):
    #         value = row[cat_num]
    #         for prior_num, prior in enumerate(self.fun_prior):
    #             alpha, beta = prior(col_num=col_num)
    #             if value in lens[col_num].index:
    #                 new_row[prior_num*len(self.cat_nums) + cat_num] = (alpha + all_cnt1[col_num].loc[value]) / (alpha + beta + lens[col_num].loc[value])
    #             else:
    #                 new_row[prior_num*len(self.cat_nums) + cat_num] = alpha/(alpha + beta)
    #     return pd.Series(new_row)

    # def _change_dataset_test_time(self, X, label, is_shuffle=True):
    #     X = np.nan_to_num(X)
    #
    #     if is_shuffle:
    #         inds = shuffle(list(range(X.shape[0])), random_state=42)
    #         X = X[inds]
    #         label = np.array(label)[inds]
    #     else:
    #         label = np.array(label)
    #
    #     cat_dataset = np.zeros((X.shape[0], len(self.cat_nums)*len(self.fun_prior)))
    #     for num, col_num in enumerate(self.cat_nums):
    #         for prior_num, fun_prior in enumerate(self.fun_prior):
    #             ones = self.groupped[col_num].sum()
    #             lens = self.groupped[col_num].count()
    #             cat_dataset[:, prior_num*len(self.cat_nums) + num] = \
    #                 self._change_col_time(ones, lens, X[:, col_num], label, *fun_prior(col_num=col_num))
    #
    #     if len(self.fun_prior) > 1:
    #         X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
    #         return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label
    #     else:
    #         X[:, self.cat_nums] = cat_dataset
    #         return X, label

    # def _change_test_dataset(self, X, label, is_shuffle):
    #     if is_shuffle:
    #         inds = shuffle(list(range(X.shape[0])), random_state=42)
    #         X = X[inds]
    #         label = np.array(label)[inds]
    #     else:
    #         label = np.array(label)
    #
    #     all_cnt1 = {}
    #     lens = {}
    #     for col_num in self.cat_nums:
    #         all_cnt1[col_num] = self.groupped[col_num].sum()
    #         lens[col_num] = self.groupped[col_num].count()
    #
    #     apply_row = partial(self.apply_row, all_cnt1=all_cnt1, lens=lens)
    #     cat_dataset = pd.DataFrame(X[:, self.cat_nums]).apply(apply_row, axis=1).values
    #
    #     if len(self.fun_prior) > 1:
    #         X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
    #         return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label
    #     else:
    #         X[:, self.cat_nums] = cat_dataset
    #         return X, label

    def _change_test_dataset(self, X, label, is_shuffle):
        if is_shuffle:
            inds = shuffle(list(range(X.shape[0])), random_state=42)
            X = X[inds]
            label = np.array(label)[inds]
        else:
            label = np.array(label)

        cat_dataset = pd.DataFrame(np.zeros((X.shape[0], len(self.cat_nums)*len(self.fun_prior))))
        for cat_num, col_num in enumerate(self.cat_nums):
            values = np.unique(X[:, col_num])
            all_cnt1 = self.groupped[col_num].sum()['target']
            lens = self.groupped[col_num].count()['target']
            for value in values:
                for prior_num, prior in enumerate(self.fun_prior):
                    alpha, beta = prior(col_num=col_num)
                    if value in lens.index:
                        cur_alpha = alpha + all_cnt1[value]
                        cur_beta = alpha + beta + lens[value] - cur_alpha
                    else:
                        cur_alpha = alpha
                        cur_beta = beta
                    size = (X[:, col_num] == value).sum()
                    cat_dataset.loc[X[:, col_num] == value, prior_num * len(self.cat_nums) + cat_num] = self._replace_fun(cur_alpha, cur_beta, size)

        if len(self.fun_prior) > 1:
            X[:, self.cat_nums] = cat_dataset.values[:, :len(self.cat_nums)]
            return np.hstack((X, cat_dataset.values[:, len(self.cat_nums):])), label
        else:
            X[:, self.cat_nums] = cat_dataset
            return X, label

    def _replace_fun(self, alpha, beta, size):
        return alpha/(alpha + beta)

    def _change_learn_dataset(self, X, label):
        data = pd.DataFrame(X)
        data['target'] = label
        cat_dataset = np.zeros((X.shape[0], len(self.fun_prior)*len(self.cat_nums)))
        for cat_num, col_num in enumerate(self.cat_nums):
            for num, prior in enumerate(self.fun_prior):
                alpha, beta = prior(col_num=col_num)
                cat_dataset[:, num*len(self.cat_nums) + cat_num] = self._change_learn_col(col_num, X[:, col_num], X.shape[0], alpha, beta)

        X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
        if len(self.fun_prior) > 1:
            return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label
        else:
            return X, label

    def _change_learn_col(self, col_num, col, col_size, alpha, beta):
        groupby_data = self.groupped[col_num]['target']
        return (groupby_data.transform('sum') + alpha)/(groupby_data.transform('count') + alpha + beta)

    # def _change_dataset_learn_time(self, X, label):
    #     data = pd.DataFrame(X)
    #     data['target'] = label
    #     cat_dataset = np.zeros((X.shape[0], len(self.fun_prior)*len(self.cat_nums)))
    #     for cat_num, col_num in enumerate(self.cat_nums):
    #         for num, prior in enumerate(self.fun_prior):
    #             alpha, beta = prior(col_num=col_num)
    #             ones = pd.DataFrame(columns=['target'])
    #             lens = pd.DataFrame(columns=['target'])
    #             cat_dataset[:, num*len(self.cat_nums) + cat_num] = self._change_col_time(ones, lens, X[:, col_num], label, alpha, beta)
    #     X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
    #     if len(self.fun_prior) > 1:
    #         return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label
    #     else:
    #         return X, label

    def _change_dataset_learn_time(self, X, label):
        data = pd.DataFrame(X)
        data['target'] = label
        cat_dataset = np.zeros((X.shape[0], len(self.fun_prior)*len(self.cat_nums)))
        # cat_dataset2 = np.zeros((X.shape[0], len(self.fun_prior)*len(self.cat_nums)))
        for cat_num, col_num in enumerate(self.cat_nums):
            for num, prior in enumerate(self.fun_prior):
                alpha, beta = prior(col_num=col_num)
                cat_dataset[:, num*len(self.cat_nums) + cat_num] = self._change_learn_col_time(col_num, X[:, col_num], label, alpha, beta)

        X[:, self.cat_nums] = cat_dataset[:, :len(self.cat_nums)]
        if len(self.fun_prior) > 1:
            return np.hstack((X, cat_dataset[:, len(self.cat_nums):])), label
        else:
            return X, label

    def _change_learn_col_time(self, col_num, col, label, alpha, beta):
        groupby_data = self.groupped[col_num]

        return (groupby_data['target'].transform(lambda x: x.shift().cumsum()).fillna(0).values + alpha) / \
            (groupby_data[col_num].cumcount().values + alpha + beta)
