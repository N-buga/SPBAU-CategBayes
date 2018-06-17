# coding=utf-8
from __future__ import print_function

import os
import numpy as np
import time

import sys

from catboost import CatBoostClassifier, Pool
import multiprocessing

from eval.bayes_wrapper import BayesWrapper, BayesTimeWrapper
from eval.bayes_catboost_wrapper import BayesCatboostWrapper, BayesCatboostTimeWrapper
from eval.catboost_wrapper import CatboostTimeWrapper, CatboostWrapper
from eval.fool_bayes_wrapper import FoolBayesWrapper, FoolBayesTimeWrapper

dir_datasets = ['amazon', 'adult', 'appet', 'kick']

learn_name = 'train_full3.01'
test_name = 'test3.01'
cd_file = 'train_full3.cd'

learn_catboost = 'train_catboost'
test_catboost = 'test_catboost'
learn_catboost_time = 'train_catboost_time'
test_catboost_time = 'test_catboost_time'
learn_bayes = 'train_bayes'
test_bayes = 'test_bayes'
learn_bayes_time = 'train_bayes_time'
test_bayes_time = 'test_bayes_time'
learn_fool = 'train_fool'
test_fool = 'test_fool'
learn_fool_time = 'train_fool_time'
test_fool_time = 'test_fool_time'
learn_bayes_catboost = 'train_bayes_catboost'
test_bayes_catboost = 'test_bayes_catboost'
learn_bayes_catboost_time = 'train_bayes_catboost_time'
test_bayes_catboost_time = 'test_bayes_catboost_time'

dataset_description = {
    CatboostWrapper: 'Catboost(3 columns with constants 0, 0.5, 1) without time'.encode('utf-8'),
    CatboostTimeWrapper: 'Catboost(3 columns with constants 0, 0.5, 1) with time'.encode('utf-8'),
    BayesWrapper: 'Emperical bayes without time'.encode('utf-8'),
    BayesTimeWrapper: 'Emperical bayes with time'.encode('utf-8'),
    FoolBayesWrapper: 'Fool bayes(aprior parameters is calculated through mean target value of all dataset) without time'.encode('utf-8'),
    FoolBayesTimeWrapper: 'Fool bayes(aprior parameters is calculated through mean target value of all dataset) with time'.encode('utf-8'),
    BayesCatboostWrapper: 'Emperical bayes column plus catboost columns without time'.encode('utf-8'),
    BayesCatboostTimeWrapper: 'Emperical bayes column plus catboost columns with time'.encode('utf-8'),

    None: 'None'
}

def catboost_test(dir_, cur_learn_name, cur_test_name, clazz, learning_rate=None, border_count=128, cnt_models=1,
                  file_result_to=sys.stdout, file_info_to=sys.stdout, iterations=1500):
    full_learn_name = os.path.join(dir_, cur_learn_name)
    full_test_name = os.path.join(dir_, cur_test_name)

    if not os.path.exists(full_learn_name):
        source_learn_pool = Pool(data=os.path.join(dir_, learn_name), column_description=os.path.join(dir_, cd_file))
        source_test_pool = Pool(data=os.path.join(dir_, test_name), column_description=os.path.join(dir_, cd_file))
        cl = clazz()
        beg = time.time()
        learn_pool = cl.handle_learn_pool(source_learn_pool)
        test_pool = cl.handle_test_pool(source_test_pool)
        end = time.time()
        print('!!!time: {}'.format(end - beg), file=file_info_to)
        print('priors: {}'.format(cl.prior), file=file_info_to)
        print('prior scores: {}'.format(cl.score), file=file_info_to)
        file_info_to.flush()
        learn_label = learn_pool.get_label()
        learn_features = learn_pool.get_features()
        learn_data = np.zeros((len(learn_label), len(learn_features[0]) + 1))
        learn_data[:, 0] = learn_label
        learn_data[:, 1:] = learn_features
        np.savetxt(full_learn_name, learn_data, delimiter='\t', fmt='%.10f')
        test_label = test_pool.get_label()
        test_features = test_pool.get_features()
        test_data = np.zeros((len(test_label), len(test_features[0]) + 1))
        test_data[:, 0] = test_label
        test_data[:, 1:] = test_features
        np.savetxt(full_test_name, test_data, delimiter='\t', fmt='%.10f')

    learn_pool = Pool(data=full_learn_name)
    test_pool = Pool(data=full_test_name)

    scores = []
    auc = []
    logloss = []
    times =[]
    tree_counts = []
    for seed in range(cnt_models):
        print(seed)
        # print(len(learn_pool.get_features()), len(learn_pool.get_features()[0]))
        # print(len(test_pool.get_features()), len(test_pool.get_features()[0]))
        beg = time.time()
        cat = CatBoostClassifier(max_ctr_complexity=1, custom_metric='AUC', boosting_type='Plain', random_seed=seed, border_count=border_count, iterations=iterations, learning_rate=learning_rate, thread_count=multiprocessing.cpu_count())
        cat.fit(learn_pool, eval_set=(test_pool), use_best_model=True)
        end = time.time()
        X_test = test_pool.get_features()
        y_test = test_pool.get_label()

        tree_counts.append(cat.tree_count_)
        scores.append(cat.score(X_test, y_test))
        metrics = cat.eval_metrics(test_pool, ['AUC', 'Logloss'], eval_period=cat.tree_count_ - 1)
        print('overfit={}; acc={}; AUC={}; logloss={}; learn_time={}'.format(cat.tree_count_, scores[-1], metrics['AUC'][1], metrics['Logloss'][1], end - beg), file=file_result_to)
        file_result_to.flush()
        auc.append(metrics['AUC'][1])
        logloss.append(metrics['Logloss'][1])
        times.append(end - beg)
    if len(tree_counts) != 0:
        print('mean tree_count: {}'.format(sum(tree_counts)/len(tree_counts)), file=file_result_to)
        return sum(scores)/len(scores), sum(auc)/len(auc), sum(logloss)/len(logloss), sum(times)/len(times)
    else:
        return 0, 0, 0, 0


class Params:
    def __init__(self, dir_dataset, cur_learn_name, cur_test_name, step, border_count, clazz):
        self.border_count = border_count
        self.step = step
        self.test_name = cur_test_name
        self.learn_name = cur_learn_name
        self.dir_dataset = dir_dataset
        self.clazz = clazz

    def run(self, file_result, file_info, cnt_models=10, iterations=1500):
        print(dataset_description[self.clazz], file=file_result)
        print(dataset_description[self.clazz], file=file_info)
        file_result.flush()
        file_info.flush()
        print(self.__dict__, file=file_result)
        file_result.flush()
        score, auc, logloss, mean_time = \
            catboost_test(self.dir_dataset, self.learn_name, self.test_name, self.clazz, self.step,
                          border_count=self.border_count, cnt_models=cnt_models,
                          file_result_to=file_result, file_info_to=file_info, iterations=iterations)
        print('mean_acc={}, mean_auc={}, mean_logloss={}, mean_time={}'.format(score, auc, logloss, mean_time), file=file_result)
        file_result.flush()

if __name__ == "__main__":
    with open("results.txt", 'w') as file_result_to:
        with open("dataset_info.txt", 'a') as file_info_to:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', file=file_info_to)

            Params(dir_datasets[0], learn_catboost, test_catboost, 0.00055, border_count=32, clazz=CatboostWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[0], learn_catboost_time, test_catboost_time, 0.024, border_count=32, clazz=CatboostTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[0], learn_bayes, test_bayes, 0.00075, border_count=32, clazz=BayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[0], learn_bayes_time, test_bayes_time, 0.013, border_count=32, clazz=BayesTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[0], learn_fool, test_fool, 0.00058, border_count=32, clazz=FoolBayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[0], learn_fool_time, test_fool_time, 0.016, border_count=32, clazz=FoolBayesTimeWrapper)\
                .run(file_result_to, file_info_to)

            Params(dir_datasets[6], learn_catboost, test_catboost, 0.00668, border_count=32, clazz=CatboostWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[6], learn_catboost_time, test_catboost_time, 0.0302, border_count=32, clazz=CatboostTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[6], learn_bayes, test_bayes, 0.0208, border_count=32, clazz=BayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[6], learn_bayes_time, test_bayes_time, 0.0205, border_count=32, clazz=BayesTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[6], learn_fool, test_fool, 0.0172, border_count=32, clazz=FoolBayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[6], learn_fool_time, test_fool_time, 0.025, border_count=32, clazz=FoolBayesTimeWrapper)\
                .run(file_result_to, file_info_to)

            Params(dir_datasets[1], learn_catboost, test_catboost, 0.023, border_count=32, clazz=CatboostWrapper)\
                .run(file_result_to, file_info_to, iterations=1500)
            Params(dir_datasets[1], learn_catboost_time, test_catboost_time, 0.0174, border_count=32, clazz=CatboostTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[1], learn_bayes, test_bayes, 0.0243, border_count=32, clazz=BayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[1], learn_bayes_time, test_bayes_time, 0.018, border_count=32, clazz=BayesTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[1], learn_fool, test_fool, 0.021, border_count=32, clazz=FoolBayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[1], learn_fool_time, test_fool_time, 0.015, border_count=32, clazz=FoolBayesTimeWrapper)\
                .run(file_result_to, file_info_to)

            Params(dir_datasets[2], learn_catboost, test_catboost, 0.001, border_count=32, clazz=CatboostWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[2], learn_catboost_time, test_catboost_time, 0.016, border_count=32, clazz=CatboostTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[2], learn_bayes, test_bayes, 0.0008, border_count=32, clazz=BayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[2], learn_bayes_time, test_bayes_time, 0.0175, border_count=32, clazz=BayesTimeWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[2], learn_fool, test_fool, 0.00075, border_count=32, clazz=FoolBayesWrapper)\
                .run(file_result_to, file_info_to)
            Params(dir_datasets[2], learn_fool_time, test_fool_time, 0.015, border_count=32, clazz=FoolBayesTimeWrapper)\
                .run(file_result_to, file_info_to)
