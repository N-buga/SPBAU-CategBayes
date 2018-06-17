from __future__ import print_function

import os
import numpy as np
import sys
from catboost import Pool, CatBoostClassifier
import time
import multiprocessing
import json

from plotly.offline import plot
from scipy.stats import wilcoxon
from eval.bayes_wrapper import BayesWrapper, BayesTimeWrapper
from eval.bayes_catboost_wrapper import BayesCatboostWrapper, BayesCatboostTimeWrapper
from eval.catboost_wrapper import CatboostTimeWrapper, CatboostWrapper
from eval.fool_bayes_wrapper import FoolBayesWrapper, FoolBayesTimeWrapper

dir_datasets = ['amazon', 'adult', 'appet', 'kdd98', 'click', 'upsel', 'kick', 'test_data', 'comb_amazon']


def create_learning_curves_plot(curves, case_description, eval_step=1):
    """
    :param offset: First iteration to plot
    :return: plotly Figure with learning curves for each fold
    """
    import plotly.graph_objs as go

    traces = []

    for num, scores_curve in enumerate(curves):
        first_idx = int(len(scores_curve) * 0.1)
        traces.append(go.Scatter(x=[i * int(eval_step) for i in range(first_idx, len(scores_curve))],
                                 y=scores_curve[first_idx:],
                                 mode='lines',
                                 name='Split #{}'.format(num)))

    layout = go.Layout(
        title='Learning curves for case {}'.format(case_description),
        hovermode='closest',
        xaxis=dict(
            title='Iteration',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Metric',
            ticklen=5,
            gridwidth=2,
        ),
        showlegend=True
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig


def catboost_bootstrap(dir_, learn_name, test_name, cd_file, classes, learning_rate=None, border_count=32, cnt_values=20,
                       file_result_to=sys.stdout, file_info_to=sys.stdout, iterations=1500):
    logloss = {}
    auc = {}
    for clazz in classes:
        print('class={}'.format(clazz.WRAPPER_NAME))
        print('class={}; step={}'.format(clazz.WRAPPER_NAME, learning_rate[clazz]), file=file_result_to)
        file_result_to.flush()
        auc[clazz.WRAPPER_NAME] = []
        logloss[clazz.WRAPPER_NAME] = []
        tree_counts = []
        logloss_curves = []
        auc_curves = []

        cl = clazz()
        source_learn_pool = Pool(data=os.path.join(dir_, learn_name), column_description=os.path.join(dir_, cd_file))
        beg = time.time()
        learn_pool = cl.handle_learn_pool(source_learn_pool)
        end = time.time()
        print('!!!time: {}'.format(end - beg), file=file_info_to)
        print('priors: {}'.format(cl.prior), file=file_info_to)
        print('prior scores: {}'.format(cl.score), file=file_info_to)
        file_info_to.flush()

        source_test_pool = Pool(data=os.path.join(dir_, test_name), column_description=os.path.join(dir_, cd_file))
        source_test_label = np.array(source_test_pool.get_label())
        source_test_features = np.array(source_test_pool.get_features())

        cat = CatBoostClassifier(max_ctr_complexity=1, custom_metric='AUC', boosting_type='Plain', random_seed=0,
                                 border_count=border_count, iterations=iterations, learning_rate=learning_rate[clazz],
                                 thread_count=multiprocessing.cpu_count())
        beg = time.time()
        cat.fit(learn_pool, use_best_model=True)
        end = time.time()

        for seed in range(cnt_values):
            idx = list(range(source_test_features.shape[0]))
            np.random.seed(seed*10 + 300)
            boot_idx = np.random.choice(idx, len(idx), replace=True)
            boot_test_features = source_test_features[boot_idx]
            boot_test_label = source_test_label[boot_idx]
            X, y = cl.handle_test_matrix(boot_test_features, boot_test_label, False)
            metrics = cat.eval_metrics(Pool(X, y), ['Logloss', 'AUC'], eval_period=1, thread_count=multiprocessing.cpu_count())
            for num, loss in enumerate(metrics['Logloss']):
                print('iter={:10}:     loss={:.10}'.format(num + 1, loss))
            cnt_trees = np.argmin(metrics['Logloss'])
            print('choose cnt_trees={}'.format(cnt_trees))
            print('overfit={}; AUC={}; logloss={}'.format(cnt_trees, metrics['AUC'][cnt_trees], metrics['Logloss'][cnt_trees]), file=file_result_to)
            tree_counts.append(cnt_trees)
            file_result_to.flush()
            logloss_curves.append(metrics['Logloss'])
            auc_curves.append(metrics['AUC'])
            auc[clazz.WRAPPER_NAME].append(metrics['AUC'][cnt_trees])
            logloss[clazz.WRAPPER_NAME].append(metrics['Logloss'][cnt_trees])

        print('class={}, learn_time={}, mean_tree_count={}'.format(clazz.WRAPPER_NAME, end - beg, sum(tree_counts)/len(tree_counts)), file=file_result_to)
        print('mean_AUC={}, mean_logloss={}'.format(sum(auc[clazz.WRAPPER_NAME])/len(auc[clazz.WRAPPER_NAME]), sum(logloss[clazz.WRAPPER_NAME])/len(logloss[clazz.WRAPPER_NAME])), file=file_result_to)
        file_result_to.flush()

        logloss_fig = create_learning_curves_plot(logloss_curves, 'logloss {}'.format(clazz.WRAPPER_NAME))
        auc_fig = create_learning_curves_plot(auc_curves, 'AUC {}'.format(clazz.WRAPPER_NAME))
        logloss_file = os.path.join(dir_, 'fig_{}_{}'.format('Logloss', clazz.WRAPPER_NAME))
        AUC_file = os.path.join(dir_, 'fig_{}_{}'.format('AUC', clazz.WRAPPER_NAME))
        plot(logloss_fig, filename=logloss_file, auto_open=False)
        plot(auc_fig, filename=AUC_file, auto_open=False)

    file_name = os.path.join(dir_, 'boot.txt')
    with open(file_name, 'w') as file_to:
        json.dump(auc, file_to)

    for cl1 in classes:
        for cl2 in classes:
            stat, p_value = wilcoxon(auc[cl1.WRAPPER_NAME], auc[cl2.WRAPPER_NAME], zero_method="pratt")
            print('for {} & {}: stat: {}, p_value: {}'.format(cl1.WRAPPER_NAME, cl2.WRAPPER_NAME, stat, p_value), file=file_result_to)

if __name__ == '__main__':
    learning_rate = {
        'kick': {
            CatboostWrapper: 0.004,
            CatboostTimeWrapper: 0.029,
            BayesWrapper: 0.0217,
            BayesTimeWrapper: 0.019,
            FoolBayesWrapper: 0.013,
            FoolBayesTimeWrapper: 0.0168
        },

        'amazon': {
            CatboostWrapper: 0.00055,
            CatboostTimeWrapper: 0.0204,
            BayesWrapper: 0.00075,
            BayesTimeWrapper: 0.008,
            FoolBayesWrapper: 0.00058,
            FoolBayesTimeWrapper: 0.016
        },

        'adult': {
            CatboostWrapper: 0.025,
            CatboostTimeWrapper: 0.0186,
            BayesWrapper: 0.0249,
            BayesTimeWrapper: 0.019,
            FoolBayesWrapper: 0.0205,
            FoolBayesTimeWrapper: 0.016
        },

        'appet': {
            CatboostWrapper: 0.001,
            CatboostTimeWrapper: 0.014,
            BayesWrapper: 0.00075,
            BayesTimeWrapper: 0.0175,
            FoolBayesWrapper: 0.00075,
            FoolBayesTimeWrapper: 0.015
        }
    }

    with open('amazon_boot_results.txt', 'w') as file_result:
        with open('amazon_boot_info.txt', 'w') as file_info:
            catboost_bootstrap('amazon', 'train_full3.01', 'test3.01', 'train_full3.cd',
                               [CatboostWrapper,
                                CatboostTimeWrapper,
                                BayesWrapper,
                                BayesTimeWrapper,
                                FoolBayesWrapper, FoolBayesTimeWrapper
                                ],
                                learning_rate=learning_rate['amazon'],
                                file_result_to=file_result,
                                file_info_to=file_info,
                                cnt_values=25
                               )

    with open('kick_boot_results.txt', 'w') as file_result:
        with open('kick_boot_info.txt', 'w') as file_info:
            catboost_bootstrap('kick', 'train_full3.01', 'test3.01', 'train_full3.cd',
                               [CatboostWrapper,
                                CatboostTimeWrapper,
                                BayesWrapper, BayesTimeWrapper,
                                FoolBayesWrapper,
                                FoolBayesTimeWrapper],
                                learning_rate=learning_rate['kick'],
                                file_result_to=file_result,
                                file_info_to=file_info,
                                cnt_values=25
                               )

    with open('adult_boot_results.txt', 'w') as file_result:
        with open('adult_boot_info.txt', 'w') as file_info:
            catboost_bootstrap('adult', 'train_full3.01', 'test3.01', 'train_full3.cd',
                               [
                                   CatboostWrapper,
                                   CatboostTimeWrapper,
                                   BayesWrapper,
                                   BayesTimeWrapper,
                                   FoolBayesWrapper,
                                   FoolBayesTimeWrapper
                               ],
                               learning_rate=learning_rate['adult'],
                               file_result_to=file_result,
                               file_info_to=file_info,
                               cnt_values=25
                               )

    with open('appet_boot_results.txt', 'w') as file_result:
        with open('appet_boot_info.txt', 'w') as file_info:
            catboost_bootstrap('appet', 'train_full3.01', 'test3.01', 'train_full3.cd',
                               [
                                  CatboostWrapper,
                                  CatboostTimeWrapper,
                                  BayesWrapper,
                                  BayesTimeWrapper,
                                  FoolBayesWrapper,
                                  FoolBayesTimeWrapper
                               ],
                               learning_rate=learning_rate['appet'],
                               file_result_to=file_result,
                               file_info_to=file_info,
                               cnt_values=25
                               )