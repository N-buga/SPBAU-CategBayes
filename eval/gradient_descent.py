from functools import partial

import numpy as np
from numpy import linalg

from scipy.optimize import minimize
from scipy.special import polygamma
from scipy.special import gammaln


def _expect_score(alpha, beta, lens, cnt_1):
    m = lens.shape[0]
    score = gammaln(alpha + cnt_1).sum() + gammaln(lens - cnt_1 + beta).sum() - gammaln(lens + alpha + beta).sum() + \
            m * gammaln(alpha + beta) - m * gammaln(alpha) - m * gammaln(beta)
    return score


def _calc_der_alpha(alpha, beta, lens, cnt_1, koeff=1):
    m = lens.shape[0]
    der_alpha = (koeff*polygamma(0, alpha + cnt_1)).sum() - (koeff*polygamma(0, lens + alpha + beta)).sum() +\
    (koeff*np.full((m,), polygamma(0, alpha + beta))).sum() - (np.full((m,), polygamma(0, alpha))*koeff).sum()
    return der_alpha


def _calc_der_beta(alpha, beta, lens, cnt_1, koeff=1):
    m = lens.shape[0]
    der_beta = (koeff*polygamma(0, lens - cnt_1 + beta)).sum() - (koeff*polygamma(0, lens + alpha + beta)).sum() +\
    (koeff*np.full((m,), polygamma(0, alpha + beta))).sum() - (koeff*np.full((m,), polygamma(0, beta))).sum()
    return der_beta


def _calc_der_alpha_alpha(alpha, beta, lens, cnt_1, koeff=1):
    m = lens.shape[0]
    der_alpha_alpha = (koeff*polygamma(1, alpha + cnt_1)).sum() - (koeff*polygamma(1, lens + alpha + beta)).sum() +\
    (koeff*np.full((m,), polygamma(1, alpha + beta))).sum() - (koeff*np.full((m,), polygamma(1, alpha))).sum()
    return der_alpha_alpha


def _calc_der_alpha_beta(alpha, beta, lens, cnt_1, koeff=1):
    m = lens.shape[0]
    der_alpha_beta = - (koeff*polygamma(1, lens + alpha + beta)).sum() +\
    (koeff*np.full((m,), polygamma(1, alpha + beta))).sum()
    return der_alpha_beta


def _calc_der_beta_beta(alpha, beta, lens, cnt_1, koeff=1):
    m = lens.shape[0]
    der_beta_beta = (koeff*polygamma(1, lens - cnt_1 + beta)).sum() - (koeff*polygamma(1, lens + alpha + beta)).sum() +\
    (koeff*np.full((m,), polygamma(1, alpha + beta))).sum() - (koeff*np.full((m,), polygamma(1, beta))).sum()
    return der_beta_beta


def _calc_var(alpha, beta):
    return alpha*beta/((alpha + beta)*(alpha + beta)*(alpha + beta + 1))


def _calc_exp(alpha, beta):
    return alpha/(alpha + beta)


def get_approx_newton(lens, cnt_1, alpha_init, beta_init, min_border=0.5, max_border=150):
    optimize_result = minimize(fun=lambda x: -_expect_score(x[0], x[1], lens, cnt_1),
             x0=np.array([alpha_init, beta_init]),
             method='TNC',
             jac=lambda x: np.array([-_calc_der_alpha(x[0], x[1], lens, cnt_1), -_calc_der_beta(x[0], x[1], lens, cnt_1)]),
             hess=lambda x: np.array([[-_calc_der_alpha_alpha(x[0], x[1], lens, cnt_1), -_calc_der_alpha_beta(x[0], x[1], lens, cnt_1)],
                             [-_calc_der_alpha_beta(x[0], x[1], lens, cnt_1)], -_calc_der_beta_beta(x[0], x[1], lens, cnt_1)]),
             bounds=[(min_border, max_border), (min_border, max_border)],
            options={'iprint': 1, 'disp': True, 'maxiter': 20})

    alpha, beta = optimize_result.x[0], optimize_result.x[1]
    initial_score = _expect_score(alpha_init, beta_init, lens, cnt_1)
    print('Initial values: alpha={}\tbeta={}\tscore={}'.format(alpha_init, beta_init, initial_score))
    print('Optimization result: alpha={}\tbeta={}\tscore={}\tsuccess={}'.format(alpha, beta, optimize_result.fun, optimize_result.success))
    if -initial_score < optimize_result.fun:
        print('Use initial values')
        return alpha_init, beta_init, initial_score
    else:
        print('Use optimized values')
        return alpha, beta, _expect_score(alpha, beta, lens, cnt_1)


# def _make_step(alpha, beta, delta_alpha, delta_beta, score_func, min_step=0.0001):
#     init_score = score_func(alpha, beta)
#     new_alpha = alpha + delta_alpha
#     new_beta = beta + delta_beta
#     cur_score = score_func(new_alpha, new_beta)
#     i = 1
#     while cur_score <= init_score and max(delta_alpha/(i*i), delta_beta/(i*i)) >= min_step:
#         new_alpha = alpha + 1/(i*i)*delta_alpha
#         new_beta = beta + 1/(i*i)*delta_beta
#         cur_score = score_func(new_alpha, new_beta)
#         i += 1
#
#     if cur_score > init_score:
#         return new_alpha, new_beta
#     else:
#         return alpha, beta
#
#
# def get_approx_gd(lens, cnt_1, alpha_init, beta_init, steps_count=5000, init_step=1, max_val=150, eps=0.01):
#     score_func = partial(_expect_score, lens=lens, cnt_1=cnt_1)
#     alpha = alpha_init
#     beta = beta_init
#
#     beta_ans = beta
#     alpha_ans = alpha
#     max_score = score_func(alpha, beta)
#
#     for i in range(steps_count):
#         cur_score = score_func(alpha, beta)
#         # if i % 100 == 0:
#         #     print('step = {}'.format(i))
#         #     print('alpha = {}, beta = {}'.format(alpha, beta))
#         #     print('alpha_ans = {}, beta_ans= {}'.format(alpha_ans, beta_ans))
#         #     print('score: {}'.format(cur_score))
#         if cur_score > max_score:
#             max_score = cur_score
#             alpha_ans = alpha
#             beta_ans = beta
#
#         der_alpha = _calc_der_alpha(alpha, beta, lens, cnt_1)
#         der_beta = _calc_der_beta(alpha, beta, lens, cnt_1)
#
#         step = init_step
#
#         koeff = 1
#         if alpha + step * der_alpha <= eps:
#             koeff = abs((alpha / 2) / (step * der_alpha))
#
#         if beta + step * der_beta <= eps:
#             koeff = min(koeff, abs((beta / 2) / (step * der_beta)))
#
#         if alpha + step * der_alpha > max_val:
#             koeff = min(koeff, abs((max_val - alpha) / (step * der_alpha)))
#
#         if beta + step * der_beta > max_val:
#             koeff = min(koeff, abs((max_val - beta) / (step * der_beta)))
#
#         alpha, beta = _make_step(alpha, beta, koeff * step * der_alpha, koeff * step * der_beta,
#                                 score_func)  # alpha + max_step*der_alpha, beta + max_step*der_beta
#
#     return alpha_ans, beta_ans, max_score
#
#
# def get_approx_newton(lens, cnt_1, alpha_init, beta_init, step=1, max_val=150, eps=0.01):
#     score_func = partial(_expect_score, lens=lens, cnt_1=cnt_1)
#
#     alpha, beta, cur_score = get_approx_gd(lens, cnt_1, alpha_init, beta_init, 50)
#
#     beta_ans = beta
#     alpha_ans = alpha
#     max_score = cur_score
#
#     for i in range(10):
#         cur_score = score_func(alpha, beta)
#         # print(i)
#         # print('alpha = {}, beta = {}'.format(alpha, beta))
#         # print('score: {}'.format(cur_score))
#
#         der_alpha = _calc_der_alpha(alpha, beta, lens, cnt_1)
#         der_beta = _calc_der_beta(alpha, beta, lens, cnt_1)
#
#         matrix = np.array(
#             [[_calc_der_alpha_alpha(alpha, beta, lens, cnt_1), _calc_der_alpha_beta(alpha, beta, lens, cnt_1)],
#              [_calc_der_alpha_beta(alpha, beta, lens, cnt_1), _calc_der_beta_beta(alpha, beta, lens, cnt_1)]])
#
#         inv_matrix = linalg.inv(matrix)
#
#         koeff = 1
#         delta_alpha = - step * (inv_matrix[0, 0] * der_alpha + inv_matrix[0, 1] * der_beta)
#         delta_beta = - step * (inv_matrix[1, 0] * der_alpha + inv_matrix[1, 1] * der_beta)
#
#         if alpha + delta_alpha <= eps:
#             koeff = abs((alpha / 2) / (delta_alpha))
#
#         if beta + delta_beta <= eps:
#             koeff = min(koeff, abs((beta / 2) / (delta_beta)))
#
#         if alpha + delta_alpha > max_val:
#             koeff = min(koeff, abs((max_val - alpha) / (delta_alpha)))
#
#         if beta + delta_beta > max_val:
#             koeff = min(koeff, abs((max_val - beta) / (delta_beta)))
#
#         alpha, beta = _make_step(alpha, beta, koeff * delta_alpha, koeff * delta_beta, score_func)
#     return alpha, beta, cur_score
