import numpy as np


def grad_desc(f, df, learning_rate, iteration, w_0):
    w_curr = w_0
    for i in range(iteration):
        g = df(w_curr)
        w_curr = w_curr - learning_rate * g
    return w_curr


def momentum(f, df, learning_rate, iteration, w_0, alpha=1/1000):
    w_curr = w_0
    exp = None
    for i in range(iteration):
        g = df(w_curr)
        if exp is None:
            exp = alpha * g
        else:
            exp = (1 - alpha) * g + alpha * exp
        w_curr = w_curr - learning_rate*exp
    return w_curr


def newton_method(f, df, ddf, learning_rate, iteration, w_0, eps=10**-6):
    w_curr = w_0
    for i in range(iteration):
        g = df(w_curr)
        hess = ddf(w_curr)
        hess_stable = hess + np.eye(len(w_0)) * eps
        step = np.linalg.inv(hess_stable) @ g
        w_curr = w_curr - learning_rate * step
    return w_curr

