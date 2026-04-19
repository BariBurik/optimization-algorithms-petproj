import numpy as np

def f(w):
    x = w[0]
    y = w[1]
    return x**2 + y**2 - 2*x*y**2

def df(w):
    x = w[0]
    y = w[1]
    x = 2*x - 2*y**2
    y = 2*y - 4*x*y
    return np.array([x, y])

def ddf(w):
    x = w[0]
    y = w[1]
    x_x = 2
    x_y = - 4 * y
    y_y = 2 - 4 * x
    y_x = - 4 * y
    return np.array([[x_x, x_y], [y_x, y_y]])