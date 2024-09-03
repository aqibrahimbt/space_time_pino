from abstract import (Solution, Mapping)
import numpy as np


def identity(x):
    return x


def parareal(coarse: Solution, fine: Solution, c2f: Mapping = identity, f2c: Mapping = identity):
    def f(y, t):
        m = t.size
        y_n = [None] * m
        y_n[0] = y[0]
        for i in range(1, m):
            y_n[i] = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i])) \
                + fine(y[i-1], t[i-1], t[i]) \
                - c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
        return y_n
    return f


def parareal_np(coarse: Solution, fine: Solution, c2f: Mapping = identity, f2c: Mapping = identity):
    def f(y, t):
        m = t.size
        y_n = np.zeros_like(y)
        y_n[0] = y[0]
        for i in range(1, m):
            y_n[i] = c2f(coarse(f2c(y_n[i-1]), t[i-1], t[i])) \
                + fine(y[i-1], t[i-1], t[i]) \
                - c2f(coarse(f2c(y[i-1]), t[i-1], t[i]))
        return y_n
    return f
