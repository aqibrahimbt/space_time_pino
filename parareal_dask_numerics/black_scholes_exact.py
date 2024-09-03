from abstract import (Problem)
from typing import Callable
from numpy.typing import NDArray
import numpy as np



def black_scholes(r: float, K: float, sigma: float) -> Problem:
    def f(u, t):
        S = np.linspace(90.0, 110.0, 100)
        C = u * np.ones_like(S)
        dS_dt = r * S - (C / S) * (S - K) - 0.5 * sigma**2 * S**2
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])
    return f


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from plotnine import ggplot, geom_line, aes

    from forward_euler import crank_nicolson_black_scholes, implicit_euler_black_scholes
    from iterate_solution import iterate_solution
    from tabulate_solution import tabulate_np

    H = 0.001
    r = 0.05
    K = 10.0
    # system = harmonic_oscillator(OMEGA0, ZETA)
    system = black_scholes(0.05, 10.0, 0.2)

    def coarse(y, t0, t1):
        # return forward_euler(system)(y, t0, t1)
        return implicit_euler_black_scholes(system)(y, t0, t1)
    
    # fine :: Solution[NDArray]
    def fine(y, t0, t1):
        return iterate_solution(implicit_euler_black_scholes(system), H)(y, t0, t1)

    y0 = np.linspace(90.0, 110.0, 100)  # Example: Start from 90 and go up to 110

    t = np.linspace(0.0, 1, 20)
    exact_result = np.zeros((len(t), 2))
    test = fine(y0, 0.0, 1.0)
    print(test.shape)
    test1 = coarse(y0, 0.0, 1.0)
    print(test1.shape)
    # euler_result = tabulate_np(fine, y0, t)
    # print(euler_result)

    # data = pd.DataFrame({
    #     "time": t,
    #     "exact_q": exact_result[:, 0],
    #     "exact_p": exact_result[:, 1],
    #     "euler_q": euler_result[:, 0],
    #     "euler_p": euler_result[:, 1]})

    # plot = ggplot(data) \
    #     + geom_line(aes("time", "exact_q")) \
    #     + geom_line(aes("time", "euler_q"), color="#000088")
    # plot.save("plot.svg")
