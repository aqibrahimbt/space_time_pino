from abstract import (Vector, Problem, Solution)
import numpy as np
from scipy.optimize import fsolve

def forward_euler(f: Problem) -> Solution:
    """Forward-Euler solver."""
    def step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of Euler method."""
        return y + (t_1 - t_0) * f(y, t_0)
    return step
# ~\~ end


def implicit_euler_black_scholes(f: Problem) -> Solution:
    """Implicit Euler solver for Black-Scholes equation."""
    def f(u, t):
        r = 0.05
        K = 10.0
        S = np.linspace(90.0, 110.0, 100)
        C = u * np.ones_like(S)
        dS_dt = r * S - (C / S) * (S - K)
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])

    def step(u: Vector, t_0: float, t_1: float) -> Vector:
        def implicit_equation(u_new):
            result = np.zeros_like(u_new)
            f_values = f(u_new, t_1)
            for i in range(u_new.shape[0]):
                result[i] = u_new[i] - u[i] - (t_1 - t_0) * f_values[i, 1]
            return result # Flatten the result for fsolve

        u_new = fsolve(implicit_equation, u)
        return u_new

    return step


def crank_nicolson_black_scholes(f: Problem) -> Solution:
    """Crank-Nicolson solver for Black-Scholes equation."""
    def f(u, t):
        r = 0.05
        K = 10.0
        S = np.linspace(90.0, 110.0, 100)
        C = u * np.ones_like(S)
        dS_dt = r * S - (C / S) * (S - K)
        dC_dt = -r * C
        return np.column_stack([dS_dt, dC_dt])

    def step(u: Vector, t_0: float, t_1: float) -> Vector:
        def implicit_equation(u_new):
            result = np.zeros_like(u_new)
            f_values_old = f(u, t_0)
            f_values_new = f(u_new, t_1)
            for i in range(u_new.shape[0]):
                result[i] = u_new[i] - u[i] - 0.5 * (t_1 - t_0) * (f_values_old[i, 1] + f_values_new[i, 1])
            return result # Not flattening the result for fsolve

        u_new = fsolve(implicit_equation, u)
        return u_new

    return step