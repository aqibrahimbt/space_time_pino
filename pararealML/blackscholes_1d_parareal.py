import numpy as np

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.parareal import *
from pararealml.utils.time import mpi_time


diff_eq = BlackScholesEquation()
mesh = Mesh([(0.0, 100.0)], [0.1])

# Dirichlet boundary conditions
lower_boundary_condition = DirichletBoundaryCondition(
    lambda x, t: np.zeros((len(x), 1)), is_static=True
)

upper_boundary_condition = DirichletBoundaryCondition(
    lambda x, t: np.full((len(x), 1), 100), is_static=True
)

# Neumann boundary conditions
left_boundary_condition = NeumannBoundaryCondition(
    lambda x, t: np.zeros((len(x), 1)), is_static=True
)

right_boundary_condition = NeumannBoundaryCondition(
    lambda x, t: np.zeros((len(x), 1)), is_static=True
)

# bcs = [
#     (lower_boundary_condition, upper_boundary_condition),
#     (left_boundary_condition, right_boundary_condition),
# ]

bcs = [
    (
        NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        DirichletBoundaryCondition(lambda x, t: np.full((len(x), 1),100)),
    )
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
ic = GaussianInitialCondition(
    cp, [(np.array([5.0]), np.array([[0.5]]))], [5.0]
)

ivp = InitialValueProblem(cp, (0.0, 1.0), ic)

f = FDMOperator(RK4(), CrankNicolsonMethod(), 0.0001)
g = FDMOperator(RK4(), CrankNicolsonMethod(), 0.01)
p = PararealOperator(f, g, 0.0025)

f_solution = f.solve(ivp)
g_solution = g.solve(ivp)


mpi_time("f")(f.solve)(ivp)
mpi_time("g")(g.solve)(ivp)
mpi_time("parareal")(p.solve)(ivp)
