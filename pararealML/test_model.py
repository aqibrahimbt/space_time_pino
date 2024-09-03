import numpy as np
import tensorflow as tf

import numpy as np

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.physics_informed import *
# from pararealml.operators.parareal import *
# from pararealml.utils.time import mpi_time



diff_eq = BlackScholesEquation()
mesh = Mesh([(0.0, 10.0)], [0.1])

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

f = FDMOperator(RK4(), CrankNicolsonMethod(), 0.001)
g = FDMOperator(RK4(), CrankNicolsonMethod(), 0.01)


import pickle
# Load the model
loaded_model = tf.keras.models.load_model("dump/model")

# Load the PhysicsInformedMLOperator instance
with open("dump/instance", "rb") as f:
    loaded_piml = pickle.load(f)

# Now you can use the loaded model and PhysicsInformedMLOperator instance
# For example, you can call the solve method on the loaded PhysicsInformedMLOperator instance
# result = loaded_piml.solve(cp, t_interval)

# loaded_model = tf.keras.models.load_model("black_1d_pidon", custom_objects={
#                                           "PhysicsInformedMLOperator": PhysicsInformedMLOperator})
# print(loaded_model.summary())

# loaded_model.solve(ivp)
