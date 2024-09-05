import numpy as np
import tensorflow as tf
from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.physics_informed import *
from pararealml.operators.parareal import *
# from pararealml.utils.time import mpi_time
import pickle
from keras import layers

#TODO: Not the best way to do this, but it works for now. To simplify later and refacor to make it more general


diff_eq = MultiDimensionalBlackScholesEquation()
mesh = Mesh([(0.0, 10.0), (0.0, 10.0)], [0.1, 0.1])  # 2D mesh
bcs = [
    (
        NeumannBoundaryCondition(lambda x, t: np.zeros((len(x), 1))),
        DirichletBoundaryCondition(lambda x, t: np.full((len(x), 1), 100)),
    )
] * 2  # Boundary conditions for both dimensions

cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0.0, 1.0)
fdm = FDMOperator(RK4(), CrankNicolsonMethod(), 0.0001)
sampler = UniformRandomCollocationPointSampler()
piml = PhysicsInformedMLOperator(sampler, 0.001, True)

training_y_0_functions = [
    MarginalBetaProductInitialCondition(cp, [[(p, p), (q, q)]]).y_0
    for p in np.arange(1.2, 4.0, 0.4)
    for q in np.arange(1.2, 4.0, 0.4)
]

fno_model = FNO2D(modes1=16, modes2=16, width=64)
print(fno_model)
# print(training_y_0_functions)
# piml.train(
#     cp,
#     t_interval,
#     training_data_args=DataArgs(
#         y_0_functions=training_y_0_functions,
#         n_domain_points=1000,  # Increased for 2D
#         n_boundary_points=200,  # Increased for 2D
#         n_batches=1,
#     ),
#     model_args=ModelArgs(
#         model=fno_model,
#         ic_loss_weight=10.0,
#     ),
#     optimization_args=OptimizationArgs(
#         optimizer=tf.optimizers.Adam(
#             learning_rate=tf.optimizers.schedules.ExponentialDecay(
#                 2e-3, decay_steps=25, decay_rate=0.98
#             )
#         ),
#         epochs=20, 
#     ),
# )

# # Define space decomposition and composition functions
# def space_decomposition(domain, n_subdomains):
#     x_split = np.array_split(domain[0], n_subdomains[0])
#     y_split = np.array_split(domain[1], n_subdomains[1])
#     return [np.meshgrid(x, y) for x in x_split for y in y_split]

# def space_composition(subdomains):
#     return np.concatenate([np.concatenate(sd, axis=1) for sd in subdomains], axis=0)


# # Save the model and PhysicsInformedMLOperator
# fno_model.save("dump/fno2d_model")
# with open("dump/fno2d_instance", "wb") as f:
#     pickle.dump(piml, f)

# ic = GaussianInitialCondition(
#     cp, [(np.array([5.0, 5.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))], [5.0]
# )
# ivp = InitialValueProblem(cp, (0.0, 1.0), ic)
# f = FDMOperator(RK4(), CrankNicolsonMethod(), 0.001)
# g = FDMOperator(RK4(), CrankNicolsonMethod(), 0.01)
# g_star = piml
# # Create SpaceTimePararealOperator
# p = SpaceTimePararealOperator(
#     f, 
#     g_star, 
#     space_decomposition=lambda domain: space_decomposition(domain, (2, 2)),  # Decompose into 2x2 grid
#     space_composition=space_composition,
#     termination_condition=0.0025,
#     max_iterations=10
# )

# # Solve using different methods and measure time
# mpi_time("fine")(f.solve)(ivp)
# mpi_time("coarse")(g.solve)(ivp)
# mpi_time("coarse_ml")(piml.solve)(ivp)
# mpi_time("parareal")(p.solve)(ivp)

# # Generate and save plots for different initial conditions
# for p_val in [2.0, 3.5, 5.0]:
#     for q_val in [2.0, 3.5, 5.0]:
#         ic = MarginalBetaProductInitialCondition(cp, [[(p_val, p_val), (q_val, q_val)]])
#         ivp = InitialValueProblem(cp, t_interval, ic)
        
#         # FDM solution
#         fdm_solution = f.solve(ivp)
#         for i, plot in enumerate(fdm_solution.generate_plots()):
#             plot.save(f"diff_2d_fdm_{p_val:.1f}_{q_val:.1f}_{i}").close()
        
#         # PIML (FNO) solution
#         piml_solution = piml.solve(ivp)
#         for i, plot in enumerate(piml_solution.generate_plots()):
#             plot.save(f"diff_2d_fno_{p_val:.1f}_{q_val:.1f}_{i}").close()
        
#         # Space-Time Parareal solution
#         parareal_solution = p.solve(ivp)
#         for i, plot in enumerate(parareal_solution.generate_plots()):
#             plot.save(f"diff_2d_parareal_{p_val:.1f}_{q_val:.1f}_{i}").close()
