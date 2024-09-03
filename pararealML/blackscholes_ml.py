import numpy as np
import tensorflow as tf
from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.physics_informed import *
from pararealml.operators.parareal import *
from pararealml.utils.time import mpi_time
import pickle

#TODO: Not the best way to do this, but it works for now. To simplify later and refacor to make it more general

class FNO2D(tf.keras.Model):
    def __init__(self, modes1, modes2, width):
        super(FNO2D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fc0 = tf.keras.layers.Dense(self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = tf.keras.layers.Dense(self.width)
        self.w1 = tf.keras.layers.Dense(self.width)
        self.w2 = tf.keras.layers.Dense(self.width)
        self.w3 = tf.keras.layers.Dense(self.width)
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc0(x)
        x = tf.transpose(x, perm=[0, 2, 3, 1])  # (batch, x, y, channel)
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = tf.nn.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = tf.transpose(x, perm=[0, 3, 1, 2])  # (batch, channel, x, y)
        x = self.fc1(x)
        x = tf.nn.gelu(x)
        x = self.fc2(x)
        return x

class SpectralConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = self.add_weight(shape=(in_channels, out_channels, self.modes1, self.modes2, 2),
                                        initializer=tf.initializers.RandomNormal(0.0, self.scale),
                                        trainable=True)
        self.weights2 = self.add_weight(shape=(in_channels, out_channels, self.modes1, self.modes2, 2),
                                        initializer=tf.initializers.RandomNormal(0.0, self.scale),
                                        trainable=True)

    def call(self, x):
        batchsize = tf.shape(x)[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = tf.signal.rfft2d(x)

        # Multiply relevant Fourier modes
        out_ft = tf.zeros(tf.concat([[batchsize, self.out_channels], tf.shape(x_ft)[2:]], axis=0), dtype=tf.complex64)
        out_ft = tf.tensor_scatter_nd_update(
            out_ft, 
            tf.constant([[i, j] for i in range(min(self.modes1, tf.shape(out_ft)[2])) 
                               for j in range(min(self.modes2, tf.shape(out_ft)[3]))]), 
            tf.einsum("bixy,ioxy->boxy", 
                      tf.cast(x_ft[:, :, :self.modes1, :self.modes2], dtype=tf.complex64),
                      tf.complex(self.weights1[:, :, :, :, 0], self.weights1[:, :, :, :, 1]))
        )

        # Return to physical space
        x = tf.signal.irfft2d(out_ft, fft_length=tf.shape(x)[1:3])
        return x

diff_eq = BlackScholesEquation()
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

piml.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=1000,  # Increased for 2D
        n_boundary_points=200,  # Increased for 2D
        n_batches=1,
    ),
    model_args=ModelArgs(
        model=fno_model,
        ic_loss_weight=10.0,
    ),
    optimization_args=OptimizationArgs(
        optimizer=tf.optimizers.Adam(
            learning_rate=tf.optimizers.schedules.ExponentialDecay(
                2e-3, decay_steps=25, decay_rate=0.98
            )
        ),
        epochs=20, 
    ),
)

# Save the model and PhysicsInformedMLOperator
fno_model.save("dump/fno2d_model")
with open("dump/fno2d_instance", "wb") as f:
    pickle.dump(piml, f)

ic = GaussianInitialCondition(
    cp, [(np.array([5.0, 5.0]), np.array([[0.5, 0.0], [0.0, 0.5]]))], [5.0]
)
ivp = InitialValueProblem(cp, (0.0, 1.0), ic)
f = FDMOperator(RK4(), CrankNicolsonMethod(), 0.001)
g = FDMOperator(RK4(), CrankNicolsonMethod(), 0.01)
g_star = piml
# p = PararealOperator(f, g_star, 0.0025) # time only
p = SpaceTimePararealOperator(f, g_star, 0.0025, 0.0025, 0.0025) # space-time

mpi_time("fine")(f.solve)(ivp)
mpi_time("coarse")(g.solve)(ivp)
mpi_time("coarse_ml")(piml.solve)(ivp)
mpi_time("parareal")(p.solve)(ivp)

for p in [2.0, 3.5, 5.0]:
    for q in [2.0, 3.5, 5.0]:
        ic = MarginalBetaProductInitialCondition(cp, [[(p, p), (q, q)]])
        ivp = InitialValueProblem(cp, t_interval, ic)
        fdm_solution = fdm.solve(ivp)
        for i, plot in enumerate(fdm_solution.generate_plots()):
            plot.save(f"diff_2d_fdm_{p:.1f}_{q:.1f}_{i}").close()
        piml_solution = piml.solve(ivp)
        for i, plot in enumerate(piml_solution.generate_plots()):
            plot.save(f"diff_2d_fno_{p:.1f}_{q:.1f}_{i}").close()