from typing import Optional
import numpy as np
import tensorflow as tf

from pararealml import *
from pararealml.operators.fdm import *
from pararealml.operators.ml.physics_informed import *

diff_eq = DiffusionEquation(1, 0.2)
mesh = Mesh([(0.0, 1.0)], (0.1,))
bcs = [
    (
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
        NeumannBoundaryCondition(
            lambda x, t: np.zeros((len(x), 1)), is_static=True
        ),
    ),
]
cp = ConstrainedProblem(diff_eq, mesh, bcs)
t_interval = (0.0, 0.5)

fdm = FDMOperator(RK4(), CrankNicolsonMethod(), 0.0001
)

sampler = UniformRandomCollocationPointSampler()
piml = PhysicsInformedMLOperator(sampler, 0.001, True)
training_y_0_functions = [
    MarginalBetaProductInitialCondition(cp, [[(p, p)]]).y_0
    for p in np.arange(1.2, 6.0, 0.2)
]


    
def calculate_output_shape(input_shape, fft_length):
    """
    Calculate the output shape of a 2D Fourier transform.
    
    :param input_shape: shape of the input tensor (batch_size, height, width)
    :param fft_length: the desired size for the Fourier transform (num_modes, num_modes)
    :return: the shape after the Fourier transform
    """
    # Unpack the input shape (batch_size, height, width)
    batch_size, height, width = input_shape
    
    # Calculate the output shape of rfft2d
    # For real FFT, the last dimension is halved + 1 (because it's a real-to-complex transformation)
    output_height = fft_length[0]
    output_width = fft_length[1] // 2 + 1  # Real FFT halves the width and adds 1
    
    # Return the new shape: (batch_size, output_height, output_width)
    return (batch_size, output_height, output_width)

class FNO(tf.keras.Model):

    def __init__(
        self,
        trunk_net: tf.keras.Model,
        combiner_net: tf.keras.Model,
        input_size: int,
        num_modes: int,
    ):

        super(FNO, self).__init__()
        self._trunk_net = trunk_net
        self._combiner_net = combiner_net
        self._input_size = input_size
        self._num_modes = num_modes

    @property
    def trunk_net(self) -> tf.keras.Model:
        """
        The model's trunk net that processes the Fourier coefficients.
        """
        return self._trunk_net

    @property
    def combiner_net(self) -> tf.keras.Model:
        """
        The model's combiner net that combines the outputs of the branch and trunk nets.
        """
        return self._combiner_net


    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            mask: Optional[tf.Tensor] = None,
        ) -> tf.Tensor:
        
        # Check if inputs are 2D (batch_size, input_size), and reshape accordingly
        if len(inputs.shape) == 2:  # If it's 2D
            batch_size = tf.shape(inputs)[0]  # Dynamically get the batch size at runtime
            input_size = tf.shape(inputs)[1]  # Dynamically get the input size
            
            # Calculate height and width based on input_size (assuming input_size is a perfect square)
            height = tf.cast(tf.math.sqrt(tf.cast(input_size, tf.float32)), tf.int32)
            width = height  # For square-shaped inputs
            
            # Reshape inputs to (batch_size, height, width)
            inputs = tf.reshape(inputs, (batch_size, height, width))

        # Perform the Fourier transform on the reshaped input data
        output_shape = calculate_output_shape(tf.shape(inputs), [self._num_modes, self._num_modes])
        
        x = tf.keras.layers.Lambda(
            lambda inputs: tf.signal.rfft2d(inputs, fft_length=[self._num_modes, self._num_modes]),
            output_shape=output_shape
        )(inputs)

        # Process the Fourier coefficients using the trunk net
        trunk_output = self._trunk_net(x, training=training, mask=mask)

        # Combine the outputs of the branch and trunk nets using the combiner net
        combiner_input = tf.concat([x, trunk_output], axis=1)
        
        return self._combiner_net(combiner_input, training=training, mask=mask)



# Example usage:
num_modes = 10  # Define the number of Fourier modes
fno = FNO(
    trunk_net=tf.keras.Sequential(
        [tf.keras.layers.InputLayer((2 * num_modes,))] + [tf.keras.layers.Dense(50, activation="tanh") for _ in range(8)]
    ),
    combiner_net=tf.keras.Sequential(
        [tf.keras.layers.InputLayer((100,)), tf.keras.layers.Dense(diff_eq.y_dimension)]
    ),
    input_size=np.prod(cp.y_vertices_shape).item(),
    num_modes=num_modes,
)

piml.train(
    cp,
    t_interval,
    training_data_args=DataArgs(
        y_0_functions=training_y_0_functions,
        n_domain_points=500,
        n_boundary_points=100,
        n_batches=1,
    ),
    model_args=ModelArgs(
        model=fno,
        ic_loss_weight=10.0,
    ),
    optimization_args=OptimizationArgs(
        optimizer=tf.optimizers.Adam(
            learning_rate=tf.optimizers.schedules.ExponentialDecay(
                2e-3, decay_steps=25, decay_rate=0.98
            )
        ),
        epochs=5000,
    ),
)

# for p in [2.0, 3.5, 5.0]:
#     ic = MarginalBetaProductInitialCondition(cp, [[(p, p)]])
#     ivp = InitialValueProblem(cp, t_interval, ic)

#     fdm_solution = fdm.solve(ivp)
#     for i, plot in enumerate(fdm_solution.generate_plots()):
#         plot.save("diff_1d_fdm_{:.1f}_{}".format(p, i)).close()

#     piml_solution = piml.solve(ivp)
#     for i, plot in enumerate(piml_solution.generate_plots()):
#         plot.save("diff_1d_pidon_{:.1f}_{}".format(p, i)).close()