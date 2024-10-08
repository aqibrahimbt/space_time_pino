from __future__ import annotations

from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf

from pararealml.constrained_problem import ConstrainedProblem
from pararealml.initial_condition import VectorizedInitialConditionFunction
from pararealml.initial_value_problem import (
    InitialValueProblem,
    TemporalDomainInterval,
)
from pararealml.operator import Operator, discretize_time_domain
from pararealml.operators.ml.physics_informed.collocation_point_sampler import (  # noqa: 501
    CollocationPointSampler,
)
from pararealml.operators.ml.physics_informed.dataset import Dataset
from pararealml.operators.ml.physics_informed.physics_informed_regressor import (  # noqa: 501
    PhysicsInformedRegressor,
)
from pararealml.solution import Solution


class PhysicsInformedMLOperator(Operator):

    def __init__(
        self,
        sampler: CollocationPointSampler,
        d_t: float,
        vertex_oriented: bool,
        auto_regressive: bool = False,
    ):
        super(PhysicsInformedMLOperator, self).__init__(d_t, vertex_oriented)
        self._sampler = sampler
        self._auto_regressive = auto_regressive
        self._model: Optional[PhysicsInformedRegressor] = None

    @property
    def auto_regressive(self) -> bool:
        return self._auto_regressive

    @property
    def model(self) -> Optional[PhysicsInformedRegressor]:
        return self._model

    @model.setter
    def model(self, model: Optional[PhysicsInformedRegressor]):
        self._model = model

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        cp = ivp.constrained_problem
        diff_eq = cp.differential_equation

        t = discretize_time_domain(ivp.t_interval, self._d_t)[1:]

        if diff_eq.x_dimension:
            x = cp.mesh.all_index_coordinates(
                self._vertex_oriented, flatten=True
            )
            x_tensor = tf.convert_to_tensor(x, tf.float32)
            u = ivp.initial_condition.y_0(x).reshape((1, -1))
            u_tensor = tf.tile(
                tf.convert_to_tensor(u, tf.float32), (x.shape[0], 1)
            )
        else:
            x_tensor = None
            u = np.array([ivp.initial_condition.y_0(None)])
            u_tensor = tf.convert_to_tensor(u, tf.float32)

        t_tensor = tf.constant(
            self._d_t if self._auto_regressive else t[0],
            dtype=tf.float32,
            shape=(u_tensor.shape[0], 1),
        )

        y_shape = cp.y_shape(self._vertex_oriented)
        y = np.empty((len(t),) + y_shape)

        for i, t_i in enumerate(t):
            y_i_tensor = self._infer((u_tensor, t_tensor, x_tensor))
            y[i, ...] = y_i_tensor.numpy().reshape(y_shape)

            if i < len(t) - 1:
                if self._auto_regressive:
                    u_tensor = (
                        tf.tile(
                            tf.reshape(y_i_tensor, (1, -1)),
                            (x_tensor.shape[0], 1),
                        )
                        if diff_eq.x_dimension
                        else tf.reshape(y_i_tensor, u_tensor.shape)
                    )
                else:
                    t_tensor = tf.constant(
                        t[i + 1],
                        dtype=tf.float32,
                        shape=(u_tensor.shape[0], 1),
                    )

        return Solution(
            ivp, t, y, vertex_oriented=self._vertex_oriented, d_t=self._d_t
        )

    def train(
        self,
        cp: ConstrainedProblem,
        t_interval: TemporalDomainInterval,
        training_data_args: DataArgs,
        optimization_args: OptimizationArgs,
        model_args: Optional[ModelArgs] = None,
        validation_data_args: Optional[DataArgs] = None,
        test_data_args: Optional[DataArgs] = None,
    ) -> Tuple[tf.keras.callbacks.History, Optional[Sequence[float]]]:

        if model_args is None and self._model is None:
            raise ValueError(
                "the model arguments cannot be None if the operator's model "
                "is None"
            )
        
        # print("training_data_args: ", training_data_args)

        if self._auto_regressive:
            if t_interval != (0.0, self._d_t):
                raise ValueError(
                    "in auto-regressive mode, the training time interval "
                    f"{t_interval} must range from 0 to the time step size of "
                    f"the operator ({self._d_t})"
                )

            diff_eq = cp.differential_equation
            t_symbol = diff_eq.symbols.t
            eq_sys = diff_eq.symbolic_equation_system
            if any([t_symbol in rhs.free_symbols for rhs in eq_sys.rhs]):
                raise ValueError(
                    "auto-regressive mode is not compatible with differential "
                    "equations whose right-hand sides contain any t terms"
                )

            if (
                diff_eq.x_dimension
                and not cp.are_all_boundary_conditions_static
            ):
                raise ValueError(
                    "auto-regressive mode is not compatible with dynamic "
                    "boundary conditions"
                )

        training_dataset = self._create_dataset(
            cp, t_interval, training_data_args
        )
        validation_dataset = self._create_dataset(
            cp, t_interval, validation_data_args
        )
        test_dataset = self._create_dataset(cp, t_interval, test_data_args)
        # print("training_dataset: ", training_dataset)

        model = (
            self._model
            if model_args is None
            else PhysicsInformedRegressor(
                cp=cp,
                model=model_args.model,
                diff_eq_loss_weight=model_args.diff_eq_loss_weight,
                ic_loss_weight=model_args.ic_loss_weight,
                bc_loss_weight=model_args.bc_loss_weight,
                vertex_oriented=self._vertex_oriented,
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.get(optimization_args.optimizer)
        )
        history = model.fit(
            training_dataset,
            epochs=optimization_args.epochs,
            steps_per_epoch=training_data_args.n_batches,
            validation_data=validation_dataset,
            validation_steps=validation_data_args.n_batches
            if validation_data_args
            else None,
            validation_freq=optimization_args.validation_frequency,
            callbacks=optimization_args.callbacks,
            verbose=optimization_args.verbose,
        )

        test_loss = (
            model.evaluate(
                test_dataset,
                steps=test_data_args.n_batches,
                verbose=optimization_args.verbose,
            )
            if test_dataset
            else None
        )

        self._model = model

        return history, test_loss

    @tf.function
    def _infer(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]
    ) -> tf.Tensor:
        return self.model.__call__(inputs)

    def _create_dataset(
        self,
        cp: ConstrainedProblem,
        t_interval: Tuple[float, float],
        data_args: Optional[DataArgs],
    ) -> Optional[Generator[Sequence[Sequence[tf.Tensor]], None, None]]:

        if not data_args:
            return None

        dataset = Dataset(
            cp=cp,
            t_interval=t_interval,
            y_0_functions=data_args.y_0_functions,
            point_sampler=self._sampler,
            n_domain_points=data_args.n_domain_points,
            n_boundary_points=data_args.n_boundary_points,
            vertex_oriented=self._vertex_oriented,
        )
        iterator = dataset.get_iterator(
            n_batches=data_args.n_batches,
            n_ic_repeats=data_args.n_ic_repeats,
            shuffle=data_args.shuffle,
        )
        return iterator.to_infinite_generator()

    def get_config(self):
        config = {
            'model': tf.keras.utils.serialize_keras_object(self._model),
            'cp': tf.keras.utils.serialize_keras_object(self._cp),
            'diff_eq_loss_weight': self._diff_eq_loss_weights,
            'ic_loss_weight': self._ic_loss_weights,
            'bc_loss_weight': self._bc_loss_weights,
        }
        base_config = super(PhysicsInformedMLOperator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config['model'])
        cp = tf.keras.utils.deserialize_keras_object(config['cp'])
        diff_eq_loss_weight = config['diff_eq_loss_weight']
        ic_loss_weight = config['ic_loss_weight']
        bc_loss_weight = config['bc_loss_weight']

        return cls(
            model=model,
            cp=cp,
            diff_eq_loss_weight=diff_eq_loss_weight,
            ic_loss_weight=ic_loss_weight,
            bc_loss_weight=bc_loss_weight,
        )



class DataArgs(NamedTuple):

    y_0_functions: Iterable[VectorizedInitialConditionFunction]
    n_domain_points: int
    n_batches: int
    n_boundary_points: int = 0
    n_ic_repeats: int = 1
    shuffle: bool = True


class ModelArgs(NamedTuple):
    model: tf.keras.Model
    diff_eq_loss_weight: Union[float, Sequence[float]] = 1.0
    ic_loss_weight: Union[float, Sequence[float]] = 1.0
    bc_loss_weight: Union[float, Sequence[float]] = 1.0


class OptimizationArgs(NamedTuple):
    
    optimizer: Union[str, Dict[str, Any], tf.optimizers.Optimizer]
    epochs: int
    validation_frequency: int = 1
    callbacks: Sequence[tf.keras.callbacks.Callback] = ()
    verbose: Union[str, int] = "auto"
