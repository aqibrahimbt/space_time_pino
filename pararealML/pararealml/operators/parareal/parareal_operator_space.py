import numpy as np
from mpi4py import MPI

from pararealML.pararealml.constrained_problem import ConstrainedProblem
from pararealML.pararealml.mesh import Mesh
from pararealml.initial_condition import DiscreteInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator, discretize_time_domain
from pararealml.solution import Solution
import numpy as np
from mpi4py import MPI
from typing import List, Union, Sequence, Callable
import sys

class SpaceTimePararealOperator(Operator):
    """
    A parallel-in-space-and-time differential equation solver framework based on
    an extended Parareal algorithm.
    """

    def __init__(
        self,
        f: Operator,
        g: Operator,
        space_decomposition: Callable[[np.ndarray], List[np.ndarray]],
        space_composition: Callable[[List[np.ndarray]], np.ndarray],
        termination_condition: Union[
            float, Sequence[float], Callable[[np.ndarray, np.ndarray], bool]
        ] = None,
        max_iterations: int = sys.maxsize,
    ):
        """
        :param f: the fine operator
        :param g: the coarse operator
        :param space_decomposition: function to decompose the spatial domain
        :param space_composition: function to compose the spatial sub-domains
        :param termination_condition: the termination condition
        :param max_iterations: the maximum number of iterations to perform
        """
        super(SpaceTimePararealOperator, self).__init__(f.d_t, f.vertex_oriented)

        self._f = f
        self._g = g
        self._space_decomposition = space_decomposition
        self._space_composition = space_composition
        self._termination_condition = termination_condition
        self._max_iterations = max_iterations

    def _should_terminate(
        self, old_y_end_points: np.ndarray, new_y_end_points: np.ndarray
    ) -> bool:
        if callable(self._termination_condition):
            return self._termination_condition(
                old_y_end_points, new_y_end_points
            )

        y_dim = old_y_end_points.shape[-1]

        if isinstance(self._termination_condition, Sequence):
            if len(self._termination_condition) != y_dim:
                raise ValueError(
                    "length of update tolerances "
                    f"({len(self._termination_condition)}) must match number "
                    f"of y dimensions ({y_dim})"
                )

            update_tolerances = np.array(self._termination_condition)

        else:
            update_tolerances = np.array([self._termination_condition] * y_dim)

        max_diff_norms = np.empty(y_dim)
        for y_ind in range(y_dim):
            max_diff_norm = 0.0
            for new_y_end_point, old_y_end_point in zip(
                new_y_end_points[..., y_ind], old_y_end_points[..., y_ind]
            ):
                max_diff_norm = np.maximum(
                    max_diff_norm,
                    np.sqrt(
                        np.square(new_y_end_point - old_y_end_point).mean()
                    ),
                )

            max_diff_norms[y_ind] = max_diff_norm

        return all(max_diff_norms < update_tolerances)

    def solve(
        self, ivp: InitialValueProblem, parallel_enabled: bool = True
    ) -> Solution:
        if not parallel_enabled:
            return self._f.solve(ivp)

        comm = MPI.COMM_WORLD
        size = comm.size
        rank = comm.rank

        # Determine the number of processes for time and space decomposition
        time_procs = int(np.sqrt(size))
        space_procs = size // time_procs

        # Create subcommunicators for time and space
        time_comm = comm.Split(rank // space_procs, rank % space_procs)
        space_comm = comm.Split(rank % space_procs, rank // space_procs)

        f = self._f
        g = self._g
        t_interval = ivp.t_interval
        delta_t = (t_interval[1] - t_interval[0]) / time_procs

        # Space decomposition
        full_domain = ivp.constrained_problem.mesh.vertices
        sub_domains = self._space_decomposition(full_domain)
        local_domain = sub_domains[space_comm.rank]

        # Create sub-IVP for the local space-time domain
        local_ivp = self._create_local_ivp(ivp, local_domain, time_comm.rank, delta_t)

        vertex_oriented = self._vertex_oriented
        y_shape = local_ivp.constrained_problem.y_shape(vertex_oriented)

        # Time slice computations
        time_slice_border_points = np.linspace(
            t_interval[0], t_interval[1], time_procs + 1
        )

        # Initial coarse solve
        y_coarse_end_points = self._coarse_solve(local_ivp, g, time_slice_border_points)

        y_border_points = np.concatenate(
            [
                local_ivp.initial_condition.discrete_y_0(vertex_oriented)[np.newaxis],
                y_coarse_end_points,
            ]
        )

        sub_y_fine = None
        corrections = np.empty((time_procs, *y_shape))

        for i in range(min(time_procs, self._max_iterations)):
            # Fine solve on local space-time domain
            sub_y_fine = self._fine_solve(local_ivp, f, time_comm.rank, y_border_points)

            # Compute and gather corrections
            correction = sub_y_fine[-1] - y_coarse_end_points[time_comm.rank]
            time_comm.Allgather([correction, MPI.DOUBLE], [corrections, MPI.DOUBLE])

            # Update solution
            old_y_end_points = np.copy(y_border_points[1:])
            self._update_solution(y_border_points, y_coarse_end_points, corrections, time_comm, i)

            # Check termination condition
            if self._should_terminate(old_y_end_points, y_border_points[1:]):
                break

        # Gather final solution across space
        t = discretize_time_domain(local_ivp.t_interval, f.d_t)[1:]
        local_y_fine = np.empty((len(t), *y_shape))
        sub_y_fine += y_border_points[time_comm.rank + 1] - sub_y_fine[-1]
        time_comm.Allgather([sub_y_fine, MPI.DOUBLE], [local_y_fine, MPI.DOUBLE])

        global_y_fine = self._space_composition(space_comm.allgather(local_y_fine))

        return Solution(
            ivp, t, global_y_fine, vertex_oriented=vertex_oriented, d_t=f.d_t
        )

    def _create_local_ivp(self, global_ivp, local_domain, time_rank, delta_t):
        # Create a local IVP for the given space-time subdomain
        global_cp = global_ivp.constrained_problem
        global_mesh = global_cp.mesh
        global_t_interval = global_ivp.t_interval

        # Create a new mesh for the local domain
        local_mesh = Mesh(local_domain, global_mesh.step_sizes)

        # Create a new constrained problem with the local mesh
        local_cp = ConstrainedProblem(global_cp.diff_eq, local_mesh, global_cp.bcs)

        # Calculate the local time interval
        local_t_start = global_t_interval[0] + time_rank * delta_t
        local_t_end = local_t_start + delta_t

        # Create the initial condition for the local IVP
        if time_rank == 0:
            # Use the global initial condition for the first time slice
            local_ic = global_ivp.initial_condition
        else:
            # For other time slices, we'll update this later in the algorithm
            local_ic = DiscreteInitialCondition(local_cp, np.zeros_like(local_domain), self.vertex_oriented)

        return InitialValueProblem(local_cp, (local_t_start, local_t_end), local_ic)

    def _coarse_solve(self, local_ivp, g, time_slice_border_points):
        # Perform coarse solve on the local space-time domain
        coarse_solution = g.solve(local_ivp)
        
        # Extract the solution at the time slice border points
        coarse_t = coarse_solution.t
        coarse_y = coarse_solution.y
        
        y_coarse_end_points = np.array([
            coarse_y[np.argmin(np.abs(coarse_t - t))]
            for t in time_slice_border_points[1:]  # Exclude the start point
        ])
        
        return y_coarse_end_points

    def _fine_solve(self, local_ivp, f, time_rank, y_border_points):
        # Perform fine solve on the local space-time domain
        
        # Update the initial condition for this time slice
        local_ivp.initial_condition = DiscreteInitialCondition(
            local_ivp.constrained_problem,
            y_border_points[time_rank],
            self.vertex_oriented
        )
        
        # Solve using the fine operator
        fine_solution = f.solve(local_ivp)
        
        return fine_solution.y

    def _update_solution(self, y_border_points, y_coarse_end_points, corrections, time_comm, iteration):
        # Update the solution based on corrections
        size = time_comm.size
        rank = time_comm.rank
        
        old_y_end_points = np.copy(y_border_points[1:])
        
        for j in range(iteration, size):
            if j > iteration:
                # Recompute coarse solution for future time slices
                local_ivp = self._create_local_ivp(
                    self.global_ivp,  # Assume this is stored as an attribute
                    self.local_domain,  # Assume this is stored as an attribute
                    j,
                    self.delta_t  # Assume this is stored as an attribute
                )
                local_ivp.initial_condition = DiscreteInitialCondition(
                    local_ivp.constrained_problem,
                    y_border_points[j],
                    self.vertex_oriented
                )
                y_coarse_end_points[j] = self._coarse_solve(
                    local_ivp, self._g, [local_ivp.t_interval[0], local_ivp.t_interval[1]]
                )[0]

            y_border_points[j + 1] = y_coarse_end_points[j] + corrections[j]
        
        # Broadcast the updated y_border_points to all processes in the time communicator
        time_comm.Bcast(y_border_points, root=0)
        
        return old_y_end_points