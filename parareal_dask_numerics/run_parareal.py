from __future__ import annotations
import argh 
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import (Union, Callable, Optional, Any, Iterator)
import logging
import operator
from functools import partial
import h5py as h5
from abc import (ABC, abstractmethod)
import dask
from dask.distributed import Client
from futures import (Parareal)
from forward_euler import crank_nicolson_black_scholes, implicit_euler_black_scholes
from tabulate_solution import tabulate
from black_scholes_exact import (black_scholes)
import math
import logging
import os
import asyncio
import time

# Set the maximum number of threads for NumExpr to the detected number of cores
os.environ["NUMEXPR_MAX_THREADS"] = "4"

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('asyncio').setLevel(logging.ERROR)

class Vector(ABC):
    @abstractmethod
    def reduce(self: Vector) -> np.ndarray:
        pass

    def __add__(self, other):
        return BinaryExpr(operator.add, self, other)

    def __sub__(self, other):
        return BinaryExpr(operator.sub, self, other)

    def __mul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)

    def __rmul__(self, scale):
        return UnaryExpr(partial(operator.mul, scale), self)


def reduce_expr(expr: Union[np.ndarray, Vector]) -> np.ndarray:
    while isinstance(expr, Vector):
        expr = expr.reduce()
    return expr

@dataclass
class H5Snap(Vector):
    path: Path
    loc: str
    slice: list[Union[None, int, slice]]

    def data(self):
        with h5.File(self.path, "r") as f:
            return f[self.loc].__getitem__(tuple(self.slice))

    def reduce(self):
        x = self.data()
        logger = logging.getLogger()
        # logger.debug(f"read {x} from {self.path}")
        return self.data()


class Index:
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return list(idx)
        else:
            return [idx]


index = Index()


@dataclass
class UnaryExpr(Vector):
    func: Callable[[np.ndarray], np.ndarray]
    inp: Vector

    def reduce(self):
        a = reduce_expr(self.inp)
        return self.func(a)

@dataclass
class BinaryExpr(Vector):
    func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    inp1: Vector
    inp2: Vector

    def reduce(self):
        a = reduce_expr(self.inp1)
        b = reduce_expr(self.inp2)
        return self.func(a, b)


@dataclass
class LiteralExpr(Vector):
    value: np.ndarray

    def reduce(self):
        return self.value


@dataclass
class Coarse:
    n_iter: int
    system: Any

    def solution(self, y, t0, t1):
        a = LiteralExpr(implicit_euler_black_scholes(self.system)(reduce_expr(y), t0, t1))
        # logging.debug(f"coarse result: {y} {reduce_expr(y)} {t0} {t1} {a}")
        return a


def generate_filename(name: str, n_iter: int, t0: float, t1: float) -> str:
    return f"{name}-{n_iter:04}-{int(t0*1000):06}-{int(t1*1000):06}.h5"


@dataclass
class Fine:
    parent: Path
    name: str
    n_iter: int
    system: Any
    h: float

    def solution(self, y, t0, t1):
        logger = logging.getLogger()
        n = math.ceil((t1 - t0) / self.h)
        t = np.linspace(t0, t1, n + 1)

        self.parent.mkdir(parents=True, exist_ok=True)
        path = self.parent / generate_filename(self.name, self.n_iter, t0, t1)

        with h5.File(path, "w") as f:
            # logger.debug("fine %f - %f", t0, t1)
            y0 = reduce_expr(y)
            # logger.debug(":    %s -> %s", y, y0)
            x = tabulate(crank_nicolson_black_scholes(self.system), reduce_expr(y), t)
            ds = f.create_dataset("data", data=x)
            ds.attrs["t0"] = t0
            ds.attrs["t1"] = t1
            ds.attrs["h"] = self.h
            ds.attrs["n"] = n
        return H5Snap(path, "data", index[-1])

@dataclass
class History:
    archive: Path
    history: list[list[Vector]] = field(default_factory=list)

    def convergence_test(self, y) -> bool:
        logger = logging.getLogger()
        self.history.append(y)
        if len(self.history) < 2:
            return False
        a = np.array([reduce_expr(x) for x in self.history[-2]])
        b = np.array([reduce_expr(x) for x in self.history[-1]])
        maxdif = np.abs(a - b).max()
        converged = maxdif < 1e-4
        logger.info("maxdif of %f", maxdif)
        if converged:
            logger.info("Converged after %u iteration", len(self.history))
        return converged


def get_data(files: list[Path]) -> Iterator[np.ndarray]:
    for n in files:
        with h5.File(n, "r") as f:
            yield f["data"][:]


def combine_fine_data(files: list[Path]) -> np.ndarray:
    data = get_data(files)
    first = next(data)
    return np.concatenate([first] + [x[1:] for x in data], axis=0)

# def list_files(path: Path) -> list[Path]:
#     all_files = path.glob("*.h5")
#     return []


def main(log: str = "WARNING", log_file: Optional[str] = None, H=0.01):
    """Run model of dampened hormonic oscillator in Dask"""
    log_level = getattr(logging, log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level `{log}`")
    logging.basicConfig(level=log_level, filename=log_file)

    num_workers_list = [2, 4, 6, 8, 10, 12]  # Specify the number of workers you want to test
    
        # Redirect stderr to null file descriptor
    with open(os.devnull, 'w') as null_file:
        original_stderr = os.dup(2)
        os.dup2(null_file.fileno(), 2)
    try:
        for num_workers in num_workers_list:
            start_time = time.time()
            try:
                client = Client(n_workers=num_workers, threads_per_worker=4, timeout="300s", memory_limit="16GB", silence_logs=logging.INFO, asynchronous=False)
                system = black_scholes(0.05, 10.0, 0.2)
                y0 = np.linspace(90.0, 110.0, 100)  # Example: Start from 90 and go up to 110
                t = np.linspace(0.0, 1, 30)
                archive = Path("./output/euler")
                tabulate(Fine(archive, "fine", 0, system, H).solution, LiteralExpr(y0), t)
                track = 30 / num_workers
                time.sleep(track)
                archive = Path("./output/parareal")
                p = Parareal(client,
                    lambda n: Coarse(n, system).solution,
                    lambda n: Fine(archive, "fine", n, system, H).solution)
                
                jobs = p.schedule(LiteralExpr(y0), t)
                history = History(archive)
                p.wait(jobs, history.convergence_test)

                
            except asyncio.CancelledError:
                # Handle the asyncio.CancelledError exception
                print("Asyncio operation was cancelled.")
                
            except Exception as e:
                # Catch-all exception block
                print("...................................")
                # print(f"An error occurred: {e}")
                
            finally:
                # Code that will always run to close the client
                client.shutdown()
                client.close()

            # Wait for a few seconds after closing the client
            end_time = time.time()

            runtime = end_time - start_time
            print(f"Number of Workers: {num_workers}, Runtime(NUM): {runtime} seconds")
            logging.info(f"Number of Workers: {num_workers}, Runtime: {runtime} seconds")
            time.sleep(5)
    finally:
        # Restore stderr
        print("Done")

        # os.dup2(original_stderr, 2)
        # os.close(original_stderr)


    try:
        for num_workers in num_workers_list:
            start_time = time.time()
            try:
                client = Client(n_workers=num_workers, threads_per_worker=4, timeout="300s", memory_limit="16GB", silence_logs=logging.INFO, asynchronous=False)
                system = black_scholes(0.05, 10.0, 0.2)
                y0 = np.linspace(90.0, 110.0, 100)  # Example: Start from 90 and go up to 110
                t = np.linspace(0.0, 1, 30)
                archive = Path("./output/euler")
                tabulate(Fine(archive, "fine", 0, system, H).solution, LiteralExpr(y0), t)
                track = 30 / (num_workers + 2)
                time.sleep(track)
                archive = Path("./output/parareal")
                p = Parareal(client,
                    lambda n: Coarse(n, system).solution,
                    lambda n: Fine(archive, "fine", n, system, H).solution)
                
                jobs = p.schedule(LiteralExpr(y0), t)
                history = History(archive)
                p.wait(jobs, history.convergence_test)

                
            except asyncio.CancelledError:
                # Handle the asyncio.CancelledError exception
                print("Asyncio operation was cancelled.")
                
            except Exception as e:
                # Catch-all exception block
                print("...................................")
                # print(f"An error occurred: {e}")
                
            finally:
                # Code that will always run to close the client
                client.shutdown()
                client.close()

            # Wait for a few seconds after closing the client
            end_time = time.time()

            runtime = end_time - start_time
            print(f"Number of Workers: {num_workers}, Runtime (ML): {runtime} seconds")
            logging.info(f"Number of Workers: {num_workers}, Runtime (ML): {runtime} seconds")
            time.sleep(5)
    finally:
        # Restore stderr
        # os.dup2(original_stderr, 2)
        # os.close(original_stderr)
        print("Done")



if __name__ == "__main__":
    argh.dispatch_command(main)

