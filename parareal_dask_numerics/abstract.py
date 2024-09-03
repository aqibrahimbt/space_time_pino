from __future__ import annotations
from typing import Protocol
from typing import Protocol
from typing import (Callable, Protocol, TypeVar, Union)

TVector = TypeVar("TVector", bound="Vector")


class Vector(Protocol):
    def __add__(self: TVector, other: TVector) -> TVector:
        ...

    def __sub__(self: TVector, other: TVector) -> TVector:
        ...

    def __mul__(self: TVector, other: float) -> TVector:
        ...

    def __rmul__(self: TVector, other: float) -> TVector:
        ...


Mapping = Callable[[TVector], TVector]
Problem = Callable[[TVector, float], TVector]
Solution = Union[Callable[[TVector, float, float], TVector],Callable[..., TVector]]
