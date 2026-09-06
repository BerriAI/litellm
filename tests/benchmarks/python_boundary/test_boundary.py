from collections.abc import Callable, Iterator
from typing import Final, Protocol

import pytest
from cases import Case, boundary_cases, payloads


class Benchmark(Protocol):
    def __call__(self, function: Callable[[], object]) -> object: ...


@pytest.fixture(scope="session")
def cases() -> Iterator[tuple[Case, ...]]:
    with boundary_cases() as values:
        yield values


SIMULATION_IDS: Final = tuple(
    f"{operation}/{label}" for label, _ in payloads() for operation in ("decode", "encode", "roundtrip")
) + ("typed_response",)
WALLTIME_IDS: Final = tuple(f"{operation}/{label}" for label, _ in payloads() for operation in ("sync", "async")) + (
    "gil_roundtrip",
    "sync_threads/4",
    "async_concurrent/8",
    "async_concurrent/32",
)


@pytest.mark.parametrize("name", SIMULATION_IDS)
def test_conversion(benchmark: Benchmark, cases: tuple[Case, ...], name: str) -> None:
    case: Final = next(case for case in cases if case.name == name)
    assert benchmark(case.call) == case.expected


@pytest.mark.parametrize("name", WALLTIME_IDS)
def test_execution(benchmark: Benchmark, cases: tuple[Case, ...], name: str) -> None:
    case: Final = next(case for case in cases if case.name == name)
    assert benchmark(case.call) == case.expected
