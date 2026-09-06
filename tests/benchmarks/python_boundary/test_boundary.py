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


PAYLOAD_IDS: Final = tuple(label for label, _ in payloads())


def check_case(benchmark: Benchmark, cases: tuple[Case, ...], name: str) -> None:
    case: Final = next(case for case in cases if case.name == name)
    assert benchmark(case.call) == case.expected


@pytest.mark.parametrize("payload", PAYLOAD_IDS)
def test_python_to_rust(benchmark: Benchmark, cases: tuple[Case, ...], payload: str) -> None:
    check_case(benchmark, cases, f"decode/{payload}")


@pytest.mark.parametrize("payload", PAYLOAD_IDS)
def test_rust_to_python(benchmark: Benchmark, cases: tuple[Case, ...], payload: str) -> None:
    check_case(benchmark, cases, f"encode/{payload}")


@pytest.mark.parametrize("payload", PAYLOAD_IDS)
def test_conversion_roundtrip(benchmark: Benchmark, cases: tuple[Case, ...], payload: str) -> None:
    check_case(benchmark, cases, f"roundtrip/{payload}")


@pytest.mark.parametrize("payload", PAYLOAD_IDS)
def test_sync_bridge(benchmark: Benchmark, cases: tuple[Case, ...], payload: str) -> None:
    check_case(benchmark, cases, f"sync/{payload}")


@pytest.mark.parametrize("payload", PAYLOAD_IDS)
def test_async_bridge(benchmark: Benchmark, cases: tuple[Case, ...], payload: str) -> None:
    check_case(benchmark, cases, f"async/{payload}")


def test_rust_to_python_response(benchmark: Benchmark, cases: tuple[Case, ...]) -> None:
    check_case(benchmark, cases, "typed_response")


def test_gil_release_reacquire(benchmark: Benchmark, cases: tuple[Case, ...]) -> None:
    check_case(benchmark, cases, "gil_roundtrip")


def test_sync_bridge_batch_4_threads(benchmark: Benchmark, cases: tuple[Case, ...]) -> None:
    check_case(benchmark, cases, "sync_threads/4")


@pytest.mark.parametrize("count", (8, 32), ids=("8-calls", "32-calls"))
def test_async_bridge_batch(benchmark: Benchmark, cases: tuple[Case, ...], count: int) -> None:
    check_case(benchmark, cases, f"async_concurrent/{count}")
