import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final, Generic, ParamSpec, TypeVar

_P: Final = ParamSpec("_P")
_R: Final = TypeVar("_R")


@dataclass(slots=True)
class SingleFlight(Generic[_R]):
    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: Future[_R] | None = None

    def run(self, fetch: Callable[[], _R]) -> _R:
        leader: Final[Future[_R]] = Future()
        with self.lock:
            flight: Final = leader if self.in_flight is None else self.in_flight
            self.in_flight = flight
        if flight is not leader:
            return flight.result()
        try:
            result: Final = fetch()
        except BaseException as error:
            leader.set_exception(error)
            raise
        else:
            leader.set_result(result)
            return result
        finally:
            with self.lock:
                self.in_flight = None


@dataclass(frozen=True, slots=True)
class SingleFlightCache(Generic[_P, _R]):
    cached: Callable[..., _R]
    clear: Callable[[], None]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(
        self,
        *args: _P.args,
        **kwargs: _P.kwargs,  # kwargs-ok: forwarded verbatim to the cached builder, typed by its own signature
    ) -> _R:
        with self.lock:
            return self.cached(*args, **kwargs)

    def cache_clear(self) -> None:
        self.clear()


def single_flight_lru_cache(build: Callable[_P, _R]) -> SingleFlightCache[_P, _R]:
    cached: Final = lru_cache(maxsize=128)(build)
    return SingleFlightCache(cached, cached.cache_clear)
