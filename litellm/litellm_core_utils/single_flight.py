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
    flights_started: int = 0
    last_flight_failed: bool = False

    def run(self, fetch: Callable[[], _R], while_busy: _R | None = None) -> _R:
        leader: Final[Future[_R]] = Future()
        with self.lock:
            in_flight: Final = self.in_flight
            joined_flight: Final = self.flights_started
            retry_if_it_fails: Final = not self.last_flight_failed
            if in_flight is None:
                self._take_off(leader)
        if in_flight is None:
            return self._lead(leader, fetch)
        if while_busy is not None:
            return while_busy
        try:
            return in_flight.result()
        except Exception:
            retry: Final = self._claim_retry(joined_flight) if retry_if_it_fails else None
            if retry is None:
                raise
            retry_flight, leads_retry = retry
            return self._lead(retry_flight, fetch) if leads_retry else retry_flight.result()

    def _take_off(self, leader: Future[_R]) -> None:
        self.in_flight = leader
        self.flights_started += 1

    def _claim_retry(self, failed_flight: int) -> tuple[Future[_R], bool] | None:
        leader: Final[Future[_R]] = Future()
        with self.lock:
            in_flight: Final = self.in_flight
            if in_flight is not None:
                return in_flight, False
            if self.flights_started > failed_flight and self.last_flight_failed:
                return None
            self._take_off(leader)
        return leader, True

    def _lead(self, flight: Future[_R], fetch: Callable[[], _R]) -> _R:
        try:
            result: Final = fetch()
        except BaseException as error:
            self._land(failed=True)
            flight.set_exception(error)
            raise
        self._land(failed=False)
        flight.set_result(result)
        return result

    def _land(self, failed: bool) -> None:
        with self.lock:
            self.in_flight = None
            self.last_flight_failed = failed


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
