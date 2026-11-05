from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock
from typing import Final, TypeAlias

from typing_extensions import assert_never

MAX_TRACKED_DEPLOYMENTS: Final = 10_000
MAX_SERIES_PER_DEPLOYMENT: Final = 8


class DeploymentState(IntEnum):
    HEALTHY = 0
    PARTIAL_OUTAGE = 1
    COMPLETE_OUTAGE = 2


@dataclass(frozen=True, slots=True)
class DeploymentStateSeries:
    litellm_model_name: str
    model_id: str | None
    api_base: str | None
    api_provider: str


@dataclass(frozen=True, slots=True)
class Serving:
    written: tuple[DeploymentStateSeries, ...] = ()


@dataclass(frozen=True, slots=True)
class CoolingDown:
    written: tuple[DeploymentStateSeries, ...]


SeriesHistory: TypeAlias = Serving | CoolingDown


def including(
    written: tuple[DeploymentStateSeries, ...], series: DeploymentStateSeries
) -> tuple[DeploymentStateSeries, ...]:
    return (*(known for known in written if known != series), series)


def stuck_at_two(history: SeriesHistory) -> tuple[DeploymentStateSeries, ...]:
    match history:
        case Serving():
            return ()
        case CoolingDown(written):
            return written
        case _:
            assert_never(history)


class DeploymentStateSeriesTracker:
    """
    A cooldown marks each series written since the last success, as its config labels can differ from the request
    labels. The next success clears them. Serving deployments are capped, and a cooling deployment is kept until its
    next success.
    """

    def __init__(
        self,
        write: Callable[[DeploymentState, str, str | None, str | None, str], None],
        lock: AbstractContextManager[object] | None = None,
    ) -> None:
        self._write = write
        self._serving: OrderedDict[str | None, Serving] = OrderedDict()  # mutable-ok: O(1) LRU
        self._cooling: dict[str | None, CoolingDown] = {}  # mutable-ok: emptied by each deployment's next success
        self._lock: AbstractContextManager[object] = Lock() if lock is None else lock

    def healthy(self, series: DeploymentStateSeries) -> None:
        with self._lock:
            history: Final = self._history(series.model_id)
            self._write_all(DeploymentState.HEALTHY, including(stuck_at_two(history), series))
            self._remember(series.model_id, Serving(written=(series,)))

    def partial_outage(self, series: DeploymentStateSeries) -> None:
        with self._lock:
            history: Final = self._history(series.model_id)
            listed: Final = including(history.written, series)
            tracked: Final = listed[-MAX_SERIES_PER_DEPLOYMENT:]
            match history:
                case Serving():
                    self._write_all(DeploymentState.PARTIAL_OUTAGE, (series,))
                    self._remember(series.model_id, Serving(written=tracked))
                case CoolingDown():
                    self._write_all(DeploymentState.HEALTHY, listed[:-MAX_SERIES_PER_DEPLOYMENT])
                    self._write_all(DeploymentState.COMPLETE_OUTAGE, (series,))
                    self._remember(series.model_id, CoolingDown(written=tracked))
                case _:
                    assert_never(history)

    def complete_outage(self, derived_from_config: DeploymentStateSeries) -> None:
        with self._lock:
            history: Final = self._history(derived_from_config.model_id)
            marked: Final = history.written or (derived_from_config,)
            self._write_all(DeploymentState.COMPLETE_OUTAGE, marked)
            self._remember(derived_from_config.model_id, CoolingDown(written=marked))

    def _write_all(self, state: DeploymentState, series: Iterable[DeploymentStateSeries]) -> None:
        for item in series:
            self._write(state, item.litellm_model_name, item.model_id, item.api_base, item.api_provider)

    def _history(self, model_id: str | None) -> SeriesHistory:
        cooling: Final = self._cooling.get(model_id)
        return self._serving.get(model_id, Serving()) if cooling is None else cooling

    def _remember(self, model_id: str | None, history: SeriesHistory) -> None:
        match history:
            case Serving():
                self._cooling.pop(model_id, None)
                self._serving[model_id] = history
                self._serving.move_to_end(model_id)
                if len(self._serving) > MAX_TRACKED_DEPLOYMENTS:
                    self._serving.popitem(last=False)
            case CoolingDown():
                self._serving.pop(model_id, None)
                self._cooling[model_id] = history
            case _:
                assert_never(history)
