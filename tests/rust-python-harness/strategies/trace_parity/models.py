from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from ...shared.parity.recorded_http import RecordedHttpResponse
from ...shared.reporting.models import SdkFunction
from ...shared.tracing.steps import Engine, TraceContract, TraceMapping

TraceMode = Literal["sync", "async"]
TraceFailureSource = Literal["python", "rust", "harness"]


@dataclass(frozen=True, slots=True)
class RouteFixture:
    kwargs: dict[str, object]
    provider_responses: tuple[RecordedHttpResponse, ...]


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route: SdkFunction
    python_entrypoints: tuple[str, str]
    rust_entrypoints: tuple[str, str]
    fixture: Callable[[Engine, str], RouteFixture]


@dataclass(frozen=True, slots=True)
class GatewayRouteSpec:
    route: SdkFunction


TraceRouteSpec: TypeAlias = RouteSpec | GatewayRouteSpec


@dataclass(frozen=True, slots=True)
class TraceScenario:
    name: str
    fixture: Callable[[Engine, str], RouteFixture]
    mappings: tuple[TraceMapping, ...]
    modes: tuple[TraceMode, ...] = ("sync", "async")
    contract: TraceContract = TraceContract()
    sync_mappings: tuple[TraceMapping, ...] | None = None
    async_mappings: tuple[TraceMapping, ...] | None = None

    def mappings_for(self, mode: TraceMode) -> tuple[TraceMapping, ...]:
        selected: Final = self.async_mappings if mode == "async" else self.sync_mappings
        return self.mappings if selected is None else selected


@dataclass(frozen=True, slots=True)
class TraceSuite:
    route: TraceRouteSpec
    scenarios: tuple[TraceScenario, ...]


@dataclass(frozen=True, slots=True)
class TraceExecutionFailure:
    engine: TraceFailureSource
    message: str
