from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, cast

from ...shared.parity.recorded_http import RecordedResponse
from ...shared.reporting.models import SdkFunction

TraceFailureSource = Literal["python", "harness"]


@dataclass(frozen=True, slots=True)
class RouteFixture:
    kwargs: dict[str, object]
    provider_responses: tuple[RecordedResponse, ...]
    expected_failure: bool = False
    consume_stream: bool = False
    environment: tuple[tuple[str, str], ...] = ()

    def derive(
        self,
        *,
        kwargs: Mapping[str, object] | None = None,
        provider_responses: tuple[RecordedResponse, ...] | None = None,
        expected_failure: bool | None = None,
        consume_stream: bool | None = None,
    ) -> RouteFixture:
        return RouteFixture(
            kwargs={**self.kwargs, **(kwargs or {})},
            provider_responses=self.provider_responses if provider_responses is None else provider_responses,
            expected_failure=self.expected_failure if expected_failure is None else expected_failure,
            consume_stream=self.consume_stream if consume_stream is None else consume_stream,
            environment=self.environment,
        )

    def with_body(self, **updates: object) -> RouteFixture:
        raw_body: Final = self.kwargs.get("body")
        if not isinstance(raw_body, dict):
            raise ValueError("route fixture does not contain an object body")
        body: Final = cast(dict[str, object], raw_body)
        return self.derive(kwargs={"body": {**body, **updates}})


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route: SdkFunction
    python_entrypoints: tuple[str, str]
    fixture: Callable[[str], RouteFixture]


@dataclass(frozen=True, slots=True)
class TraceScenario:
    name: str
    fixture: Callable[[str], RouteFixture]
    asynchronous: bool


@dataclass(frozen=True, slots=True)
class TraceSuite:
    route: RouteSpec
    scenarios: tuple[TraceScenario, ...]


@dataclass(frozen=True, slots=True)
class TraceExecutionFailure:
    engine: TraceFailureSource
    message: str
