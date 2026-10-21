from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Final

import pytest

from litellm.rust_bridge import bindings, observation, runtime
from litellm.rust_bridge.catalog import Route, RouteContext, RouteRule
from litellm.rust_bridge.configuration import Decision, Rollout


@dataclass
class Collector:
    events: tuple[observation.RouteObservation, ...] = ()

    def record(self, event: observation.RouteObservation) -> None:
        observation.record_event(event)
        self.events = (*self.events, event)


def binding() -> bindings.NativeBinding[object]:
    bound: Final = bindings.NativeBinding("observation", validate=lambda _: None)
    bound.override(lambda: "native")
    return bound


def test_runtime_observes_context_decision_and_terminal_engine() -> None:
    collector: Final = Collector()
    token: Final = observation.set_observer(collector)
    try:
        with observation.request_scope("trace"), observation.trace_scope():
            assert (
                runtime.run(
                    RouteContext(Route.MESSAGES, provider="anthropic", model="model"),
                    binding=binding(),
                    native=lambda hook: hook(),
                    python=lambda: "python",
                    rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
                )
                == "native"
            )
        selected: Final = next(event for event in collector.events if event.event == "selected")
        terminal: Final = next(event for event in collector.events if event.event == "terminal")
        assert selected.context == RouteContext(Route.MESSAGES, provider="anthropic", model="model")
        assert selected.decision is Decision.RUST_REQUIRED
        assert terminal.engine == "rust"
        assert terminal.outcome == "success"
    finally:
        observation.reset_observer(token)


def test_execution_error_is_observed_as_failure_and_re_raised() -> None:
    collector: Final = Collector()
    token: Final = observation.set_observer(collector)
    failure: Final = ValueError("native failure")
    try:
        with (
            observation.request_scope("trace"),
            observation.trace_scope(),
            pytest.raises(ValueError, match="native failure") as raised,
        ):
            runtime.run(
                RouteContext(Route.MESSAGES),
                binding=binding(),
                native=lambda hook: (_ for _ in ()).throw(failure),
                python=lambda: "python",
                rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
            )
        assert raised.value is failure
        terminal: Final = next(event for event in collector.events if event.event == "terminal")
        assert (terminal.outcome, terminal.reason) == ("failure", "execution_error")
    finally:
        observation.reset_observer(token)


@pytest.mark.asyncio
async def test_async_request_scopes_are_isolated() -> None:
    collector: Final = Collector()
    token: Final = observation.set_observer(collector)

    async def call(trace_id: str) -> None:
        with observation.request_scope(trace_id), observation.trace_scope():
            await runtime.arun(
                RouteContext(Route.MESSAGES, model=trace_id),
                binding=binding(),
                native=lambda hook: asyncio.sleep(0, result=hook()),
                python=lambda: asyncio.sleep(0, result="python"),
                rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
            )

    try:
        await asyncio.gather(call("one"), call("two"))
        selected: Final = tuple(event for event in collector.events if event.event == "selected")
        assert {event.trace_id for event in selected} == {"one", "two"}
        assert {event.context.model for event in selected if event.context is not None} == {"one", "two"}
    finally:
        observation.reset_observer(token)


@pytest.mark.asyncio
async def test_copied_worker_context_records_dispatch_events_in_request_trace() -> None:
    collector: Final = Collector()
    token: Final = observation.set_observer(collector)
    try:
        with observation.request_scope("thread"), observation.trace_scope():
            await asyncio.to_thread(
                lambda: runtime.run(
                    RouteContext(Route.MESSAGES, model="thread"),
                    binding=binding(),
                    native=lambda hook: hook(),
                    python=lambda: "python",
                    rules=(RouteRule(Route.MESSAGES, Rollout.RUST_REQUIRED),),
                )
            )
            events: Final = observation.current_events()
        assert {event.event for event in events} == {"selected", "terminal"}
        assert {event.trace_id for event in events} == {"thread"}
    finally:
        observation.reset_observer(token)


def test_nested_attempts_have_parent_depth_and_cleanup() -> None:
    collector: Final = Collector()
    token: Final = observation.set_observer(collector)
    try:
        with observation.request_scope("trace"), observation.trace_scope():
            with observation.observed_attempt(RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED) as parent:
                with observation.observed_attempt(RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED) as child:
                    observation.observe_terminal(
                        child, RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED, engine="rust"
                    )
                observation.observe_terminal(parent, RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED, engine="rust")
            with observation.observed_attempt(RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED) as next_attempt:
                observation.observe_terminal(
                    next_attempt, RouteContext(Route.MESSAGES), Decision.RUST_REQUIRED, engine="rust"
                )
        nested: Final = tuple(event for event in collector.events if event.event == "terminal")
        assert nested[0].depth == 1
        assert nested[1].depth == 0
        assert nested[0].parent_attempt_id == nested[1].attempt_id
        assert nested[2].depth == 0
        assert nested[2].parent_attempt_id is None
    finally:
        observation.reset_observer(token)
