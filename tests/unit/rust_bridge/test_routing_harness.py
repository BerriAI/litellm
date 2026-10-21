from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.routing import (
    _RoutingTraceMiddleware,
    _TraceObserver,
    assert_routing_matches_policy,
    strict_current_policy,
    trace_path,
)
from pydantic import TypeAdapter
from starlette.types import Receive, Scope, Send

from litellm.rust_bridge.catalog import Feature, Route, RouteRule, Target
from litellm.rust_bridge.configuration import Decision, Rollout, reset_rust_configuration


@pytest.mark.parametrize(
    ("switch", "optional_rollout"),
    ((True, Rollout.RUST_REQUIRED), (False, Rollout.PYTHON_ONLY)),
)
def test_strict_policy_resolves_synthetic_rollouts_in_order(
    monkeypatch: pytest.MonkeyPatch, switch: bool, optional_rollout: Rollout
) -> None:
    monkeypatch.setenv("LITELLM_RUST", str(switch))
    reset_rust_configuration()
    rules: Final = (
        RouteRule(
            Route.MESSAGES,
            Rollout.RUST_OPT_IN,
            providers=frozenset({"anthropic"}),
            targets=frozenset({Target.MESSAGES}),
        ),
        RouteRule(
            Route.MESSAGES,
            Rollout.PYTHON_ONLY,
            features=frozenset({Feature.SAFEGUARDS_CLASSIFIER}),
        ),
        RouteRule(Route.RESPONSES, Rollout.RUST_OPT_OUT, models=frozenset({"synthetic"})),
        RouteRule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"openai"})),
    )
    resolved: Final = strict_current_policy(rules)
    reset_rust_configuration()

    assert tuple(rule.route for rule in resolved) == tuple(rule.route for rule in rules)
    assert tuple(rule.rollout for rule in resolved) == (
        optional_rollout,
        Rollout.PYTHON_ONLY,
        optional_rollout,
        Rollout.RUST_REQUIRED,
    )
    assert tuple((rule.providers, rule.models, rule.targets, rule.features) for rule in resolved) == tuple(
        (rule.providers, rule.models, rule.targets, rule.features) for rule in rules
    )


def _trace(
    directory: Path,
    *,
    trace_id: str,
    rollout: Rollout,
    decision: Decision,
    engine: str,
) -> Path:
    destination: Final = directory / f"trace-{trace_id}.json"
    destination.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "policy": [
                    {
                        "route": Route.MESSAGES.value,
                        "rollout": rollout.value,
                        "providers": None,
                        "models": None,
                        "targets": None,
                        "features": None,
                    }
                ],
                "events": [
                    {
                        "event": "terminal",
                        "trace_id": trace_id,
                        "attempt_id": "attempt",
                        "parent_attempt_id": None,
                        "depth": 0,
                        "route": Route.MESSAGES.value,
                        "context": {
                            "route": Route.MESSAGES.value,
                            "provider": "anthropic",
                            "model": "synthetic",
                            "target": Target.MESSAGES.value,
                            "features": [],
                        },
                        "decision": decision.value,
                        "outcome": "success",
                        "reason": None,
                        "engine": engine,
                    },
                    {
                        "event": "request",
                        "trace_id": trace_id,
                        "attempt_id": None,
                        "parent_attempt_id": None,
                        "depth": 0,
                        "route": None,
                        "context": None,
                        "decision": None,
                        "outcome": "completed",
                        "reason": None,
                        "engine": None,
                    },
                ],
            }
        )
    )
    return destination


def test_policy_assertion_tracks_the_injected_python_or_native_rule(tmp_path: Path) -> None:
    python_trace: Final = _trace(
        tmp_path,
        trace_id="python",
        rollout=Rollout.PYTHON_ONLY,
        decision=Decision.PYTHON,
        engine="python",
    )
    assert_routing_matches_policy({"x-integration-routing-id": "python"}, python_trace)

    native_trace: Final = _trace(
        tmp_path,
        trace_id="native",
        rollout=Rollout.RUST_REQUIRED,
        decision=Decision.RUST_REQUIRED,
        engine="rust",
    )
    assert_routing_matches_policy(
        {"x-integration-routing-id": "native", "x-litellm-rust": "true"},
        native_trace,
    )


@pytest.mark.asyncio
async def test_middleware_generates_and_persists_a_request_trace(tmp_path: Path) -> None:
    rules: Final = (RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),)
    observer: Final = _TraceObserver(rules, tmp_path)

    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        assert scope["type"] == "http"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"x-integration-routing-id", b"client-value")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    transport: Final = httpx.ASGITransport(app=_RoutingTraceMiddleware(app, observer))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response: Final = await client.get("/", headers={"x-integration-routing-id": "client-value"})

    trace_id: Final = response.headers["x-integration-routing-id"]
    assert trace_id != "client-value"
    assert trace_path(tmp_path, trace_id).exists()
    document: Final = TypeAdapter(dict[str, object]).validate_python(
        json.loads(trace_path(tmp_path, trace_id).read_text())
    )
    events: Final = TypeAdapter(list[dict[str, object]]).validate_python(document["events"])
    assert events[-1]["outcome"] == "no_dispatch"


def test_policy_assertion_rejects_missing_trace_and_mismatched_engine(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="routing trace missing"):
        assert_routing_matches_policy({"x-integration-routing-id": "missing"}, tmp_path / "trace-missing.json")

    trace: Final = _trace(
        tmp_path,
        trace_id="mismatch",
        rollout=Rollout.RUST_REQUIRED,
        decision=Decision.RUST_REQUIRED,
        engine="python",
    )
    with pytest.raises(AssertionError):
        assert_routing_matches_policy(
            {"x-integration-routing-id": "mismatch", "x-litellm-rust": "true"},
            trace,
        )


def test_no_dispatch_trace_validates_a_python_exclusion_without_inventing_a_terminal(
    tmp_path: Path,
) -> None:
    trace_id: Final = "early"
    trace: Final = tmp_path / f"trace-{trace_id}.json"
    trace.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "policy": [
                    {
                        "route": Route.MESSAGES.value,
                        "rollout": Rollout.PYTHON_ONLY.value,
                        "providers": None,
                        "models": None,
                        "targets": None,
                        "features": None,
                    }
                ],
                "events": [
                    {
                        "event": "early",
                        "trace_id": trace_id,
                        "attempt_id": None,
                        "parent_attempt_id": None,
                        "depth": 0,
                        "route": Route.MESSAGES.value,
                        "context": None,
                        "decision": None,
                        "outcome": None,
                        "reason": "no_candidate",
                        "engine": None,
                    },
                    {
                        "event": "request",
                        "trace_id": trace_id,
                        "attempt_id": None,
                        "parent_attempt_id": None,
                        "depth": 0,
                        "route": None,
                        "context": None,
                        "decision": None,
                        "outcome": "no_dispatch",
                        "reason": None,
                        "engine": None,
                    },
                ],
            }
        )
    )

    assert_routing_matches_policy({"x-integration-routing-id": trace_id}, trace)


def _request_event(trace_id: str, outcome: str) -> dict[str, object]:
    return {
        "event": "request",
        "trace_id": trace_id,
        "attempt_id": None,
        "parent_attempt_id": None,
        "depth": 0,
        "route": None,
        "context": None,
        "decision": None,
        "outcome": outcome,
        "reason": None,
        "engine": None,
    }


def _early_event(trace_id: str, reason: str) -> dict[str, object]:
    return {
        "event": "early",
        "trace_id": trace_id,
        "attempt_id": None,
        "parent_attempt_id": None,
        "depth": 0,
        "route": Route.MESSAGES.value,
        "context": None,
        "decision": None,
        "outcome": None,
        "reason": reason,
        "engine": None,
    }


def _terminal_event(trace_id: str, outcome: str, decision: Decision, engine: str) -> dict[str, object]:
    return {
        "event": "terminal",
        "trace_id": trace_id,
        "attempt_id": f"attempt-{engine}-{outcome}",
        "parent_attempt_id": None,
        "depth": 0,
        "route": Route.MESSAGES.value,
        "context": {
            "route": Route.MESSAGES.value,
            "provider": "anthropic",
            "model": "synthetic",
            "target": Target.MESSAGES.value,
            "features": [],
        },
        "decision": decision.value,
        "outcome": outcome,
        "reason": None if outcome == "success" else "execution_error",
        "engine": engine,
    }


def _event_trace(
    directory: Path,
    *,
    trace_id: str,
    rollout: Rollout,
    events: tuple[dict[str, object], ...],
) -> Path:
    destination: Final = directory / f"trace-{trace_id}.json"
    destination.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "policy": [
                    {
                        "route": Route.MESSAGES.value,
                        "rollout": rollout.value,
                        "providers": None,
                        "models": None,
                        "targets": None,
                        "features": None,
                    }
                ],
                "events": events,
            }
        )
    )
    return destination


@pytest.mark.parametrize(
    ("trace_id", "rollout", "events", "headers"),
    (
        ("missing-dispatch", Rollout.PYTHON_ONLY, (_request_event("missing-dispatch", "no_dispatch"),), {}),
        (
            "detector-failure",
            Rollout.PYTHON_ONLY,
            (_early_event("detector-failure", "detector_failure"), _request_event("detector-failure", "no_dispatch")),
            {},
        ),
        (
            "projection-failure",
            Rollout.PYTHON_ONLY,
            (
                _early_event("projection-failure", "projection_failure"),
                _request_event("projection-failure", "no_dispatch"),
            ),
            {},
        ),
        (
            "failed-only",
            Rollout.RUST_REQUIRED,
            (
                _terminal_event("failed-only", "failure", Decision.RUST_REQUIRED, "rust"),
                _request_event("failed-only", "completed"),
            ),
            {"x-litellm-rust": "true"},
        ),
        (
            "false-early-marker",
            Rollout.PYTHON_ONLY,
            (_early_event("false-early-marker", "no_candidate"), _request_event("false-early-marker", "no_dispatch")),
            {"x-litellm-rust": "true"},
        ),
        (
            "optional",
            Rollout.RUST_OPT_IN,
            (_terminal_event("optional", "success", Decision.RUST_WITH_FALLBACK, "rust"),),
            {"x-litellm-rust": "true"},
        ),
        (
            "ambiguous",
            Rollout.RUST_REQUIRED,
            (
                _terminal_event("ambiguous", "success", Decision.RUST_REQUIRED, "rust"),
                _terminal_event("ambiguous", "success", Decision.RUST_REQUIRED, "rust-2"),
                _request_event("ambiguous", "completed"),
            ),
            {"x-litellm-rust": "true"},
        ),
        (
            "trace-mismatch",
            Rollout.PYTHON_ONLY,
            (
                _terminal_event("different-trace", "success", Decision.PYTHON, "python"),
                _request_event("trace-mismatch", "completed"),
            ),
            {},
        ),
    ),
)
def test_policy_assertion_rejects_unproven_dispatch_outcomes(
    tmp_path: Path,
    trace_id: str,
    rollout: Rollout,
    events: tuple[dict[str, object], ...],
    headers: dict[str, str],
) -> None:
    trace: Final = _event_trace(tmp_path, trace_id=trace_id, rollout=rollout, events=events)

    with pytest.raises(AssertionError):
        assert_routing_matches_policy({"x-integration-routing-id": trace_id, **headers}, trace)


def test_failed_root_attempt_can_precede_one_successful_root_attempt(tmp_path: Path) -> None:
    trace_id: Final = "retry"
    trace: Final = _event_trace(
        tmp_path,
        trace_id=trace_id,
        rollout=Rollout.RUST_REQUIRED,
        events=(
            _terminal_event(trace_id, "failure", Decision.RUST_REQUIRED, "rust"),
            _terminal_event(trace_id, "success", Decision.RUST_REQUIRED, "rust"),
            _request_event(trace_id, "completed"),
        ),
    )

    assert_routing_matches_policy(
        {"x-integration-routing-id": trace_id, "x-litellm-rust": "true"},
        trace,
    )
