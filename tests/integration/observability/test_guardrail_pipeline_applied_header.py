from __future__ import annotations

import ast
import asyncio
import itertools
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import pytest
import yaml

from tests.integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from tests.integration._support.otlp_sink import Span, SpanSinks, recorded_spans
from tests.integration._support.process import owned_proxy
from tests.integration._support.upstream import aws_event_stream_frame
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_HEADER: Final = "x-litellm-applied-guardrails"
_VENDOR_PATH: Final = "/beta/litellm_basic_guardrail_api"
_PROMPT_SHIELD_PATH: Final = "/contentsafety/text:shieldPrompt"
_PROMPT_SHIELD: Final = "prompt-shield-step"
_REQUEST_FIRST: Final = "request-first-step"
_REQUEST_SECOND: Final = "request-second-step"
_REQUEST_BLOCKER: Final = "request-blocking-step"
_REQUEST_ERRORING: Final = "request-erroring-step"
_RESPONSE_FIRST: Final = "response-first-step"
_RESPONSE_SECOND: Final = "response-second-step"
_RESPONSE_BLOCKER: Final = "response-blocking-step"
_LIVE_FIRST: Final = "live-first-step"
_LIVE_SECOND: Final = "live-second-step"
_UNGATED_STEP: Final = "ungated-step"
_JUDGE: Final = "judge-step"
_JUDGE_MODEL: Final = "pipeline-judge"
_STOPS_AFTER_FIRST: Final = "stops-after-first"
_RUNS_EVERY_STEP: Final = "runs-every-step"
_BLOCKS_AT_FIRST: Final = "blocks-at-first"
_RESPONSE_STOPS_AFTER_FIRST: Final = "response-stops-after-first"
_SHIELDS_THEN_CHECKS: Final = "shields-then-checks"
_MOVES_PAST_FIRST: Final = "moves-past-first"
_RESPONSE_MOVES_PAST_FIRST: Final = "response-moves-past-first"
_SCANS_AFTER_RELEASE: Final = "scans-after-release"
_CANNOT_HOLD_STREAM: Final = "cannot-hold-stream"
_ERRORS_PAST_FIRST: Final = "errors-past-first"
_JUDGES_THEN_CHECKS: Final = "judges-then-checks"
_ENDPOINTS: Final = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
_BEDROCK_MODEL: Final = "pipeline-bedrock"
_BEDROCK_MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    vendor: Wire
    upstream: Wire
    model: str
    traces: str


def _completion_chunk(identity: str, delta: dict[str, str], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _event(event: dict[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _response(identity: str) -> dict[str, object]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "guarded", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
    }


def _responses_reply(request: Request) -> Reply:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    response: Final = _response(identity)
    if json.loads(request.body).get("stream") is not True:
        return Reply(body=json.dumps(response).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _event(
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**response, "status": "in_progress", "output": []},
                }
            ),
            _event(
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": f"msg_{identity}",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "guarded",
                }
            ),
            _event({"type": "response.completed", "sequence_number": 2, "response": response}),
        ),
    )


def _bedrock_event(event_type: str, payload: dict[str, object]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload).encode(),
    )


def _bedrock_reply(request: Request) -> Reply:
    action: Final = request.target.rsplit("/", 1)[1]
    if action == "converse-stream":
        return Reply(
            content_type="application/vnd.amazon.eventstream",
            chunks=(
                _bedrock_event("messageStart", {"role": "assistant"}),
                _bedrock_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "guarded"}}),
                _bedrock_event("contentBlockStop", {"contentBlockIndex": 0}),
                _bedrock_event("messageStop", {"stopReason": "end_turn"}),
                _bedrock_event(
                    "metadata", {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}, "metrics": {}}
                ),
            ),
        )
    if action == "invoke":
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": _BEDROCK_MODEL_ID,
                    "content": [{"type": "text", "text": "guarded"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
        )
    assert action == "converse", request.target
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "guarded"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _completion(request: Request) -> Reply:
    if request.target.startswith("/model/"):
        return _bedrock_reply(request)
    if request.target == "/v1/responses":
        return _responses_reply(request)
    if request.target == "/v1/images/generations":
        return Reply(body=json.dumps({"created": 1, "data": [{"b64_json": "aW1n", "revised_prompt": None}]}).encode())
    assert request.target == "/v1/chat/completions", request.target
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    if json.loads(request.body).get("stream") is True:
        return Reply(
            content_type="text/event-stream",
            chunks=(
                _completion_chunk(identity, {"role": "assistant", "content": "guarded"}, None),
                _completion_chunk(identity, {}, "stop"),
                b"data: [DONE]\n\n",
            ),
        )
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "guarded"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
    )


def _vendor(request: Request) -> Reply:
    if request.target == f"/{_JUDGE}/v1/decisions":
        return Reply(
            body=json.dumps(
                {
                    "model": "gpt-6-luna",
                    "answers": [{"type": "predicate", "name": "unsafe_output", "probability": 0.01}],
                    "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
                }
            ).encode()
        )
    if request.target.startswith(f"/{_PROMPT_SHIELD}{_PROMPT_SHIELD_PATH}?"):
        return Reply(
            body=json.dumps({"userPromptAnalysis": {"attackDetected": False}, "documentsAnalysis": []}).encode()
        )
    if request.target.startswith(f"/{_UNGATED_STEP}/"):
        return Reply(body=json.dumps({"finalAction": "NONE", "appliedPolicies": []}).encode())
    assert request.target.endswith(_VENDOR_PATH), request.target
    if request.target.startswith(f"/{_REQUEST_ERRORING}/"):
        return Reply(status=503, body=json.dumps({"error": "guardrail service unavailable"}).encode())
    if request.target.startswith((f"/{_REQUEST_BLOCKER}/", f"/{_RESPONSE_BLOCKER}/")):
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "first step refused"}).encode())
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _guardrail(name: str, mode: str, vendor: Wire, **params: object) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": mode,
            "default_on": False,
            "api_base": f"{vendor.url}/{name}",
            "api_key": "synthetic-guardrail-key",
            **params,
        },
    }


def _policy(mode: str, steps: tuple[tuple[str, str, str], ...], *, on_error: str | None = None) -> dict[str, object]:
    return {
        "guardrails": {"add": [guardrail for guardrail, _, _ in steps]},
        "pipeline": {
            "mode": mode,
            "steps": [
                {"guardrail": guardrail, "on_pass": on_pass, "on_fail": on_fail, "on_error": on_error}
                for guardrail, on_pass, on_fail in steps
            ],
        },
    }


def _config(directory: Path, vendor: Wire, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["litellm_settings"] = {**configuration.get("litellm_settings", {}), "callbacks": ["otel"]}
    configuration["model_list"] = [
        *configuration.get("model_list", []),
        {
            "model_name": _BEDROCK_MODEL,
            "litellm_params": {
                "model": f"bedrock/{_BEDROCK_MODEL_ID}",
                "api_base": upstream.url,
                "aws_access_key_id": "AKIASYNTHETICPROVIDER",
                "aws_secret_access_key": "synthetic-provider-secret",
                "aws_region_name": "us-east-1",
            },
        },
        {
            "model_name": _JUDGE_MODEL,
            "litellm_params": {
                "model": "openai/gpt-6-luna",
                "api_base": f"{vendor.url}/{_JUDGE}/v1",
                "api_key": "synthetic-judge-key",
            },
        },
    ]
    configuration["guardrails"] = [
        *(
            _guardrail(name, "pre_call", vendor)
            for name in (_REQUEST_FIRST, _REQUEST_SECOND, _REQUEST_BLOCKER, _REQUEST_ERRORING)
        ),
        *(_guardrail(name, "post_call", vendor) for name in (_RESPONSE_FIRST, _RESPONSE_SECOND, _RESPONSE_BLOCKER)),
        *(
            _guardrail(name, "post_call", vendor, streaming_buffer_until_moderated=False)
            for name in (_LIVE_FIRST, _LIVE_SECOND)
        ),
        {
            "guardrail_name": _UNGATED_STEP,
            "litellm_params": {
                "guardrail": "dynamoai",
                "mode": "post_call",
                "default_on": False,
                "api_base": f"{vendor.url}/{_UNGATED_STEP}",
                "api_key": "synthetic-guardrail-key",
            },
        },
        {
            "guardrail_name": _JUDGE,
            "litellm_params": {
                "guardrail": "decision_model",
                "mode": "post_call",
                "default_on": False,
                "decision_model": _JUDGE_MODEL,
                "checks": [
                    {
                        "name": "unsafe_output",
                        "instructions": "Does the response say anything unsafe?",
                        "action": "block",
                        "threshold": 0.5,
                    }
                ],
            },
        },
        {
            "guardrail_name": _PROMPT_SHIELD,
            "litellm_params": {
                "guardrail": "azure/prompt_shield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": f"{vendor.url}/{_PROMPT_SHIELD}",
                "api_key": "synthetic-guardrail-key",
            },
        },
    ]
    configuration["policies"] = {
        _STOPS_AFTER_FIRST: _policy(
            "pre_call", ((_REQUEST_FIRST, "allow", "block"), (_REQUEST_SECOND, "next", "block"))
        ),
        _RUNS_EVERY_STEP: _policy("pre_call", ((_REQUEST_FIRST, "next", "block"), (_REQUEST_SECOND, "next", "block"))),
        _BLOCKS_AT_FIRST: _policy(
            "pre_call", ((_REQUEST_BLOCKER, "next", "block"), (_REQUEST_SECOND, "next", "block"))
        ),
        _RESPONSE_STOPS_AFTER_FIRST: _policy(
            "post_call", ((_RESPONSE_FIRST, "allow", "block"), (_RESPONSE_SECOND, "next", "block"))
        ),
        _SHIELDS_THEN_CHECKS: _policy(
            "pre_call", ((_PROMPT_SHIELD, "next", "block"), (_REQUEST_FIRST, "next", "block"))
        ),
        _MOVES_PAST_FIRST: _policy(
            "pre_call", ((_REQUEST_BLOCKER, "next", "next"), (_REQUEST_SECOND, "next", "block"))
        ),
        _RESPONSE_MOVES_PAST_FIRST: _policy(
            "post_call", ((_RESPONSE_BLOCKER, "next", "next"), (_RESPONSE_SECOND, "next", "block"))
        ),
        _SCANS_AFTER_RELEASE: _policy("post_call", ((_LIVE_FIRST, "next", "next"), (_LIVE_SECOND, "next", "next"))),
        _CANNOT_HOLD_STREAM: _policy(
            "post_call", ((_UNGATED_STEP, "next", "block"), (_RESPONSE_SECOND, "next", "block"))
        ),
        _JUDGES_THEN_CHECKS: _policy("post_call", ((_JUDGE, "next", "block"), (_RESPONSE_SECOND, "next", "block"))),
        _ERRORS_PAST_FIRST: _policy(
            "pre_call", ((_REQUEST_ERRORING, "next", "block"), (_REQUEST_SECOND, "next", "block")), on_error="next"
        ),
    }
    configuration["policy_attachments"] = [{"policy": policy, "tags": [policy]} for policy in configuration["policies"]]
    path: Final = directory / "guardrail-pipeline-applied-header.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory, audit_sinks: SpanSinks) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("guardrail-pipeline-applied-header")
    with (
        wire_server(_vendor) as vendor,
        wire_server(_completion) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory,
            {"OTEL_EXPORTER": "http/json", "OTEL_ENDPOINT": audit_sinks.operator, "OTEL_BSP_SCHEDULE_DELAY": "100"},
            config=_config(directory, vendor, upstream),
            workers=2,
        ) as owned,
        owned.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{upstream.url}/v1", api_key="sk-fixture")
        yield _Rig(owned, vendor, upstream, model, audit_sinks.operator)


def _body(path: str, model: str, stream: bool) -> dict[str, object]:
    prompt: Final = f"hello {uuid.uuid4().hex}"
    if path == "/v1/responses":
        return {"model": model, "input": prompt, "stream": stream}
    messages: Final = [{"role": "user", "content": prompt}]
    if path == "/v1/messages":
        return {"model": model, "max_tokens": 20, "messages": messages, "stream": stream}
    return {"model": model, "messages": messages, "stream": stream}


def _ask_as_new_team(
    rig: _Rig,
    tag: str,
    *,
    stream: bool = False,
    path: str = "/v1/chat/completions",
    extra: Mapping[str, object] = MappingProxyType({}),
) -> tuple[str, httpx.Response]:
    rig.vendor.drain()
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"tags": [tag]})
        key: Final = scenario.key(team_id=team)
        return team, rig.gateway.request("POST", path, {**_body(path, rig.model, stream), **extra}, key=key)


def _ask_as_team_tagged(
    rig: _Rig,
    tag: str,
    *,
    stream: bool = False,
    path: str = "/v1/chat/completions",
    extra: Mapping[str, object] = MappingProxyType({}),
) -> httpx.Response:
    return _ask_as_new_team(rig, tag, stream=stream, path=path, extra=extra)[1]


def _steps_that_ran(rig: _Rig) -> tuple[str, ...]:
    return tuple(request.target.split("/")[1] for request in rig.vendor.drain())


def test_a_request_pipeline_that_allows_at_its_first_step_names_only_that_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _STOPS_AFTER_FIRST)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_FIRST,)
    assert len(rig.upstream.drain()) == 1
    assert response.headers.get(_HEADER) == _REQUEST_FIRST, dict(response.headers)


def test_a_streamed_request_pipeline_that_allows_at_its_first_step_names_only_that_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _STOPS_AFTER_FIRST, stream=True)
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    assert _steps_that_ran(rig) == (_REQUEST_FIRST,)
    assert len(rig.upstream.drain()) == 1
    assert response.headers.get(_HEADER) == _REQUEST_FIRST, dict(response.headers)


def test_a_request_pipeline_that_runs_every_step_names_every_step_in_order(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _RUNS_EVERY_STEP)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_FIRST, _REQUEST_SECOND)
    assert response.headers.get(_HEADER) == f"{_REQUEST_FIRST},{_REQUEST_SECOND}", dict(response.headers)


def test_a_request_pipeline_names_a_prompt_shield_step_that_ran_before_a_generic_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _SHIELDS_THEN_CHECKS)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_PROMPT_SHIELD, _REQUEST_FIRST)
    assert set(response.headers.get(_HEADER, "").split(",")) == {_PROMPT_SHIELD, _REQUEST_FIRST}, dict(response.headers)


def test_a_request_pipeline_that_blocks_at_its_first_step_names_only_that_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _BLOCKS_AT_FIRST)
    assert response.status_code == 400, response.text
    assert _steps_that_ran(rig) == (_REQUEST_BLOCKER,)
    assert rig.upstream.drain() == ()
    assert response.headers.get(_HEADER) == _REQUEST_BLOCKER, dict(response.headers)


def test_a_response_pipeline_that_allows_at_its_first_step_names_only_that_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _RESPONSE_STOPS_AFTER_FIRST)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_RESPONSE_FIRST,)
    assert response.headers.get(_HEADER) == _RESPONSE_FIRST, dict(response.headers)


def test_a_streamed_response_pipeline_that_allows_at_its_first_step_names_only_that_step(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _RESPONSE_STOPS_AFTER_FIRST, stream=True)
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    assert _steps_that_ran(rig) == (_RESPONSE_FIRST,)
    assert response.headers.get(_HEADER) == _RESPONSE_FIRST, dict(response.headers)


def test_a_request_pipeline_names_a_step_that_flagged_the_request_before_moving_on(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _MOVES_PAST_FIRST)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_BLOCKER, _REQUEST_SECOND)
    assert len(rig.upstream.drain()) == 1
    assert set(response.headers.get(_HEADER, "").split(",")) == {_REQUEST_BLOCKER, _REQUEST_SECOND}, dict(
        response.headers
    )


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_a_response_pipeline_names_a_step_that_flagged_the_response_before_moving_on(rig: _Rig, stream: bool) -> None:
    response: Final = _ask_as_team_tagged(rig, _RESPONSE_MOVES_PAST_FIRST, stream=stream)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_RESPONSE_BLOCKER, _RESPONSE_SECOND)
    assert set(response.headers.get(_HEADER, "").split(",")) == {_RESPONSE_BLOCKER, _RESPONSE_SECOND}, dict(
        response.headers
    )


def test_a_stream_released_before_its_pipeline_scans_it_names_the_steps_in_its_headers(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _SCANS_AFTER_RELEASE, stream=True)
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    assert _steps_that_ran(rig) == (_LIVE_FIRST, _LIVE_SECOND)
    assert set(response.headers.get(_HEADER, "").split(",")) == {_LIVE_FIRST, _LIVE_SECOND}, dict(response.headers)


def test_a_stream_its_pipeline_cannot_hold_names_the_steps_that_run_on_their_own(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _CANNOT_HOLD_STREAM, stream=True)
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    assert _RESPONSE_SECOND in _steps_that_ran(rig)
    assert set(response.headers.get(_HEADER, "").split(",")) == {_UNGATED_STEP, _RESPONSE_SECOND}, dict(
        response.headers
    )


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("path", ["/v1/responses", "/v1/messages"])
def test_a_request_pipeline_that_allows_at_its_first_step_names_only_that_step_on_every_endpoint(
    rig: _Rig, path: str, stream: bool
) -> None:
    response: Final = _ask_as_team_tagged(rig, _STOPS_AFTER_FIRST, stream=stream, path=path)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_FIRST,)
    assert len(rig.upstream.drain()) == 1
    assert response.headers.get(_HEADER) == _REQUEST_FIRST, dict(response.headers)


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("path", ["/v1/responses", "/v1/messages"])
def test_a_response_pipeline_that_allows_at_its_first_step_names_only_that_step_on_every_endpoint(
    rig: _Rig, path: str, stream: bool
) -> None:
    response: Final = _ask_as_team_tagged(rig, _RESPONSE_STOPS_AFTER_FIRST, stream=stream, path=path)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_RESPONSE_FIRST,)
    assert response.headers.get(_HEADER) == _RESPONSE_FIRST, dict(response.headers)


def test_a_request_pipeline_names_a_step_whose_service_errored_before_moving_on(rig: _Rig) -> None:
    response: Final = _ask_as_team_tagged(rig, _ERRORS_PAST_FIRST)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_ERRORING, _REQUEST_SECOND)
    assert len(rig.upstream.drain()) == 1
    assert set(response.headers.get(_HEADER, "").split(",")) == {_REQUEST_ERRORING, _REQUEST_SECOND}, dict(
        response.headers
    )


@pytest.mark.parametrize(
    "forged",
    [
        pytest.param(
            {
                "metadata": {
                    "applied_guardrails": [_REQUEST_SECOND],
                    "_pre_call_executed_guardrails": [f"deadbeef:{_REQUEST_SECOND}", _REQUEST_SECOND],
                }
            },
            id="names-in-its-metadata",
        ),
        pytest.param({"guardrails": [_REQUEST_SECOND]}, id="asks-for-the-skipped-step"),
    ],
)
def test_a_request_cannot_get_a_step_that_never_ran_named(rig: _Rig, forged: Mapping[str, object]) -> None:
    response: Final = _ask_as_team_tagged(rig, _STOPS_AFTER_FIRST, extra=forged)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_REQUEST_FIRST,)
    assert response.headers.get(_HEADER) == _REQUEST_FIRST, dict(response.headers)


def _spans_naming_guardrails(rig: _Rig, since: int, team: str) -> tuple[Span, ...]:
    spans: Final = recorded_spans(rig.traces, since)[1]
    return tuple(
        span
        for span in spans
        if span["attributes"].get("metadata.user_api_key_team_id") == team
        and "metadata.applied_guardrails" in span["attributes"]
    )


@pytest.mark.parametrize(
    ("tag", "path", "stream", "expected"),
    [
        pytest.param(_STOPS_AFTER_FIRST, "/v1/chat/completions", False, [_REQUEST_FIRST], id="request-pipeline"),
        pytest.param(_STOPS_AFTER_FIRST, "/v1/chat/completions", True, [_REQUEST_FIRST], id="request-pipeline-stream"),
        pytest.param(_STOPS_AFTER_FIRST, "/v1/responses", True, [_REQUEST_FIRST], id="request-pipeline-responses"),
        pytest.param(
            _RESPONSE_STOPS_AFTER_FIRST, "/v1/chat/completions", False, [_RESPONSE_FIRST], id="response-pipeline"
        ),
        pytest.param(
            _RESPONSE_STOPS_AFTER_FIRST, "/v1/chat/completions", True, [_RESPONSE_FIRST], id="response-pipeline-stream"
        ),
        pytest.param(
            _RESPONSE_STOPS_AFTER_FIRST, "/v1/messages", True, [_RESPONSE_FIRST], id="response-pipeline-messages"
        ),
    ],
)
def test_the_spend_log_and_the_trace_name_only_the_steps_that_ran(
    rig: _Rig, tag: str, path: str, stream: bool, expected: list[str]
) -> None:
    since: Final = recorded_spans(rig.traces)[0]
    team, response = _ask_as_new_team(rig, tag, stream=stream, path=path)
    assert response.status_code == 200, response.text
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    spans: Final = eventually(lambda: _spans_naming_guardrails(rig, since, team), bool, seconds=30)
    assert object_value(rows[0]["metadata"]).get("applied_guardrails") == expected, rows
    assert [ast.literal_eval(str(span["attributes"]["metadata.applied_guardrails"])) for span in spans] == [
        expected
    ] * len(spans), spans


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
def test_a_response_pipeline_names_a_judge_step_that_records_its_own_results(rig: _Rig, stream: bool) -> None:
    team, response = _ask_as_new_team(rig, _JUDGES_THEN_CHECKS, stream=stream)
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_JUDGE, _RESPONSE_SECOND)
    assert sorted(response.headers.get(_HEADER, "").split(",")) == [_JUDGE, _RESPONSE_SECOND], dict(response.headers)
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert sorted(object_value(rows[0]["metadata"]).get("applied_guardrails") or []) == [_JUDGE, _RESPONSE_SECOND]


def _bedrock_body(action: str) -> dict[str, object]:
    prompt: Final = f"hello {uuid.uuid4().hex}"
    if action == "invoke":
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    return {"messages": [{"role": "user", "content": [{"text": prompt}]}]}


@pytest.mark.parametrize("action", ["converse", "invoke", "converse-stream"])
def test_a_pass_through_route_names_its_response_pipeline_up_front_because_its_headers_are_built_first(
    rig: _Rig, action: str
) -> None:
    rig.vendor.drain()
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"tags": [_RESPONSE_STOPS_AFTER_FIRST]})
        response: Final = rig.gateway.request(
            "POST", f"/bedrock/model/{_BEDROCK_MODEL}/{action}", _bedrock_body(action), key=scenario.key(team_id=team)
        )
    assert response.status_code == 200, response.text
    assert _steps_that_ran(rig) == (_RESPONSE_FIRST,)
    assert [request.target.rsplit("/", 1)[1] for request in rig.upstream.drain()] == [action]
    assert sorted(response.headers.get(_HEADER, "").split(",")) == [_RESPONSE_FIRST, _RESPONSE_SECOND], dict(
        response.headers
    )
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert sorted(object_value(rows[0]["metadata"]).get("applied_guardrails") or []) == [
        _RESPONSE_FIRST,
        _RESPONSE_SECOND,
    ], rows


def test_an_image_generation_names_its_response_pipeline_up_front_because_its_spend_log_is_built_first(
    rig: _Rig,
) -> None:
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        image_model: Final = scenario.model(
            model="openai/dall-e-3", api_base=f"{rig.upstream.url}/v1", api_key="sk-fixture"
        )
        team: Final = scenario.team(metadata={"tags": [_RESPONSE_STOPS_AFTER_FIRST]})
        response: Final = rig.gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": image_model, "prompt": f"a cat {uuid.uuid4().hex}", "size": "1024x1024"},
            key=scenario.key(team_id=team),
        )
    assert response.status_code == 200, response.text
    assert [request.target for request in rig.upstream.drain()] == ["/v1/images/generations"]
    assert sorted(response.headers.get(_HEADER, "").split(",")) == [_RESPONSE_FIRST, _RESPONSE_SECOND], dict(
        response.headers
    )
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert sorted(object_value(rows[0]["metadata"]).get("applied_guardrails") or []) == [
        _RESPONSE_FIRST,
        _RESPONSE_SECOND,
    ], rows


def _agent_card(name: str, url: str) -> dict[str, object]:
    return {
        "protocolVersion": "0.3",
        "name": name,
        "description": "Scripted agent that answers every message with one text part",
        "version": "1.0.0",
        "url": url,
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


@pytest.fixture(scope="module")
def agent(rig: _Rig) -> Iterator[str]:
    name: Final = f"pipeline-agent-{uuid.uuid4().hex}"

    def answer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(_agent_card(name, f"{peer.url}/")).encode())
        message: Final = {
            "kind": "message",
            "role": "agent",
            "messageId": uuid.uuid4().hex,
            "parts": [{"kind": "text", "text": "guarded"}],
        }
        return Reply(
            body=json.dumps({"jsonrpc": "2.0", "id": json.loads(request.body)["id"], "result": message}).encode()
        )

    with wire_server(answer) as peer:
        identity: Final = str(
            rig.gateway.post(
                "/v1/agents", {"agent_name": name, "agent_card_params": _agent_card(name, f"{peer.url}/")}
            )["agent_id"]
        )
        yield identity
        rig.gateway.request("DELETE", f"/v1/agents/{identity}")


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        pytest.param(_SCANS_AFTER_RELEASE, [_LIVE_FIRST, _LIVE_SECOND], id="every-step-passes"),
        pytest.param(
            _RESPONSE_MOVES_PAST_FIRST, [_RESPONSE_BLOCKER, _RESPONSE_SECOND], id="first-step-flags-and-moves-on"
        ),
    ],
)
def test_an_agent_message_names_every_response_step_that_ran_in_the_spend_log_and_the_trace(
    rig: _Rig, agent: str, tag: str, expected: list[str]
) -> None:
    since: Final = recorded_spans(rig.traces)[0]
    rig.vendor.drain()
    message: Final = {
        "kind": "message",
        "role": "user",
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": f"hello {uuid.uuid4().hex}"}],
    }
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"tags": [tag]})
        response: Final = rig.gateway.request(
            "POST",
            f"/a2a/{agent}",
            {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "message/send", "params": {"message": message}},
            key=scenario.key(team_id=team),
            headers={"Accept": "application/json"},
        )
    assert response.status_code == 200, response.text
    assert list(_steps_that_ran(rig)) == expected
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE team_id = %s', (team,)),
        lambda found: len(found) == 1,
        seconds=70,
    )
    spans: Final = eventually(lambda: _spans_naming_guardrails(rig, since, team), bool, seconds=30)
    assert sorted(object_value(rows[0]["metadata"]).get("applied_guardrails") or []) == expected, rows
    assert [sorted(ast.literal_eval(str(span["attributes"]["metadata.applied_guardrails"]))) for span in spans] == [
        expected
    ] * len(spans), spans


def _openai_chat(base: str, key: str, model: str) -> str | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        openai.OpenAI(api_key=key, base_url=f"{base}/v1", max_retries=0, http_client=transport) as client,
    ):
        response: Final = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": f"hello {uuid.uuid4().hex}"}]
        )
        return response.headers.get(_HEADER)


def _openai_responses(base: str, key: str, model: str) -> str | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        openai.OpenAI(api_key=key, base_url=f"{base}/v1", max_retries=0, http_client=transport) as client,
    ):
        return client.responses.with_raw_response.create(model=model, input=f"hello {uuid.uuid4().hex}").headers.get(
            _HEADER
        )


def _async_openai_streamed_chat(base: str, key: str, model: str) -> str | None:
    async def run() -> str | None:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False) as transport,
            openai.AsyncOpenAI(api_key=key, base_url=f"{base}/v1", max_retries=0, http_client=transport) as client,
            client.chat.completions.with_streaming_response.create(
                model=model, messages=[{"role": "user", "content": f"hello {uuid.uuid4().hex}"}], stream=True
            ) as response,
        ):
            lines: Final = [line async for line in response.iter_lines()]
            assert "data: [DONE]" in lines, lines
            return response.headers.get(_HEADER)

    return asyncio.run(run())


def _async_openai_streamed_responses(base: str, key: str, model: str) -> str | None:
    async def run() -> str | None:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False) as transport,
            openai.AsyncOpenAI(api_key=key, base_url=f"{base}/v1", max_retries=0, http_client=transport) as client,
            client.responses.with_streaming_response.create(
                model=model, input=f"hello {uuid.uuid4().hex}", stream=True
            ) as response,
        ):
            lines: Final = [line async for line in response.iter_lines()]
            assert any("response.completed" in line for line in lines), lines
            return response.headers.get(_HEADER)

    return asyncio.run(run())


def _anthropic_streamed_messages(base: str, key: str, model: str) -> str | None:
    with (
        httpx.Client(timeout=30, trust_env=False) as transport,
        anthropic.Anthropic(api_key=key, base_url=base, max_retries=0, http_client=transport) as client,
        client.messages.with_streaming_response.create(
            model=model,
            max_tokens=20,
            messages=[{"role": "user", "content": f"hello {uuid.uuid4().hex}"}],
            stream=True,
        ) as response,
    ):
        lines: Final = tuple(response.iter_lines())
        assert any("message_stop" in line for line in lines), lines
        return response.headers.get(_HEADER)


def _async_anthropic_messages(base: str, key: str, model: str) -> str | None:
    async def run() -> str | None:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False) as transport,
            anthropic.AsyncAnthropic(api_key=key, base_url=base, max_retries=0, http_client=transport) as client,
        ):
            response: Final = await client.messages.with_raw_response.create(
                model=model, max_tokens=20, messages=[{"role": "user", "content": f"hello {uuid.uuid4().hex}"}]
            )
            return response.headers.get(_HEADER)

    return asyncio.run(run())


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(_openai_chat, id="openai-chat"),
        pytest.param(_async_openai_streamed_chat, id="async-openai-chat-stream"),
        pytest.param(_openai_responses, id="openai-responses"),
        pytest.param(_async_openai_streamed_responses, id="async-openai-responses-stream"),
        pytest.param(_anthropic_streamed_messages, id="anthropic-messages-stream"),
        pytest.param(_async_anthropic_messages, id="async-anthropic-messages"),
    ],
)
def test_every_sdk_client_sees_only_the_step_that_ran(rig: _Rig, client: Callable[[str, str, str], str | None]) -> None:
    rig.vendor.drain()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(team_id=scenario.team(metadata={"tags": [_STOPS_AFTER_FIRST]}))
        header: Final = client(str(rig.gateway.client.base_url).rstrip("/"), key, rig.model)
    assert _steps_that_ran(rig) == (_REQUEST_FIRST,)
    assert header == _REQUEST_FIRST


async def _send_concurrently(
    rig: _Rig, keys: Mapping[str, str], cases: tuple[tuple[str, str, bool], ...]
) -> tuple[httpx.Response, ...]:
    async with httpx.AsyncClient(base_url=rig.gateway.client.base_url, timeout=60, trust_env=False) as client:
        return tuple(
            await asyncio.gather(
                *(
                    client.post(
                        path, json=_body(path, rig.model, stream), headers={"Authorization": f"Bearer {keys[tag]}"}
                    )
                    for tag, path, stream in cases
                )
            )
        )


def test_a_concurrent_burst_names_only_each_requests_own_steps(rig: _Rig) -> None:
    expected: Final = MappingProxyType(
        {
            _STOPS_AFTER_FIRST: _REQUEST_FIRST,
            _RUNS_EVERY_STEP: f"{_REQUEST_FIRST},{_REQUEST_SECOND}",
            _RESPONSE_STOPS_AFTER_FIRST: _RESPONSE_FIRST,
        }
    )
    cases: Final = tuple(itertools.product(expected, _ENDPOINTS, (False, True))) * 2
    with rig.gateway.scenario() as scenario:
        keys: Final = MappingProxyType(
            {tag: scenario.key(team_id=scenario.team(metadata={"tags": [tag]})) for tag in expected}
        )
        responses: Final = asyncio.run(_send_concurrently(rig, keys, cases))
    rig.vendor.drain()
    rig.upstream.drain()
    assert [
        (tag, path, stream, response.status_code, response.headers.get(_HEADER))
        for (tag, path, stream), response in zip(cases, responses, strict=True)
    ] == [(tag, path, stream, 200, expected[tag]) for tag, path, stream in cases]
