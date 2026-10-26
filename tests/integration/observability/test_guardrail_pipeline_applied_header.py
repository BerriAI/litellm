from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_HEADER: Final = "x-litellm-applied-guardrails"
_VENDOR_PATH: Final = "/beta/litellm_basic_guardrail_api"
_PROMPT_SHIELD_PATH: Final = "/contentsafety/text:shieldPrompt"
_PROMPT_SHIELD: Final = "prompt-shield-step"
_REQUEST_FIRST: Final = "request-first-step"
_REQUEST_SECOND: Final = "request-second-step"
_REQUEST_BLOCKER: Final = "request-blocking-step"
_RESPONSE_FIRST: Final = "response-first-step"
_RESPONSE_SECOND: Final = "response-second-step"
_RESPONSE_BLOCKER: Final = "response-blocking-step"
_LIVE_FIRST: Final = "live-first-step"
_LIVE_SECOND: Final = "live-second-step"
_UNGATED_STEP: Final = "ungated-step"
_STOPS_AFTER_FIRST: Final = "stops-after-first"
_RUNS_EVERY_STEP: Final = "runs-every-step"
_BLOCKS_AT_FIRST: Final = "blocks-at-first"
_RESPONSE_STOPS_AFTER_FIRST: Final = "response-stops-after-first"
_SHIELDS_THEN_CHECKS: Final = "shields-then-checks"
_MOVES_PAST_FIRST: Final = "moves-past-first"
_RESPONSE_MOVES_PAST_FIRST: Final = "response-moves-past-first"
_SCANS_AFTER_RELEASE: Final = "scans-after-release"
_CANNOT_HOLD_STREAM: Final = "cannot-hold-stream"


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    vendor: Wire
    upstream: Wire
    model: str


def _completion_chunk(identity: str, delta: dict[str, str], finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _completion(request: Request) -> Reply:
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
    if request.target.startswith(f"/{_PROMPT_SHIELD}{_PROMPT_SHIELD_PATH}?"):
        return Reply(
            body=json.dumps({"userPromptAnalysis": {"attackDetected": False}, "documentsAnalysis": []}).encode()
        )
    if request.target.startswith(f"/{_UNGATED_STEP}/"):
        return Reply(body=json.dumps({"finalAction": "NONE", "appliedPolicies": []}).encode())
    assert request.target.endswith(_VENDOR_PATH), request.target
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


def _policy(mode: str, steps: tuple[tuple[str, str, str], ...]) -> dict[str, object]:
    return {
        "guardrails": {"add": [guardrail for guardrail, _, _ in steps]},
        "pipeline": {
            "mode": mode,
            "steps": [
                {"guardrail": guardrail, "on_pass": on_pass, "on_fail": on_fail}
                for guardrail, on_pass, on_fail in steps
            ],
        },
    }


def _config(directory: Path, vendor: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["guardrails"] = [
        *(_guardrail(name, "pre_call", vendor) for name in (_REQUEST_FIRST, _REQUEST_SECOND, _REQUEST_BLOCKER)),
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
    }
    configuration["policy_attachments"] = [{"policy": policy, "tags": [policy]} for policy in configuration["policies"]]
    path: Final = directory / "guardrail-pipeline-applied-header.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("guardrail-pipeline-applied-header")
    with (
        wire_server(_vendor) as vendor,
        wire_server(_completion) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_config(directory, vendor)) as owned,
        owned.scenario() as scenario,
    ):
        yield _Rig(owned, vendor, upstream, scenario.model(api_base=f"{upstream.url}/v1", api_key="sk-fixture"))


def _ask_as_team_tagged(rig: _Rig, tag: str, *, stream: bool = False) -> httpx.Response:
    rig.vendor.drain()
    rig.upstream.drain()
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"tags": [tag]})
        key: Final = scenario.key(team_id=team)
        return rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": f"hello {uuid.uuid4().hex}"}],
                "stream": stream,
            },
            key=key,
        )


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
