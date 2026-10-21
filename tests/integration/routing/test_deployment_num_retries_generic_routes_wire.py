import json
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.anthropic_sse import (
    Attempts,
    event_type,
    event_types,
    parse_sse,
    user_prompt,
)
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.openai_wire import (
    answering_model_discovery,
    chat_reply,
    openai_error,
    posted_targets,
    responses_reply,
)
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_TEXT: Final = "Hello"
_OPENAI_MODEL: Final = "gpt-4o-mini"
_PROVIDER_KEY: Final = "integration-provider-key"
_GEMINI_MODEL: Final = "gemini-2.5-flash"
_GEMINI_KEY: Final = "synthetic-gemini-key"
_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])


def _marker() -> str:
    return "generic-retry-" + uuid.uuid4().hex


def _openai_upstream(
    marker: str, target: str, attempts: Attempts, served: Callable[[int, bool], Reply]
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", target), request
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["model"] == _OPENAI_MODEL, body
        assert "num_retries" not in body, body
        attempt: Final = attempts.record(marker)
        if attempt == 1:
            return openai_error(500)
        return served(attempt, body.get("stream") is True)

    return answering_model_discovery(respond)


def _responses_served(marker: str) -> Callable[[int, bool], Reply]:
    def served(attempt: int, streamed: bool) -> Reply:
        return responses_reply(f"resp_{marker}_a{attempt}", _OPENAI_MODEL, _TEXT, stream=streamed)

    return served


def _chat_served(marker: str) -> Callable[[int, bool], Reply]:
    def served(attempt: int, streamed: bool) -> Reply:
        return chat_reply(f"chatcmpl-{marker}-a{attempt}", _OPENAI_MODEL, _TEXT, stream=streamed)

    return served


def _success_rows(model: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda rows: len(rows) >= 1,
        seconds=70,
    )


def _assert_two_attempts_one_success(wire: Wire, target: str, model: str) -> None:
    assert posted_targets(wire) == (target,) * 2
    assert [row["status"] for row in _success_rows(model)] == ["success"]


def _responses_text(response_text: str, stream: bool) -> str:
    if not stream:
        payload: Final = object_value(json.loads(response_text))
        content: Final = _OBJECTS.validate_python(_OBJECTS.validate_python(payload["output"])[0]["content"])
        return str(content[0]["text"])
    events: Final = parse_sse(response_text)
    assert event_types(events)[-1] == "response.completed", events
    return "".join(str(event.data["delta"]) for event in events if event_type(event) == "response.output_text.delta")


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_responses_rejected_before_the_stream_opens_is_retried_per_the_deployment_budget(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    served: Final = _responses_served(marker)
    with (
        wire_server(_openai_upstream(marker, "/v1/responses", attempts, served)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=1)
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": marker, "stream": stream})
        assert response.status_code == 200, response.text
        assert _responses_text(response.text, stream) == _TEXT, response.text
        _assert_two_attempts_one_success(wire, "/v1/responses", model)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_chat_completions_control_keeps_retrying_per_the_deployment_budget(gateway: Gateway, stream: bool) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    served: Final = _chat_served(marker)
    with (
        wire_server(_openai_upstream(marker, "/v1/chat/completions", attempts, served)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=1)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], "stream": stream},
        )
        assert response.status_code == 200, response.text
        assert f"chatcmpl-{marker}-a2" in response.text, response.text
        assert _TEXT in response.text, response.text
        _assert_two_attempts_one_success(wire, "/v1/chat/completions", model)


def test_responses_request_budget_still_wins_over_a_zero_deployment_budget(gateway: Gateway) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    served: Final = _responses_served(marker)
    with (
        wire_server(_openai_upstream(marker, "/v1/responses", attempts, served)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": marker, "num_retries": 1})
        assert response.status_code == 200, response.text
        assert _responses_text(response.text, False) == _TEXT, response.text
        _assert_two_attempts_one_success(wire, "/v1/responses", model)


def _vllm_passthrough_upstream(marker: str, attempts: Attempts) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions"), request
        body: Final = object_value(json.loads(request.body))
        assert user_prompt(body) == marker, body
        attempt: Final = attempts.record(marker)
        if attempt == 1:
            return openai_error(500)
        return chat_reply(f"chatcmpl-{marker}-a{attempt}", _OPENAI_MODEL, _TEXT, stream=body.get("stream") is True)

    return answering_model_discovery(respond)


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vllm_passthrough_rejected_before_it_opens_is_retried_per_the_deployment_budget(
    gateway: Gateway, stream: bool
) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_vllm_passthrough_upstream(marker, attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_OPENAI_MODEL}", api_base=wire.url + "/v1", num_retries=1)
        response: Final = gateway.request(
            "POST",
            "/vllm/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": marker}], **({"stream": True} if stream else {})},
        )
        assert response.status_code == 200, response.text
        assert f"chatcmpl-{marker}-a2" in response.text, response.text
        assert _TEXT in response.text, response.text
        assert posted_targets(wire) == ("/v1/chat/completions",) * 2


def _gemini_upstream(marker: str, attempts: Attempts) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request
        assert request.target.split("?")[0] == f"/models/{_GEMINI_MODEL}:generateContent", request.target
        assert request.headers["x-goog-api-key"] == _GEMINI_KEY, request.headers
        body: Final = object_value(json.loads(request.body))
        assert body["contents"] == [{"role": "user", "parts": [{"text": marker}]}], body
        attempt: Final = attempts.record(marker)
        if attempt == 1:
            return Reply(
                status=500,
                body=json.dumps({"error": {"code": 500, "message": "scripted", "status": "INTERNAL"}}).encode(),
            )
        return Reply(
            body=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": f"{_TEXT} a{attempt}"}], "role": "model"},
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
                    "modelVersion": _GEMINI_MODEL,
                }
            ).encode()
        )

    return respond


def test_gemini_generate_content_rejected_is_retried_per_the_deployment_budget(gateway: Gateway) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_gemini_upstream(marker, attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"gemini/{_GEMINI_MODEL}", api_base=wire.url, api_key=_GEMINI_KEY, num_retries=1
        )
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": marker}]}]},
        )
        assert response.status_code == 200, response.text
        assert f"{_TEXT} a2" in response.text, response.text
        assert [request.target.split("?")[0] for request in wire.drain()] == [
            f"/models/{_GEMINI_MODEL}:generateContent"
        ] * 2
        assert [row["status"] for row in _success_rows(model)] == ["success"]


def _fine_tuning_list_upstream(marker: str, attempts: Attempts) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        assert request.target.split("?")[0] == "/v1/fine_tuning/jobs", request.target
        assert request.headers["authorization"] == f"Bearer {_PROVIDER_KEY}", request.headers
        attempt: Final = attempts.record(marker)
        if attempt == 1:
            return openai_error(500)
        return Reply(body=json.dumps({"object": "list", "data": [], "has_more": False}).encode())

    return answering_model_discovery(respond)


def test_fine_tuning_jobs_list_rejected_is_retried_per_the_deployment_budget(gateway: Gateway) -> None:
    marker: Final = _marker()
    attempts: Final = Attempts()
    with wire_server(_fine_tuning_list_upstream(marker, attempts)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=1)
        response: Final = gateway.request(
            "GET", "/v1/fine_tuning/jobs", params={"target_model_names": model, "limit": "5"}
        )
        assert response.status_code == 200, response.text
        assert object_value(json.loads(response.text))["data"] == [], response.text
        assert [request.target.split("?")[0] for request in wire.drain() if request.method == "GET"] == [
            "/v1/fine_tuning/jobs"
        ] * 2
