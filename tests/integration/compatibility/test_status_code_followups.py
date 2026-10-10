from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import BinaryResponse, JsonResponse

_ROUTER_DEFAULTS: Final[dict[str, JsonValue]] = {
    "prompt": "router default prompt",
    "input": "router default input",
    "max_tokens": 32,
}
_MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg-$UNIQUE_ID",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "scripted"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_CHAT: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-$UNIQUE_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"},
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
_VECTOR_STORE_FILE: Final[dict[str, JsonValue]] = {
    "id": "file-audit",
    "object": "vector_store.file",
    "created_at": 1,
    "usage_bytes": 0,
    "vector_store_id": "vs-audit",
    "status": "completed",
    "last_error": None,
    "attributes": {},
}
_SEVENTEEN_ATTRIBUTES: Final[dict[str, JsonValue]] = {f"key_{index:02d}": "v" for index in range(17)}


def _openai_error(message: str, param: str) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        status=400,
        body={"error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}},
    )


class _Observations:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(
                client.get(f"{self.url}/__observations?include_method=true").json()
            )
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self.items = (*self.items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self.items

    def provider_calls(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            item
            for item in self.items
            if f"/{identity}/" in str(item.get("path"))
            and not (item.get("method") == "GET" and str(item.get("path", "")).endswith(("/v1/models", "/models")))
        )


def _scripted(scenario: Scenario, prefix: str, response: JsonResponse | BinaryResponse) -> tuple[str, ScenarioHandle]:
    identity: Final = f"{prefix}-{uuid.uuid4().hex}"
    handle: Final = register_scenario(identity, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return identity, handle


def _outbound(gateway: Gateway, identity: str) -> dict[str, JsonValue]:
    observations: Final = _Observations(gateway.upstream_url)
    eventually(observations.read, lambda _items: len(observations.provider_calls(identity)) == 1, seconds=20)
    return object_value(observations.provider_calls(identity)[0]["body"])


def _assert_no_provider_call(gateway: Gateway, identity: str) -> None:
    observations: Final = _Observations(gateway.upstream_url)
    observations.read()
    assert observations.provider_calls(identity) == ()


def _assert_no_call_to_deployment(gateway: Gateway, model_name: str) -> None:
    observations: Final = _Observations(gateway.upstream_url)
    observations.read()
    calls: Final = tuple(
        f"{item.get('method')} {item.get('path')}"
        for item in observations.items
        if f"/{model_name}-" in str(item.get("path")) and not str(item.get("path")).endswith("/models")
    )
    assert calls == (), calls


def _post(gateway: Gateway, path: str, body: dict[str, JsonValue]) -> httpx.Response:
    return gateway.client.post(path, json=body, headers={"Authorization": f"Bearer {gateway.key}"})


def _invalid_request(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_python(response.json())["error"])
    assert error.get("type") == "invalid_request_error", response.text
    return error


def _write_config(path: Path, config: dict[str, JsonValue]) -> Path:
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def router_default_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    """A proxy whose router-wide defaults name positional and route-required params,
    with client-side `user_config` allowed so a null value can be sent."""
    directory: Final = tmp_path_factory.mktemp("router-defaults")
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        image_identity, image_handle = _scripted(
            scenario, "followup-image", JsonResponse(content_type="application/json", body={"created": 1, "data": []})
        )
        text_identity, text_handle = _scripted(
            scenario, "followup-text", JsonResponse(content_type="application/json", body={"choices": []})
        )
        speech_identity, speech_handle = _scripted(
            scenario, "followup-speech", BinaryResponse(content_type="audio/mpeg", length=16)
        )
        message_identity, message_handle = _scripted(
            scenario, "followup-message", JsonResponse(content_type="application/json", body=_MESSAGE)
        )
        config: Final = _write_config(
            directory / "router-defaults.yaml",
            {
                "model_list": [
                    {
                        "model_name": "followup-image",
                        "litellm_params": {
                            "model": "openai/gpt-image-1",
                            "api_base": image_handle.api_base(),
                            "api_key": image_identity,
                        },
                    },
                    {
                        "model_name": "followup-text",
                        "litellm_params": {
                            "model": "openai/gpt-3.5-turbo-instruct",
                            "api_base": text_handle.api_base(),
                            "api_key": text_identity,
                        },
                    },
                    {
                        "model_name": "followup-speech",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini-tts",
                            "api_base": speech_handle.api_base(),
                            "api_key": speech_identity,
                        },
                    },
                    {
                        "model_name": "followup-message",
                        "litellm_params": {
                            "model": "anthropic/claude-haiku-4-5",
                            "api_base": message_handle.api_base(),
                            "api_key": message_identity,
                        },
                    },
                ],
                "router_settings": {"default_litellm_params": _ROUTER_DEFAULTS},
                "general_settings": {"allow_client_side_credentials": True},
            },
        )
        with owned_proxy_process(gateway, directory, {}, config=config) as owned:
            yield Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)


@pytest.fixture(scope="module")
def wildcard_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    """A proxy whose only deployment is the `*` wildcard."""
    directory: Final = tmp_path_factory.mktemp("wildcard")
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        identity, handle = _scripted(
            scenario, "followup-wildcard", JsonResponse(content_type="application/json", body=_MESSAGE)
        )
        config: Final = _write_config(
            directory / "wildcard.yaml",
            {
                "model_list": [
                    {
                        "model_name": "*",
                        "litellm_params": {"model": "openai/*", "api_base": handle.api_base(), "api_key": identity},
                    }
                ]
            },
        )
        with owned_proxy_process(gateway, directory, {}, config=config) as owned:
            yield Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url), identity


@pytest.fixture(scope="module")
def fixed_target_wildcard_proxy(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, str]]:
    """A proxy whose only deployment is a `*` wildcard pinned to one fixed upstream model."""
    directory: Final = tmp_path_factory.mktemp("fixed-wildcard")
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        identity, handle = _scripted(
            scenario, "followup-fixed-wildcard", JsonResponse(content_type="application/json", body=_CHAT)
        )
        config: Final = _write_config(
            directory / "fixed-wildcard.yaml",
            {
                "model_list": [
                    {
                        "model_name": "*",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": handle.api_base(),
                            "api_key": identity,
                        },
                    }
                ]
            },
        )
        with owned_proxy_process(gateway, directory, {}, config=config) as owned:
            yield Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url), identity


def test_ocr_multipart_without_file_returns_400(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        identity, handle = _scripted(
            scenario, "followup-ocr", JsonResponse(content_type="application/json", body={"pages": []})
        )
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=handle.api_base(), api_key=identity)
        response: Final = gateway.client.post(
            "/v1/ocr", files={"model": (None, model)}, headers={"Authorization": f"Bearer {gateway.key}"}
        )
        error: Final = _invalid_request(response)
        assert error.get("param") == "file", response.text
        _assert_no_provider_call(gateway, identity)


def test_ocr_empty_file_returns_400(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        identity, handle = _scripted(
            scenario, "followup-ocr-empty", JsonResponse(content_type="application/json", body={"pages": []})
        )
        model: Final = scenario.model(model="mistral/mistral-ocr-latest", api_base=handle.api_base(), api_key=identity)
        response: Final = gateway.request_multipart(
            "/v1/ocr", {"model": model}, {"file": ("empty.pdf", b"", "application/pdf")}
        )
        error: Final = _invalid_request(response)
        assert error.get("message") == "Uploaded file is empty", response.text
        _assert_no_provider_call(gateway, identity)


@pytest.mark.parametrize(
    ("path", "body"),
    (
        ("/v1/responses", {"input": "Test", "store": False}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]}),
    ),
    ids=("responses", "chat"),
)
def test_missing_model_with_wildcard_deployment_returns_400(
    wildcard_proxy: tuple[Gateway, str], path: str, body: dict[str, JsonValue]
) -> None:
    proxy, identity = wildcard_proxy
    response: Final = _post(proxy, path, body)
    _invalid_request(response)
    _assert_no_provider_call(proxy, identity)


def test_messages_missing_model_with_wildcard_deployment_returns_400(wildcard_proxy: tuple[Gateway, str]) -> None:
    proxy, identity = wildcard_proxy
    response: Final = _post(
        proxy, "/v1/messages", {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}
    )
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_python(response.json())["error"])
    assert error.get("type") == "invalid_request_error", response.text
    _assert_no_provider_call(proxy, identity)


def test_missing_model_with_fixed_target_wildcard_reaches_that_model(
    fixed_target_wildcard_proxy: tuple[Gateway, str],
) -> None:
    proxy, identity = fixed_target_wildcard_proxy
    response: Final = _post(proxy, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
    assert response.status_code == 200, response.text
    assert _outbound(proxy, identity).get("model") == "gpt-4o-mini"


@pytest.mark.parametrize("max_output_tokens", (-1, 0), ids=("negative", "zero"))
def test_responses_non_positive_max_output_tokens_returns_provider_400(
    gateway: Gateway, max_output_tokens: int
) -> None:
    message: Final = f"Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got {max_output_tokens} instead."
    with gateway.scenario() as scenario:
        identity, handle = _scripted(scenario, "followup-max-output", _openai_error(message, "max_output_tokens"))
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=handle.api_base(), api_key=identity)
        response: Final = _post(
            gateway,
            "/v1/responses",
            {"model": model, "input": "Say pong", "store": False, "max_output_tokens": max_output_tokens},
        )
        assert response.status_code == 400, response.text
        assert message in response.text, response.text
        assert _outbound(gateway, identity).get("max_output_tokens") == max_output_tokens


def test_responses_small_positive_max_output_tokens_is_still_raised_to_minimum(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        identity, handle = _scripted(
            scenario, "followup-max-output-min", JsonResponse(content_type="application/json", body=_RESPONSE)
        )
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=handle.api_base(), api_key=identity)
        response: Final = _post(
            gateway, "/v1/responses", {"model": model, "input": "Say pong", "store": False, "max_output_tokens": 1}
        )
        assert response.status_code == 200, response.text
        assert _outbound(gateway, identity).get("max_output_tokens") == 16


def test_team_daily_activity_end_before_start_returns_400(gateway: Gateway) -> None:
    response: Final = gateway.request(
        "GET",
        "/team/daily/activity",
        params={"start_date": "2026-10-01", "end_date": "2026-09-24", "page": "1"},
    )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": {"error": "end_date must be on or after start_date"}}, response.text


def _registered_store(gateway: Gateway, scenario: Scenario, identity: str, handle: ScenarioHandle) -> str:
    store: Final = f"vs-{uuid.uuid4().hex}"
    model: Final = scenario.model(model="openai/text-embedding-3-small", api_base=handle.api_base(), api_key=identity)
    gateway.post(
        "/vector_store/new",
        {
            "vector_store_id": store,
            "custom_llm_provider": "openai",
            "litellm_params": {"model": model, "api_base": handle.api_base(), "api_key": identity},
        },
    )
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store})
    return store


@pytest.mark.parametrize(
    ("path", "body"),
    (
        ("/v1/vector_stores/{store}/files/file-audit", {"attributes": _SEVENTEEN_ATTRIBUTES}),
        ("/v1/vector_stores/{store}/files", {"file_id": "file-audit", "attributes": _SEVENTEEN_ATTRIBUTES}),
    ),
    ids=("update", "create"),
)
def test_vector_store_file_attributes_over_provider_limit_reach_provider(
    gateway: Gateway, path: str, body: dict[str, JsonValue]
) -> None:
    message: Final = (
        "Invalid 'attributes': too many properties. "
        "Expected an object with at most 16 properties, but got an object with 17 properties instead."
    )
    with gateway.scenario() as scenario:
        identity, handle = _scripted(scenario, "followup-attributes", _openai_error(message, "attributes"))
        store: Final = _registered_store(gateway, scenario, identity, handle)
        response: Final = _post(gateway, path.format(store=store), body)
        assert response.status_code == 400, response.text
        assert message in response.text, response.text
        assert _outbound(gateway, identity).get("attributes") == _SEVENTEEN_ATTRIBUTES


@pytest.mark.parametrize(
    ("path", "body", "missing"),
    (
        ("/v1/images/generations", {"model": "followup-image"}, "prompt"),
        ("/v1/completions", {"model": "followup-text"}, "prompt"),
        ("/v1/audio/speech", {"model": "followup-speech", "voice": "alloy"}, "input"),
    ),
    ids=("image-prompt", "text-completion-prompt", "speech-input"),
)
def test_router_default_for_positional_param_returns_400(
    router_default_proxy: Gateway, path: str, body: dict[str, JsonValue], missing: str
) -> None:
    response: Final = _post(router_default_proxy, path, body)
    error: Final = _invalid_request(response)
    assert error.get("param") == missing, response.text
    _assert_no_call_to_deployment(router_default_proxy, str(body["model"]))


def test_router_default_for_model_outside_router_returns_400(router_default_proxy: Gateway) -> None:
    response: Final = _post(router_default_proxy, "/v1/moderations", {"model": "omni-moderation-2024-09-26"})
    error: Final = _invalid_request(response)
    assert error.get("param") == "input", response.text


@pytest.mark.parametrize(
    ("body", "expected_max_tokens"),
    (
        ({"messages": [{"role": "user", "content": "router default"}]}, 32),
        ({"messages": [{"role": "user", "content": "explicit"}], "max_tokens": 8}, 8),
    ),
    ids=("router-default", "explicit"),
)
def test_null_user_config_is_treated_as_absent(
    router_default_proxy: Gateway, body: dict[str, JsonValue], expected_max_tokens: int
) -> None:
    observations: Final = _Observations(router_default_proxy.upstream_url)
    observations.read()
    before: Final = len(observations.items)
    response: Final = _post(
        router_default_proxy, "/v1/messages", {"model": "followup-message", "user_config": None, **body}
    )
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_python(response.json()).get("type") == "message", response.text
    eventually(
        observations.read,
        lambda items: any("followup-message-" in str(item.get("path")) for item in items[before:]),
        seconds=20,
    )
    outbound: Final = next(
        object_value(item["body"])
        for item in observations.items[before:]
        if "followup-message-" in str(item.get("path"))
    )
    assert outbound.get("max_tokens") == expected_max_tokens
