"""An upstream OpenAI-compatible 403 reaches the caller as a permission-denied error, sent once (#34503).

`test_regression_*` fails on a build without the fix, `test_guard_*` already holds and must keep holding.
"""

import json
import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.wire import Reply, wire_server

_UPSTREAM_KEY: Final = "synthetic-openai-compatible-key"
_RETRIES: Final = 2
_FORBIDDEN_MESSAGE: Final = "insufficient permissions to access the resource"
_CHAT: Final = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hello"}]})
_EMBEDDING: Final = ("/v1/embeddings", {"input": ["hello"]})


def _forbidden_reply() -> Reply:
    return Reply(
        status=403,
        body=json.dumps(
            {"error": {"code": 403, "message": _FORBIDDEN_MESSAGE, "type": "forbidden", "param": ""}}
        ).encode(),
    )


def _openai_style_error_reply(status: int, message: str, error_type: str) -> Reply:
    return Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": error_type, "code": None}}).encode(),
    )


def _error_type_and_message(response: httpx.Response) -> tuple[str, str]:
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    return str(error["type"]), str(error["message"])


@pytest.mark.parametrize(
    ("deployment_model", "route", "payload"),
    (
        ("deepseek/deepseek-chat", *_CHAT),
        ("together_ai/BAAI/bge-base-en-v1.5", *_EMBEDDING),
        ("hostinger_router/llama-3.1-8b-instruct", *_CHAT),
        ("openai/gpt-4o-mini", *_CHAT),
    ),
    ids=("deepseek-chat", "together-ai-embedding", "hostinger-router-chat", "openai-chat"),
)
def test_regression_openai_compatible_upstream_403_is_permission_denied_and_not_retried(
    gateway: Gateway, deployment_model: str, route: str, payload: dict[str, object]
) -> None:
    """an OpenAI-compatible provider's 403 surfaces as PermissionDeniedError and is sent upstream once"""
    with wire_server(lambda _: _forbidden_reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=deployment_model, api_base=f"{wire.url}/v1", api_key=_UPSTREAM_KEY, num_retries=_RETRIES
        )
        response: Final = gateway.request("POST", route, {"model": model, **payload})
        error_type, message = _error_type_and_message(response)
        upstream_calls: Final = [(request.method, request.target) for request in wire.drain()]
        assert (response.status_code, error_type, message.split(":", 1)[0], upstream_calls) == (
            403,
            "permission_error",
            "litellm.PermissionDeniedError",
            [("POST", route)],
        ), response.text
        assert _FORBIDDEN_MESSAGE in message, response.text


def test_guard_deepseek_upstream_401_is_authentication_error_and_not_retried(gateway: Gateway) -> None:
    """a DeepSeek 401 still surfaces as AuthenticationError and is sent upstream once"""
    upstream_message: Final = f"invalid api key {uuid.uuid4().hex}"
    reply: Final = _openai_style_error_reply(401, upstream_message, "invalid_request_error")
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepseek/deepseek-chat",
            api_base=f"{wire.url}/v1",
            api_key=_UPSTREAM_KEY,
            num_retries=_RETRIES,
        )
        route, payload = _CHAT
        response: Final = gateway.request("POST", route, {"model": model, **payload})
        error_type, message = _error_type_and_message(response)
        upstream_calls: Final = [(request.method, request.target) for request in wire.drain()]
        assert (response.status_code, error_type, message.split(":", 1)[0], upstream_calls) == (
            401,
            "authentication_error",
            "litellm.AuthenticationError",
            [("POST", route)],
        ), response.text
        assert upstream_message in message, response.text


def test_guard_deepseek_upstream_429_is_rate_limit_error_and_retried(gateway: Gateway) -> None:
    """a DeepSeek 429 still surfaces as RateLimitError and is retried num_retries times"""
    upstream_message: Final = f"rate limited {uuid.uuid4().hex}"
    reply: Final = _openai_style_error_reply(429, upstream_message, "rate_limit_error")
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="deepseek/deepseek-chat",
            api_base=f"{wire.url}/v1",
            api_key=_UPSTREAM_KEY,
            num_retries=_RETRIES,
        )
        route, payload = _CHAT
        response: Final = gateway.request("POST", route, {"model": model, **payload})
        error_type, message = _error_type_and_message(response)
        upstream_calls: Final = [(request.method, request.target) for request in wire.drain()]
        assert (response.status_code, error_type, message.split(":", 1)[0], upstream_calls) == (
            429,
            "throttling_error",
            "litellm.RateLimitError",
            [("POST", route)] * (1 + _RETRIES),
        ), response.text
        assert upstream_message in message, response.text
