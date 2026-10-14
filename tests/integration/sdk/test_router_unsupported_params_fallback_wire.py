from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support.wire import Reply, Request, Wire, wire_server

import litellm
from litellm import Router

_MODEL: Final = "gpt-5-wire-unmapped"
_API_KEY: Final = "synthetic-unsupported-params-key"
_PROMPT: Final = "which deployment answers when the requested model rejects a param before any request is sent?"
_CLIENTS: Final = ("sync", "async")
_PRIMARY_TO_BACKUP: Final = [{"primary": ["backup"]}]
_UNSUPPORTED_BY_PRIMARY: Final = {"reasoning_effort": "xhigh"}


def _answers(text: str) -> Reply:
    body: Final = {
        "id": "chatcmpl-unsupported-params-wire",
        "object": "chat.completion",
        "created": 1700000000,
        "model": _MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    return Reply(body=json.dumps(body).encode())


_UPSTREAM_500: Final = Reply(status=500, body=json.dumps({"error": {"message": "overloaded"}}).encode())
_BACKUP_ANSWERS: Final = _answers("answered by the backup")


def _peer(replies: Mapping[str, Reply]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST", request.method
        deployment, _, route = request.target.lstrip("/").partition("/")
        assert route == "chat/completions", request.target
        return replies[deployment]

    return respond


def _deployments_hit(wire: Wire) -> tuple[str, ...]:
    return tuple(request.target.lstrip("/").partition("/")[0] for request in wire.drain())


def _router(wire: Wire, **settings: object) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": f"openai/{_MODEL}", "api_base": f"{wire.url}/primary", "api_key": _API_KEY},
            },
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": f"openai/{_MODEL}",
                    "api_base": f"{wire.url}/backup",
                    "api_key": _API_KEY,
                    "drop_params": True,
                },
            },
        ],
        fallbacks=_PRIMARY_TO_BACKUP,
        num_retries=0,
        disable_cooldowns=True,
        **settings,
    )


def _complete(client: str, router: Router, **request: object) -> str:
    messages: Final = [{"role": "user", "content": _PROMPT}]
    if client == "async":
        response: Final = asyncio.run(router.acompletion(model="primary", messages=messages, **request))
    else:
        response = router.completion(model="primary", messages=messages, **request)
    return str(response.choices[0].message.content)


@pytest.mark.parametrize("client", _CLIENTS)
def test_client_side_param_rejection_reaches_the_caller_without_trying_fallbacks(client: str) -> None:
    with wire_server(_peer({"backup": _BACKUP_ANSWERS})) as wire:
        router: Final = _router(wire, fallback_on_unsupported_params=False)
        with pytest.raises(litellm.UnsupportedParamsError, match="reasoning_effort"):
            _complete(client, router, **_UNSUPPORTED_BY_PRIMARY)
        assert _deployments_hit(wire) == ()


@pytest.mark.parametrize("client", _CLIENTS)
def test_client_side_param_rejection_still_falls_back_by_default(client: str) -> None:
    with wire_server(_peer({"backup": _BACKUP_ANSWERS})) as wire:
        router: Final = _router(wire)
        assert _complete(client, router, **_UNSUPPORTED_BY_PRIMARY) == "answered by the backup"
        assert _deployments_hit(wire) == ("backup",)


@pytest.mark.parametrize("client", _CLIENTS)
def test_upstream_500_still_falls_back_when_param_rejections_raise(client: str) -> None:
    with wire_server(_peer({"primary": _UPSTREAM_500, "backup": _BACKUP_ANSWERS})) as wire:
        router: Final = _router(wire, fallback_on_unsupported_params=False)
        assert _complete(client, router) == "answered by the backup"
        assert _deployments_hit(wire) == ("primary", "backup")
