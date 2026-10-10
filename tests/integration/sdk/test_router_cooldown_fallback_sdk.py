from __future__ import annotations

import json
import random
import uuid
from typing import Final

import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import Router
from litellm.types.router import DeploymentTypedDict

_MODEL: Final = "gpt-5.4"
_API_KEY: Final = "migration-fallback-test-key"
_ANSWER: Final = "served by"
_UNAUTHORIZED: Final = Reply(
    status=401,
    body=json.dumps(
        {
            "error": {
                "message": "Incorrect API key provided: invalid test key.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
    ).encode(),
)
_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _peer(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())

    deployment, _, route = request.target.lstrip("/").partition("/")
    assert route == "chat/completions", request.target
    if deployment == "primary":
        return _UNAUTHORIZED

    body: Final = _JSON.validate_json(request.body)
    return chat_reply(
        f"chatcmpl-{deployment}-{uuid.uuid4().hex}",
        _MODEL,
        f"{_ANSWER} {deployment}",
        stream=body.get("stream") is True,
    )


def _deployment(model_name: str, api_base: str, deployment_id: str) -> DeploymentTypedDict:
    return DeploymentTypedDict(
        model_name=model_name,
        litellm_params={
            "model": f"openai/{_MODEL}",
            "api_base": api_base,
            "api_key": _API_KEY,
        },
        model_info={"id": deployment_id},
    )


@pytest.mark.asyncio
async def test_fallback_response_reports_the_attempted_fallback() -> None:
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                _deployment("primary", f"{wire.url}/primary", "primary-deployment"),
                _deployment("backup", f"{wire.url}/backup", "backup-deployment"),
            ],
            fallbacks=[{"primary": ["backup"]}],
            num_retries=0,
            disable_cooldowns=True,
        )
        response: Final = await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": f"fallback {uuid.uuid4().hex}"}],
            include_fallback_errors=True,
        )
        requests: Final = wire.drain()

    assert response.choices[0].message.content == f"{_ANSWER} backup"
    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1
    assert tuple(request.target for request in requests) == (
        "/primary/chat/completions",
        "/backup/chat/completions",
    )


@pytest.mark.asyncio
async def test_cooldown_skips_only_the_failed_duplicate_model_deployment() -> None:
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                _deployment("duplicate-model", f"{wire.url}/primary", "primary-deployment"),
                _deployment("duplicate-model", f"{wire.url}/backup", "backup-deployment"),
            ],
            allowed_fails=0,
            num_retries=0,
        )
        with pytest.raises(litellm.AuthenticationError):
            await router.acompletion(
                model="primary-deployment",
                messages=[{"role": "user", "content": f"cooldown {uuid.uuid4().hex}"}],
            )
        random.seed(1)
        response: Final = await router.acompletion(
            model="duplicate-model",
            messages=[{"role": "user", "content": f"cooldown {uuid.uuid4().hex}"}],
        )
        requests: Final = wire.drain()

    assert response.choices[0].message.content == f"{_ANSWER} backup"
    assert tuple(request.target for request in requests) == (
        "/primary/chat/completions",
        "/backup/chat/completions",
    )
