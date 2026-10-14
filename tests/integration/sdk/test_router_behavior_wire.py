from __future__ import annotations

import json
import threading
import uuid
from typing import Final

import litellm
import pytest
from integration._support.openai_wire import chat_reply
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "gpt-4o-mini"
_API_KEY: Final = "scripted-router-behavior-key"


def _peer(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if "/timeout/" in request.target:
        return Reply(chunks=(b"{",), gate_after_first=threading.Event())
    if "/stream-timeout/" in request.target:
        threading.Event().wait(timeout=1.0)
        frame: Final = (
            b'data: {"id":"chatcmpl-timeout","object":"chat.completion.chunk",'
            b'"created":1,"model":"gpt-4o-mini","choices":[{"index":0,'
            b'"delta":{"role":"assistant","content":"partial"},"finish_reason":null}]}\n\n'
        )
        return Reply(
            content_type="text/event-stream",
            chunks=(frame, b"data: [DONE]\n\n"),
        )
    if request.target.split("?", 1)[0].endswith("/v1/completions"):
        prompt: Final = body.get("prompt")
        return Reply(
            body=json.dumps(
                {
                    "id": f"cmpl-{uuid.uuid4().hex}",
                    "object": "text_completion",
                    "created": 1,
                    "model": "gpt-3.5-turbo-instruct",
                    "choices": [
                        {
                            "index": 0,
                            "text": f"response to {prompt}",
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                }
            ).encode()
        )
    return chat_reply(
        f"chatcmpl-{uuid.uuid4().hex}",
        _MODEL,
        "wildcard response",
        stream=body.get("stream") is True,
    )


@pytest.mark.asyncio
async def test_router_wildcard_model_routes_to_scripted_provider() -> None:
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "openai/*",
                    "litellm_params": {
                        "model": "openai/*",
                        "api_base": f"{wire.url}/v1",
                        "api_key": _API_KEY,
                    },
                    "model_info": {"id": "wildcard-deployment"},
                }
            ],
            num_retries=0,
        )
        response: Final = await router.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "wildcard routing"}],
        )
        requests: Final = wire.drain()

    assert response.choices[0].message.content == "wildcard response"
    assert response._hidden_params["model_id"] == "wildcard-deployment"
    assert tuple(request.target for request in requests) == ("/v1/chat/completions",)
    assert json.loads(requests[0].body)["model"] == _MODEL


@pytest.mark.asyncio
async def test_router_text_completion_reuses_provider_connection() -> None:
    pytest.skip("BUG: router text completion opens a new upstream connection per request")
    prompts: Final = tuple(f"prompt-{index}" for index in range(8))
    with wire_server(_peer, keep_alive=True) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "text-model",
                    "litellm_params": {
                        "model": "text-completion-openai/gpt-3.5-turbo-instruct",
                        "api_base": f"{wire.url}/v1",
                        "api_key": _API_KEY,
                    },
                }
            ],
            num_retries=0,
        )
        responses: Final = tuple(
            [
                await router.atext_completion(model="text-model", prompt=prompt)
                for prompt in prompts
            ]
        )
        requests: Final = wire.drain()
        connections: Final = wire.connections()

    assert tuple(response.choices[0].text for response in responses) == tuple(
        f"response to {prompt}" for prompt in prompts
    )
    assert tuple(
        _JSON_OBJECT.validate_json(request.body)["prompt"] for request in requests
    ) == prompts
    assert len(requests) == len(prompts)
    assert connections == 1


@pytest.mark.asyncio
async def test_router_raises_timeout_for_scripted_slow_endpoint() -> None:
    with wire_server(_peer) as wire:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "slow-model",
                    "litellm_params": {
                        "model": f"openai/{_MODEL}",
                        "api_base": f"{wire.url}/timeout/v1",
                        "api_key": _API_KEY,
                        "timeout": 0.2,
                    },
                }
            ],
            num_retries=0,
        )
        with pytest.raises(litellm.Timeout):
            await router.acompletion(
                model="slow-model",
                messages=[{"role": "user", "content": "timeout"}],
            )
        requests: Final = wire.drain()

    assert tuple(request.target for request in requests) == (
        "/timeout/v1/chat/completions",
    )
    assert json.loads(requests[0].body)["model"] == _MODEL


def test_streaming_completion_times_out_before_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with wire_server(_peer) as wire:
        with pytest.raises(litellm.Timeout):
            litellm.completion(
                model=f"openai/{_MODEL}",
                api_base=f"{wire.url}/stream-timeout/v1",
                api_key=_API_KEY,
                timeout=0.5,
                num_retries=0,
                messages=[{"role": "user", "content": "stream timeout"}],
                stream=True,
            )
        requests: Final = wire.drain()

    assert len(requests) == 1
    assert json.loads(requests[0].body)["stream"] is True
