import asyncio
import json
import uuid
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def _sse(payload: dict[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _chat(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    if request.target.startswith("/v1/responses"):
        return Reply(
            body=json.dumps(
                {
                    "id": f"resp_{uuid.uuid4().hex}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": body["model"],
                    "output": [
                        {
                            "type": "message",
                            "id": f"msg_{uuid.uuid4().hex}",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                }
            ).encode()
        )
    assert request.target.startswith("/v1/chat/completions"), request.target
    completion_id: Final = f"chatcmpl-{uuid.uuid4().hex}"
    if body.get("stream"):
        chunk: Final[dict[str, JsonValue]] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        }
        terminal: Final[dict[str, JsonValue]] = {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": _USAGE,
        }
        return Reply(content_type="text/event-stream", chunks=(_sse(chunk), _sse(terminal), b"data: [DONE]\n\n"))
    return Reply(
        body=json.dumps(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": _USAGE,
            }
        ).encode()
    )


def _logged_api_base(model_group: str) -> JsonValue:
    rows: Final = eventually(
        lambda: read_rows('SELECT api_base FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model_group,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]["api_base"]


async def _async_stream_call_id(base_url: str, key: str, model: str, prompt: str) -> str:
    client: Final = openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0)
    async with client.chat.completions.with_streaming_response.create(
        model=model, messages=[{"role": "user", "content": prompt}], stream=True
    ) as response:
        async for _ in response.iter_lines():
            pass
        return response.headers["x-litellm-call-id"]


@pytest.mark.parametrize(
    ("surface", "logged_suffix"),
    (("chat-sync", "/"), ("chat-async-stream", ""), ("messages", "/responses"), ("responses", "/responses")),
)
def test_spend_row_records_the_deployment_api_base_on_every_surface(
    gateway: Gateway, surface: str, logged_suffix: str
) -> None:
    with wire_server(_chat) as wire, gateway.scenario() as scenario:
        api_base: Final = f"{wire.url}/v1"
        model: Final = scenario.model(api_base=api_base, num_retries=0)
        prompt: Final = f"api base audit {uuid.uuid4().hex}"
        base_url: Final = str(gateway.client.base_url)
        if surface == "chat-sync":
            raw = openai.OpenAI(base_url=f"{base_url}/v1", api_key=gateway.key, max_retries=0)
            completion = raw.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            assert completion.parse().choices[0].message.content == "ok"
            call_id = completion.headers["x-litellm-call-id"]
        elif surface == "chat-async-stream":
            call_id = asyncio.run(_async_stream_call_id(f"{base_url}/v1", gateway.key, model, prompt))
        elif surface == "messages":
            messages = anthropic.Anthropic(base_url=base_url, api_key=gateway.key, max_retries=0)
            message = messages.messages.with_raw_response.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": prompt}]
            )
            assert message.parse().content[0].type == "text"
            call_id = message.headers["x-litellm-call-id"]
        else:
            response = gateway.request("POST", "/v1/responses", {"model": model, "input": prompt})
            assert response.status_code == 200, response.text
            call_id = string_value(response.headers["x-litellm-call-id"])
        received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    assert call_id, surface
    assert _logged_api_base(model) == f"{api_base}{logged_suffix}", (surface, call_id)
