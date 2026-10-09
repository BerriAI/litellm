"""A streamed /v1/completions bills its tokens without the client asking for stream_options.include_usage.

The scripted OpenAI-compatible upstream sends a usage chunk only when the request asks for one, the way OpenAI
does, so the proxy has to ask on the client's behalf and keep the answer to itself: the spend row of a streamed
text completion with no stream_options equals the include_usage variant's and the non-streamed twin's, and the
client gets no usage frame it did not ask for
"""

import json
from pathlib import Path
from typing import Final
from uuid import uuid4

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROMPT_TOKENS: Final = 19
COMPLETION_TOKENS: Final = 5
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
EXPECTED_SPEND: Final = PROMPT_TOKENS * INPUT_RATE + COMPLETION_TOKENS * OUTPUT_RATE
USAGE: Final[dict[str, JsonValue]] = {
    "prompt_tokens": PROMPT_TOKENS,
    "completion_tokens": COMPLETION_TOKENS,
    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
}
PROMPT: Final = "Say hi"


def _sse_frame(payload: dict[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def _chunk(request_id: str, model: str, delta: dict[str, JsonValue], finish_reason: str | None) -> dict[str, JsonValue]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _respond(request: Request) -> Reply:
    if request.method == "GET" and request.target == "/v1/models":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": "gpt-5.4-mini", "object": "model"}]}).encode())
    assert request.method == "POST", request.method
    assert request.target == "/v1/chat/completions", request.target
    body: Final = json.loads(request.body)
    assert body["messages"] == [{"role": "user", "content": PROMPT}], body
    request_id: Final = f"chatcmpl-{uuid4().hex}"
    model: Final = str(body["model"])
    if body.get("stream") is not True:
        return Reply(
            body=json.dumps(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hi there"}, "finish_reason": "length"}
                    ],
                    "usage": USAGE,
                }
            ).encode()
        )
    stream_options: Final = body.get("stream_options") or {}
    usage_frames: Final = (
        (
            _sse_frame(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [],
                    "usage": USAGE,
                }
            ),
        )
        if stream_options.get("include_usage") is True
        else ()
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse_frame(_chunk(request_id, model, {"role": "assistant", "content": "Hi"}, None)),
            _sse_frame(_chunk(request_id, model, {"content": " there"}, None)),
            _sse_frame(_chunk(request_id, model, {}, "length")),
            *usage_frames,
            b"data: [DONE]\n\n",
        ),
    )


def _proxy_config(directory: Path, model: str, upstream_url: str) -> Path:
    config: Final = directory / "text_completion_stream_usage_cost_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/gpt-5.4-mini",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": "scripted-upstream",
                            "input_cost_per_token": INPUT_RATE,
                            "output_cost_per_token": OUTPUT_RATE,
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "disable_spend_logs": False,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _data_events(body: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(line.removeprefix("data:"))
        for line in body.splitlines()
        if line.startswith("data:") and not line.endswith("[DONE]")
    )


def _carries_text_or_finish(frame: dict[str, JsonValue]) -> bool:
    choices: Final = frame["choices"]
    assert isinstance(choices, list) and len(choices) == 1, frame
    choice: Final = choices[0]
    assert isinstance(choice, dict), frame
    return choice.get("text") is not None or choice.get("finish_reason") is not None


def _billed(row: dict[str, object]) -> tuple[object, object, float]:
    return (row["prompt_tokens"], row["completion_tokens"], float(str(row["spend"])))


@pytest.mark.timeout(180)
def test_text_completion_stream_without_stream_options_bills_like_its_twins(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = f"gpt-5.4-mini-{uuid4().hex}"
    body: Final[dict[str, JsonValue]] = {"model": model, "prompt": PROMPT, "max_tokens": COMPLETION_TOKENS}

    with wire_server(_respond) as wire:
        config: Final = _proxy_config(tmp_path, model, wire.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            silent: Final = candidate.request("POST", "/v1/completions", {**body, "stream": True})
            asked: Final = candidate.request(
                "POST", "/v1/completions", {**body, "stream": True, "stream_options": {"include_usage": True}}
            )
            whole: Final = candidate.request("POST", "/v1/completions", body)
            assert (silent.status_code, asked.status_code, whole.status_code) == (200, 200, 200), (
                silent.text,
                asked.text,
                whole.text,
            )
            silent_frames: Final = _data_events(silent.text)
            asked_frames: Final = _data_events(asked.text)
            request_ids: Final = (silent_frames[0]["id"], asked_frames[0]["id"], whole.json()["id"])
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" '
                    "WHERE request_id = ANY(%s)",
                    (list(request_ids),),
                ),
                lambda values: len(values) == 3,
                seconds=70,
            )
        upstream_stream_options: Final = [
            json.loads(sent.body).get("stream_options") for sent in wire.drain() if sent.method == "POST"
        ]

    assert upstream_stream_options == [{"include_usage": True}, {"include_usage": True}, None]
    assert all(_carries_text_or_finish(frame) for frame in silent_frames), silent.text
    assert all(frame.get("usage", {}) == {} or frame["usage"]["total_tokens"] == 0 for frame in silent_frames), (
        silent.text
    )
    assert asked_frames[-1]["usage"]["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS, asked.text
    billed: Final = {str(row["request_id"]): _billed(row) for row in rows}
    assert billed[str(request_ids[0])] == billed[str(request_ids[1])] == billed[str(request_ids[2])], billed
    assert billed[str(request_ids[0])] == (PROMPT_TOKENS, COMPLETION_TOKENS, pytest.approx(EXPECTED_SPEND)), billed
