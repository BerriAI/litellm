import json
import uuid
from hashlib import sha256
from typing import Final

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.deepseek_tokens import (
    CHINESE_TEXT,
    SHORT_TEXT,
    assert_counted_with_deepseek,
    chinese_minus_short,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_DEEPSEEK_V4: Final = "DeepSeek-V4-Flash-0731"


def _data_frame(frame: dict[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def _stream_without_usage(request: Request) -> Reply:
    assert request.method == "POST"
    chunk: Final[dict[str, JsonValue]] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _DEEPSEEK_V4,
    }
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _data_frame({**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}}]}),
            _data_frame({**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ),
    )


def _streamed_prompt_tokens(gateway: Gateway, key: str, model: str, text: str) -> int:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}], "stream": True},
        key=key,
    )
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (sha256(key.encode()).hexdigest(),),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["status"] == "success", rows
    prompt_tokens: Final = rows[0]["prompt_tokens"]
    assert isinstance(prompt_tokens, int), rows
    return prompt_tokens


def test_a_deepseek_v4_stream_without_usage_bills_prompt_tokens_counted_with_deepseeks_tokenizer(
    gateway: Gateway,
) -> None:
    with wire_server(_stream_without_usage) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"azure_ai/{_DEEPSEEK_V4}", api_base=wire.url, num_retries=0)
        chinese: Final = _streamed_prompt_tokens(gateway, scenario.key(models=[model]), model, CHINESE_TEXT)
        short: Final = _streamed_prompt_tokens(gateway, scenario.key(models=[model]), model, SHORT_TEXT)
        assert len(wire.drain()) == 2
    assert_counted_with_deepseek(chinese - short, chinese_minus_short)
