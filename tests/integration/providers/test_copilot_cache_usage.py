import json
import math
from collections.abc import Iterator
from typing import Final

import pytest
import yaml
from integration._support import device_login
from integration._support.client import Gateway, gateway_from_environment
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

_MODEL: Final = "github_copilot/claude-sonnet-5.5"
_CACHE: Final = {
    "token_details": [
        {"token_type": "input", "token_count": 10},
        {"token_type": "cache_read", "token_count": 80},
        {"token_type": "cache_write", "token_count": 20},
        {"token_type": "output", "token_count": 5},
    ],
}
_CHAT_USAGE: Final = {"prompt_tokens": 110, "completion_tokens": 5, "total_tokens": 115}
_NATIVE_USAGE: Final = {"input_tokens": 10, "output_tokens": 5}
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


def _provider(request: Request) -> Reply:
    body: Final = _JSON.validate_json(request.body)
    (message,) = _MESSAGES.validate_python(body["messages"])
    shape: Final = message["content"]
    if shape == "Native reply" or request.target == "/v1/messages":
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_copilot_cache",
                    "type": "message",
                    "role": "assistant",
                    "model": _MODEL.split("/", 1)[1],
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "usage": _NATIVE_USAGE,
                    "copilot_usage": _CACHE,
                }
            ).encode()
        )
    common: Final = {"id": "chatcmpl-copilot-cache", "created": 1, "model": _MODEL.split("/", 1)[1]}
    if body.get("stream"):
        chunk: Final = {
            **common,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        }
        usage_chunk: Final[dict[str, object]] = {
            **common,
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": _CHAT_USAGE,
            "copilot_usage": _CACHE,
        }
        return Reply(
            content_type="text/event-stream",
            body=f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(usage_chunk)}\n\ndata: [DONE]\n\n".encode(),
        )
    return Reply(
        body=json.dumps(
            {
                **common,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
                "usage": _CHAT_USAGE,
                "copilot_usage": _CACHE,
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def copilot_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("copilot-cache-usage")
    device_login.write_copilot_access(directory, device_login.COPILOT_ACCESS)
    with gateway_from_environment() as gateway, device_login.device_login_peers(directory, _provider) as peers:
        config: Final = directory / "proxy.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model_list": [
                        {
                            "model_name": "copilot-cache",
                            "litellm_params": {"model": _MODEL, "api_base": peers.api.url},
                            "model_info": {
                                "input_cost_per_token": 0.001,
                                "output_cost_per_token": 0.002,
                                "cache_read_input_token_cost": 0.0001,
                                "cache_creation_input_token_cost": 0.003,
                            },
                        }
                    ],
                    "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY"},
                }
            )
        )
        with owned_proxy_process(
            gateway,
            directory,
            {**peers.environment(), "GITHUB_COPILOT_TOKEN_DIR": str(directory)},
            config=config,
            workers=2,
        ) as owned:
            yield owned.gateway


@pytest.mark.parametrize(
    ("endpoint", "prompt", "stream"),
    [
        ("/v1/chat/completions", "Chat reply", False),
        ("/v1/chat/completions", "Chat reply", True),
        ("/v1/chat/completions", "Native reply", False),
        ("/v1/messages", "Native reply", False),
        ("/v1/responses", "Chat reply", False),
    ],
    ids=("chat", "chat-stream", "chat-native-reply", "messages", "responses"),
)
def test_copilot_cache_token_details_are_included_in_usage(
    copilot_gateway: Gateway, endpoint: str, prompt: str, stream: bool
) -> None:
    response: Final = copilot_gateway.request(
        "POST",
        endpoint,
        {
            "model": "copilot-cache",
            **(
                {"input": prompt, "max_output_tokens": 16}
                if endpoint == "/v1/responses"
                else {"messages": [{"role": "user", "content": prompt}], "max_tokens": 16}
            ),
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        },
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    assert "OK" in response.text
    payloads: Final = (
        tuple(_JSON.validate_json(line[6:]) for line in response.text.splitlines() if line.startswith("data: {"))
        if stream
        else (_JSON.validate_json(response.content),)
    )
    usage: Final = next(
        _JSON.validate_python(payload["usage"]) for payload in reversed(payloads) if payload.get("usage")
    )
    if endpoint == "/v1/responses":
        assert (usage["input_tokens"], usage["total_tokens"]) == (110, 115), usage
        details: Final = _JSON.validate_python(usage["input_tokens_details"] or {})
        assert (details.get("cached_tokens"), details.get("cache_write_tokens")) == (80, 20), usage
        return
    assert (usage.get("cache_read_input_tokens"), usage.get("cache_creation_input_tokens")) == (80, 20), usage
    if endpoint == "/v1/chat/completions":
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (110, 115), usage
        assert _JSON.validate_python(usage["prompt_tokens_details"])["cached_tokens"] == 80, usage
        if not stream:
            assert math.isclose(
                float(response.headers["x-litellm-response-cost"]), 10 * 0.001 + 80 * 0.0001 + 20 * 0.003 + 5 * 0.002
            ), response.headers
