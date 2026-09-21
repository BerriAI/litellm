import json
from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_event import (
    CACHE_OFF_KEY,
    SpendEventBuildError,
    SpendEventDecodeError,
    build_spend_event,
    decode_spend_event,
    is_offloadable_success,
    spend_event_callback_args,
)
from litellm.types.utils import LiteLLMBatch, ModelResponse, Usage

_BIG_PROMPT: Final = "x" * 20_000
_RESERVATION: Final = {
    "reserved_cost": 0.5,
    "entries": [{"counter_key": "key:hash", "reserved_cost": 0.5}],
    "finalized": False,
    "input_cost": 0.1,
    "input_tokens": 5000,
}


def _response(tool_name: str | None = None) -> ModelResponse:
    tool_calls: Final = (
        [{"id": "call-1", "type": "function", "function": {"name": tool_name, "arguments": "{}"}}]
        if tool_name is not None
        else None
    )
    return ModelResponse(
        id="chatcmpl-1",
        model="gpt-4o-2024-08-06",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "y" * 20_000, "tool_calls": tool_calls},
                "finish_reason": "tool_calls" if tool_name else "stop",
            }
        ],
        usage=Usage(prompt_tokens=5000, completion_tokens=4000, total_tokens=9000),
    )


def _success_kwargs(preset_cache_key: str | None = "preset-key") -> dict:
    return {
        "litellm_call_id": "call-1",
        "call_type": "acompletion",
        "model": "gpt-4o",
        "custom_llm_provider": "openai",
        "stream": False,
        "cache_hit": None,
        "response_cost": 0.0125,
        "completion_start_time": datetime(2026, 1, 1, 0, 0, 1),
        "messages": [{"role": "user", "content": _BIG_PROMPT}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "litellm_params": {
            "api_base": "https://api.openai.com",
            "preset_cache_key": preset_cache_key,
            "proxy_server_request": {"body": {"messages": [{"role": "user", "content": _BIG_PROMPT}]}},
            "metadata": {
                "user_api_key": "hash-1",
                "user_api_key_user_id": "user-1",
                "user_api_key_team_id": "team-1",
                "user_api_key_org_id": "org-1",
                "user_api_key_end_user_id": "end-user-1",
                "user_api_key_auth": UserAPIKeyAuth(api_key="hash-1", budget_reservation=dict(_RESERVATION)),
                "model_group": "gpt-4o",
                "model_info": {"id": "deployment-1"},
                "tags": ["tag-a"],
                "litellm_parent_otel_span": object(),
            },
        },
        "standard_logging_object": {
            "response_cost": 0.0125,
            "model": "gpt-4o-2024-08-06",
            "model_id": "deployment-1",
            "request_tags": ["tag-a"],
            "request_model_access_groups": ["premium"],
            "messages": [{"role": "user", "content": _BIG_PROMPT}],
            "response": {"choices": [{"message": {"content": "y" * 20_000}}]},
            "model_parameters": {"temperature": 0.1},
            "metadata": {"user_api_key_hash": "hash-1", "usage_object": {"prompt_tokens": 5000}},
            "hidden_params": {"litellm_overhead_time_ms": 3},
            "model_map_information": {},
        },
    }


def _build(kwargs: dict, response: object, store_bodies: bool = False) -> bytes:
    line: Final = build_spend_event(
        kwargs, response, datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 2), store_bodies=store_bodies
    )
    assert isinstance(line, bytes)
    return line


def test_event_is_compact_and_omits_bodies_by_default():
    line: Final = _build(_success_kwargs(), _response(tool_name="get_weather"))
    assert line.endswith(b"\n")
    assert len(line) < 4_000
    assert _BIG_PROMPT.encode() not in line
    assert b"yyyy" not in line
    decoded: Final = json.loads(line)
    assert "messages" not in decoded["standard_logging_object"]
    assert "response" not in decoded["standard_logging_object"]
    assert decoded["litellm_params"]["proxy_server_request"] is None


def test_event_carries_bodies_when_spend_logs_store_them():
    line: Final = _build(_success_kwargs(), _response(), store_bodies=True)
    decoded: Final = json.loads(line)
    assert decoded["standard_logging_object"]["messages"][0]["content"] == _BIG_PROMPT
    assert decoded["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "y" * 20_000
    assert decoded["litellm_params"]["proxy_server_request"]["body"]["messages"][0]["content"] == _BIG_PROMPT


def test_round_trip_preserves_identity_usage_reservation_and_tools():
    line: Final = _build(_success_kwargs(), _response(tool_name="get_weather"))
    event: Final = decode_spend_event(line)
    assert not isinstance(event, SpendEventDecodeError)
    args: Final = spend_event_callback_args(event)

    metadata: Final = args.kwargs["litellm_params"]["metadata"]
    assert metadata is not None
    assert (metadata["user_api_key"], metadata["user_api_key_team_id"], metadata["user_api_key_org_id"]) == (
        "hash-1",
        "team-1",
        "org-1",
    )
    assert metadata["user_api_key_budget_reservation"] == _RESERVATION
    assert "user_api_key_auth" not in metadata
    assert "litellm_parent_otel_span" not in metadata
    assert args.kwargs["standard_logging_object"]["request_model_access_groups"] == ["premium"]
    assert args.kwargs["standard_logging_object"]["response_cost"] == 0.0125
    assert args.kwargs["tools"] == ({"type": "function", "function": {"name": "get_weather"}},)
    assert args.kwargs["completion_start_time"] == datetime(2026, 1, 1, 0, 0, 1)
    assert (args.start_time, args.end_time) == (datetime(2026, 1, 1), datetime(2026, 1, 1, 0, 0, 2))
    assert args.response_obj is not None
    assert args.response_obj["id"] == "chatcmpl-1"
    assert args.response_obj["usage"]["prompt_tokens"] == 5000
    assert args.response_obj["usage"]["completion_tokens"] == 4000
    tool_calls: Final = args.response_obj["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in tool_calls] == ["get_weather"]
    assert "complete_streaming_response" not in args.kwargs


def test_streaming_event_reconstructs_complete_streaming_response():
    kwargs: Final = {**_success_kwargs(), "stream": True, "complete_streaming_response": _response()}
    event: Final = decode_spend_event(_build(kwargs, _response()))
    assert not isinstance(event, SpendEventDecodeError)
    args: Final = spend_event_callback_args(event)
    assert args.kwargs["stream"] is True
    assert args.kwargs["complete_streaming_response"] == args.response_obj


class _HashingCache(Cache):
    def __init__(self) -> None:
        pass

    def get_cache_key(self, **kwargs) -> str:
        raise AssertionError("the fast path must not hash the request body")


@pytest.mark.parametrize(
    ("cache", "preset", "expected"),
    [
        (None, "preset-key", CACHE_OFF_KEY),
        (_HashingCache(), "preset-key", "preset-key"),
        (_HashingCache(), None, None),
    ],
)
def test_event_reuses_preset_cache_key_and_never_hashes(monkeypatch, cache, preset, expected):
    monkeypatch.setattr(litellm, "cache", cache)
    decoded: Final = json.loads(_build(_success_kwargs(preset_cache_key=preset), _response()))
    assert decoded["litellm_params"]["preset_cache_key"] == expected


def test_unbuildable_kwargs_fall_back_to_in_process_tracking():
    kwargs: Final = {**_success_kwargs(), "response_cost": "not-a-number"}
    assert isinstance(
        build_spend_event(kwargs, _response(), datetime.now(), datetime.now(), False), SpendEventBuildError
    )


def test_undecodable_line_is_an_error_value():
    assert isinstance(decode_spend_event(b'{"version": 2}\n'), SpendEventDecodeError)
    assert isinstance(decode_spend_event(b"not json\n"), SpendEventDecodeError)


def test_batch_retrieves_stay_in_process():
    assert is_offloadable_success(_response()) is True
    assert is_offloadable_success(None) is True
    assert (
        is_offloadable_success(
            LiteLLMBatch(
                id="batch-1",
                completion_window="24h",
                created_at=1,
                endpoint="/v1/chat/completions",
                input_file_id="f",
                object="batch",
                status="completed",
            )
        )
        is False
    )
