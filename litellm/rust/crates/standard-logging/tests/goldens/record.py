"""Records the StandardLoggingPayload Python builds for the /v1/messages calls the Rust gateway serves.

Run from the repository root, after `uv sync --frozen --extra proxy --no-install-package litellm-admin-mcp`,
with `uv run --frozen python litellm/rust/crates/standard-logging/tests/goldens/record.py`.
Each scenario calls `get_standard_logging_object_payload` directly with the kwargs and logging object the
proxy hands it for an `anthropic_messages` call made with a bare virtual key, and writes `<scenario>.json`
beside this script.

The written text is Python's payload, with its key order, rendered the way `serde_json::to_string`
renders the same document, after these documented normalizations:

- Separators are compact and non-ASCII is written as is, as serde_json writes them.
- Floats keep Python's shortest round-trip digits but take serde_json's notation (`3e-6`, not `3e-06`).
- Values that change from run to run are masked as `MASKS` says: a cache hit's id ends in
  `_cache_hit<time>` and the logged response's `created` is 0.
- The metadata keys `get_standard_logging_metadata` copies from the request through a set
  intersection land in set-iteration order, which depends on the hash seed, so they are written in
  the order `StandardLoggingMetadata` declares them.
"""

import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Final

from fastapi import HTTPException

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging, get_standard_logging_object_payload
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.types.utils import StandardLoggingMetadata

GOLDENS: Final = Path(__file__).resolve().parent

MODEL: Final = "claude-sonnet-4-5-20250929"
MODEL_GROUP: Final = "claude-sonnet-4-5"
DEPLOYMENT_ID: Final = "deployment-1"
API_BASE: Final = "https://api.anthropic.com"
CALL_ID: Final = "call-1"
TRACE_ID: Final = "trace-1"
MESSAGE_ID: Final = "msg_01"
KEY_HASH: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
START: Final = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
FIRST_EVENT: Final = datetime(2026, 1, 1, 12, 0, 0, 750000, tzinfo=timezone.utc)
END: Final = datetime(2026, 1, 1, 12, 0, 2, 500000, tzinfo=timezone.utc)
MESSAGES: Final = ({"role": "user", "content": "Hey"},)
REQUEST_BODY: Final = {"model": MODEL_GROUP, "max_tokens": 64, "messages": list(MESSAGES)}
HEADERS: Final = {"content-type": "application/json", "user-agent": "curl/8.7.1"}

# A bare virtual key: no user, team, organization, project, customer or budget.
KEY: Final = UserAPIKeyAuth(api_key=KEY_HASH, key_alias="ci-key", spend=0.25, request_route="/v1/messages")

# How each volatile value is masked, keyed by its path in the payload.
MASKS: Final[Mapping[tuple[str, ...], Callable[[object], object]]] = {
    # a cache hit's id is `<response id>_cache_hit<time.time()>`
    ("id",): lambda value: re.sub(r"_cache_hit[0-9.]+$", "_cache_hit<time>", str(value)),
    # the logged ModelResponse stamps int(time.time()) as created, since an Anthropic body carries none
    ("response", "created"): lambda _: 0,
}


def _anthropic_body(output_tokens: int, text: str) -> dict[str, object]:
    return {
        "id": MESSAGE_ID,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _logging(*, stream: bool, routed: bool = True) -> Logging:
    """The logging object the proxy sets up for one /v1/messages call, with the metadata it stamps."""
    logging_obj: Final = Logging(
        model=MODEL_GROUP if not routed else MODEL,
        messages=list(MESSAGES),
        stream=stream,
        call_type="anthropic_messages",
        start_time=START,
        litellm_call_id=CALL_ID,
        function_id="function-1",
    )
    litellm_metadata: Final[dict[str, object]] = {
        "requester_metadata": {},
        "user_api_key": KEY_HASH,
        **LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(KEY),
        "headers": dict(HEADERS),
        "endpoint": "http://localhost:4000/v1/messages",
        "requester_ip_address": "10.0.0.1",
        "user_agent": "curl/8.7.1",
    }
    if routed:
        litellm_metadata.update(
            {
                "model_group": MODEL_GROUP,
                "model_group_size": 1,
                "deployment": f"anthropic/{MODEL}",
                "model_info": {"id": DEPLOYMENT_ID, "db_model": False},
                "api_base": API_BASE,
                "caching_groups": None,
            }
        )
    logging_obj.update_environment_variables(
        model=MODEL if routed else MODEL_GROUP,
        user=None,
        optional_params={"max_tokens": 64, **({"stream": True} if stream else {})},
        litellm_params={
            "litellm_call_id": CALL_ID,
            "litellm_trace_id": TRACE_ID,
            "litellm_metadata": litellm_metadata,
            "metadata": {},
            "proxy_server_request": {
                "url": "http://localhost:4000/v1/messages",
                "method": "POST",
                "headers": dict(HEADERS),
                "body": dict(REQUEST_BODY),
            },
            **({"api_base": API_BASE, "model_info": {"id": DEPLOYMENT_ID, "db_model": False}} if routed else {}),
        },
        custom_llm_provider="anthropic" if routed else None,
    )
    return logging_obj


def _served(logging_obj: Logging, body: Mapping[str, object], *, cache_hit: bool) -> object:
    response: Final = logging_obj.normalize_endpoint_logging_result(dict(body))
    logging_obj.model_call_details["cache_hit"] = cache_hit
    logging_obj.model_call_details["response_cost"] = logging_obj.response_cost_calculator(
        result=response, cache_hit=cache_hit
    )
    return response


def unary_success() -> object:
    logging_obj: Final = _logging(stream=False)
    response: Final = _served(logging_obj, _anthropic_body(5, "Hi"), cache_hit=False)
    return get_standard_logging_object_payload(
        kwargs=logging_obj.model_call_details,
        init_response_obj=response,
        start_time=START,
        end_time=END,
        logging_obj=logging_obj,
        status="success",
    )


def _streamed(*, disconnected: bool) -> object:
    logging_obj: Final = _logging(stream=True)
    logging_obj.update_completion_start_time(FIRST_EVENT)
    response: Final = _served(
        logging_obj, _anthropic_body(2 if disconnected else 5, "H" if disconnected else "Hi"), cache_hit=False
    )
    logging_obj.model_call_details["async_complete_streaming_response"] = response
    if disconnected:
        logging_obj.record_client_disconnect()
    return get_standard_logging_object_payload(
        kwargs=logging_obj.model_call_details,
        init_response_obj=response,
        start_time=START,
        end_time=END,
        logging_obj=logging_obj,
        status="success",
    )


def stream_success() -> object:
    return _streamed(disconnected=False)


def client_disconnect() -> object:
    """The client went away mid-stream: the proxy logs the partial stream as served, marked disconnected."""
    return _streamed(disconnected=True)


def cache_hit() -> object:
    logging_obj: Final = _logging(stream=False)
    response: Final = _served(logging_obj, _anthropic_body(5, "Hi"), cache_hit=True)
    return get_standard_logging_object_payload(
        kwargs=logging_obj.model_call_details,
        init_response_obj=response,
        start_time=START,
        end_time=END,
        logging_obj=logging_obj,
        status="success",
    )


def _failed(logging_obj: Logging, error: Exception) -> object:
    logging_obj.model_call_details["exception"] = error
    logging_obj.model_call_details["response_cost"] = 0
    return get_standard_logging_object_payload(
        kwargs=logging_obj.model_call_details,
        init_response_obj={},
        start_time=START,
        end_time=END,
        logging_obj=logging_obj,
        status="failure",
        error_str=str(error),
        original_exception=error,
    )


def provider_4xx() -> object:
    error: Final = litellm.BadRequestError(
        message='AnthropicException - {"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be greater than or equal to 1"}}',
        model=MODEL,
        llm_provider="anthropic",
    )
    return _failed(_logging(stream=False), error)


def key_limit_429() -> object:
    """The key's rpm limit refused the call before routing, so no deployment was picked."""
    error: Final = ProxyRateLimitError(
        detail=(
            f"Rate limit exceeded for api_key: {KEY_HASH}. Limit type: requests. "
            "Current limit: 1, Remaining: 0. Limit resets at: 2026-01-01 12:01:00 UTC"
        ),
        headers={
            "retry-after": "60",
            "rate_limit_type": "requests",
            "rate_limited_by": "api_key",
            "reset_at": "2026-01-01 12:01:00 UTC",
        },
        model=MODEL_GROUP,
        llm_provider="",
        descriptor_key="api_key",
    )
    assert isinstance(error, HTTPException)
    return _failed(_logging(stream=False, routed=False), error)


SCENARIOS: Final[Mapping[str, Callable[[], object]]] = {
    "unary_success": unary_success,
    "stream_success": stream_success,
    "cache_hit": cache_hit,
    "provider_4xx": provider_4xx,
    "key_limit_429": key_limit_429,
    "client_disconnect": client_disconnect,
}

_METADATA_ORDER: Final = tuple(StandardLoggingMetadata.__annotations__)


def _declared_order(metadata: Mapping[str, object]) -> dict[str, object]:
    """The metadata with the keys after `used_client_oauth_token`/`is_silent_experiment` (the ones the
    set intersection appended) in declared order, and `claude_subscription_usage` last, as Python
    always appends it last."""
    keys: Final = list(metadata)
    anchor: Final = keys.index("is_silent_experiment" if "is_silent_experiment" in keys else "used_client_oauth_token")
    head: Final = keys[: anchor + 1]
    appended: Final = sorted(
        (key for key in keys[anchor + 1 :] if key != "claude_subscription_usage"), key=_METADATA_ORDER.index
    )
    return {key: metadata[key] for key in (*head, *appended, "claude_subscription_usage")}


def _masked(value: object, path: tuple[str, ...] = ()) -> object:
    if path in MASKS:
        return MASKS[path](value)
    if isinstance(value, Mapping):
        ordered: Final = _declared_order(value) if path == ("metadata",) else value
        return {key: _masked(item, (*path, key)) for key, item in ordered.items()}
    if isinstance(value, list | tuple):
        return [_masked(item, path) for item in value]
    return value


def _float_text(value: float) -> str:
    """Python's shortest round-trip digits in the notation serde_json (ryu) writes."""
    sign, digit_tuple, exponent = Decimal(repr(value)).as_tuple()
    digits: Final = "".join(map(str, digit_tuple)).lstrip("0").rstrip("0")
    if not digits:
        return "-0.0" if sign else "0.0"
    point: Final = len("".join(map(str, digit_tuple)).lstrip("0")) + int(exponent)
    minus: Final = "-" if sign else ""
    if len(digits) <= point <= 16:
        return f"{minus}{digits}{'0' * (point - len(digits))}.0"
    if 0 < point <= 16:
        return f"{minus}{digits[:point]}.{digits[point:]}"
    if -5 < point <= 0:
        return f"{minus}0.{'0' * -point}{digits}"
    mantissa: Final = digits if len(digits) == 1 else f"{digits[0]}.{digits[1:]}"
    return f"{minus}{mantissa}e{point - 1}"


def _serde_text(value: object) -> str:
    if value is None or isinstance(value, bool | int | str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return _float_text(value)
    if isinstance(value, Mapping):
        return (
            "{"
            + ",".join(f"{json.dumps(key, ensure_ascii=False)}:{_serde_text(item)}" for key, item in value.items())
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_serde_text(item) for item in value) + "]"
    raise TypeError(f"{type(value).__name__} is not JSON")


def main() -> int:
    for name, build in SCENARIOS.items():
        payload = build()
        if payload is None:
            sys.exit(f"{name}: get_standard_logging_object_payload returned None")
        document = json.loads(json.dumps(payload, default=str))
        (GOLDENS / f"{name}.json").write_text(_serde_text(_masked(document)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
