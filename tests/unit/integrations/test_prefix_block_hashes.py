import asyncio
import copy
import json
import math
import threading
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.constants import (
    PREFIX_BLOCK_HASHES_BLOCK_SIZE,
    PREFIX_BLOCK_HASHES_MAX_CHARS,
)
from litellm.integrations.prefix_block_hashes import PrefixBlockHashes
from litellm.litellm_core_utils.litellm_logging import (
    build_custom_logger_compatible_class,
    get_custom_logger_compatible_class,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.utils import CallTypesLiteral
from litellm.utils import encode


def _hashes(data: dict[str, object], bucket: str = "metadata") -> list[str]:
    metadata: Final = TypeAdapter(dict[str, object]).validate_python(data[bucket])
    spend: Final = TypeAdapter(dict[str, object]).validate_python(metadata["spend_logs_metadata"])
    value: Final = TypeAdapter(dict[str, object]).validate_python(spend["prefix_block_hashes"])
    assert value["block_size"] == PREFIX_BLOCK_HASHES_BLOCK_SIZE
    return TypeAdapter(list[str]).validate_python(value["hash_ids"])


async def _call(data: dict[str, object], call_type: CallTypesLiteral = "completion") -> dict[str, object]:
    return await PrefixBlockHashes().async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, call_type)


@pytest.fixture(autouse=True)
def secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_PREFIX_BLOCK_HASHES_SECRET", "prefix-test-secret")
    monkeypatch.setattr(litellm, "turn_off_message_logging", False)


@pytest.mark.asyncio
async def test_shared_prefix_blocks_match_and_later_blocks_diverge() -> None:
    prefix: Final = " common" * (PREFIX_BLOCK_HASHES_BLOCK_SIZE * 3)
    first: Final = await _call({"model": "gpt-4o", "messages": [{"role": "user", "content": prefix + " first"}]})
    second: Final = await _call({"model": "gpt-4o", "messages": [{"role": "user", "content": prefix + " second"}]})
    assert _hashes(first)[:2] == _hashes(second)[:2]
    assert _hashes(first)[-1] != _hashes(second)[-1]


@pytest.mark.asyncio
async def test_tokenizer_runs_off_the_event_loop_thread() -> None:
    event_loop_thread: Final = threading.get_ident()

    def tokenizer(model: str, prompt: str) -> tuple[int, ...]:
        assert threading.get_ident() != event_loop_thread
        return (1, 2, 3)

    callback: Final = PrefixBlockHashes(tokenizer=tokenizer)
    data: Final = {"messages": [{"role": "user", "content": "hello"}]}
    result: Final = await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion")
    assert len(_hashes(result)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("tool_call_id", "name"))
async def test_tool_result_identity_changes_hash(field: str) -> None:
    first: Final = await _call({"messages": [{"role": "tool", "content": "result", field: "first"}]})
    second: Final = await _call({"messages": [{"role": "tool", "content": "result", field: "second"}]})
    assert _hashes(first) != _hashes(second)
    assert all(len(value) == 16 and int(value, 16) >= 0 for value in _hashes(first))


@pytest.mark.asyncio
async def test_secret_changes_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    data: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    first: Final = await _call(data)
    monkeypatch.setenv("LITELLM_PREFIX_BLOCK_HASHES_SECRET", "another-test-secret")
    second: Final = await _call(data)
    assert _hashes(first) != _hashes(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("repetitions", (0, 10, 1500))
async def test_includes_final_partial_block(repetitions: int) -> None:
    content: Final = " hello" * repetitions
    data: Final = await _call({"model": "gpt-4o", "messages": [{"role": "user", "content": content}]})
    tokens: Final = TypeAdapter(list[int]).validate_python(encode(model="gpt-4o", text="\nuser\n" + content))
    assert len(_hashes(data)) == math.ceil(len(tokens) / PREFIX_BLOCK_HASHES_BLOCK_SIZE)


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ("metadata", "litellm_metadata"))
@pytest.mark.parametrize("stream", (False, True))
async def test_preserves_metadata_without_mutation_and_reaches_spend_row(bucket: str, stream: bool) -> None:
    data: Final[dict[str, object]] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        bucket: {"spend_logs_metadata": {"existing": "retained"}, "tags": ["original"]},
    }
    before: Final = copy.deepcopy(data)
    result: Final = await _call(data)
    assert data == before
    assert result is not data
    timestamp: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload: Final = get_logging_payload(
        kwargs={"model": "gpt-4o", "call_type": "acompletion", "litellm_params": {bucket: result[bucket]}},
        response_obj={},
        start_time=timestamp,
        end_time=timestamp,
    )
    stored: Final = json.loads(payload["metadata"])
    assert stored["spend_logs_metadata"] == {
        "existing": "retained",
        "prefix_block_hashes": {"block_size": PREFIX_BLOCK_HASHES_BLOCK_SIZE, "hash_ids": _hashes(result, bucket)},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ("embedding", "aresponses"))
async def test_non_chat_requests_are_untouched(call_type: CallTypesLiteral) -> None:
    data: Final = {"messages": [{"role": "user", "content": "hello"}]}
    assert await _call(data, call_type) is data


@pytest.mark.asyncio
@pytest.mark.parametrize("data", ({}, {"messages": []}))
async def test_missing_messages_are_untouched(data: dict[str, object]) -> None:
    assert await _call(data) is data


@pytest.mark.asyncio
async def test_missing_secret_leaves_request_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PREFIX_BLOCK_HASHES_SECRET")
    monkeypatch.setenv("LITELLM_SALT_KEY", "not-a-hashing-secret")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "not-a-hashing-secret")
    data: Final = {"messages": [{"role": "user", "content": "hello"}]}
    assert await _call(data) is data


@pytest.mark.asyncio
@pytest.mark.parametrize("size", (PREFIX_BLOCK_HASHES_MAX_CHARS, 20 * 1024 * 1024))
async def test_oversize_prompt_skips_tokenization(size: int) -> None:
    callback: Final = PrefixBlockHashes(tokenizer=lambda model, prompt: (1,))
    data: Final = {"messages": [{"role": "user", "content": "x" * size}]}
    assert await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion") is data


@pytest.mark.asyncio
async def test_prompt_at_size_budget_still_hashes() -> None:
    callback: Final = PrefixBlockHashes(tokenizer=lambda model, prompt: (1,))
    data: Final = {"messages": [{"role": "user", "content": "x" * (PREFIX_BLOCK_HASHES_MAX_CHARS - len("\nuser\n"))}]}
    assert len(_hashes(await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion"))) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("control", ("global", "request", "metadata", "litellm_metadata", "header"))
async def test_message_redaction_controls_hashing(monkeypatch: pytest.MonkeyPatch, enabled: bool, control: str) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", enabled if control == "global" else False)
    settings: Final = {"turn_off_message_logging": enabled}
    controls: Final = (
        settings
        if control == "request"
        else {control: settings}
        if control in ("metadata", "litellm_metadata")
        else {"metadata": {"headers": {"litellm-enable-message-redaction": enabled}}}
        if control == "header"
        else {}
    )
    data: Final = {"messages": [{"role": "user", "content": "hello"}], **controls}
    result: Final = await _call(data)
    if enabled:
        assert result is data
    else:
        assert len(_hashes(result, "litellm_metadata" if control == "litellm_metadata" else "metadata")) == 1


@pytest.mark.asyncio
async def test_busy_hash_workers_skip_without_queuing_and_cancellation_keeps_limit() -> None:
    loop: Final = asyncio.get_running_loop()
    started: Final[asyncio.Queue[None]] = asyncio.Queue()
    release: Final = threading.Event()

    def tokenizer(model: str, prompt: str) -> tuple[int, ...]:
        loop.call_soon_threadsafe(started.put_nowait, None)
        release.wait()
        return (1,)

    callback: Final = PrefixBlockHashes(tokenizer=tokenizer, hash_jobs=threading.BoundedSemaphore(2))
    data: Final = {"messages": [{"role": "user", "content": "hello"}]}
    jobs: Final = tuple(
        asyncio.create_task(callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion"))
        for _ in range(2)
    )
    try:
        for _ in jobs:
            await started.get()
        assert await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion") is data
        jobs[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await jobs[0]
        assert await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion") is data
    finally:
        release.set()
        await asyncio.gather(*jobs, return_exceptions=True)


@pytest.mark.asyncio
async def test_registered_callback_is_built_reused_and_hashes_requests() -> None:
    callback: Final = build_custom_logger_compatible_class("prefix_block_hashes", None, None)
    assert isinstance(callback, PrefixBlockHashes)
    assert build_custom_logger_compatible_class("prefix_block_hashes", None, None) is callback
    assert get_custom_logger_compatible_class("prefix_block_hashes") is callback
    result: Final = await callback.async_pre_call_hook(
        UserAPIKeyAuth(), DualCache(), {"messages": [{"role": "user", "content": "hello"}]}, "completion"
    )
    assert len(_hashes(result)) == 1


@pytest.mark.asyncio
async def test_serialization_uses_sorted_tools_roles_and_only_text_parts() -> None:
    first: Final = await _call(
        {
            "model": "gpt-4o",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": "ignored"}},
                    ],
                }
            ],
        }
    )
    second: Final = await _call(
        {
            "model": "gpt-4o",
            "tools": [{"function": {"name": "lookup"}, "type": "function"}],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    changed: Final = await _call({"model": "gpt-4o", "messages": [{"role": "system", "content": "hello"}]})
    assert _hashes(first) == _hashes(second)
    assert _hashes(first) != _hashes(changed)


@pytest.mark.asyncio
async def test_tokenizer_failure_does_not_fail_request() -> None:
    data: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}

    def unavailable_tokenizer(model: str, prompt: str) -> tuple[int, ...]:
        raise RuntimeError("tokenizer unavailable")

    callback: Final = PrefixBlockHashes(tokenizer=unavailable_tokenizer)
    assert await callback.async_pre_call_hook(UserAPIKeyAuth(), DualCache(), data, "completion") is data


@pytest.mark.asyncio
async def test_tool_call_arguments_change_final_block_id() -> None:
    def request(arguments: str) -> dict[str, object]:
        return {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            ],
        }

    first: Final = await _call(request('{"city":"Paris"}'))
    second: Final = await _call(request('{"city":"London"}'))
    assert _hashes(first)[-1] != _hashes(second)[-1]
