import copy
import json
import math
from datetime import datetime, timezone
from typing import Final
from unittest.mock import patch

import pytest
import tiktoken
from pydantic import TypeAdapter

from litellm.caching.dual_cache import DualCache
from litellm.constants import PREFIX_BLOCK_HASHES_BLOCK_SIZE
from litellm.integrations.prefix_block_hashes import PrefixBlockHashes
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
def salt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "prefix-test-salt")


@pytest.mark.asyncio
async def test_shared_prefix_blocks_match_and_later_blocks_diverge() -> None:
    prefix: Final = " common" * (PREFIX_BLOCK_HASHES_BLOCK_SIZE * 3)
    first: Final = await _call({"model": "gpt-4o", "messages": [{"role": "user", "content": prefix + " first"}]})
    second: Final = await _call({"model": "gpt-4o", "messages": [{"role": "user", "content": prefix + " second"}]})
    assert _hashes(first)[:2] == _hashes(second)[:2]
    assert _hashes(first)[-1] != _hashes(second)[-1]
    assert all(len(value) == 16 and int(value, 16) >= 0 for value in _hashes(first))


@pytest.mark.asyncio
async def test_salt_changes_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    data: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    first: Final = await _call(data)
    monkeypatch.setenv("LITELLM_SALT_KEY", "another-test-salt")
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
    with patch.object(tiktoken.Encoding, "encode", side_effect=RuntimeError("tokenizer unavailable")):
        assert await _call(data) is data
