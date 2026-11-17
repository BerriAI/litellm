import asyncio
import hashlib
import hmac
import json
import os
from collections.abc import Callable, Mapping
from itertools import accumulate
from threading import BoundedSemaphore
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import (
    PREFIX_BLOCK_HASHES_BLOCK_SIZE,
    PREFIX_BLOCK_HASHES_MAX_CHARS,
    TOKEN_COUNTER_MAX_CONCURRENT_COUNTS,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.initialize_dynamic_callback_params import initialize_standard_callback_dynamic_params
from litellm.litellm_core_utils.redact_messages import (
    should_redact_message_logging,  # pyright: ignore[reportUnknownVariableType]  # legacy helper accepts an untyped dict
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral
from litellm.utils import (
    encode,  # pyright: ignore[reportUnknownVariableType]  # tokenizer SDKs have partial return types
)

_TOKEN_IDS: Final = TypeAdapter(tuple[int, ...])
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_HASH_JOBS: Final = BoundedSemaphore(TOKEN_COUNTER_MAX_CONCURRENT_COUNTS)


def _secret() -> str | None:
    return os.getenv("LITELLM_PREFIX_BLOCK_HASHES_SECRET")


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: Final = tuple(as_str_mapping(part) or {} for part in _ITEMS.validate_python(content))
    return "".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _message_text(message: object) -> str:
    fields: Final = as_str_mapping(message) or {}
    tool_fields: Final = {key: fields[key] for key in ("tool_calls", "tool_call_id", "name") if key in fields}
    tool_text: Final = json.dumps(tool_fields, sort_keys=True, separators=(",", ":")) if tool_fields else ""
    return f"\n{fields.get('role', '')}\n{_text(fields.get('content'))}{tool_text}"


def _encode_prompt(model: str, prompt: str) -> tuple[int, ...]:
    return _TOKEN_IDS.validate_python(encode(model=model, text=prompt))


def _encode_and_hash(
    tokenizer: Callable[[str, str], tuple[int, ...]], model: str, prompt: str, key: str
) -> tuple[str, ...]:
    return _hash_ids(tokenizer(model, prompt), key)


def _hash_ids(tokens: tuple[int, ...], key: str) -> tuple[str, ...]:
    blocks: Final = tuple(
        json.dumps(tokens[offset : offset + PREFIX_BLOCK_HASHES_BLOCK_SIZE], separators=(",", ":")).encode()
        for offset in range(0, len(tokens), PREFIX_BLOCK_HASHES_BLOCK_SIZE)
    )
    digests: Final = accumulate(
        blocks, lambda previous, block: hmac.new(key.encode(), previous + block, hashlib.sha256).digest(), initial=b""
    )
    return tuple(digest.hex()[:16] for digest in digests if digest)


class PrefixBlockHashes(CustomLogger):
    def __init__(
        self,
        tokenizer: Callable[[str, str], tuple[int, ...]] = _encode_prompt,
        secret: Callable[[], str | None] = _secret,
        hash_jobs: BoundedSemaphore = _HASH_JOBS,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # legacy base accepts untyped optional kwargs
        self._tokenizer: Final = tokenizer
        self._secret: Final = secret
        self._hash_jobs: Final = hash_jobs

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:  # mutable-ok: CustomLogger requires a dictionary return value
        messages: Final = data.get("messages")
        if call_type not in ("completion", "acompletion") or not isinstance(messages, list) or not messages:
            return data if isinstance(data, dict) else dict(data)
        try:
            if should_redact_message_logging(
                {
                    "litellm_params": dict(data),
                    "standard_callback_dynamic_params": initialize_standard_callback_dynamic_params(dict(data)),
                }
            ):
                verbose_logger.debug("Prefix block hashing skipped because message logging is redacted")
                return data if isinstance(data, dict) else dict(data)
            key: Final = self._secret()
            if not key:
                verbose_logger.debug("Prefix block hashing skipped because no dedicated secret is configured")
                return data if isinstance(data, dict) else dict(data)
            tools: Final = data.get("tools")
            tools_text: Final = json.dumps(tools, sort_keys=True, separators=(",", ":")) if tools else ""
            prompt: Final = tools_text + "".join(_message_text(message) for message in _ITEMS.validate_python(messages))
            if len(prompt) > PREFIX_BLOCK_HASHES_MAX_CHARS:
                verbose_logger.debug("Prefix block hashing skipped because the prompt exceeds its size budget")
                return data if isinstance(data, dict) else dict(data)
            model: Final = data.get("model")
            if not self._hash_jobs.acquire(blocking=False):
                verbose_logger.debug("Prefix block hashing skipped because hash workers are busy")
                return data if isinstance(data, dict) else dict(data)
            job: Final = asyncio.create_task(
                asyncio.to_thread(
                    _encode_and_hash, self._tokenizer, model if isinstance(model, str) else "", prompt, key
                )
            )
            job.add_done_callback(lambda completed: self._hash_jobs.release())
            hash_ids: Final = await asyncio.shield(job)
            bucket: Final = get_metadata_variable_name_from_kwargs(data)
            metadata: Final[Mapping[str, object]] = as_str_mapping(data.get(bucket)) or {}
            spend_metadata: Final = as_str_mapping(metadata.get("spend_logs_metadata")) or {}
        except Exception as error:  # noqa: BLE001  # optional observability must never fail a request
            verbose_logger.debug("Prefix block hashing skipped: %s", type(error).__name__)
            return data if isinstance(data, dict) else dict(data)
        return {
            **data,
            bucket: {
                **metadata,
                "spend_logs_metadata": {
                    **spend_metadata,
                    "prefix_block_hashes": {
                        "block_size": PREFIX_BLOCK_HASHES_BLOCK_SIZE,
                        "hash_ids": hash_ids,
                    },
                },
            },
        }
