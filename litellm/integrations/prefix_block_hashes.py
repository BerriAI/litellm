import hashlib
import hmac
import json
from collections.abc import Mapping
from itertools import accumulate
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import PREFIX_BLOCK_HASHES_BLOCK_SIZE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping, get_metadata_variable_name_from_kwargs
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import get_salt_key
from litellm.types.utils import CallTypesLiteral
from litellm.utils import (
    encode,  # pyright: ignore[reportUnknownVariableType]  # tokenizer SDKs have partial return types
)

_TOKEN_IDS: Final = TypeAdapter(tuple[int, ...])
_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: Final = tuple(as_str_mapping(part) or {} for part in _ITEMS.validate_python(content))
    return "".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")


def _message_text(message: object) -> str:
    fields: Final = as_str_mapping(message) or {}
    return f"\n{fields.get('role', '')}\n{_text(fields.get('content'))}"


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
            key: Final = get_salt_key()
            if not key:
                verbose_logger.debug("Prefix block hashing skipped because no salt key is configured")
                return data if isinstance(data, dict) else dict(data)
            tools: Final = data.get("tools")
            tools_text: Final = json.dumps(tools, sort_keys=True, separators=(",", ":")) if tools else ""
            prompt: Final = tools_text + "".join(_message_text(message) for message in _ITEMS.validate_python(messages))
            model: Final = data.get("model")
            tokens: Final = _TOKEN_IDS.validate_python(
                encode(model=model if isinstance(model, str) else "", text=prompt)
            )
            bucket: Final = get_metadata_variable_name_from_kwargs(data)
            metadata: Final[Mapping[str, object]] = as_str_mapping(data.get(bucket)) or {}
            spend_metadata: Final = as_str_mapping(metadata.get("spend_logs_metadata")) or {}
            return {
                **data,
                bucket: {
                    **metadata,
                    "spend_logs_metadata": {
                        **spend_metadata,
                        "prefix_block_hashes": {
                            "block_size": PREFIX_BLOCK_HASHES_BLOCK_SIZE,
                            "hash_ids": _hash_ids(tokens, key),
                        },
                    },
                },
            }
        except Exception as error:  # noqa: BLE001  # optional observability must never fail a request
            verbose_logger.debug("Prefix block hashing skipped: %s", type(error).__name__)
            return data if isinstance(data, dict) else dict(data)
