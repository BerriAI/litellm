from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime
from typing import Final

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter

from litellm.integrations.custom_logger import CustomLogger

IMAGE_MODEL: Final = "openai/gpt-image-2.6-sunburst"
CHAT_MODEL: Final = "openai/gpt-6.2-sol"
EMBEDDING_MODEL: Final = "openai/text-embedding-3-large"

IMAGE_B64: Final = base64.b64encode(bytes(range(256)) * 6144).decode()
CHAT_TEXT: Final = "large scripted reply " * 100_000
CHAT_ID: Final = "chatcmpl-raw-response"
EMBEDDING_VECTOR: Final[tuple[float, ...]] = (0.25,) * 100_000

IMAGE_BODY: Final[Mapping[str, JsonValue]] = {"created": 1_700_000_000, "data": [{"b64_json": IMAGE_B64}]}
EMBEDDING_BODY: Final[Mapping[str, JsonValue]] = {
    "object": "list",
    "model": "text-embedding-3-large",
    "data": [{"object": "embedding", "index": 0, "embedding": list(EMBEDDING_VECTOR)}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}


def chat_body(text: str) -> Mapping[str, JsonValue]:
    return {
        "id": CHAT_ID,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-6.2-sol",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


CHAT_BODY: Final = chat_body(CHAT_TEXT)
_JSON: Final = TypeAdapter(JsonValue)


def encoded(body: Mapping[str, JsonValue]) -> bytes:
    return json.dumps(body).encode()


def without_nulls(value: JsonValue) -> JsonValue:
    """Typed SDK models add null optional fields, which the provider never sent."""
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_nulls(item) for item in value]
    return value


def fingerprint(value: JsonValue | Mapping[str, JsonValue]) -> str:
    """Compare multi-megabyte payloads without pytest rendering a diff of them."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def raw_response_fields(raw: object) -> JsonValue:
    if isinstance(raw, BaseModel):
        return without_nulls(_JSON.validate_python(raw.model_dump(mode="json")))
    assert isinstance(raw, str), f"raw response is a {type(raw).__name__}"
    return without_nulls(_JSON.validate_json(raw))


class RawResponseProbe(CustomLogger):
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        await self._deliver("success", kwargs)

    async def _deliver(self, event: str, kwargs: Mapping[str, object]) -> None:
        payload: Final = json.dumps(
            {
                "event": event,
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "original_response": kwargs.get("original_response"),
            },
            default=str,
        )
        async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
            response: Final = await client.post(os.environ["RAW_RESPONSE_PROBE_URL"], content=payload)
            response.raise_for_status()


probe: Final = RawResponseProbe()
