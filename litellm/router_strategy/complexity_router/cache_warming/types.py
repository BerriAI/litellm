import base64
import hashlib
import json
import zlib
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

CACHE_WARMING_MARKER_KEY = "_complexity_router_cache_warming"
CACHE_WARMING_REPLAY_MARKER_KEY = "litellm_cache_warming"
CACHE_WARMING_REPLAY_TAG = "litellm_cache_warming"
CACHE_WARMING_RECORD_SCHEMA_VERSION = 1
WARM_FRESHNESS_SLACK_SECONDS = 60


class CacheWarmingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: tuple[Mapping[str, object], ...]
    system: str | tuple[Mapping[str, object], ...] | None = None
    tools: tuple[Mapping[str, object], ...] | None = None
    tool_choice: str | Mapping[str, object] | None = None
    call_surface: Literal["chat_completions", "anthropic_messages"]


class CacheWarmingAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_api_key: str | None = None
    user_api_key_hash: str | None = None
    user_api_key_user_id: str | None = None
    user_api_key_team_id: str | None = None
    user_api_key_end_user_id: str | None = None


class CacheWarmingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    payload_compressed: str
    payload_sha256: str
    token_estimate: int
    last_activity: float
    served_model: str
    attribution: CacheWarmingAttribution
    auto_router_model_name: str


def compress_payload(payload: CacheWarmingPayload) -> tuple[str, str]:
    raw = payload.model_dump_json().encode("utf-8")
    blob = base64.b64encode(zlib.compress(raw)).decode("ascii")
    sha = hashlib.sha256(json.dumps(payload.model_dump(), sort_keys=True).encode("utf-8")).hexdigest()
    return blob, sha


def decompress_payload(blob_b64: str) -> CacheWarmingPayload:
    raw = zlib.decompress(base64.b64decode(blob_b64.encode("ascii")))
    return CacheWarmingPayload.model_validate_json(raw)
