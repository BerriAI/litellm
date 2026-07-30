import base64
import hashlib
import json
import zlib
from functools import lru_cache
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_router_logger

CACHE_WARMING_REPLAY_MARKER_KEY = "litellm_cache_warming"
CACHE_WARMING_REPLAY_TAG = "litellm_cache_warming"
CACHE_WARMING_RECORD_SCHEMA_VERSION = 1
WARM_FRESHNESS_SLACK_SECONDS = 60
# Anthropic and Bedrock hold a cached prefix for about five minutes. Freshness is that TTL and nothing else:
# the operator's idle_timeout_seconds and refresh_interval_seconds may legally exceed it, and a model whose
# stamp is older than the TTL is cold no matter which of them says otherwise.
PROVIDER_PROMPT_CACHE_TTL_SECONDS = 300


def is_cache_fresh(warmed_at: float, now: float) -> bool:
    """The one definition of "the provider still holds this prefix", read by both the refresher's due-model
    calculation and the router's warm-aware pick, so a model can never be preferred as warm by one while the
    other treats it as stale."""
    return now - warmed_at < PROVIDER_PROMPT_CACHE_TTL_SECONDS


def needs_rewarming(warmed_at: float, now: float, refresh_interval_seconds: int) -> bool:
    """Due when the operator's interval has elapsed or the provider TTL is about to lapse, whichever comes
    first, so a refresh_interval longer than the TTL cannot open a window where the pick still believes a
    model is warm."""
    return now - warmed_at >= min(
        refresh_interval_seconds, PROVIDER_PROMPT_CACHE_TTL_SECONDS - WARM_FRESHNESS_SLACK_SECONDS
    )


class WarmthStamp(BaseModel):
    """When a model group was last replayed for a session, and whether that replay actually landed. A failed
    replay leaves the provider cache cold, so ``warmed`` keeps the pick from preferring it while ``at`` still
    paces the next attempt."""

    model_config = ConfigDict(extra="forbid")

    at: float
    warmed: bool


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
    user_api_key_org_id: str | None = None
    user_api_key_project_id: str | None = None
    user_api_key_end_user_id: str | None = None


class CacheWarmingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    payload_compressed: str
    payload_sha256: str
    token_estimate: int
    last_activity: float
    served_model: str
    session_id: str | None = None
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


@lru_cache(maxsize=4096)
def warn_once(message: str) -> None:
    """Package-wide warn-once; keyed on the formatted message, so per-entity messages dedupe per entity."""
    verbose_router_logger.warning(message)
