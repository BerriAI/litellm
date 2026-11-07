from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

DEFAULT_CONTEXT_CACHE_TTL_SECONDS: Final = 3600.0
_SECONDS_PER_HOUR: Final = 3600.0


class _CachedContentUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    totalTokenCount: int = 0


class _CreatedCachedContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    usageMetadata: _CachedContentUsage = _CachedContentUsage()
    createTime: datetime | None = None
    expireTime: datetime | None = None


def _reported_lifetime_seconds(created: _CreatedCachedContent) -> float | None:
    if created.createTime is None or created.expireTime is None:
        return None
    seconds: Final = (created.expireTime - created.createTime).total_seconds()
    return seconds if seconds > 0 else None


def _requested_ttl_seconds(requested_ttl: str | None) -> float | None:
    if requested_ttl is None or not requested_ttl.endswith("s"):
        return None
    try:
        seconds: Final = float(requested_ttl[:-1])
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def context_cache_storage_token_hours(created_cached_content: object, requested_ttl: str | None) -> float:
    """Token-hours a newly created cachedContents resource is billed for over its whole TTL.

    The lifetime comes from the create response (expireTime - createTime), then the TTL that was
    requested, then the provider default of one hour.
    """
    try:
        created: Final = _CreatedCachedContent.model_validate(created_cached_content)
    except ValidationError:
        return 0.0
    lifetime_seconds: Final = (
        _reported_lifetime_seconds(created)
        or _requested_ttl_seconds(requested_ttl)
        or DEFAULT_CONTEXT_CACHE_TTL_SECONDS
    )
    return created.usageMetadata.totalTokenCount * lifetime_seconds / _SECONDS_PER_HOUR
