import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, NoReturn, TypeAlias

from pydantic import Field, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    EMPTY_MAPPING,
    MAX_BATCH_FILE_RECORDS_KEY,
    MAX_BATCH_FILE_UPLOADS_PER_DAY_KEY,
    MAX_FILE_DOWNLOADS_PER_MINUTE_KEY,
)
from litellm.proxy._types import ProxyException, UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache

FileUsageSetting: TypeAlias = Literal[
    "max_batch_file_records",
    "max_batch_file_uploads_per_day",
    "max_file_downloads_per_minute",
]
LimitSource: TypeAlias = Literal["key", "team", "general_settings"]
CounterScope: TypeAlias = Literal["key", "user", "team"]

_COUNTER_PREFIX: Final = "litellm:file_usage"
_DAY_SECONDS: Final = 24 * 60 * 60
_MINUTE_SECONDS: Final = 60
_LIMIT_ADAPTER: Final = TypeAdapter(Annotated[int, Field(gt=0)])


@dataclass(frozen=True, slots=True)
class FileUsageLimit:
    setting: FileUsageSetting
    value: int
    source: LimitSource


@dataclass(frozen=True, slots=True)
class ScopedFileUsageLimit:
    scope: CounterScope
    scope_id: str
    limit: FileUsageLimit


@dataclass(frozen=True, slots=True)
class FileUsageLimitExceeded:
    limit: ScopedFileUsageLimit
    retry_after_seconds: int


def _read_limit(
    settings: Mapping[str, object] | None,
    setting: FileUsageSetting,
    source: LimitSource,
) -> FileUsageLimit | None:
    raw: Final = (settings or EMPTY_MAPPING).get(setting)
    if raw is None:
        return None
    try:
        return FileUsageLimit(setting=setting, value=_LIMIT_ADAPTER.validate_python(raw), source=source)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Ignoring invalid %s in %s; expected a positive integer",
            setting,
            source,
        )
        return None


def _key_limit(
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    setting: FileUsageSetting,
) -> FileUsageLimit | None:
    from_key: Final = _read_limit(user_api_key_dict.metadata, setting, "key")
    if from_key is not None:
        return from_key
    return _read_limit(general_settings, setting, "general_settings")


def _team_limit(user_api_key_dict: UserAPIKeyAuth, setting: FileUsageSetting) -> FileUsageLimit | None:
    return _read_limit(user_api_key_dict.team_metadata, setting, "team")


def batch_file_record_limit(
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
) -> FileUsageLimit | None:
    applicable: Final = tuple(
        limit
        for limit in (
            _key_limit(user_api_key_dict, general_settings, MAX_BATCH_FILE_RECORDS_KEY),
            _team_limit(user_api_key_dict, MAX_BATCH_FILE_RECORDS_KEY),
        )
        if limit is not None
    )
    return min(applicable, key=lambda limit: limit.value, default=None)


def _caller_counter(user_api_key_dict: UserAPIKeyAuth, limit: FileUsageLimit | None) -> ScopedFileUsageLimit | None:
    if limit is None:
        return None
    if user_api_key_dict.api_key:
        return ScopedFileUsageLimit(scope="key", scope_id=user_api_key_dict.api_key, limit=limit)
    if user_api_key_dict.user_id:
        return ScopedFileUsageLimit(scope="user", scope_id=user_api_key_dict.user_id, limit=limit)
    return None


def resolve_scoped_limits(
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    setting: FileUsageSetting,
) -> tuple[ScopedFileUsageLimit, ...]:
    team_limit: Final = _team_limit(user_api_key_dict, setting)
    candidates: Final = (
        _caller_counter(user_api_key_dict, _key_limit(user_api_key_dict, general_settings, setting)),
        ScopedFileUsageLimit(scope="team", scope_id=user_api_key_dict.team_id, limit=team_limit)
        if team_limit is not None and user_api_key_dict.team_id
        else None,
    )
    return tuple(scoped for scoped in candidates if scoped is not None)


async def _increment_all_or_none(
    cache: "InternalUsageCache",
    counters: tuple[tuple[str, ScopedFileUsageLimit], ...],
    ttl_seconds: int,
) -> ScopedFileUsageLimit | None:
    if not counters:
        return None
    (counter_key, scoped), rest = counters[0], counters[1:]
    count: Final = await cache.async_increment_cache(
        key=counter_key, value=1, litellm_parent_otel_span=None, ttl=ttl_seconds
    )
    over_here: Final = count is not None and count > scoped.limit.value
    exceeded: Final = scoped if over_here else await _increment_all_or_none(cache, rest, ttl_seconds)
    if exceeded is not None:
        await cache.async_increment_cache(key=counter_key, value=-1, litellm_parent_otel_span=None, ttl=ttl_seconds)
    return exceeded


async def consume_file_usage(
    cache: "InternalUsageCache",
    limits: tuple[ScopedFileUsageLimit, ...],
    window_seconds: int,
    subject: str,
    now: float,
) -> FileUsageLimitExceeded | None:
    window_start: Final = int(now // window_seconds) * window_seconds
    counters: Final = tuple(
        (
            f"{_COUNTER_PREFIX}:{scoped.limit.setting}:{scoped.scope}:{scoped.scope_id}:{subject}:{window_start}",
            scoped,
        )
        for scoped in limits
    )
    exceeded: Final = await _increment_all_or_none(cache, counters, window_seconds)
    if exceeded is None:
        return None
    return FileUsageLimitExceeded(
        limit=exceeded,
        retry_after_seconds=max(1, math.ceil(window_start + window_seconds - now)),
    )


def describe_limit_source(source: LimitSource) -> str:
    match source:
        case "key":
            return "in this key's metadata"
        case "team":
            return "in this team's metadata"
        case "general_settings":
            return "in general_settings"
    return assert_never(source)


def _describe_scope(scoped: ScopedFileUsageLimit) -> str:
    match scoped.scope:
        case "key":
            return "this key"
        case "user":
            return f"user {scoped.scope_id}"
        case "team":
            return f"team {scoped.scope_id}"
    return assert_never(scoped.scope)


def _raise_limit_exceeded(exceeded: FileUsageLimitExceeded, what_ran_out: str, when_it_resets: str) -> NoReturn:
    scoped: Final = exceeded.limit
    headers: Final = {"retry-after": str(exceeded.retry_after_seconds)}  # mutable-ok: ProxyException mutates it
    raise ProxyException(
        message=(
            f"{what_ran_out}: {scoped.limit.setting} is {scoped.limit.value} for {_describe_scope(scoped)} "
            f"(set {describe_limit_source(scoped.limit.source)}). {when_it_resets}"
        ),
        type="rate_limit_error",
        param=None,
        code=429,
        headers=headers,
    )


async def enforce_batch_file_upload_limit(
    cache: "InternalUsageCache",
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    clock: Callable[[], float] = time.time,
) -> None:
    limits: Final = resolve_scoped_limits(user_api_key_dict, general_settings, MAX_BATCH_FILE_UPLOADS_PER_DAY_KEY)
    if not limits:
        return
    exceeded: Final = await consume_file_usage(cache, limits, _DAY_SECONDS, "", clock())
    if exceeded is None:
        return
    _raise_limit_exceeded(
        exceeded,
        "Batch file upload limit reached, the file was not forwarded to the provider",
        f"The count resets at 00:00 UTC, in {exceeded.retry_after_seconds} seconds.",
    )


async def enforce_file_download_limit(
    cache: "InternalUsageCache",
    user_api_key_dict: UserAPIKeyAuth,
    general_settings: Mapping[str, object],
    file_id: str,
    clock: Callable[[], float] = time.time,
) -> None:
    limits: Final = resolve_scoped_limits(user_api_key_dict, general_settings, MAX_FILE_DOWNLOADS_PER_MINUTE_KEY)
    if not limits:
        return
    exceeded: Final = await consume_file_usage(cache, limits, _MINUTE_SECONDS, file_id, clock())
    if exceeded is None:
        return
    _raise_limit_exceeded(
        exceeded,
        f"Download limit reached for file {file_id}",
        f"Retry in {exceeded.retry_after_seconds} seconds.",
    )
