"""Redis-backed registry of authenticated proxy requests that are still running."""

import asyncio
import hashlib
import json
import os
import secrets
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, Final, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral

if TYPE_CHECKING:
    from redis.asyncio import Redis, RedisCluster

    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.utils import InternalUsageCache


class ActiveRequestRecord(TypedDict):
    registry_id: str
    request_id: str
    started_at: float
    model: str | None
    call_type: str | None
    streaming: bool
    route: str | None
    user_id: str | None
    user_email: str | None
    end_user_id: str | None
    organization_id: str | None
    organization_alias: str | None
    project_id: str | None
    project_alias: str | None
    team_id: str | None
    team_alias: str | None
    key_alias: str | None
    key_fingerprint: str | None
    pod: str | None


class ActiveRequestsPage(TypedDict):
    available: bool
    reason: str | None
    items: tuple[ActiveRequestRecord, ...]
    total: int
    page: int
    page_size: int
    truncated: bool


class ActiveRequestFilters(TypedDict):
    model: str | None
    user_id: str | None
    end_user_id: str | None
    organization_id: str | None
    project_id: str | None


MIN_TTL_SECONDS: Final = 300
TTL_SETTING: Final = "active_request_ttl_seconds"
INCLUDE_USER_EMAIL_SETTING: Final = "active_request_include_user_email"
CANCEL_KEY_PREFIX: Final = "litellm:active_requests:cancel:"
CANCEL_TTL_SECONDS: Final = 60
CANCEL_POLL_SECONDS: Final = 1.0


def _decode_member(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _decode_record(raw_item: bytes | str | None) -> ActiveRequestRecord | None:
    if raw_item is None:
        return None
    try:
        decoded: Final = raw_item.decode() if isinstance(raw_item, bytes) else raw_item
        parsed: Final = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return ActiveRequestRecord(**parsed)  # pyright: ignore[reportArgumentType]  # json payload written by this class


def _matches(record: ActiveRequestRecord, filters: ActiveRequestFilters) -> bool:
    return all(expected is None or str(record.get(field) or "") == expected for field, expected in filters.items())


class ActiveRequestRegistry(CustomLogger):
    """Track live requests across proxy replicas without retaining request content."""

    register_as_litellm_callback: ClassVar[bool] = False
    INDEX_KEY: ClassVar[str] = "litellm:active_requests:index"
    ITEM_KEY_PREFIX: ClassVar[str] = "litellm:active_requests:item:"
    DEFAULT_TTL_SECONDS: ClassVar[int] = 1800
    MAX_TTL_SECONDS: ClassVar[int] = 86400
    MAX_FIELD_LENGTH: ClassVar[int] = 512
    DEFAULT_MAX_SCAN_MEMBERS: ClassVar[int] = 5000

    def __init__(self, internal_usage_cache: "InternalUsageCache", max_scan_members: int | None = None) -> None:
        super().__init__()
        self.internal_usage_cache: Final = internal_usage_cache
        self.max_scan_members: Final = (
            max_scan_members if max_scan_members is not None else self.DEFAULT_MAX_SCAN_MEMBERS
        )
        self._local_tasks: dict[str, asyncio.Task[object]] = {}  # mutable-ok: per-process handles for cancellation
        self._cancel_watcher: asyncio.Task[None] | None = None  # rebind-ok: started lazily on first registration

    @staticmethod
    def _general_settings() -> Mapping[str, object]:
        from litellm.proxy.proxy_server import general_settings

        return general_settings

    @property
    def ttl_seconds(self) -> int:
        """Read at call time, since general_settings is loaded after the hooks are built."""
        raw: Final = self._general_settings().get(TTL_SETTING)
        if raw is None:
            return self.DEFAULT_TTL_SECONDS
        try:
            configured: Final = int(str(raw))
        except ValueError:
            verbose_proxy_logger.warning("Invalid general_settings.%s; using %s", TTL_SETTING, self.DEFAULT_TTL_SECONDS)
            return self.DEFAULT_TTL_SECONDS
        return min(max(configured, MIN_TTL_SECONDS), self.MAX_TTL_SECONDS)

    def _redis_cache(self) -> "RedisCache | None":
        return self.internal_usage_cache.dual_cache.redis_cache

    @classmethod
    def _include_user_email(cls) -> bool:
        return cls._general_settings().get(INCLUDE_USER_EMAIL_SETTING) is True

    @classmethod
    def _safe_string(cls, value: object, max_length: int | None = None) -> str | None:
        if value is None or isinstance(value, (dict, list, tuple, set)):
            return None
        normalized: Final = str(value).strip()
        if not normalized:
            return None
        return normalized[: max_length or cls.MAX_FIELD_LENGTH]

    @classmethod
    def _mapping_value(cls, source: object, names: Sequence[str]) -> str | None:
        if not isinstance(source, dict):
            return None
        candidates: Final = (cls._safe_string(source.get(name)) for name in names)
        return next((value for value in candidates if value is not None), None)

    @staticmethod
    def _metadata_sources(auth: UserAPIKeyAuth) -> tuple[object, ...]:
        return (auth.metadata, auth.project_metadata, auth.organization_metadata, auth.team_metadata)

    @classmethod
    def _metadata_value(cls, auth: UserAPIKeyAuth, names: Sequence[str]) -> str | None:
        candidates: Final = (cls._mapping_value(source, names) for source in cls._metadata_sources(auth))
        return next((value for value in candidates if value is not None), None)

    @staticmethod
    def _key_fingerprint(api_key: str | None) -> str | None:
        """Correlate requests from one key without exposing the credential itself.

        Virtual keys reach here already hashed, but custom auth can return the raw
        credential, so this hashes unconditionally rather than slicing a prefix.
        """
        if not api_key:
            return None
        return hashlib.sha256(api_key.encode()).hexdigest()[:12]

    @classmethod
    def build_record(
        cls,
        data: Mapping[str, object],
        auth: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
        started_at: float | None = None,
        registry_id: str = "",
    ) -> ActiveRequestRecord:
        return ActiveRequestRecord(
            registry_id=registry_id,
            request_id=cls._safe_string(data.get("litellm_call_id")) or "",
            started_at=started_at if started_at is not None else time.time(),
            model=cls._safe_string(data.get("model")),
            call_type=cls._safe_string(call_type),
            streaming=bool(data.get("stream", False)),
            route=cls._safe_string(auth.request_route),
            user_id=cls._safe_string(auth.user_id),
            user_email=cls._safe_string(auth.user_email) if cls._include_user_email() else None,
            end_user_id=cls._safe_string(auth.end_user_id)
            or cls._metadata_value(auth, ("user_api_key_end_user_id", "end_user_id")),
            organization_id=cls._safe_string(auth.org_id),
            organization_alias=cls._mapping_value(
                auth.organization_metadata, ("organization_alias", "organization_name")
            ),
            project_id=cls._safe_string(auth.project_id),
            project_alias=cls._mapping_value(auth.project_metadata, ("project_alias", "project_name")),
            team_id=cls._safe_string(auth.team_id),
            team_alias=cls._safe_string(auth.team_alias),
            key_alias=cls._safe_string(auth.key_alias),
            key_fingerprint=cls._key_fingerprint(auth.api_key),
            pod=cls._safe_string(os.getenv("HOSTNAME"), max_length=253),
        )

    def _index_key(self, redis_cache: "RedisCache") -> str:
        return redis_cache.check_and_fix_namespace(self.INDEX_KEY)

    def _item_key(self, redis_cache: "RedisCache", registry_id: str) -> str:
        return redis_cache.check_and_fix_namespace(f"{self.ITEM_KEY_PREFIX}{registry_id}")

    async def register(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: Mapping[str, object],
        call_type: CallTypesLiteral,
        registry_id: str | None = None,
        started_at: float | None = None,
    ) -> str | None:
        redis_cache: Final = self._redis_cache()
        if not data.get("litellm_call_id") or redis_cache is None:
            return None

        resolved_id: Final = registry_id if registry_id is not None else secrets.token_hex(16)
        try:
            client: Final = redis_cache.init_async_client()
            record: Final = self.build_record(
                data, user_api_key_dict, call_type, started_at=started_at, registry_id=resolved_id
            )
            async with client.pipeline(transaction=False) as pipe:
                pipe.set(self._item_key(redis_cache, resolved_id), json.dumps(record), ex=self.ttl_seconds)
                pipe.zadd(
                    self._index_key(redis_cache),
                    {resolved_id: record["started_at"]},  # mutable-ok: redis-py zadd takes a mapping
                )
                pipe.expire(self._index_key(redis_cache), self.ttl_seconds * 2)
                await pipe.execute()
        except Exception:  # noqa: BLE001  # observability must never fail a model request
            verbose_proxy_logger.exception("Failed to register active request")
            return None
        self._track_locally(resolved_id)
        return resolved_id

    async def remove(self, registry_id: str | None) -> None:
        redis_cache: Final = self._redis_cache()
        if not registry_id or redis_cache is None:
            return
        try:
            client: Final = redis_cache.init_async_client()
            async with client.pipeline(transaction=False) as pipe:
                pipe.delete(self._item_key(redis_cache, registry_id))
                pipe.zrem(self._index_key(redis_cache), registry_id)
                await pipe.execute()
        except Exception:  # noqa: BLE001  # cleanup must never alter the HTTP response
            verbose_proxy_logger.exception("Failed to remove active request")
        self._local_tasks.pop(registry_id, None)

    async def _read_members(
        self,
        client: "Redis | RedisCluster",
        index_key: str,
        start: int,
        page_size: int,
        has_filters: bool,
    ) -> tuple[tuple[str, ...], bool]:
        if not has_filters:
            members: Final = await client.zrevrange(index_key, start, start + page_size - 1)
            return tuple(_decode_member(member) for member in members), False

        scanned: Final = await client.zrevrange(index_key, 0, self.max_scan_members - 1)
        truncated: Final = len(scanned) >= self.max_scan_members
        if truncated:
            verbose_proxy_logger.warning(
                "Active request index exceeds %s members; filtered results are truncated",
                self.max_scan_members,
            )
        return tuple(_decode_member(member) for member in scanned), truncated

    @staticmethod
    async def _drop_stale(client: "Redis | RedisCluster", index_key: str, stale: Iterable[str]) -> None:
        stale_members: Final = tuple(stale)
        if stale_members:
            await client.zrem(index_key, *stale_members)

    def _track_locally(self, registry_id: str) -> None:
        """Only the worker serving a request can cancel it, so it keeps the handle."""
        task: Final = asyncio.current_task()
        if task is not None:
            self._local_tasks[registry_id] = task
        self._ensure_cancel_watcher()

    def _ensure_cancel_watcher(self) -> None:
        if self._cancel_watcher is not None and not self._cancel_watcher.done():
            return
        if self._redis_cache() is None:
            return
        self._cancel_watcher = asyncio.create_task(self._watch_for_cancellations())

    def _cancel_key(self, redis_cache: "RedisCache", registry_id: str) -> str:
        return redis_cache.check_and_fix_namespace(f"{CANCEL_KEY_PREFIX}{registry_id}")

    async def _watch_for_cancellations(self) -> None:
        """Poll for cancellations of the requests this worker is serving.

        Polling rather than pub/sub on purpose: REDIS_SOCKET_TIMEOUT defaults to
        100ms, which kills a blocking subscription immediately.
        """
        while True:
            await asyncio.sleep(CANCEL_POLL_SECONDS)
            if not self._local_tasks:
                return
            try:
                await self._cancel_flagged_requests()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  # a failed poll must not take the worker down
                verbose_proxy_logger.exception("Active request cancel watcher failed a poll")

    async def _cancel_flagged_requests(self) -> None:
        redis_cache: Final = self._redis_cache()
        if redis_cache is None:
            return
        owned: Final = tuple(self._local_tasks)
        client: Final = redis_cache.init_async_client()
        flags: Final = await client.mget(tuple(self._cancel_key(redis_cache, rid) for rid in owned))
        flagged: Final = tuple(rid for rid, flag in zip(owned, flags) if flag is not None)
        for registry_id in flagged:
            self._cancel_owned_task(registry_id)
        if flagged:
            await client.delete(*(self._cancel_key(redis_cache, rid) for rid in flagged))

    def _cancel_owned_task(self, registry_id: str) -> None:
        task: Final = self._local_tasks.get(registry_id)
        if task is None or task.done():
            return
        verbose_proxy_logger.info("Cancelling active request %s on admin request", registry_id)
        task.cancel()

    async def request_cancel(self, registry_id: str) -> bool:
        """Flag a request for cancellation by whichever worker is serving it.

        A True here means the flag was written, not that the upstream call has
        already unwound; the worker that owns the task does the cancelling.
        """
        redis_cache: Final = self._redis_cache()
        if redis_cache is None:
            return False
        try:
            client: Final = redis_cache.init_async_client()
            known: Final = await client.zscore(self._index_key(redis_cache), registry_id)
            if known is None:
                return False
            await client.set(self._cancel_key(redis_cache, registry_id), "1", ex=CANCEL_TTL_SECONDS)
        except Exception:  # noqa: BLE001  # a failed cancel must not surface as a 500
            verbose_proxy_logger.exception("Failed to flag an active request for cancellation")
            return False
        self._cancel_owned_task(registry_id)
        return True

    async def list_requests(
        self,
        *,
        model: str | None = None,
        user_id: str | None = None,
        end_user_id: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> ActiveRequestsPage:
        redis_cache: Final = self._redis_cache()
        if redis_cache is None:
            return ActiveRequestsPage(
                available=False,
                reason="Redis is not configured; cross-replica active requests are unavailable.",
                items=(),
                total=0,
                page=page,
                page_size=page_size,
                truncated=False,
            )

        filters: Final = ActiveRequestFilters(
            model=model,
            user_id=user_id,
            end_user_id=end_user_id,
            organization_id=organization_id,
            project_id=project_id,
        )
        has_filters: Final = any(value is not None for value in filters.values())
        start: Final = (page - 1) * page_size

        client: Final = redis_cache.init_async_client()
        index_key: Final = self._index_key(redis_cache)
        await client.zremrangebyscore(index_key, "-inf", time.time() - self.ttl_seconds)

        indexed_total: Final = 0 if has_filters else await client.zcard(index_key)
        members, truncated = await self._read_members(client, index_key, start, page_size, has_filters)
        item_keys: Final = tuple(self._item_key(redis_cache, member) for member in members)
        raw_items: Final = await client.mget(item_keys) if item_keys else ()

        decoded: Final = tuple(zip(members, (_decode_record(raw_item) for raw_item in raw_items)))
        await self._drop_stale(client, index_key, (member for member, record in decoded if record is None))

        matching: Final = tuple(record for _, record in decoded if record is not None and _matches(record, filters))
        stale_count: Final = sum(1 for _, record in decoded if record is None)

        return ActiveRequestsPage(
            available=True,
            reason=None,
            items=matching[start : start + page_size] if has_filters else matching,
            total=len(matching) if has_filters else max(0, indexed_total - stale_count),
            page=page,
            page_size=page_size,
            truncated=truncated,
        )
