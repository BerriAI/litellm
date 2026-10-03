import asyncio
import hashlib
import inspect
import ipaddress
import socket
import threading
import time
import urllib.request
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import wraps
from types import MappingProxyType
from typing import Concatenate, Final, Literal, ParamSpec, Protocol, TypeAlias, TypeVar, cast

import httpx

from litellm._logging import verbose_router_logger
from litellm.constants import (
    KUBERNETES_POD_DISCOVERY_IDLE_EVICTION_SECONDS,
    KUBERNETES_POD_DISCOVERY_REFRESH_INTERVAL_SECONDS,
    KUBERNETES_POD_ROUTING_KEY,
    SESSION_ID_GENERATED_METADATA_KEY,
)

_SocketAddress: TypeAlias = tuple[str, int] | tuple[str, int, int, int] | tuple[int, bytes]
_AddressInfo: TypeAlias = tuple[socket.AddressFamily, socket.SocketKind, int, str, _SocketAddress]
_CacheKey: TypeAlias = tuple[str, int | None]
_PodSelection: TypeAlias = tuple[str, int, Literal["round_robin", "session_affinity", "session_affinity_retry"]]
_DeploymentT = TypeVar("_DeploymentT")


def _retry_count_from_metadata(metadata: object) -> int | None:
    if not isinstance(metadata, Mapping):
        return None
    metadata_mapping: Final[Mapping[str, object]] = metadata
    retry_count: Final = metadata_mapping.get("request_retry_count")
    if type(retry_count) is int and retry_count >= 0:
        return retry_count
    return None


@dataclass(frozen=True, slots=True)
class _PodSet:
    ips: tuple[str, ...]
    resolved_at: float
    last_used_at: float
    cursor: int = 0


class KubernetesPodDiscovery:
    def __init__(
        self,
        refresh_interval_seconds: float = KUBERNETES_POD_DISCOVERY_REFRESH_INTERVAL_SECONDS,
        idle_eviction_seconds: float = KUBERNETES_POD_DISCOVERY_IDLE_EVICTION_SECONDS,
        proxy_environment: Callable[[], Mapping[str, str]] = urllib.request.getproxies_environment,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.refresh_interval_seconds = refresh_interval_seconds
        self.idle_eviction_seconds = idle_eviction_seconds
        self.proxy_environment = proxy_environment
        self.clock = clock
        self._cache: Mapping[_CacheKey, _PodSet] = MappingProxyType({})
        self._refreshing: frozenset[_CacheKey] = frozenset()
        self._https_warning_hosts: frozenset[str] = frozenset()
        self._proxy_warning_hosts: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def _begin_refresh(self, key: _CacheKey, now: float) -> tuple[_PodSet | None, bool]:
        with self._lock:
            self._cache = MappingProxyType(
                {
                    cache_key: pod_set
                    for cache_key, pod_set in self._cache.items()
                    if cache_key in self._refreshing or now - pod_set.last_used_at <= self.idle_eviction_seconds
                }
            )
            cached: Final = self._cache.get(key)
            refreshing: Final = key in self._refreshing
            if cached is not None and (now - cached.resolved_at < self.refresh_interval_seconds or refreshing):
                return cached, False
            if refreshing:
                return None, False
            self._refreshing = self._refreshing | {key}
            return None, True

    def resolve_deployment(
        self,
        deployment: _DeploymentT,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> _DeploymentT:
        eligible: Final = self._eligible(deployment)
        if eligible is None:
            return deployment
        url, key, deployment_mapping = eligible
        host, port = key
        now: Final = self.clock()
        session_id: Final = self._session_id(request_kwargs)
        retry_count: Final = self._request_retry_count(request_kwargs)
        cached, should_refresh = self._begin_refresh(key, now)
        if cached is not None:
            return self._deployment_with_cached_ip(
                deployment, deployment_mapping, key, url, now, session_id, retry_count
            )
        if not should_refresh:
            return deployment

        try:
            records: Final = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            return self._apply_lookup(deployment, deployment_mapping, key, url, error, now, session_id, retry_count)
        else:
            return self._apply_lookup(deployment, deployment_mapping, key, url, records, now, session_id, retry_count)
        finally:
            with self._lock:
                self._refreshing = self._refreshing - {key}

    async def async_resolve_deployment(
        self,
        deployment: _DeploymentT,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> _DeploymentT:
        eligible: Final = self._eligible(deployment)
        if eligible is None:
            return deployment
        url, key, deployment_mapping = eligible
        host, port = key
        now: Final = self.clock()
        session_id: Final = self._session_id(request_kwargs)
        retry_count: Final = self._request_retry_count(request_kwargs)
        cached, should_refresh = self._begin_refresh(key, now)
        if cached is not None:
            return self._deployment_with_cached_ip(
                deployment, deployment_mapping, key, url, now, session_id, retry_count
            )
        if not should_refresh:
            return deployment

        try:
            result: Final = await self._async_getaddrinfo(host, port)
            return self._apply_lookup(deployment, deployment_mapping, key, url, result, now, session_id, retry_count)
        finally:
            with self._lock:
                self._refreshing = self._refreshing - {key}

    @staticmethod
    def _session_id(request_kwargs: Mapping[str, object] | None) -> str | None:
        if request_kwargs is None:
            return None
        metadata_dicts: Final = tuple(
            cast(Mapping[str, object], metadata)  # cast-ok: runtime dict check precedes metadata lookup
            for metadata in (request_kwargs.get("litellm_metadata"), request_kwargs.get("metadata"))
            if isinstance(metadata, dict)
        )
        if any(metadata.get(SESSION_ID_GENERATED_METADATA_KEY) for metadata in metadata_dicts):
            return None
        metadata_session_id: Final = next(
            (
                session_id
                for metadata in metadata_dicts
                if (session_id := metadata.get("session_id")) is not None and session_id != ""
            ),
            None,
        )
        if metadata_session_id is not None:
            return str(metadata_session_id)
        top_level_session_id: Final = request_kwargs.get("litellm_session_id")
        if top_level_session_id is None or top_level_session_id == "":
            return None
        return str(top_level_session_id)

    @staticmethod
    def _request_retry_count(request_kwargs: Mapping[str, object] | None) -> int:
        if request_kwargs is None:
            return 0
        retry_counts: Final = tuple(
            _retry_count_from_metadata(request_kwargs.get(metadata_key))
            for metadata_key in ("litellm_metadata", "metadata")
        )
        return next((retry_count for retry_count in retry_counts if retry_count is not None), 0)

    def _eligible(self, deployment: object) -> tuple[httpx.URL, _CacheKey, Mapping[str, object]] | None:
        if not isinstance(deployment, Mapping):
            return None
        typed_deployment: Final = cast(  # cast-ok: runtime Mapping check precedes typed lookup
            Mapping[str, object], deployment
        )
        target: Final = self._target(typed_deployment)
        if target is None:
            return None
        url, host, port, target_deployment = target
        if url.scheme == "https":
            self._warn_https_once(host)
            return None
        if url.scheme != "http" or self._is_ip_literal(host):
            return None
        return (url, (host, port), target_deployment)

    @staticmethod
    def _target(
        deployment_mapping: Mapping[str, object],
    ) -> tuple[httpx.URL, str, int | None, Mapping[str, object]] | None:
        raw_params: Final = deployment_mapping.get("litellm_params")
        if not isinstance(raw_params, Mapping):
            return None
        params: Final = cast(  # cast-ok: runtime Mapping validation precedes the string field checks
            Mapping[str, object], raw_params
        )
        if params.get("kubernetes_pod_discovery") is not True:
            return None
        raw_api_base: Final = params.get("api_base")
        if not isinstance(raw_api_base, str):
            return None
        try:
            url: Final = httpx.URL(raw_api_base)
        except httpx.InvalidURL:
            return None
        return (url, url.host, url.port, deployment_mapping)

    @staticmethod
    async def _async_getaddrinfo(host: str, port: int | None) -> Sequence[_AddressInfo] | OSError:
        try:
            return await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            return error

    def _apply_lookup(
        self,
        deployment: _DeploymentT,
        deployment_mapping: Mapping[str, object],
        key: _CacheKey,
        url: httpx.URL,
        result: Sequence[_AddressInfo] | OSError,
        now: float,
        session_id: str | None,
        retry_count: int,
    ) -> _DeploymentT:
        if isinstance(result, OSError):
            if isinstance(result, socket.gaierror) and self._is_authoritative_no_pods(result):
                self._store(key, (), self.clock())
                return deployment
            self._stamp_failed_refresh(key, self.clock())
            verbose_router_logger.debug("Kubernetes pod discovery DNS refresh failed for %s: %s", key[0], result)
            return self._deployment_with_cached_ip(
                deployment, deployment_mapping, key, url, now, session_id, retry_count
            )
        ips: Final = self._pod_ips(result)
        self._store(key, ips, self.clock())
        if not ips:
            return deployment
        return self._deployment_with_cached_ip(deployment, deployment_mapping, key, url, now, session_id, retry_count)

    @staticmethod
    def _is_ip_literal(host: str) -> bool:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_authoritative_no_pods(error: socket.gaierror) -> bool:
        no_pod_errors: Final = (socket.EAI_NONAME, socket.EAI_NODATA)
        return error.errno in no_pod_errors

    @staticmethod
    def _pod_ip(record: _AddressInfo) -> str | None:
        address: Final = record[4][0]
        return address if isinstance(address, str) else None

    @classmethod
    def _pod_ips(cls, records: Sequence[_AddressInfo]) -> tuple[str, ...]:
        addresses: Final = tuple(cls._pod_ip(record) for record in records)
        return tuple(sorted({address for address in addresses if address is not None}))

    def _store(self, key: _CacheKey, ips: tuple[str, ...], resolved_at: float) -> None:
        with self._lock:
            cached: Final = self._cache.get(key)
            cursor: Final = cached.cursor % len(ips) if cached is not None and cached.ips and ips else 0
            self._cache = MappingProxyType(
                {
                    **self._cache,
                    key: _PodSet(ips=ips, resolved_at=resolved_at, last_used_at=resolved_at, cursor=cursor),
                }
            )

    def _stamp_failed_refresh(self, key: _CacheKey, resolved_at: float) -> None:
        with self._lock:
            cached: Final = self._cache.get(key)
            if cached is not None:
                self._cache = MappingProxyType({**self._cache, key: replace(cached, resolved_at=resolved_at)})

    def _next_ip(
        self,
        key: _CacheKey,
        now: float,
        session_id: str | None,
        retry_count: int,
    ) -> _PodSelection | None:
        with self._lock:
            cached: Final = self._cache.get(key)
            if cached is None or not cached.ips:
                return None
            pod_count: Final = len(cached.ips)
            ranked_ips: Final = (
                tuple(
                    sorted(
                        cached.ips,
                        key=lambda address: (
                            -int.from_bytes(
                                hashlib.blake2b(f"{session_id}\x00{address}".encode(), digest_size=8).digest(),
                                "big",
                            ),
                            address,
                        ),
                    )
                )
                if session_id is not None
                else ()
            )
            ip: Final = cached.ips[cached.cursor] if session_id is None else ranked_ips[retry_count % pod_count]
            next_cursor: Final = (cached.cursor + 1) % len(cached.ips) if session_id is None else cached.cursor
            selection: Final = (
                "round_robin"
                if session_id is None
                else "session_affinity_retry"
                if retry_count > 0
                else "session_affinity"
            )
            self._cache = MappingProxyType(
                {
                    **self._cache,
                    key: replace(cached, cursor=next_cursor, last_used_at=max(cached.last_used_at, now)),
                }
            )
            return ip, pod_count, selection

    def _deployment_with_cached_ip(
        self,
        deployment: _DeploymentT,
        deployment_mapping: Mapping[str, object],
        key: _CacheKey,
        url: httpx.URL,
        now: float,
        session_id: str | None,
        retry_count: int,
    ) -> _DeploymentT:
        selection: Final = self._next_ip(key, now, session_id, retry_count)
        if selection is None:
            return deployment
        ip, pod_count, selection_kind = selection
        if self._proxy_bypasses_only_service_host(key[0], ip):
            self._warn_proxy_once(key[0])
            return deployment
        params: Final = deployment_mapping.get("litellm_params")
        if not isinstance(params, Mapping):
            return deployment
        typed_params: Final = cast(  # cast-ok: runtime Mapping validation precedes the copied provider parameters
            Mapping[str, object], params
        )
        return cast(  # cast-ok: resolving produces a copied dict for the selector's Mapping type
            _DeploymentT,
            {
                **deployment_mapping,
                "litellm_params": {**typed_params, "api_base": str(url.copy_with(host=ip))},
                KUBERNETES_POD_ROUTING_KEY: {
                    "service_host": key[0],
                    "pod_ip": ip,
                    "pod_count": pod_count,
                    "selection": selection_kind,
                },
            },
        )

    def _proxy_bypasses_only_service_host(self, host: str, ip: str) -> bool:
        proxies: Final = self.proxy_environment()
        if not (proxies.get("http") or proxies.get("all")):
            return False
        no_proxy: Final = proxies.get("no", "")
        entries: Final = tuple(entry.strip() for entry in no_proxy.split(","))
        service_bypassed: Final = "*" in entries or urllib.request.proxy_bypass_environment(host, dict(proxies))
        pod_bypassed: Final = urllib.request.proxy_bypass_environment(ip, dict(proxies))
        return service_bypassed and not pod_bypassed

    def _warn_https_once(self, host: str) -> None:
        with self._lock:
            should_warn: Final = host not in self._https_warning_hosts
            if should_warn:
                self._https_warning_hosts = self._https_warning_hosts | {host}
        if should_warn:
            verbose_router_logger.warning(
                "Kubernetes pod discovery is disabled for HTTPS host %s because substituting an IP "
                "breaks TLS hostname verification",
                host,
            )

    def _warn_proxy_once(self, host: str) -> None:
        with self._lock:
            should_warn: Final = host not in self._proxy_warning_hosts
            if should_warn:
                self._proxy_warning_hosts = self._proxy_warning_hosts | {host}
        if should_warn:
            verbose_router_logger.warning(
                "Kubernetes pod discovery keeps the service hostname %s: NO_PROXY exempts it but not its pod IPs",
                host,
            )


class _HasPodDiscovery(Protocol):
    kubernetes_pod_discovery: KubernetesPodDiscovery


_P = ParamSpec("_P")
_R = TypeVar("_R")
_S = TypeVar("_S", bound=_HasPodDiscovery)


def _request_kwargs_from_call(
    signature: inspect.Signature, *args: object, **kwargs: object
) -> Mapping[str, object] | None:
    try:
        request_kwargs: Final = signature.bind_partial(*args, **kwargs).arguments.get(
            "request_kwargs",
            kwargs.get("request_kwargs"),
        )
    except TypeError:
        return _mapping_request_kwargs(kwargs.get("request_kwargs"))
    return _mapping_request_kwargs(request_kwargs)


def _mapping_request_kwargs(request_kwargs: object) -> Mapping[str, object] | None:
    if not isinstance(request_kwargs, Mapping):
        return None
    return cast(  # cast-ok: router selector request kwargs use string keys
        Mapping[str, object],
        request_kwargs,
    )


def resolve_pods_after(
    fn: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    signature: Final = inspect.signature(fn)

    @wraps(fn)
    def wrapped(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        deployment: Final = fn(self, *args, **kwargs)
        return self.kubernetes_pod_discovery.resolve_deployment(
            deployment,
            _request_kwargs_from_call(signature, self, *args, **kwargs),
        )

    return wrapped


def async_resolve_pods_after(
    fn: Callable[Concatenate[_S, _P], Awaitable[_R]],
) -> Callable[Concatenate[_S, _P], Coroutine[object, object, _R]]:
    signature: Final = inspect.signature(fn)

    @wraps(fn)
    async def wrapped(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        deployment: Final = await fn(self, *args, **kwargs)
        return await self.kubernetes_pod_discovery.async_resolve_deployment(
            deployment,
            _request_kwargs_from_call(signature, self, *args, **kwargs),
        )

    return wrapped


def resolve_pods_after_bound(
    fn: Callable[_P, _R],
    discovery: KubernetesPodDiscovery,
) -> Callable[_P, _R]:
    signature: Final = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        deployment: Final = fn(*args, **kwargs)
        return discovery.resolve_deployment(
            deployment,
            _request_kwargs_from_call(signature, *args, **kwargs),
        )

    return wrapped


def async_resolve_pods_after_bound(
    fn: Callable[_P, Awaitable[_R]],
    discovery: KubernetesPodDiscovery,
) -> Callable[_P, Coroutine[object, object, _R]]:
    signature: Final = inspect.signature(fn)

    @wraps(fn)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        deployment: Final = await fn(*args, **kwargs)
        return await discovery.async_resolve_deployment(
            deployment,
            _request_kwargs_from_call(signature, *args, **kwargs),
        )

    return wrapped
