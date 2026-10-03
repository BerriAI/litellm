import asyncio
import copy
import hashlib
import json
import logging
import re
import socket
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from itertools import chain, count, repeat
from typing import Final, TypeAlias, cast

import httpx
import pytest
import respx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.constants import KUBERNETES_POD_ROUTING_KEY, SESSION_ID_GENERATED_METADATA_KEY
from litellm.router import CustomRoutingStrategyBase, Router
from litellm.router_utils.kubernetes_pod_discovery import (
    KubernetesPodDiscovery,
    async_resolve_pods_after,
    async_resolve_pods_after_bound,
    resolve_pods_after,
    resolve_pods_after_bound,
)

_SERVICE_URL: Final = "http://vllm-headless.ns.svc.cluster.local:8000/v1"
_THREE_POD_CHAT_PATTERN: Final = re.compile(r"http://(?:10\.0\.0\.1|10\.0\.0\.2|10\.0\.0\.3):8000/v1/chat/completions")
_SocketAddress: TypeAlias = tuple[str, int] | tuple[str, int, int, int]
_AddrInfo: TypeAlias = tuple[socket.AddressFamily, socket.SocketKind, int, str, _SocketAddress]
_NO_POD_ERRNOS: Final = tuple(sorted((socket.EAI_NONAME, socket.EAI_NODATA)))
_CHAT_RESPONSE: Final = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "my-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _deployment(api_base: str = _SERVICE_URL, kubernetes_pod_discovery: bool | None = True) -> dict[str, object]:
    params: Final = {
        "model": "openai/my-model",
        "api_base": api_base,
        "api_key": "fake",
        **({"kubernetes_pod_discovery": kubernetes_pod_discovery} if kubernetes_pod_discovery is not None else {}),
    }
    return {
        "model_name": "gpu-model",
        "litellm_params": params,
        "model_info": {"id": "registered-id"},
    }


class _ModelListRoutingStrategy(CustomRoutingStrategyBase):
    def __init__(self, router: Router) -> None:
        self._router: Final = router

    def get_available_deployment(self, *args: object, **kwargs: object) -> Mapping[str, object]:
        return self._router.model_list[0]

    async def async_get_available_deployment(self, *args: object, **kwargs: object) -> Mapping[str, object]:
        return self._router.model_list[0]


def _record(ip: str, port: int) -> _AddrInfo:
    address: Final = (ip, port, 0, 0) if ":" in ip else (ip, port)
    family: Final = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", address)


def _records(*ips: str, port: int = 8000) -> list[_AddrInfo]:
    return [_record(ip, port) for ip in ips]


def _clock(ticks: tuple[float, ...]) -> Callable[[], float]:
    timestamps: Final = iter(ticks)

    def monotonic() -> float:
        return next(timestamps)

    return monotonic


def _proxy_environment(no_proxy: str) -> Callable[[], Mapping[str, str]]:
    return lambda: {"http": "http://proxy.example:8080", "no": no_proxy}


def _empty_proxy_environment() -> Mapping[str, str]:
    return {}


def _patch_router_pod_discovery(
    monkeypatch: pytest.MonkeyPatch,
    *,
    clock: Callable[[], float] | None = None,
    refresh_interval_seconds: float | None = None,
) -> None:
    def create_discovery() -> KubernetesPodDiscovery:
        if refresh_interval_seconds is not None and clock is not None:
            return KubernetesPodDiscovery(
                refresh_interval_seconds=refresh_interval_seconds,
                clock=clock,
                proxy_environment=_empty_proxy_environment,
            )
        if refresh_interval_seconds is not None:
            return KubernetesPodDiscovery(
                refresh_interval_seconds=refresh_interval_seconds,
                proxy_environment=_empty_proxy_environment,
            )
        if clock is not None:
            return KubernetesPodDiscovery(clock=clock, proxy_environment=_empty_proxy_environment)
        return KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    monkeypatch.setattr("litellm.router.KubernetesPodDiscovery", create_discovery)


def _stub_sync_dns(monkeypatch: pytest.MonkeyPatch, *ips: str) -> None:
    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records(*ips, port=int(port or 8000))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def _api_base(deployment: Mapping[str, object]) -> str:
    params: Final = deployment["litellm_params"]
    assert isinstance(params, Mapping)
    api_base: Final = params["api_base"]
    assert isinstance(api_base, str)
    return api_base


def _ranked_session_ips(session_id: str) -> tuple[str, ...]:
    ips: Final = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
    return tuple(
        sorted(
            ips,
            key=lambda ip: (
                -int.from_bytes(hashlib.blake2b(f"{session_id}\x00{ip}".encode(), digest_size=8).digest(), "big"),
                ip,
            ),
        )
    )


def _request_body(content: bytes) -> Mapping[str, object]:
    body: Final = json.loads(content)
    assert isinstance(body, dict)
    return cast(Mapping[str, object], body)


def _cache_async_client(router: Router) -> AsyncOpenAI:
    client: Final = AsyncOpenAI(api_key="fake", base_url=_SERVICE_URL)
    router.cache.set_cache(key="registered-id_async_client", value=client, local_only=True)
    return client


def _cache_sync_client(router: Router) -> OpenAI:
    client: Final = OpenAI(api_key="fake", base_url=_SERVICE_URL)
    router.cache.set_cache(key="registered-id_client", value=client, local_only=True)
    return client


async def _router_acompletion(router: Router, **request_kwargs: object) -> None:
    await router.acompletion(
        model="gpu-model",
        messages=[{"role": "user", "content": "hello"}],
        **request_kwargs,
    )


def _router_completion(router: Router, **request_kwargs: object) -> None:
    router.completion(
        model="gpu-model",
        messages=[{"role": "user", "content": "hello"}],
        **request_kwargs,
    )


def test_sync_resolution_round_robins_sorted_ips_without_mutating_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Final = _deployment()
    before: Final = copy.deepcopy(original)
    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        assert type == socket.SOCK_STREAM
        return _records("10.0.0.3", "10.0.0.1", "10.0.0.2", "10.0.0.1")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=30,
        proxy_environment=_empty_proxy_environment,
    )

    results: Final = tuple(discovery.resolve_deployment(original) for _ in range(3))

    assert tuple(_api_base(result) for result in results) == (
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.2:8000/v1",
        "http://10.0.0.3:8000/v1",
    )
    assert original == before
    assert results[0] is not original
    assert results[0]["model_info"] is original["model_info"]


@pytest.mark.asyncio
async def test_async_resolution_round_robins_sorted_ips_and_preserves_url_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Final = _deployment(api_base=f"{_SERVICE_URL}/custom?version=1")
    before: Final = copy.deepcopy(original)
    loop: Final = asyncio.get_running_loop()
    lookups: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return _records("10.0.0.3", "10.0.0.1", "10.0.0.2", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=30,
        proxy_environment=_empty_proxy_environment,
    )

    results: Final = (
        await discovery.async_resolve_deployment(original),
        await discovery.async_resolve_deployment(original),
        await discovery.async_resolve_deployment(original),
    )

    assert tuple(_api_base(result) for result in results) == (
        "http://10.0.0.1:8000/v1/custom?version=1",
        "http://10.0.0.2:8000/v1/custom?version=1",
        "http://10.0.0.3:8000/v1/custom?version=1",
    )
    assert original == before
    assert results[0] is not original
    assert results[0]["model_info"] is original["model_info"]


def test_refresh_reuses_cached_set_until_interval_then_uses_new_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock: Final = _clock((10.0, 10.0, 14.99, 15.0, 15.0))
    responses: Final = iter((_records("10.0.0.1", "10.0.0.2"), _records("10.0.0.3")))
    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return list(next(responses))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    first: Final = _api_base(discovery.resolve_deployment(_deployment()))
    cached: Final = _api_base(discovery.resolve_deployment(_deployment()))
    refreshed: Final = _api_base(discovery.resolve_deployment(_deployment()))

    assert (first, cached, refreshed) == (
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.2:8000/v1",
        "http://10.0.0.3:8000/v1",
    )
    assert next(lookups) == 2


def test_refresh_carries_round_robin_cursor_into_new_ip_set(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Final = _clock((0.0, 0.0, 5.0, 5.0))
    responses: Final = iter((_records("10.0.0.1", "10.0.0.2", "10.0.0.3"), _records("10.0.0.4", "10.0.0.5")))

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return list(next(responses))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    first: Final = _api_base(discovery.resolve_deployment(_deployment()))
    refreshed: Final = _api_base(discovery.resolve_deployment(_deployment()))

    assert (first, refreshed) == ("http://10.0.0.1:8000/v1", "http://10.0.0.5:8000/v1")


def test_idle_cache_eviction_restarts_round_robin_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Final = _clock((0.0, 0.0, 0.0, 11.0, 11.0))
    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        idle_eviction_seconds=10,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    first: Final = _api_base(discovery.resolve_deployment(_deployment()))
    second: Final = _api_base(discovery.resolve_deployment(_deployment()))
    after_eviction: Final = _api_base(discovery.resolve_deployment(_deployment()))

    assert (first, second, after_eviction) == (
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.2:8000/v1",
        "http://10.0.0.1:8000/v1",
    )
    assert next(lookups) == 2


def test_active_pod_set_survives_idle_eviction_when_refresh_interval_is_longer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values: Final = iter(chain((0.0, 0.0, 200.0, 400.0, 550.0), repeat(550.0)))

    def clock() -> float:
        return next(clock_values)

    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=600,
        idle_eviction_seconds=300,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    api_bases: Final = tuple(_api_base(discovery.resolve_deployment(_deployment())) for _ in range(4))

    assert api_bases == (
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.2:8000/v1",
        "http://10.0.0.3:8000/v1",
        "http://10.0.0.1:8000/v1",
    )
    assert next(lookups) == 1


def test_slow_dns_refresh_does_not_move_last_use_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    clock_values: Final = iter(chain((0.0, 250.0), repeat(500.0)))

    def clock() -> float:
        return next(clock_values)

    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=600,
        idle_eviction_seconds=300,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    api_bases: Final = tuple(_api_base(discovery.resolve_deployment(_deployment())) for _ in range(2))

    assert api_bases == ("http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1")
    assert next(lookups) == 1


def test_proxy_bypassing_only_service_hostname_keeps_api_base_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _stub_sync_dns(monkeypatch, "10.244.1.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_proxy_environment(".svc.cluster.local"))
    original: Final = _deployment()

    with caplog.at_level(logging.WARNING):
        first: Final = discovery.resolve_deployment(original)
        second: Final = discovery.resolve_deployment(original)

    assert first is original
    assert second is original
    assert _api_base(first) == _SERVICE_URL
    assert sum("NO_PROXY exempts it but not its pod IPs" in record.getMessage() for record in caplog.records) == 1


def test_proxy_with_pod_cidr_in_no_proxy_keeps_service_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sync_dns(monkeypatch, "10.244.1.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_proxy_environment(".svc.cluster.local,10.244.0.0/16"))

    result: Final = discovery.resolve_deployment(_deployment())

    assert _api_base(result) == _SERVICE_URL


def test_proxy_with_exact_pod_ip_in_no_proxy_keeps_pod_ip_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sync_dns(monkeypatch, "10.244.1.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_proxy_environment(".svc.cluster.local,10.244.1.3"))

    result: Final = discovery.resolve_deployment(_deployment())

    assert _api_base(result) == "http://10.244.1.3:8000/v1"


def test_proxy_with_empty_no_proxy_keeps_pod_ip_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sync_dns(monkeypatch, "10.244.1.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_proxy_environment(""))

    result: Final = discovery.resolve_deployment(_deployment())

    assert _api_base(result) == "http://10.244.1.3:8000/v1"


def test_empty_proxy_environment_keeps_pod_ip_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sync_dns(monkeypatch, "10.244.1.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    result: Final = discovery.resolve_deployment(_deployment())

    assert _api_base(result) == "http://10.244.1.3:8000/v1"


@pytest.mark.parametrize("no_pods_errno", _NO_POD_ERRNOS)
def test_authoritative_no_pods_uses_service_url_and_transient_failure_keeps_cached_pods(
    monkeypatch: pytest.MonkeyPatch, no_pods_errno: int
) -> None:
    clock: Final = _clock((0.0, 0.0, 0.0, 0.0, 5.0, 5.0))
    missing: Final = _deployment()
    no_pod_lookups: Final = count()

    def missing_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(no_pod_lookups)
        raise socket.gaierror(no_pods_errno, "missing")

    monkeypatch.setattr(socket, "getaddrinfo", missing_getaddrinfo)
    discovery_without_pods: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )

    assert discovery_without_pods.resolve_deployment(missing) is missing
    assert discovery_without_pods.resolve_deployment(missing) is missing
    assert _api_base(missing) == _SERVICE_URL
    assert next(no_pod_lookups) == 1

    responses: Final = iter((_records("10.0.0.4"), socket.gaierror(socket.EAI_AGAIN, "temporary")))

    def transient_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        response: Final = next(responses)
        if isinstance(response, socket.gaierror):
            raise response
        return list(response)

    monkeypatch.setattr(socket, "getaddrinfo", transient_getaddrinfo)
    discovery_with_pods: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )
    first: Final = _api_base(discovery_with_pods.resolve_deployment(_deployment()))
    after_failure: Final = _api_base(discovery_with_pods.resolve_deployment(_deployment()))

    assert (first, after_failure) == ("http://10.0.0.4:8000/v1", "http://10.0.0.4:8000/v1")


@pytest.mark.parametrize(
    ("api_base", "flag"),
    (
        pytest.param(_SERVICE_URL, None, id="flag-absent"),
        pytest.param(_SERVICE_URL, False, id="flag-false"),
        pytest.param("https://vllm-headless.ns.svc.cluster.local:8000/v1", True, id="https"),
        pytest.param("http://10.0.0.9:8000/v1", True, id="ip-literal"),
        pytest.param("http://[zzzz]", True, id="malformed-url"),
    ),
)
def test_unsupported_deployments_are_returned_unchanged_without_dns(
    monkeypatch: pytest.MonkeyPatch, api_base: str, flag: bool | None
) -> None:
    deployment: Final = _deployment(api_base=api_base, kubernetes_pod_discovery=flag)
    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    assert discovery.resolve_deployment(deployment) is deployment
    assert next(lookups) == 0


@pytest.mark.asyncio
async def test_non_mapping_deployments_are_returned_unchanged() -> None:
    deployment: Final = object()
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    assert discovery.resolve_deployment(deployment) is deployment
    assert await discovery.async_resolve_deployment(deployment) is deployment


def test_https_discovery_warning_is_emitted_once_per_host(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    deployment: Final = _deployment(api_base="https://vllm-headless.ns.svc.cluster.local:8000/v1")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    results: Final = (
        discovery.resolve_deployment(deployment),
        discovery.resolve_deployment(deployment),
    )
    warnings: Final = tuple(
        record
        for record in caplog.records
        if "Kubernetes pod discovery is disabled for HTTPS host" in record.getMessage()
    )

    assert results == (deployment, deployment)
    assert all(result is deployment for result in results)
    assert len(warnings) == 1


def test_ipv6_pod_address_is_bracketed_in_url(monkeypatch: pytest.MonkeyPatch) -> None:
    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("2001:db8::2")

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    result: Final = discovery.resolve_deployment(_deployment())

    assert _api_base(result) == "http://[2001:db8::2]:8000/v1"


@pytest.mark.asyncio
async def test_concurrent_async_refresh_uses_cached_ips_while_refresh_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock: Final = _clock((0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0))
    loop: Final = asyncio.get_running_loop()
    refresh_started: Final = asyncio.Event()
    finish_refresh: Final = asyncio.Event()
    lookups: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        lookup_number: Final = next(lookups)
        if lookup_number == 0:
            return _records("10.0.0.1", port=int(port or 0))
        if lookup_number > 1:
            return _records("10.0.0.3", port=int(port or 0))
        refresh_started.set()
        await finish_refresh.wait()
        return _records("10.0.0.2", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )
    first: Final = _api_base(await discovery.async_resolve_deployment(_deployment()))
    refresh_task: Final = asyncio.create_task(discovery.async_resolve_deployment(_deployment()))
    await refresh_started.wait()
    concurrent: Final = _api_base(await discovery.async_resolve_deployment(_deployment()))
    finish_refresh.set()
    refreshed: Final = _api_base(await refresh_task)

    assert (first, concurrent, refreshed) == (
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.1:8000/v1",
        "http://10.0.0.2:8000/v1",
    )
    assert next(lookups) == 2


@pytest.mark.parametrize("cache_existing", (False, True), ids=("cold-cache", "stale-cache"))
def test_concurrent_sync_refresh_uses_cached_ips_or_service_url_while_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    cache_existing: bool,
) -> None:
    ticks: Final = (0.0, 0.0, 5.0, 5.0, 5.0) if cache_existing else (0.0, 0.0, 0.0)
    clock: Final = _clock(ticks)
    refresh_started: Final = threading.Event()
    finish_refresh: Final = threading.Event()
    lookups: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        lookup_number: Final = next(lookups)
        if cache_existing and lookup_number == 0:
            return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))
        refresh_started.set()
        assert finish_refresh.wait(timeout=3)
        return _records("10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    discovery: Final = KubernetesPodDiscovery(
        refresh_interval_seconds=5,
        clock=clock,
        proxy_environment=_empty_proxy_environment,
    )
    if cache_existing:
        assert _api_base(discovery.resolve_deployment(_deployment())) == "http://10.0.0.1:8000/v1"

    refresh_result: Final = Future[dict[str, object]]()

    def refresh() -> None:
        refresh_result.set_result(discovery.resolve_deployment(_deployment()))

    thread: Final = threading.Thread(target=refresh, daemon=True)
    thread.start()
    try:
        assert refresh_started.wait(timeout=2)
        concurrent: Final = discovery.resolve_deployment(_deployment())
        assert _api_base(concurrent) == ("http://10.0.0.2:8000/v1" if cache_existing else _SERVICE_URL)
        assert next(lookups) == (2 if cache_existing else 1)
    finally:
        finish_refresh.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert _api_base(refresh_result.result(timeout=2)) == "http://10.0.0.3:8000/v1"


@pytest.mark.asyncio
async def test_router_sends_pod_hosts_without_forwarding_discovery_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock: Final = _clock((0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0))
    loop: Final = asyncio.get_running_loop()
    async_lookups: Final = count()
    sync_lookups: Final = count()

    async def async_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(async_lookups)
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    def sync_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(sync_lookups)
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", async_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", sync_getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=clock)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(
            url__regex=re.compile(
                r"http://(?:10\.0\.0\.1|10\.0\.0\.2|vllm-headless\.ns\.svc\.cluster\.local):8000/v1/chat/completions"
            )
        ).mock(return_value=httpx.Response(200, json=_CHAT_RESPONSE))
        discovery_enabled_router: Final = Router(model_list=[_deployment()])
        control_router: Final = Router(model_list=[_deployment(kubernetes_pod_discovery=None)])
        cached_async_client: Final = AsyncOpenAI(api_key="fake", base_url=_SERVICE_URL)
        cached_sync_client: Final = OpenAI(api_key="fake", base_url=_SERVICE_URL)
        discovery_enabled_router.cache.set_cache(
            key="registered-id_async_client",
            value=cached_async_client,
            local_only=True,
        )
        discovery_enabled_router.cache.set_cache(
            key="registered-id_client",
            value=cached_sync_client,
            local_only=True,
        )
        for _ in range(4):
            await discovery_enabled_router.acompletion(
                model="gpu-model", messages=[{"role": "user", "content": "hello"}]
            )
        discovery_enabled_router.completion(model="gpu-model", messages=[{"role": "user", "content": "hello"}])
        await control_router.acompletion(model="gpu-model", messages=[{"role": "user", "content": "hello"}])

        hosts: Final = tuple(call.request.url.host for call in route.calls)
        bodies: Final = tuple(_request_body(call.request.content) for call in route.calls)
        await cached_async_client.close()
        cached_sync_client.close()

    assert hosts == (
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.1",
        "vllm-headless.ns.svc.cluster.local",
    )
    assert all("kubernetes_pod_discovery" not in body for body in bodies)
    assert next(async_lookups) == 1
    assert next(sync_lookups) == 1


@pytest.mark.asyncio
async def test_router_session_requests_stick_to_one_pod_for_sync_and_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()
    async_lookups: Final = count()
    sync_lookups: Final = count()

    async def async_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(async_lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    def sync_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(sync_lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", async_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", sync_getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        async_router: Final = Router(model_list=[_deployment()])
        sync_router: Final = Router(model_list=[_deployment()])
        async_client: Final = _cache_async_client(async_router)
        sync_client: Final = _cache_sync_client(sync_router)
        for _ in range(6):
            await _router_acompletion(async_router, metadata={"session_id": "s1"})
        for _ in range(6):
            _router_completion(sync_router, metadata={"session_id": "s1"})
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await async_client.close()
        sync_client.close()

    assert len(set(hosts[:6])) == 1
    assert len(set(hosts[6:])) == 1
    assert next(async_lookups) == 1
    assert next(sync_lookups) == 1


@pytest.mark.asyncio
async def test_router_session_mapping_repeats_across_pods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()
    lookups: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    session_ids: Final = tuple(f"session-{index}" for index in range(30))
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for session_id in session_ids:
            await _router_acompletion(router, metadata={"session_id": session_id})
        await _router_acompletion(router)
        for session_id in session_ids:
            await _router_acompletion(router, metadata={"session_id": session_id})
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    first_pass: Final = hosts[: len(session_ids)]
    second_pass: Final = hosts[len(session_ids) + 1 :]
    assert len(set(first_pass)) > 1
    assert hosts[len(session_ids)] == "10.0.0.1"
    assert second_pass == first_pass
    assert next(lookups) == 1


@pytest.mark.asyncio
async def test_generated_session_id_uses_round_robin_pod_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for _ in range(6):
            await _router_acompletion(
                router,
                metadata={
                    "session_id": "generated-session",
                    SESSION_ID_GENERATED_METADATA_KEY: True,
                },
            )
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert hosts == ("10.0.0.1", "10.0.0.2", "10.0.0.3") * 2


@pytest.mark.asyncio
async def test_empty_session_id_uses_round_robin_pod_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for _ in range(6):
            await _router_acompletion(router, metadata={"session_id": ""})
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert hosts == ("10.0.0.1", "10.0.0.2", "10.0.0.3") * 2


@pytest.mark.asyncio
async def test_session_requests_do_not_advance_round_robin_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        await _router_acompletion(router)
        await _router_acompletion(router, metadata={"session_id": "s1"})
        await _router_acompletion(router, metadata={"session_id": "s1"})
        await _router_acompletion(router)
        await _router_acompletion(router, metadata={"session_id": "s1"})
        await _router_acompletion(router)
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert tuple(hosts[index] for index in (0, 3, 5)) == (
        "10.0.0.1",
        "10.0.0.2",
        "10.0.0.3",
    )


@pytest.mark.asyncio
async def test_session_pod_membership_change_remaps_only_affected_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()
    lookups: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        lookup_number: Final = next(lookups)
        ips: Final = ("10.0.0.1", "10.0.0.2", "10.0.0.3") if lookup_number == 0 else ("10.0.0.2", "10.0.0.3")
        return _records(*ips, port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    session_ids: Final = tuple(f"session-{index}" for index in range(60))
    _patch_router_pod_discovery(
        monkeypatch,
        refresh_interval_seconds=10,
        clock=_clock((0.0,) * 62 + (10.0,) * 10),
    )
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for session_id in session_ids:
            await _router_acompletion(router, metadata={"session_id": session_id})
        initial_hosts: Final = tuple(call.request.url.host for call in route.calls)
        removed_session_ids: Final = tuple(
            session_id for session_id, host in zip(session_ids, initial_hosts) if host == "10.0.0.1"
        )
        surviving_session_ids: Final = tuple(
            session_id for session_id, host in zip(session_ids, initial_hosts) if host == "10.0.0.2"
        )
        assert removed_session_ids
        assert len(surviving_session_ids) >= 2
        await _router_acompletion(router)
        await _router_acompletion(router, metadata={"session_id": surviving_session_ids[0]})
        await _router_acompletion(router, metadata={"session_id": removed_session_ids[0]})
        await _router_acompletion(router, metadata={"session_id": surviving_session_ids[1]})
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert hosts[60] == "10.0.0.1"
    assert hosts[61] == "10.0.0.2"
    assert hosts[62] in ("10.0.0.2", "10.0.0.3")
    assert hosts[63] == "10.0.0.2"
    assert next(lookups) == 2


@pytest.mark.asyncio
async def test_custom_routing_strategy_keeps_sync_and_async_sessions_on_one_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()
    async_lookups: Final = count()
    sync_lookups: Final = count()

    async def async_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(async_lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    def sync_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(sync_lookups)
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", async_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", sync_getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        async_router: Final = Router(model_list=[_deployment()])
        sync_router: Final = Router(model_list=[_deployment()])
        async_router.set_custom_routing_strategy(_ModelListRoutingStrategy(async_router))
        sync_router.set_custom_routing_strategy(_ModelListRoutingStrategy(sync_router))
        async_client: Final = _cache_async_client(async_router)
        sync_client: Final = _cache_sync_client(sync_router)
        for _ in range(6):
            await _router_acompletion(async_router, metadata={"session_id": "custom-session"})
        for _ in range(6):
            _router_completion(sync_router, metadata={"session_id": "custom-session"})
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await async_client.close()
        sync_client.close()

    assert len(set(hosts[:6])) == 1
    assert len(set(hosts[6:])) == 1
    assert next(async_lookups) == 1
    assert next(sync_lookups) == 1


@pytest.mark.asyncio
async def test_top_level_litellm_session_id_keeps_requests_on_one_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for _ in range(6):
            await _router_acompletion(router, litellm_session_id="s1")
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert len(set(hosts)) == 1


@pytest.mark.asyncio
async def test_generated_metadata_ignores_top_level_litellm_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=lambda: 0.0)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(
            return_value=httpx.Response(200, json=_CHAT_RESPONSE)
        )
        router: Final = Router(model_list=[_deployment()])
        client: Final = _cache_async_client(router)
        for _ in range(6):
            await _router_acompletion(
                router,
                litellm_session_id="s1",
                metadata={
                    "session_id": "s1",
                    SESSION_ID_GENERATED_METADATA_KEY: True,
                },
            )
        hosts: Final = tuple(call.request.url.host for call in route.calls)
        await client.close()

    assert hosts == ("10.0.0.1", "10.0.0.2", "10.0.0.3") * 2


@pytest.mark.asyncio
async def test_custom_routing_strategy_resolves_pod_hosts_for_sync_and_async_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock: Final = _clock((0.0,) * 40)
    loop: Final = asyncio.get_running_loop()
    async_lookups: Final = count()
    sync_lookups: Final = count()

    async def async_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(async_lookups)
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    def sync_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        next(sync_lookups)
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    monkeypatch.setattr(loop, "getaddrinfo", async_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", sync_getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=clock)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(
            url__regex=re.compile(
                r"http://(?:10\.0\.0\.1|10\.0\.0\.2|vllm-headless\.ns\.svc\.cluster\.local):8000/v1/chat/completions"
            )
        ).mock(return_value=httpx.Response(200, json=_CHAT_RESPONSE))
        async_router: Final = Router(model_list=[_deployment()])
        sync_router: Final = Router(model_list=[_deployment()])
        async_router.set_custom_routing_strategy(_ModelListRoutingStrategy(async_router))
        sync_router.set_custom_routing_strategy(_ModelListRoutingStrategy(sync_router))
        for _ in range(4):
            await async_router.acompletion(model="gpu-model", messages=[{"role": "user", "content": "hello"}])
        for _ in range(4):
            sync_router.completion(model="gpu-model", messages=[{"role": "user", "content": "hello"}])
        hosts: Final = tuple(call.request.url.host for call in route.calls)

    assert hosts == ("10.0.0.1", "10.0.0.2") * 4
    assert next(async_lookups) == 1
    assert next(sync_lookups) == 1


@pytest.mark.asyncio
async def test_unresolved_router_selectors_keep_service_hostname() -> None:
    router: Final = Router(model_list=[_deployment()])

    sync_selected: Final = router._get_available_deployment_unresolved(model="gpu-model", request_kwargs={})
    async_selected: Final = await router._async_get_available_deployment_unresolved(
        model="gpu-model",
        request_kwargs={},
    )

    assert _api_base(sync_selected) == _SERVICE_URL
    assert _api_base(async_selected) == _SERVICE_URL


@pytest.mark.asyncio
async def test_router_async_retry_uses_next_discovered_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Final = _clock((0.0,) * 8)
    loop: Final = asyncio.get_running_loop()
    attempts: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    def response_for(request: httpx.Request) -> httpx.Response:
        attempt: Final = next(attempts)
        status_code: Final = 500 if attempt == 0 else 200
        body: Final = {"error": {"message": "retryable", "type": "server_error"}} if attempt == 0 else _CHAT_RESPONSE
        return httpx.Response(status_code, json=body, request=request)

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=clock)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(
            url__regex=re.compile(r"http://(?:10\.0\.0\.1|10\.0\.0\.2):8000/v1/chat/completions")
        ).mock(side_effect=response_for)
        router: Final = Router(model_list=[_deployment()], num_retries=1)

        response: Final = await router.acompletion(
            model="gpu-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        hosts: Final = tuple(call.request.url.host for call in route.calls)

    assert response.choices[0].message.content == "ok"
    assert hosts == ("10.0.0.1", "10.0.0.2")


def test_router_sync_retry_uses_next_discovered_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    clock: Final = _clock((0.0,) * 8)
    attempts: Final = count()

    def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", port=int(port or 0))

    def response_for(request: httpx.Request) -> httpx.Response:
        attempt: Final = next(attempts)
        status_code: Final = 500 if attempt == 0 else 200
        body: Final = {"error": {"message": "retryable", "type": "server_error"}} if attempt == 0 else _CHAT_RESPONSE
        return httpx.Response(status_code, json=body, request=request)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch, clock=clock)

    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(
            url__regex=re.compile(r"http://(?:10\.0\.0\.1|10\.0\.0\.2):8000/v1/chat/completions")
        ).mock(side_effect=response_for)
        router: Final = Router(model_list=[_deployment()], num_retries=1)

        response: Final = router.completion(
            model="gpu-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        hosts: Final = tuple(call.request.url.host for call in route.calls)

    assert response.choices[0].message.content == "ok"
    assert hosts == ("10.0.0.1", "10.0.0.2")


@pytest.mark.parametrize(
    ("litellm_retry_count", "metadata_retry_count", "expected_rank"),
    [
        (0, None, 0),
        (None, None, 0),
        (True, None, 0),
        ("1", None, 0),
        (-1, None, 0),
        (1, None, 1),
        (True, 1, 1),
        (1, 0, 1),
    ],
)
def test_session_retry_count_selects_ranked_pod(
    monkeypatch: pytest.MonkeyPatch,
    litellm_retry_count: object,
    metadata_retry_count: object,
    expected_rank: int,
) -> None:
    _stub_sync_dns(monkeypatch, "10.0.0.1", "10.0.0.2", "10.0.0.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)
    request_kwargs: Final = {
        "litellm_metadata": {
            "session_id": "s1",
            **({"request_retry_count": litellm_retry_count} if litellm_retry_count is not None else {}),
        },
        "metadata": ({"request_retry_count": metadata_retry_count} if metadata_retry_count is not None else {}),
    }

    resolved: Final = discovery.resolve_deployment(_deployment(), request_kwargs)

    assert httpx.URL(_api_base(resolved)).host == _ranked_session_ips("s1")[expected_rank]


@pytest.mark.asyncio
async def test_router_session_retry_uses_next_rendezvous_pod(monkeypatch: pytest.MonkeyPatch) -> None:
    loop: Final = asyncio.get_running_loop()
    attempts: Final = count()

    async def getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    def response_for(request: httpx.Request) -> httpx.Response:
        attempt: Final = next(attempts)
        status_code: Final = 500 if attempt == 0 else 200
        body: Final = {"error": {"message": "retryable", "type": "server_error"}} if attempt == 0 else _CHAT_RESPONSE
        return httpx.Response(status_code, json=body, request=request)

    monkeypatch.setattr(loop, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    _patch_router_pod_discovery(monkeypatch)

    session_id: Final = "retry-session"
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(url__regex=_THREE_POD_CHAT_PATTERN).mock(side_effect=response_for)
        router: Final = Router(model_list=[_deployment()], num_retries=1)

        response: Final = await router.acompletion(
            model="gpu-model",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"session_id": session_id},
        )
        hosts: Final = tuple(call.request.url.host for call in route.calls)

    assert response.choices[0].message.content == "ok"
    assert hosts == _ranked_session_ips(session_id)[:2]


@pytest.mark.asyncio
async def test_positional_request_kwargs_reach_all_pod_resolver_decorators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop: Final = asyncio.get_running_loop()

    async def async_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    def sync_getaddrinfo(
        host: str | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[_AddrInfo]:
        return _records("10.0.0.1", "10.0.0.2", "10.0.0.3", port=int(port or 0))

    class _Selector:
        def __init__(self, discovery: KubernetesPodDiscovery) -> None:
            self.kubernetes_pod_discovery: Final = discovery

        @resolve_pods_after
        def sync(self, deployment: dict[str, object], request_kwargs: Mapping[str, object]) -> dict[str, object]:
            return deployment

        @async_resolve_pods_after
        async def async_call(
            self,
            deployment: dict[str, object],
            request_kwargs: Mapping[str, object],
        ) -> dict[str, object]:
            return deployment

    class _BoundSelector:
        def sync(self, deployment: dict[str, object], request_kwargs: Mapping[str, object]) -> dict[str, object]:
            return deployment

        async def async_call(
            self,
            deployment: dict[str, object],
            request_kwargs: Mapping[str, object],
        ) -> dict[str, object]:
            return deployment

    monkeypatch.setattr(loop, "getaddrinfo", async_getaddrinfo)
    monkeypatch.setattr(socket, "getaddrinfo", sync_getaddrinfo)
    session_id: Final = next(
        candidate
        for candidate in (f"positional-{index}" for index in range(100))
        if _ranked_session_ips(candidate)[0] != "10.0.0.1"
    )
    request_kwargs: Final = {"metadata": {"session_id": session_id}}
    sync_selector: Final = _Selector(KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment))
    async_selector: Final = _Selector(KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment))
    bound_selector: Final = _BoundSelector()
    bound_sync: Final = resolve_pods_after_bound(
        bound_selector.sync,
        KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment),
    )
    bound_async: Final = async_resolve_pods_after_bound(
        bound_selector.async_call,
        KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment),
    )

    sync_result: Final = sync_selector.sync(_deployment(), request_kwargs)
    async_result: Final = await async_selector.async_call(_deployment(), request_kwargs)
    bound_sync_result: Final = bound_sync(_deployment(), request_kwargs)
    bound_async_result: Final = await bound_async(_deployment(), request_kwargs)

    assert (
        tuple(
            httpx.URL(_api_base(result)).host
            for result in (sync_result, async_result, bound_sync_result, bound_async_result)
        )
        == (_ranked_session_ips(session_id)[0],) * 4
    )


def test_direct_session_id_keyword_does_not_enable_session_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_sync_dns(monkeypatch, "10.0.0.1", "10.0.0.2", "10.0.0.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)
    request_kwargs: Final = {"session_id": "s1", "messages": [{"role": "user", "content": "hello"}]}

    selected_ips: Final = tuple(
        httpx.URL(_api_base(discovery.resolve_deployment(_deployment(), request_kwargs))).host for _ in range(3)
    )

    assert selected_ips == ("10.0.0.1", "10.0.0.2", "10.0.0.3")


@pytest.mark.parametrize(
    ("request_kwargs", "selection"),
    [
        ({}, "round_robin"),
        ({"metadata": {"session_id": "s1"}}, "session_affinity"),
        ({"metadata": {"session_id": "s1", "request_retry_count": 1}}, "session_affinity_retry"),
    ],
)
def test_resolved_deployment_records_pod_selection(
    monkeypatch: pytest.MonkeyPatch,
    request_kwargs: dict[str, object],
    selection: str,
) -> None:
    _stub_sync_dns(monkeypatch, "10.0.0.1", "10.0.0.2", "10.0.0.3")
    discovery: Final = KubernetesPodDiscovery(proxy_environment=_empty_proxy_environment)

    resolved: Final = discovery.resolve_deployment(_deployment(), request_kwargs)
    pod_ip: Final = httpx.URL(_api_base(resolved)).host

    assert resolved[KUBERNETES_POD_ROUTING_KEY] == {
        "service_host": httpx.URL(_SERVICE_URL).host,
        "pod_ip": pod_ip,
        "pod_count": 3,
        "selection": selection,
    }


@pytest.mark.parametrize("fallback", ["https", "no_proxy", "empty_ips", "dns_failure"])
def test_unresolved_deployment_does_not_record_pod_selection(
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
) -> None:
    if fallback == "empty_ips":
        _stub_sync_dns(monkeypatch)
    if fallback == "dns_failure":

        def failing_getaddrinfo(
            host: str | None,
            port: str | int | None,
            family: int = 0,
            type: int = 0,
            proto: int = 0,
            flags: int = 0,
        ) -> list[_AddrInfo]:
            raise socket.gaierror(socket.EAI_AGAIN, "temporary failure")

        monkeypatch.setattr(socket, "getaddrinfo", failing_getaddrinfo)
    if fallback == "no_proxy":
        _stub_sync_dns(monkeypatch, "10.0.0.1")

    proxy_environment: Final = (
        _proxy_environment("vllm-headless.ns.svc.cluster.local") if fallback == "no_proxy" else _empty_proxy_environment
    )
    api_base: Final = "https://vllm-headless.ns.svc.cluster.local:8000/v1" if fallback == "https" else _SERVICE_URL
    discovery: Final = KubernetesPodDiscovery(proxy_environment=proxy_environment)
    deployment: Final = _deployment(api_base=api_base)

    resolved: Final = discovery.resolve_deployment(deployment)

    assert resolved is deployment
    assert KUBERNETES_POD_ROUTING_KEY not in resolved
