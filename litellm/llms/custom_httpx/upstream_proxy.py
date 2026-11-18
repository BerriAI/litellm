import ipaddress
import re
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx

UPSTREAM_PROXY_PARAM: Final = "upstream_proxy"
_ENV_REFERENCE_PREFIX: Final = "os.environ/"
_SCHEME_SEPARATOR: Final = "://"
_SUPPORTED_SCHEMES: Final = frozenset(("http", "https"))
_PROXY_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_HOSTNAME: Final = re.compile(
    r"(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)
UpstreamProxyProtocol: TypeAlias = Literal["http", "https"]
_UPSTREAM_PROXY: Final[ContextVar[str | None]] = ContextVar("litellm_upstream_proxy", default=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedUpstreamProxy:
    name: str
    protocol: UpstreamProxyProtocol
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    description: str | None = None

    def connect_url(self) -> str:
        return str(
            httpx.URL(
                scheme=self.protocol,
                host=self.host,
                port=self.port,
                username=self.username or "",
                password=(self.password or "") if self.username else "",
            )
        )


class UpstreamProxyRegistry:
    def __init__(self) -> None:
        self._by_name: Mapping[str, NamedUpstreamProxy] = MappingProxyType({})

    def replace(self, proxies: Iterable[NamedUpstreamProxy]) -> None:
        self._by_name = MappingProxyType({proxy.name: proxy for proxy in proxies})

    def get(self, name: str) -> NamedUpstreamProxy | None:
        return self._by_name.get(name)

    def all(self) -> tuple[NamedUpstreamProxy, ...]:
        return tuple(self._by_name.values())


UPSTREAM_PROXY_REGISTRY: Final = UpstreamProxyRegistry()


def parse_upstream_proxy_url(value: str) -> str:
    try:
        url: Final = httpx.URL(value.strip())
    except httpx.InvalidURL as e:
        raise ValueError(f"{UPSTREAM_PROXY_PARAM} is not a valid URL: {e}") from e
    if url.scheme not in _SUPPORTED_SCHEMES or not url.host:
        raise ValueError(
            f"{UPSTREAM_PROXY_PARAM} must be an http:// or https:// proxy URL with a host, got scheme {url.scheme!r}"
        )
    if url.userinfo:
        raise ValueError(
            f"{UPSTREAM_PROXY_PARAM} cannot carry proxy credentials in the URL, because a model echoes the value back "
            "in /model/info. Add the proxy under Upstream Proxies with a username and password and reference it by name"
        )
    return str(url)


def validate_upstream_proxy_name(value: str) -> str:
    if _PROXY_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{value!r} is not a valid upstream proxy name: use letters, digits, '.', '_' or '-', at most 64 characters"
        )
    return value


def validate_upstream_proxy_host(value: str) -> str:
    host: Final = value.strip().removeprefix("[").removesuffix("]")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if _HOSTNAME.fullmatch(host) is None:
        raise ValueError(f"{value!r} is not a valid proxy host: use a hostname, IPv4 or IPv6 address")
    return host.lower()


def validate_configured_upstream_proxy(value: str | None) -> str | None:
    if value is None or value.startswith(_ENV_REFERENCE_PREFIX):
        return value
    if _SCHEME_SEPARATOR in value:
        return parse_upstream_proxy_url(value)
    return validate_upstream_proxy_name(value)


def resolve_upstream_proxy(configured: str, registry: UpstreamProxyRegistry) -> str:
    if _SCHEME_SEPARATOR in configured:
        return parse_upstream_proxy_url(configured)
    named: Final = registry.get(configured)
    if named is None:
        raise ValueError(f"{UPSTREAM_PROXY_PARAM}={configured!r} names no proxy configured under Upstream Proxies")
    return named.connect_url()


def current_upstream_proxy() -> str | None:
    return _UPSTREAM_PROXY.get()


@contextmanager
def upstream_proxy_scope(
    configured: object,
    transport_honors_it: Callable[[], bool],
    registry: UpstreamProxyRegistry = UPSTREAM_PROXY_REGISTRY,
) -> Generator[None]:
    """Routes the upstream calls made inside the scope through `configured`; unset keeps the enclosing choice."""
    if configured is None or configured == "":
        yield
        return
    if not isinstance(configured, str):
        raise ValueError(f"{UPSTREAM_PROXY_PARAM} must be a string, got {type(configured).__name__}")
    if not transport_honors_it():
        raise ValueError(
            f"{UPSTREAM_PROXY_PARAM} only applies to async calls on the aiohttp transport, which this call does not use"
        )
    token: Final = _UPSTREAM_PROXY.set(resolve_upstream_proxy(configured, registry))
    try:
        yield
    finally:
        _UPSTREAM_PROXY.reset(token)
