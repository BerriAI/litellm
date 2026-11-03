from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import assert_never

from litellm.rust_bridge.configuration import ENV_NAME

GATEWAY_FD: Final = "_LITELLM_GATEWAY_FD"
GATEWAY_PORT: Final = "_LITELLM_GATEWAY_PORT"
PROXY_HEADERS: Final = "_LITELLM_GATEWAY_PROXY_HEADERS"
TRUSTED_PROXIES: Final = "FORWARDED_ALLOW_IPS"
DEFAULT_TRUSTED_PROXIES: Final = "127.0.0.1"
RUST_HOP: Final = "127.0.0.1"


def _blank_as_unset(value: object) -> object:
    stripped: Final = value.strip() if isinstance(value, str) else value
    return None if stripped == "" else stripped


_Switch: TypeAlias = Annotated[bool | None, BeforeValidator(_blank_as_unset)]


class _ForwardEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True)

    rust: _Switch = Field(None, validation_alias=ENV_NAME)
    listener_fd: int | None = Field(None, validation_alias=GATEWAY_FD)
    listener_port: int | None = Field(None, validation_alias=GATEWAY_PORT)
    proxy_headers: bool = Field(True, validation_alias=PROXY_HEADERS)
    trusted_proxies: str = Field(DEFAULT_TRUSTED_PROXIES, validation_alias=TRUSTED_PROXIES)


@dataclass(frozen=True, slots=True)
class Disabled:
    pass


@dataclass(frozen=True, slots=True)
class Supervisor:
    pass


@dataclass(frozen=True, slots=True)
class InheritedFd:
    fd: int


@dataclass(frozen=True, slots=True)
class AssignedPort:
    port: int


ChildListener: TypeAlias = InheritedFd | AssignedPort


@dataclass(frozen=True, slots=True)
class Supervised:
    listener: ChildListener
    proxy_headers: bool
    trusted_proxies: str


RustForwardMode: TypeAlias = Disabled | Supervisor | Supervised


def _listener_environment(listener: ChildListener) -> tuple[str, str]:
    match listener:
        case InheritedFd(fd):
            return GATEWAY_FD, str(fd)
        case AssignedPort(port):
            return GATEWAY_PORT, str(port)
        case _:
            assert_never(listener)


def supervised_environment(
    listener: ChildListener, proxy_headers: bool, trusted_proxies: str | None
) -> Mapping[str, str]:
    name, value = _listener_environment(listener)
    return MappingProxyType(
        {
            name: value,
            PROXY_HEADERS: str(proxy_headers).lower(),
            **({} if trusted_proxies is None else {TRUSTED_PROXIES: trusted_proxies}),
        }
    )


def rust_forward_mode(environ: Mapping[str, str]) -> RustForwardMode:
    parsed: Final = _ForwardEnvironment.model_validate(environ)
    if parsed.listener_fd is not None:
        return Supervised(InheritedFd(parsed.listener_fd), parsed.proxy_headers, parsed.trusted_proxies)
    if parsed.listener_port is not None:
        return Supervised(AssignedPort(parsed.listener_port), parsed.proxy_headers, parsed.trusted_proxies)
    if parsed.rust:
        return Supervisor()
    return Disabled()


# A supervised server's TCP peer is always Rust on loopback, so server-level proxy headers would trust any
# client's forwarding headers. They stay off there and ``peer.wrap`` applies the operator's trust instead.


def _uvicorn_listener(listener: ChildListener) -> Mapping[str, object]:
    match listener:
        case InheritedFd(fd):
            return MappingProxyType({"fd": fd})
        case AssignedPort(port):
            return MappingProxyType({"host": RUST_HOP, "port": port})
        case _:
            assert_never(listener)


def uvicorn_options(mode: RustForwardMode, factory: str) -> Mapping[str, object]:
    match mode:
        case Supervised(listener):
            return MappingProxyType(
                {"app": factory, "factory": True, **_uvicorn_listener(listener), "proxy_headers": False}
            )
        case Disabled() | Supervisor():
            return MappingProxyType({})
        case _:
            assert_never(mode)


def _uvicorn_listener_flags(listener: ChildListener) -> tuple[str, ...]:
    match listener:
        case InheritedFd(fd):
            return ("--fd", str(fd))
        case AssignedPort(port):
            return ("--host", RUST_HOP, "--port", str(port))
        case _:
            assert_never(listener)


def uvicorn_flags(mode: RustForwardMode) -> tuple[str, ...]:
    match mode:
        case Supervised(listener):
            return (*_uvicorn_listener_flags(listener), "--no-proxy-headers")
        case Disabled() | Supervisor():
            return ()
        case _:
            assert_never(mode)


def _gunicorn_listener(listener: ChildListener) -> str:
    match listener:
        case InheritedFd(fd):
            return f"fd://{fd}"
        case AssignedPort(port):
            return f"{RUST_HOP}:{port}"
        case _:
            assert_never(listener)


def gunicorn_bind(mode: RustForwardMode, public: str) -> str:
    match mode:
        case Supervised(listener):
            return _gunicorn_listener(listener)
        case Disabled() | Supervisor():
            return public
        case _:
            assert_never(mode)


def server_proxy_headers(mode: RustForwardMode) -> bool:
    match mode:
        case Supervised():
            return False
        case Disabled() | Supervisor():
            return True
        case _:
            assert_never(mode)
