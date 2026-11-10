import contextlib
import os
import signal
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.proxy.server_listener import LOOPBACK_HOST, AssignedPort, InheritedFd, PrivateListener
from litellm.rust_bridge.configuration import ENV_NAME, RustSwitch

SUPERVISED_ENV: Final = "_LITELLM_GATEWAY_SUPERVISED"
RUST_HOP: Final = LOOPBACK_HOST


class _Switch(BaseModel):
    model_config = ConfigDict(frozen=True)

    rust: RustSwitch = Field(None, validation_alias=ENV_NAME)


class Mode(Enum):
    DISABLED = auto()
    SUPERVISOR = auto()


@dataclass(frozen=True, slots=True)
class Supervised:
    listener: PrivateListener
    trusted_proxies: str


RustForwardMode: TypeAlias = Mode | Supervised
_SUPERVISED: Final = TypeAdapter(Supervised)


def supervised_environment(supervised: Supervised) -> Mapping[str, str]:
    return MappingProxyType({SUPERVISED_ENV: _SUPERVISED.dump_json(supervised).decode()})


def rust_forward_mode(environ: Mapping[str, str]) -> RustForwardMode:
    switch: Final = _Switch.model_validate(environ)
    supervised: Final = environ.get(SUPERVISED_ENV)
    if supervised is not None:
        return _SUPERVISED.validate_json(supervised)
    return Mode.SUPERVISOR if switch.rust else Mode.DISABLED


def _wait_for_eof(fd: int) -> None:
    with contextlib.suppress(OSError):
        while os.read(fd, 4096):
            pass


def _stop_when_closed(fd: int, stop: Callable[[], None]) -> None:
    _wait_for_eof(fd)
    stop()


def _terminate_self() -> None:
    signal.raise_signal(signal.SIGTERM)


def stop_with_supervisor(mode: RustForwardMode, fd: int = 0, stop: Callable[[], None] = _terminate_self) -> None:
    """Shut the supervised server down gracefully once the supervisor's end of its stdin closes, so it never outlives
    a supervisor that was killed. Call it only in the process the supervisor started, not in its workers."""
    match mode:
        case Supervised():
            threading.Thread(target=_stop_when_closed, args=(fd, stop), name="supervisor-stdin", daemon=True).start()
        case Mode.DISABLED | Mode.SUPERVISOR:
            pass
        case _:
            assert_never(mode)


def environment_error(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in detail['loc'])} is invalid: {detail['msg']}"
        for detail in error.errors(include_url=False)
    )


# A supervised server's TCP peer is always Rust on loopback, so server-level proxy headers would trust any
# client's forwarding headers. They stay off there and ``peer.wrap`` applies the operator's trust instead.


def _uvicorn_listener(listener: PrivateListener) -> Mapping[str, object]:
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
        case Mode.DISABLED | Mode.SUPERVISOR:
            return MappingProxyType({})
        case _:
            assert_never(mode)


def _uvicorn_listener_flags(listener: PrivateListener) -> tuple[str, ...]:
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
        case Mode.DISABLED | Mode.SUPERVISOR:
            return ()
        case _:
            assert_never(mode)


def private_listener(mode: RustForwardMode) -> PrivateListener | None:
    match mode:
        case Supervised(listener):
            return listener
        case Mode.DISABLED | Mode.SUPERVISOR:
            return None
        case _:
            assert_never(mode)
