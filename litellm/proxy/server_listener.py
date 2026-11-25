from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

from typing_extensions import assert_never

LOOPBACK_HOST: Final = "127.0.0.1"
NO_TRUSTED_PROXIES: Final = ""


@dataclass(frozen=True, slots=True)
class InheritedFd:
    fd: int


@dataclass(frozen=True, slots=True)
class AssignedPort:
    port: int


PrivateListener: TypeAlias = InheritedFd | AssignedPort


def _gunicorn_bind(listener: PrivateListener) -> str:
    match listener:
        case InheritedFd(fd):
            return f"fd://{fd}"
        case AssignedPort(port):
            return f"{LOOPBACK_HOST}:{port}"
        case _:
            assert_never(listener)


def gunicorn_overrides(listener: PrivateListener | None) -> Mapping[str, str]:
    if listener is None:
        return MappingProxyType({})
    return MappingProxyType({"bind": _gunicorn_bind(listener), "forwarded_allow_ips": NO_TRUSTED_PROXIES})
