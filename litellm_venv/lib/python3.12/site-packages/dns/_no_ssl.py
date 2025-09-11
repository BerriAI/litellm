import enum
from typing import Any

CERT_NONE = 0


class TLSVersion(enum.IntEnum):
    TLSv1_2 = 12


class WantReadException(Exception):
    pass


class WantWriteException(Exception):
    pass


class SSLWantReadError(Exception):
    pass


class SSLWantWriteError(Exception):
    pass


class SSLContext:
    def __init__(self) -> None:
        self.minimum_version: Any = TLSVersion.TLSv1_2
        self.check_hostname: bool = False
        self.verify_mode: int = CERT_NONE

    def wrap_socket(self, *args, **kwargs) -> "SSLSocket":  # type: ignore
        raise Exception("no ssl support")  # pylint: disable=broad-exception-raised

    def set_alpn_protocols(self, *args, **kwargs):  # type: ignore
        raise Exception("no ssl support")  # pylint: disable=broad-exception-raised


class SSLSocket:
    def pending(self) -> bool:
        raise Exception("no ssl support")  # pylint: disable=broad-exception-raised

    def do_handshake(self) -> None:
        raise Exception("no ssl support")  # pylint: disable=broad-exception-raised

    def settimeout(self, value: Any) -> None:
        pass

    def getpeercert(self) -> Any:
        raise Exception("no ssl support")  # pylint: disable=broad-exception-raised

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def create_default_context(*args, **kwargs) -> SSLContext:  # type: ignore
    raise Exception("no ssl support")  # pylint: disable=broad-exception-raised
