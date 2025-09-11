import os
import socket


_hostname = os.getenv("DD_HOSTNAME", "")  # type: str


def get_hostname():
    # type: () -> str
    global _hostname
    if not _hostname:
        _hostname = socket.gethostname()
    return _hostname


def _reset():
    global _hostname
    _hostname = ""
