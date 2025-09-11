import socket
from typing import Any  # noqa:F401

from .compat import httplib
from .http import HTTPConnectionMixin


class UDSHTTPConnection(HTTPConnectionMixin, httplib.HTTPConnection):
    """An HTTP connection established over a Unix Domain Socket."""

    # It's "important" to keep the hostname and port arguments here; while there are not used by the connection
    # mechanism, they are actually used as HTTP headers such as `Host`.
    def __init__(
        self,
        path,  # type: str
        *args,  # type: Any
        **kwargs,  # type: Any
    ):
        # type: (...) -> None
        super(UDSHTTPConnection, self).__init__(*args, **kwargs)
        self.path = path

    def connect(self):
        # type: () -> None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.path)
        self.sock = sock
