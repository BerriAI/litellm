from typing import TYPE_CHECKING

from aiohttp import TCPConnector

if TYPE_CHECKING:
    from aiohttp.client_reqrep import ConnectionKey
    from aiohttp.connector import Connection
    from aiohttp.tracing import Trace


class HardenedTCPConnector(TCPConnector):
    """
    TCPConnector that refuses to hand out keepalive connections which have been
    flagged for closing while sitting idle in the pool.

    aiohttp's BaseConnector._get only checks that a pooled connection is still
    connected and within the keepalive window; it does not re-check the
    protocol's should_close flag. A connection is only pooled while should_close
    is False, but the flag can flip to True afterwards - most notably via the
    aiohttp 3.14.x regression (aio-libs/aiohttp#12953) where a stray sock_read
    timer fires on an already-released connection, stamps a SocketTimeoutError on
    the ResponseHandler and sets should_close without closing the transport. The
    next request then reuses the poisoned connection and fails instantly with a
    sub-millisecond "Connection timed out", spanning every provider that shares
    the pool.

    Re-checking should_close at acquisition time drops such connections and keeps
    pulling until a clean one (or none) is found, independent of the aiohttp
    version in use.
    """

    async def _get(self, key: "ConnectionKey", traces: "list[Trace]") -> "Connection | None":
        conn = await super()._get(key, traces)
        if conn is None:
            return None
        protocol = conn.protocol
        if protocol is not None and protocol.should_close:
            conn.close()
            return await self._get(key, traces)
        return conn
