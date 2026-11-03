from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final, TypeAlias

from typing_extensions import assert_never

from litellm.proxy.rust_forward.mode import RUST_HOP, Disabled, RustForwardMode, Supervised, Supervisor

if TYPE_CHECKING:
    from uvicorn._types import ASGIReceiveCallable, ASGISendCallable, Scope, WWWScope

PEER_HEADER: Final = b"x-litellm-rust-peer"

ASGIApp: TypeAlias = Callable[..., Awaitable[None]]


def _restore_peer(scope: "WWWScope") -> None:
    headers: Final = tuple(scope["headers"])
    peer: Final = next((value for name, value in headers if name == PEER_HEADER), None)
    if peer is None:
        return
    remaining: Final = tuple((name, value) for name, value in headers if name != PEER_HEADER)
    scope["headers"] = remaining  # rebind-ok: the peer header is private to the Rust hop
    client: Final = scope["client"]
    if client is None or client[0] != RUST_HOP:
        return
    host, _, port = peer.decode("latin-1").rpartition(":")
    scope["client"] = (host.strip("[]"), int(port))  # rebind-ok: uvicorn's access log reads this same scope


class RustPeerMiddleware:
    """Make Python see each request as if the client had connected to it directly.

    Rust owns the public port, so Python's peer is always Rust on loopback. Rust passes the real
    TCP peer in ``x-litellm-rust-peer`` and leaves every forwarding header untouched. This puts the
    real peer back into the scope, then runs uvicorn's own ``ProxyHeadersMiddleware`` with the trust
    settings Python would have used on the public port.
    """

    def __init__(self, app: ASGIApp, mode: Supervised) -> None:
        from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

        self.app: Final = ProxyHeadersMiddleware(app, mode.trusted_proxies) if mode.proxy_headers else app

    async def __call__(self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable") -> None:
        if scope["type"] != "lifespan":
            _restore_peer(scope)
        await self.app(scope, receive, send)


def wrap(app: ASGIApp, mode: RustForwardMode) -> ASGIApp:
    match mode:
        case Supervised():
            return RustPeerMiddleware(app, mode)
        case Disabled() | Supervisor():
            return app
        case _:
            assert_never(mode)
