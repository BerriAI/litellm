from typing import (
    TYPE_CHECKING,
    Final,
    TypeAlias,
    cast,  # noqa: TID251  # the one Starlette-to-uvicorn ASGI conversion in served
)

from typing_extensions import assert_never

from litellm.proxy.rust_forward.mode import RUST_HOP, Mode, RustForwardMode, Supervised

if TYPE_CHECKING:
    import starlette.types
    from uvicorn._types import ASGI3Application, ASGIReceiveCallable, ASGISendCallable, Scope, WWWScope

PEER_HEADER: Final = b"x-litellm-rust-peer"

ASGIApp: TypeAlias = "ASGI3Application"


def served(app: "starlette.types.ASGIApp") -> ASGIApp:
    return cast(  # cast-ok: Starlette and uvicorn type the same ASGI 3 callable with unassignable scope/receive/send
        "ASGI3Application", app
    )


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

        self.app: Final = ProxyHeadersMiddleware(app, mode.trusted_proxies)

    async def __call__(self, scope: "Scope", receive: "ASGIReceiveCallable", send: "ASGISendCallable") -> None:
        if scope["type"] != "lifespan":
            _restore_peer(scope)
        await self.app(scope, receive, send)


def wrap(app: "starlette.types.ASGIApp", mode: RustForwardMode) -> ASGIApp:
    converted: Final = served(app)
    match mode:
        case Supervised():
            return RustPeerMiddleware(converted, mode)
        case Mode.DISABLED | Mode.SUPERVISOR:
            return converted
        case _:
            assert_never(mode)
