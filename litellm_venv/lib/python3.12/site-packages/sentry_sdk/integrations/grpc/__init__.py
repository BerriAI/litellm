from functools import wraps

import grpc
from grpc import Channel, Server, intercept_channel
from grpc.aio import Channel as AsyncChannel
from grpc.aio import Server as AsyncServer

from sentry_sdk.integrations import Integration

from .client import ClientInterceptor
from .server import ServerInterceptor
from .aio.server import ServerInterceptor as AsyncServerInterceptor
from .aio.client import (
    SentryUnaryUnaryClientInterceptor as AsyncUnaryUnaryClientInterceptor,
)
from .aio.client import (
    SentryUnaryStreamClientInterceptor as AsyncUnaryStreamClientIntercetor,
)

from typing import TYPE_CHECKING, Any, Optional, Sequence

# Hack to get new Python features working in older versions
# without introducing a hard dependency on `typing_extensions`
# from: https://stackoverflow.com/a/71944042/300572
if TYPE_CHECKING:
    from typing import ParamSpec, Callable
else:
    # Fake ParamSpec
    class ParamSpec:
        def __init__(self, _):
            self.args = None
            self.kwargs = None

    # Callable[anything] will return None
    class _Callable:
        def __getitem__(self, _):
            return None

    # Make instances
    Callable = _Callable()

P = ParamSpec("P")


def _wrap_channel_sync(func: Callable[P, Channel]) -> Callable[P, Channel]:
    "Wrapper for synchronous secure and insecure channel."

    @wraps(func)
    def patched_channel(*args: Any, **kwargs: Any) -> Channel:
        channel = func(*args, **kwargs)
        if not ClientInterceptor._is_intercepted:
            ClientInterceptor._is_intercepted = True
            return intercept_channel(channel, ClientInterceptor())
        else:
            return channel

    return patched_channel


def _wrap_intercept_channel(func: Callable[P, Channel]) -> Callable[P, Channel]:
    @wraps(func)
    def patched_intercept_channel(
        channel: Channel, *interceptors: grpc.ServerInterceptor
    ) -> Channel:
        if ClientInterceptor._is_intercepted:
            interceptors = tuple(
                [
                    interceptor
                    for interceptor in interceptors
                    if not isinstance(interceptor, ClientInterceptor)
                ]
            )
        else:
            interceptors = interceptors
        return intercept_channel(channel, *interceptors)

    return patched_intercept_channel  # type: ignore


def _wrap_channel_async(func: Callable[P, AsyncChannel]) -> Callable[P, AsyncChannel]:
    "Wrapper for asynchronous secure and insecure channel."

    @wraps(func)
    def patched_channel(  # type: ignore
        *args: P.args,
        interceptors: Optional[Sequence[grpc.aio.ClientInterceptor]] = None,
        **kwargs: P.kwargs,
    ) -> Channel:
        sentry_interceptors = [
            AsyncUnaryUnaryClientInterceptor(),
            AsyncUnaryStreamClientIntercetor(),
        ]
        interceptors = [*sentry_interceptors, *(interceptors or [])]
        return func(*args, interceptors=interceptors, **kwargs)  # type: ignore

    return patched_channel  # type: ignore


def _wrap_sync_server(func: Callable[P, Server]) -> Callable[P, Server]:
    """Wrapper for synchronous server."""

    @wraps(func)
    def patched_server(  # type: ignore
        *args: P.args,
        interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None,
        **kwargs: P.kwargs,
    ) -> Server:
        interceptors = [
            interceptor
            for interceptor in interceptors or []
            if not isinstance(interceptor, ServerInterceptor)
        ]
        server_interceptor = ServerInterceptor()
        interceptors = [server_interceptor, *(interceptors or [])]
        return func(*args, interceptors=interceptors, **kwargs)  # type: ignore

    return patched_server  # type: ignore


def _wrap_async_server(func: Callable[P, AsyncServer]) -> Callable[P, AsyncServer]:
    """Wrapper for asynchronous server."""

    @wraps(func)
    def patched_aio_server(  # type: ignore
        *args: P.args,
        interceptors: Optional[Sequence[grpc.ServerInterceptor]] = None,
        **kwargs: P.kwargs,
    ) -> Server:
        server_interceptor = AsyncServerInterceptor()
        interceptors = (server_interceptor, *(interceptors or []))
        return func(*args, interceptors=interceptors, **kwargs)  # type: ignore

    return patched_aio_server  # type: ignore


class GRPCIntegration(Integration):
    identifier = "grpc"

    @staticmethod
    def setup_once() -> None:
        import grpc

        grpc.insecure_channel = _wrap_channel_sync(grpc.insecure_channel)
        grpc.secure_channel = _wrap_channel_sync(grpc.secure_channel)
        grpc.intercept_channel = _wrap_intercept_channel(grpc.intercept_channel)

        grpc.aio.insecure_channel = _wrap_channel_async(grpc.aio.insecure_channel)
        grpc.aio.secure_channel = _wrap_channel_async(grpc.aio.secure_channel)

        grpc.server = _wrap_sync_server(grpc.server)
        grpc.aio.server = _wrap_async_server(grpc.aio.server)
