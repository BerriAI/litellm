import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.grpc.consts import SPAN_ORIGIN
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_CUSTOM
from sentry_sdk.utils import event_from_exception

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any, Optional


try:
    import grpc
    from grpc import HandlerCallDetails, RpcMethodHandler
    from grpc.aio import AbortError, ServicerContext
except ImportError:
    raise DidNotEnable("grpcio is not installed")


class ServerInterceptor(grpc.aio.ServerInterceptor):  # type: ignore
    def __init__(self, find_name=None):
        # type: (ServerInterceptor, Callable[[ServicerContext], str] | None) -> None
        self._find_method_name = find_name or self._find_name

        super().__init__()

    async def intercept_service(self, continuation, handler_call_details):
        # type: (ServerInterceptor, Callable[[HandlerCallDetails], Awaitable[RpcMethodHandler]], HandlerCallDetails) -> Optional[Awaitable[RpcMethodHandler]]
        self._handler_call_details = handler_call_details
        handler = await continuation(handler_call_details)
        if handler is None:
            return None

        if not handler.request_streaming and not handler.response_streaming:
            handler_factory = grpc.unary_unary_rpc_method_handler

            async def wrapped(request, context):
                # type: (Any, ServicerContext) -> Any
                name = self._find_method_name(context)
                if not name:
                    return await handler(request, context)

                # What if the headers are empty?
                transaction = Transaction.continue_from_headers(
                    dict(context.invocation_metadata()),
                    op=OP.GRPC_SERVER,
                    name=name,
                    source=TRANSACTION_SOURCE_CUSTOM,
                    origin=SPAN_ORIGIN,
                )

                with sentry_sdk.start_transaction(transaction=transaction):
                    try:
                        return await handler.unary_unary(request, context)
                    except AbortError:
                        raise
                    except Exception as exc:
                        event, hint = event_from_exception(
                            exc,
                            mechanism={"type": "grpc", "handled": False},
                        )
                        sentry_sdk.capture_event(event, hint=hint)
                        raise

        elif not handler.request_streaming and handler.response_streaming:
            handler_factory = grpc.unary_stream_rpc_method_handler

            async def wrapped(request, context):  # type: ignore
                # type: (Any, ServicerContext) -> Any
                async for r in handler.unary_stream(request, context):
                    yield r

        elif handler.request_streaming and not handler.response_streaming:
            handler_factory = grpc.stream_unary_rpc_method_handler

            async def wrapped(request, context):
                # type: (Any, ServicerContext) -> Any
                response = handler.stream_unary(request, context)
                return await response

        elif handler.request_streaming and handler.response_streaming:
            handler_factory = grpc.stream_stream_rpc_method_handler

            async def wrapped(request, context):  # type: ignore
                # type: (Any, ServicerContext) -> Any
                async for r in handler.stream_stream(request, context):
                    yield r

        return handler_factory(
            wrapped,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _find_name(self, context):
        # type: (ServicerContext) -> str
        return self._handler_call_details.method
