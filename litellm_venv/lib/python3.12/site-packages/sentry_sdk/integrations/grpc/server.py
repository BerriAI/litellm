import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.grpc.consts import SPAN_ORIGIN
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_CUSTOM

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Optional
    from google.protobuf.message import Message

try:
    import grpc
    from grpc import ServicerContext, HandlerCallDetails, RpcMethodHandler
except ImportError:
    raise DidNotEnable("grpcio is not installed")


class ServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    def __init__(self, find_name=None):
        # type: (ServerInterceptor, Optional[Callable[[ServicerContext], str]]) -> None
        self._find_method_name = find_name or ServerInterceptor._find_name

        super().__init__()

    def intercept_service(self, continuation, handler_call_details):
        # type: (ServerInterceptor, Callable[[HandlerCallDetails], RpcMethodHandler], HandlerCallDetails) -> RpcMethodHandler
        handler = continuation(handler_call_details)
        if not handler or not handler.unary_unary:
            return handler

        def behavior(request, context):
            # type: (Message, ServicerContext) -> Message
            with sentry_sdk.isolation_scope():
                name = self._find_method_name(context)

                if name:
                    metadata = dict(context.invocation_metadata())

                    transaction = Transaction.continue_from_headers(
                        metadata,
                        op=OP.GRPC_SERVER,
                        name=name,
                        source=TRANSACTION_SOURCE_CUSTOM,
                        origin=SPAN_ORIGIN,
                    )

                    with sentry_sdk.start_transaction(transaction=transaction):
                        try:
                            return handler.unary_unary(request, context)
                        except BaseException as e:
                            raise e
                else:
                    return handler.unary_unary(request, context)

        return grpc.unary_unary_rpc_method_handler(
            behavior,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    @staticmethod
    def _find_name(context):
        # type: (ServicerContext) -> str
        return context._rpc_event.call_details.method.decode()
