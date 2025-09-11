from typing import Callable, Union, AsyncIterable, Any

from grpc.aio import (
    UnaryUnaryClientInterceptor,
    UnaryStreamClientInterceptor,
    ClientCallDetails,
    UnaryUnaryCall,
    UnaryStreamCall,
    Metadata,
)
from google.protobuf.message import Message

import sentry_sdk
from sentry_sdk.consts import OP
from sentry_sdk.integrations.grpc.consts import SPAN_ORIGIN


class ClientInterceptor:
    @staticmethod
    def _update_client_call_details_metadata_from_scope(
        client_call_details: ClientCallDetails,
    ) -> ClientCallDetails:
        if client_call_details.metadata is None:
            client_call_details = client_call_details._replace(metadata=Metadata())
        elif not isinstance(client_call_details.metadata, Metadata):
            # This is a workaround for a GRPC bug, which was fixed in grpcio v1.60.0
            # See https://github.com/grpc/grpc/issues/34298.
            client_call_details = client_call_details._replace(
                metadata=Metadata.from_tuple(client_call_details.metadata)
            )
        for (
            key,
            value,
        ) in sentry_sdk.get_current_scope().iter_trace_propagation_headers():
            client_call_details.metadata.add(key, value)
        return client_call_details


class SentryUnaryUnaryClientInterceptor(ClientInterceptor, UnaryUnaryClientInterceptor):  # type: ignore
    async def intercept_unary_unary(
        self,
        continuation: Callable[[ClientCallDetails, Message], UnaryUnaryCall],
        client_call_details: ClientCallDetails,
        request: Message,
    ) -> Union[UnaryUnaryCall, Message]:
        method = client_call_details.method

        with sentry_sdk.start_span(
            op=OP.GRPC_CLIENT,
            name="unary unary call to %s" % method.decode(),
            origin=SPAN_ORIGIN,
        ) as span:
            span.set_data("type", "unary unary")
            span.set_data("method", method)

            client_call_details = self._update_client_call_details_metadata_from_scope(
                client_call_details
            )

            response = await continuation(client_call_details, request)
            status_code = await response.code()
            span.set_data("code", status_code.name)

            return response


class SentryUnaryStreamClientInterceptor(
    ClientInterceptor, UnaryStreamClientInterceptor  # type: ignore
):
    async def intercept_unary_stream(
        self,
        continuation: Callable[[ClientCallDetails, Message], UnaryStreamCall],
        client_call_details: ClientCallDetails,
        request: Message,
    ) -> Union[AsyncIterable[Any], UnaryStreamCall]:
        method = client_call_details.method

        with sentry_sdk.start_span(
            op=OP.GRPC_CLIENT,
            name="unary stream call to %s" % method.decode(),
            origin=SPAN_ORIGIN,
        ) as span:
            span.set_data("type", "unary stream")
            span.set_data("method", method)

            client_call_details = self._update_client_call_details_metadata_from_scope(
                client_call_details
            )

            response = await continuation(client_call_details, request)
            # status_code = await response.code()
            # span.set_data("code", status_code)

            return response
