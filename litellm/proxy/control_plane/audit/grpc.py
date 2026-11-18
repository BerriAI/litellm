from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import grpc
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceResponse
from opentelemetry.proto.collector.logs.v1.logs_service_pb2_grpc import (  # pyright: ignore[reportMissingTypeStubs]  # generated SDK stub
    LogsServiceStub,
)
from opentelemetry.sdk._logs.export import LogRecordExporter

if TYPE_CHECKING:
    from grpc import _CallFuture, _TResponse  # pyright: ignore[reportPrivateUsage]  # gRPC interceptor return contract

_RETRYABLE_CODES: Final = frozenset(
    {
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
        grpc.StatusCode.ABORTED,
        grpc.StatusCode.OUT_OF_RANGE,
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DATA_LOSS,
    }
)


@dataclass(slots=True)
class GrpcResponses(grpc.UnaryUnaryClientInterceptor):
    rejected: int = 0
    permanent_status: str | None = None

    def intercept_unary_unary(
        self,
        continuation: Callable[[grpc.ClientCallDetails, object], _CallFuture[_TResponse]],
        client_call_details: grpc.ClientCallDetails,
        request: object,
    ) -> _CallFuture[_TResponse]:
        call: Final = continuation(client_call_details, request)
        try:
            if isinstance(call, grpc.RpcError):
                raise call
            response: Final = call.result()
        except grpc.RpcError:
            code: Final = call.code()
            if code != grpc.StatusCode.OK and code not in _RETRYABLE_CODES:
                self.permanent_status = f"gRPC {code.name}"
            raise
        if isinstance(response, ExportLogsServiceResponse):
            self.rejected += response.partial_success.rejected_log_records
        return call


def capture_grpc_responses(exporter: LogRecordExporter) -> GrpcResponses | None:
    if not isinstance(exporter, OTLPLogExporter):
        return None
    if exporter._channel is None:  # pyright: ignore[reportPrivateUsage]  # initialized by the SDK constructor
        return None
    responses: Final = GrpcResponses()

    def stub(channel: grpc.Channel) -> LogsServiceStub:
        return LogsServiceStub(grpc.intercept_channel(channel, responses))

    exporter._stub = stub  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # rebind-ok: SDK exposes responses only through its stub
    exporter._client = stub(exporter._channel)  # pyright: ignore[reportPrivateUsage]  # rebind-ok: preserve the configured SDK channel and retries
    return responses
