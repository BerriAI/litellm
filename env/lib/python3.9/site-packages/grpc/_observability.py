# Copyright 2023 The gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import contextlib
import logging
import threading
from typing import Any, Generator, Generic, List, Optional, TypeVar

from grpc._cython import cygrpc as _cygrpc

_LOGGER = logging.getLogger(__name__)

_channel = Any  # _channel.py imports this module.
ClientCallTracerCapsule = TypeVar("ClientCallTracerCapsule")
ServerCallTracerFactoryCapsule = TypeVar("ServerCallTracerFactoryCapsule")

_plugin_lock: threading.RLock = threading.RLock()
_OBSERVABILITY_PLUGIN: Optional["ObservabilityPlugin"] = None
_SERVICES_TO_EXCLUDE: List[bytes] = [
    b"google.monitoring.v3.MetricService",
    b"google.devtools.cloudtrace.v2.TraceService",
]


class ObservabilityPlugin(
    Generic[ClientCallTracerCapsule, ServerCallTracerFactoryCapsule],
    metaclass=abc.ABCMeta,
):
    """Abstract base class for observability plugin.

    *This is a semi-private class that was intended for the exclusive use of
     the gRPC team.*

    The ClientCallTracerCapsule and ClientCallTracerCapsule created by this
    plugin should be inject to gRPC core using observability_init at the
    start of a program, before any channels/servers are built.

    Any future methods added to this interface cannot have the
    @abc.abstractmethod annotation.

    Attributes:
      _stats_enabled: A bool indicates whether tracing is enabled.
      _tracing_enabled: A bool indicates whether stats(metrics) is enabled.
    """

    _tracing_enabled: bool = False
    _stats_enabled: bool = False

    @abc.abstractmethod
    def create_client_call_tracer(
        self, method_name: bytes
    ) -> ClientCallTracerCapsule:
        """Creates a ClientCallTracerCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called after a call was created, the ClientCallTracer created
        by this method will be saved to call context.

        The ClientCallTracer is an object which implements `grpc_core::ClientCallTracer`
        interface and wrapped in a PyCapsule using `client_call_tracer` as name.

        Args:
        method_name: The method name of the call in byte format.

        Returns:
        A PyCapsule which stores a ClientCallTracer object.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_client_call_tracer(
        self, client_call_tracer: ClientCallTracerCapsule
    ) -> None:
        """Deletes the ClientCallTracer stored in ClientCallTracerCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called at the end of the call to destroy the ClientCallTracer.

        The ClientCallTracer is an object which implements `grpc_core::ClientCallTracer`
        interface and wrapped in a PyCapsule using `client_call_tracer` as name.

        Args:
        client_call_tracer: A PyCapsule which stores a ClientCallTracer object.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_trace_context(
        self, trace_id: str, span_id: str, is_sampled: bool
    ) -> None:
        """Saves the trace_id and span_id related to the current span.

        After register the plugin, if tracing is enabled, this method will be
        called after the server finished sending response.

        This method can be used to propagate census context.

        Args:
        trace_id: The identifier for the trace associated with the span as a
         32-character hexadecimal encoded string,
         e.g. 26ed0036f2eff2b7317bccce3e28d01f
        span_id: The identifier for the span as a 16-character hexadecimal encoded
         string. e.g. 113ec879e62583bc
        is_sampled: A bool indicates whether the span is sampled.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def create_server_call_tracer_factory(
        self,
    ) -> ServerCallTracerFactoryCapsule:
        """Creates a ServerCallTracerFactoryCapsule.

        After register the plugin, if tracing or stats is enabled, this method
        will be called by calling observability_init, the ServerCallTracerFactory
        created by this method will be registered to gRPC core.

        The ServerCallTracerFactory is an object which implements
        `grpc_core::ServerCallTracerFactory` interface and wrapped in a PyCapsule
        using `server_call_tracer_factory` as name.

        Returns:
        A PyCapsule which stores a ServerCallTracerFactory object.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def record_rpc_latency(
        self, method: str, rpc_latency: float, status_code: Any
    ) -> None:
        """Record the latency of the RPC.

        After register the plugin, if stats is enabled, this method will be
        called at the end of each RPC.

        Args:
        method: The fully-qualified name of the RPC method being invoked.
        rpc_latency: The latency for the RPC, equals to the time between
         when the client invokes the RPC and when the client receives the status.
        status_code: An element of grpc.StatusCode in string format representing the
         final status for the RPC.
        """
        raise NotImplementedError()

    def set_tracing(self, enable: bool) -> None:
        """Enable or disable tracing.

        Args:
        enable: A bool indicates whether tracing should be enabled.
        """
        self._tracing_enabled = enable

    def set_stats(self, enable: bool) -> None:
        """Enable or disable stats(metrics).

        Args:
        enable: A bool indicates whether stats should be enabled.
        """
        self._stats_enabled = enable

    @property
    def tracing_enabled(self) -> bool:
        return self._tracing_enabled

    @property
    def stats_enabled(self) -> bool:
        return self._stats_enabled

    @property
    def observability_enabled(self) -> bool:
        return self.tracing_enabled or self.stats_enabled


@contextlib.contextmanager
def get_plugin() -> Generator[Optional[ObservabilityPlugin], None, None]:
    """Get the ObservabilityPlugin in _observability module.

    Returns:
    The ObservabilityPlugin currently registered with the _observability
    module. Or None if no plugin exists at the time of calling this method.
    """
    with _plugin_lock:
        yield _OBSERVABILITY_PLUGIN


def set_plugin(observability_plugin: Optional[ObservabilityPlugin]) -> None:
    """Save ObservabilityPlugin to _observability module.

    Args:
    observability_plugin: The ObservabilityPlugin to save.

    Raises:
      ValueError: If an ObservabilityPlugin was already registered at the
      time of calling this method.
    """
    global _OBSERVABILITY_PLUGIN  # pylint: disable=global-statement
    with _plugin_lock:
        if observability_plugin and _OBSERVABILITY_PLUGIN:
            raise ValueError("observability_plugin was already set!")
        _OBSERVABILITY_PLUGIN = observability_plugin


def observability_init(observability_plugin: ObservabilityPlugin) -> None:
    """Initialize observability with provided ObservabilityPlugin.

    This method have to be called at the start of a program, before any
    channels/servers are built.

    Args:
    observability_plugin: The ObservabilityPlugin to use.

    Raises:
      ValueError: If an ObservabilityPlugin was already registered at the
      time of calling this method.
    """
    set_plugin(observability_plugin)
    try:
        _cygrpc.set_server_call_tracer_factory(observability_plugin)
    except Exception:  # pylint:disable=broad-except
        _LOGGER.exception("Failed to set server call tracer factory!")


def observability_deinit() -> None:
    """Clear the observability context, including ObservabilityPlugin and
    ServerCallTracerFactory

    This method have to be called after exit observability context so that
    it's possible to re-initialize again.
    """
    set_plugin(None)
    _cygrpc.clear_server_call_tracer_factory()


def delete_call_tracer(client_call_tracer_capsule: Any) -> None:
    """Deletes the ClientCallTracer stored in ClientCallTracerCapsule.

    This method will be called at the end of the call to destroy the ClientCallTracer.

    The ClientCallTracer is an object which implements `grpc_core::ClientCallTracer`
    interface and wrapped in a PyCapsule using `client_call_tracer` as the name.

    Args:
    client_call_tracer_capsule: A PyCapsule which stores a ClientCallTracer object.
    """
    with get_plugin() as plugin:
        if not (plugin and plugin.observability_enabled):
            return
        plugin.delete_client_call_tracer(client_call_tracer_capsule)


def maybe_record_rpc_latency(state: "_channel._RPCState") -> None:
    """Record the latency of the RPC, if the plugin is registered and stats is enabled.

    This method will be called at the end of each RPC.

    Args:
    state: a grpc._channel._RPCState object which contains the stats related to the
    RPC.
    """
    # TODO(xuanwn): use channel args to exclude those metrics.
    for exclude_prefix in _SERVICES_TO_EXCLUDE:
        if exclude_prefix in state.method.encode("utf8"):
            return
    with get_plugin() as plugin:
        if not (plugin and plugin.stats_enabled):
            return
        rpc_latency_s = state.rpc_end_time - state.rpc_start_time
        rpc_latency_ms = rpc_latency_s * 1000
        plugin.record_rpc_latency(state.method, rpc_latency_ms, state.code)
