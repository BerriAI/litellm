# pyright: reportMissingModuleSource=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # protobuf repeated fields lack precise runtime typing

import grpc

from litellm._logging import verbose_proxy_logger
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc
from litellm.python_extension_host.constants import PROTOCOL_MAJOR, PROTOCOL_MINOR, TOKEN_METADATA_KEY

from .config import ExtensionHostSettings
from .manifest import ExtensionManifest


@dataclass(frozen=True, slots=True)
class ExtensionHostHealth:
    healthy: bool
    reason: str | None = None


class PythonExtensionClient:
    def __init__(self, settings: ExtensionHostSettings, manifest: ExtensionManifest) -> None:
        self.settings: Final = settings
        self.manifest: Final = manifest
        target: Final = settings.endpoint.removeprefix("http://").removeprefix("https://")
        self._channel: Final = grpc.aio.insecure_channel(target)
        self._stub: Final = pb_grpc.PythonExtensionHostStub(self._channel)
        self._metadata: Final = ((TOKEN_METADATA_KEY, settings.token),)
        self._queue: Final[asyncio.Queue[pb.CallbackEvent]] = asyncio.Queue(settings.callback_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._recovery: asyncio.Task[None] | None = None
        self._recovery_lock: Final = asyncio.Lock()
        self._health = ExtensionHostHealth(False, "not connected")
        self._bypass_counts: dict[tuple[str, str], int] = {}  # mutable-ok: LiteLLM compatibility payload
        self._descriptor_hooks: dict[str, frozenset[str]] = {}  # mutable-ok: active revision descriptors
        self._closed = False

    @property
    def health(self) -> ExtensionHostHealth:
        return self._health

    @property
    def bypass_counts(
        self,
    ) -> dict[tuple[str, str], int]:  # mutable-ok: LiteLLM compatibility payload
        return dict(self._bypass_counts)  # mutable-ok: LiteLLM compatibility payload

    def descriptor_hooks(self, plugin_id: str) -> frozenset[str]:
        return self._descriptor_hooks.get(plugin_id, frozenset())

    async def start(self) -> tuple[pb.ExtensionDescriptor, ...]:
        descriptors: tuple[pb.ExtensionDescriptor, ...] = ()  # rebind-ok: invocation-scoped RPC state
        try:
            await asyncio.wait_for(self._channel.channel_ready(), self.settings.connect_timeout_seconds)
            descriptors = await self._activate()  # rebind-ok: invocation-scoped RPC state
        except TimeoutError:
            self._mark_unhealthy("connect timeout")
            self._schedule_recovery()
        except grpc.aio.AioRpcError as error:
            if error.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                self._mark_unhealthy(error.code().name)
                self._schedule_recovery()
            else:
                raise RuntimeError(f"extension host startup failed: {error.details()}") from error
        self._worker = asyncio.create_task(self._callback_worker(), name="python-extension-callbacks")
        return descriptors

    async def close(self) -> None:
        self._closed = True
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        if self._recovery is not None:
            self._recovery.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recovery
        await self._channel.close()

    async def retire(self, revision_id: str) -> None:
        try:
            await self._stub.RetireRevision(
                pb.RetireRevisionRequest(revision_id=revision_id),
                timeout=self.settings.hook_timeout_seconds,
                metadata=self._metadata,
            )
        except grpc.aio.AioRpcError:
            return

    async def execute_guardrail(self, invocation: pb.GuardrailInvocation) -> pb.GuardrailResult:
        try:
            result: Final = await self._stub.ExecuteGuardrail(
                invocation,
                timeout=self.settings.hook_timeout_seconds,
                metadata=self._metadata,
            )
        except (grpc.aio.AioRpcError, TimeoutError) as error:
            reason: Final = _rpc_reason(error)
            self._record_bypass(invocation.plugin_id, pb.HookPhase.Name(invocation.hook_phase), reason)
            self._schedule_recovery()
            return pb.GuardrailResult(
                operation=pb.OperationResult(ok=True),
                decision=pb.GUARDRAIL_DECISION_ALLOW,
            )
        else:
            if result.operation.ok:
                self._health = ExtensionHostHealth(True)
            else:
                self._record_bypass(
                    invocation.plugin_id,
                    pb.HookPhase.Name(invocation.hook_phase),
                    "extension_error",
                )
            return result

    def enqueue_callback(self, event: pb.CallbackEvent) -> bool:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._record_bypass(event.plugin_id, "callback", "queue_full")
            return False
        else:
            return True

    async def transform_stream(self, frames: AsyncIterator[pb.StreamFrame]) -> AsyncIterator[pb.StreamFrame]:
        call: Final = self._stub.TransformStream(frames, metadata=self._metadata)
        try:
            async for frame in call:
                yield frame
        except grpc.aio.AioRpcError as error:
            self._record_bypass("stream", "transform", error.code().name)
            self._schedule_recovery()
        finally:
            if not call.done():
                call.cancel()

    async def _activate(self) -> tuple[pb.ExtensionDescriptor, ...]:
        capabilities: Final = await self._stub.GetCapabilities(
            pb.GetCapabilitiesRequest(protocol_major=PROTOCOL_MAJOR, protocol_minor=PROTOCOL_MINOR),
            timeout=self.settings.connect_timeout_seconds,
            metadata=self._metadata,
        )
        if capabilities.protocol_major != PROTOCOL_MAJOR:
            raise RuntimeError(
                f"extension host protocol major {capabilities.protocol_major} does not match {PROTOCOL_MAJOR}"
            )
        has_callbacks: Final = any(spec.kind == pb.EXTENSION_KIND_CALLBACK for spec in self.manifest.specs)
        if has_callbacks and not capabilities.supports_callback_batching:
            raise RuntimeError("extension host does not support callback batching")
        if (
            capabilities.max_callback_batch_size
            and self.settings.callback_batch_size > capabilities.max_callback_batch_size
        ):
            raise RuntimeError("python_extension_host.callback_batch_size exceeds the host capability")
        if self.settings.gateway_listen is not None and not capabilities.supports_cache:
            raise RuntimeError("extension host was not configured for reverse cache access")
        response: pb.PrepareRevisionResponse = (  # rebind-ok: invocation-scoped RPC state
            await self._stub.PrepareRevision(  # rebind-ok: invocation-scoped RPC state
                pb.PrepareRevisionRequest(
                    revision_id=self.manifest.revision_id,
                    extensions=self.manifest.specs,
                ),
                timeout=self.settings.hook_timeout_seconds,
                metadata=self._metadata,
            )
        )
        if not response.operation.ok and response.operation.error_code != pb.ERROR_CODE_ALREADY_EXISTS:
            raise RuntimeError(f"extension manifest rejected: {response.operation.error_message}")
        streaming_hooks: Final = {  # mutable-ok: LiteLLM compatibility payload
            "async_post_call_streaming_hook",
            "async_post_call_streaming_iterator_hook",
        }
        if not capabilities.supports_duplex_streaming and any(
            streaming_hooks.intersection(cast(Iterable[str], descriptor.hooks))  # cast-ok: validated protobuf boundary
            for descriptor in cast(  # cast-ok: validated protobuf boundary
                Iterable[pb.ExtensionDescriptor], response.extensions
            )  # cast-ok: validated protobuf boundary
        ):
            raise RuntimeError("extension host does not support required duplex streaming hooks")
        commit: Final = await self._stub.CommitRevision(
            pb.CommitRevisionRequest(revision_id=self.manifest.revision_id),
            timeout=self.settings.hook_timeout_seconds,
            metadata=self._metadata,
        )
        if not commit.ok:
            raise RuntimeError(f"extension manifest commit failed: {commit.error_message}")
        self._health = ExtensionHostHealth(True)
        extensions: Final = cast(  # cast-ok: validated protobuf boundary
            Iterable[pb.ExtensionDescriptor], response.extensions
        )  # cast-ok: validated protobuf boundary
        descriptors: Final = tuple(extensions)
        self._descriptor_hooks = {  # mutable-ok: active revision descriptors
            descriptor.id: frozenset(descriptor.hooks) for descriptor in descriptors
        }
        return descriptors

    async def _callback_worker(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]  # mutable-ok: LiteLLM compatibility payload
            await asyncio.sleep(0.01)
            while len(batch) < self.settings.callback_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                response = await self._stub.PublishCallbackEvents(  # rebind-ok: callback batch response
                    pb.PublishCallbackEventsRequest(events=batch),
                    timeout=self.settings.hook_timeout_seconds,
                    metadata=self._metadata,
                )
            except (grpc.aio.AioRpcError, TimeoutError) as error:
                reason = _rpc_reason(error)
                for event in batch:
                    self._record_bypass(event.plugin_id, "callback", reason)
                self._schedule_recovery()
            else:
                operations = tuple(  # cast-ok: validated protobuf repeated field # rebind-ok: callback batch operations
                    cast(Iterable[pb.OperationResult], response.operations)  # cast-ok: protobuf repeated field
                )
                all_ok = len(operations) == len(batch)
                for index, event in enumerate(batch):
                    if index >= len(operations):
                        self._record_bypass(event.plugin_id, "callback", "missing_operation")
                        continue
                    operation = operations[index]  # rebind-ok: each callback has one operation
                    if not operation.ok:
                        all_ok = False
                        self._record_bypass(
                            event.plugin_id,
                            "callback",
                            _operation_reason(operation),
                        )
                if all_ok:
                    self._health = ExtensionHostHealth(True)
            finally:
                for _ in batch:
                    self._queue.task_done()

    def _schedule_recovery(self) -> None:
        if self._closed or (self._recovery is not None and not self._recovery.done()):
            return
        self._recovery = asyncio.create_task(self._recover(), name="python-extension-recovery")

    async def _recover(self) -> None:
        async with self._recovery_lock:
            delay = 0.25  # rebind-ok: invocation-scoped RPC state
            while not self._closed:
                try:
                    await asyncio.wait_for(self._channel.channel_ready(), self.settings.connect_timeout_seconds)
                    await self._activate()
                except (grpc.aio.AioRpcError, RuntimeError, TimeoutError) as error:
                    self._mark_unhealthy(_rpc_reason(error))
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5.0)
                else:
                    return

    def _record_bypass(self, plugin_id: str, hook: str, reason: str) -> None:
        key: Final = (plugin_id, f"{hook}:{reason}")
        self._bypass_counts[key] = self._bypass_counts.get(key, 0) + 1
        self._mark_unhealthy(reason)
        verbose_proxy_logger.warning(
            "python extension host bypass plugin=%s hook=%s reason=%s", plugin_id, hook, reason
        )

    def _mark_unhealthy(self, reason: str) -> None:
        self._health = ExtensionHostHealth(False, reason)


def _rpc_reason(error: BaseException) -> str:
    return error.code().name if isinstance(error, grpc.aio.AioRpcError) else str(error)


def _operation_reason(operation: pb.OperationResult) -> str:
    return operation.error_message or "extension_error"
