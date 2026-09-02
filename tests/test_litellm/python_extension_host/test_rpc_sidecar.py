from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import grpc
import pytest
import pytest_asyncio

from litellm.exceptions import GuardrailRaisedException
from litellm.extensions.cache_gateway import GatewayServices, InvocationCacheRegistry
from litellm.extensions.client import PythonExtensionClient
from litellm.extensions.config import ExtensionHostSettings, settings_from_config
from litellm.extensions.manifest import build_manifest
from litellm.extensions.runtime import configure_extension_runtime
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc
from litellm.python_extension_host.constants import TOKEN_METADATA_KEY
from litellm.python_extension_host.service import PythonExtensionHostService

MODULE = "tests.test_litellm.python_extension_host.test_rpc_sidecar"
TOKEN = "test-extension-token"
METADATA = ((TOKEN_METADATA_KEY, TOKEN),)


class GuardrailFixture(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        data["hosted"] = True
        data["received_plaintext_key"] = bool(user_api_key_dict.api_key)
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        response["post_call"] = data["model"]
        return response


class BlockingGuardrailFixture(CustomGuardrail):
    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        raise GuardrailRaisedException(
            guardrail_name="blocking",
            message="blocked by fixture",
            blocked_content=True,
        )


class CacheGuardrailFixture(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        data["cached"] = await cache.async_get_cache("seed")
        await cache.async_set_cache("written", {"ok": True}, ttl=30)
        return data


class FakeCache:
    def __init__(self):
        self.values: dict[str, object] = {"seed": {"value": 7}}

    async def async_get_cache(self, key: str, **kwargs: object) -> object | None:
        return self.values.get(key)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.values[key] = value


class UnsupportedFixture(CustomLogger):
    async def async_pre_request_hook(self, model, messages, kwargs):
        return kwargs


class CallbackFixture(CustomLogger):
    events: list[tuple[str, str]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.events.append((kwargs["model"], response_obj["id"]))


function_events: list[str] = []


async def callback_function(kwargs, response_obj, start_time, end_time):
    function_events.append(kwargs["model"])


class StreamFixture(CustomLogger):
    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict, response, request_data
    ) -> AsyncIterator[dict[str, str]]:
        async for chunk in response:
            yield {"value": chunk["value"].upper()}
            yield {"value": "!"}


class ChunkStreamFixture(CustomLogger):
    async def async_post_call_streaming_hook(self, user_api_key_dict, response):
        return {"value": response["value"].upper()}


@pytest_asyncio.fixture(loop_scope="function")
async def host_stub():
    server = grpc.aio.server()
    pb_grpc.add_PythonExtensionHostServicer_to_server(PythonExtensionHostService(TOKEN), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    await channel.channel_ready()
    try:
        yield pb_grpc.PythonExtensionHostStub(channel)
    finally:
        await channel.close()
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_capabilities_require_auth_and_negotiate_version(host_stub):
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await host_stub.GetCapabilities(pb.GetCapabilitiesRequest(protocol_major=1, protocol_minor=0))
    assert error.value.code() == grpc.StatusCode.UNAUTHENTICATED

    capabilities = await host_stub.GetCapabilities(
        pb.GetCapabilitiesRequest(protocol_major=1, protocol_minor=3), metadata=METADATA
    )
    assert capabilities.protocol_major == 1
    assert capabilities.protocol_minor == 0
    assert capabilities.supports_duplex_streaming is True
    assert "async_pre_call_hook" in capabilities.supported_hooks


@pytest.mark.asyncio
async def test_prepare_commit_execute_replace_and_retire(host_stub):
    prepare = await host_stub.PrepareRevision(
        pb.PrepareRevisionRequest(
            revision_id="revision-1",
            extensions=[_spec("guardrail", pb.EXTENSION_KIND_GUARDRAIL, "GuardrailFixture")],
        ),
        metadata=METADATA,
    )
    assert prepare.operation.ok is True
    assert prepare.extensions[0].hooks == ["async_post_call_success_hook", "async_pre_call_hook"]

    prepared_again = await host_stub.PrepareRevision(
        pb.PrepareRevisionRequest(
            revision_id="revision-1",
            extensions=[_spec("guardrail", pb.EXTENSION_KIND_GUARDRAIL, "GuardrailFixture")],
        ),
        metadata=METADATA,
    )
    assert prepared_again.operation.error_code == pb.ERROR_CODE_ALREADY_EXISTS
    assert prepared_again.extensions[0].hooks == prepare.extensions[0].hooks

    inactive = await host_stub.ExecuteGuardrail(_guardrail_request("revision-1"), metadata=METADATA)
    assert inactive.decision == pb.GUARDRAIL_DECISION_ERROR

    committed = await host_stub.CommitRevision(pb.CommitRevisionRequest(revision_id="revision-1"), metadata=METADATA)
    assert committed.ok is True
    replaced = await host_stub.ExecuteGuardrail(_guardrail_request("revision-1"), metadata=METADATA)
    assert replaced.decision == pb.GUARDRAIL_DECISION_REPLACE_REQUEST
    replaced_request = json.loads(replaced.request_json)
    assert replaced_request["hosted"] is True
    assert replaced_request["received_plaintext_key"] is False

    active_retire = await host_stub.RetireRevision(
        pb.RetireRevisionRequest(revision_id="revision-1"), metadata=METADATA
    )
    assert active_retire.error_code == pb.ERROR_CODE_INVALID_ARGUMENT

    await host_stub.PrepareRevision(
        pb.PrepareRevisionRequest(revision_id="revision-2", extensions=[]), metadata=METADATA
    )
    await host_stub.CommitRevision(pb.CommitRevisionRequest(revision_id="revision-2"), metadata=METADATA)
    retired = await host_stub.RetireRevision(pb.RetireRevisionRequest(revision_id="revision-1"), metadata=METADATA)
    assert retired.ok is True


@pytest.mark.asyncio
async def test_recognized_guardrail_exception_becomes_block(host_stub):
    await _activate(
        host_stub,
        "block-revision",
        [_spec("blocking", pb.EXTENSION_KIND_GUARDRAIL, "BlockingGuardrailFixture")],
    )
    request = _guardrail_request("block-revision", plugin_id="blocking")
    request.hook_phase = pb.HOOK_PHASE_DURING_CALL
    result = await host_stub.ExecuteGuardrail(request, metadata=METADATA)
    assert result.operation.ok is True
    assert result.decision == pb.GUARDRAIL_DECISION_BLOCK
    assert result.public_error.status_code == 400
    assert "blocked by fixture" in result.public_error.message


@pytest.mark.asyncio
async def test_prepare_rejects_any_unsupported_override(host_stub):
    response = await host_stub.PrepareRevision(
        pb.PrepareRevisionRequest(
            revision_id="unsupported",
            extensions=[_spec("unsupported", pb.EXTENSION_KIND_CALLBACK, "UnsupportedFixture")],
        ),
        metadata=METADATA,
    )
    assert response.operation.ok is False
    assert response.operation.error_code == pb.ERROR_CODE_LOAD_FAILED
    assert "async_pre_request_hook" in response.operation.error_message


@pytest.mark.asyncio
async def test_callback_batch_supports_logger_and_function(host_stub):
    CallbackFixture.events.clear()
    function_events.clear()
    specs = [
        _spec("logger", pb.EXTENSION_KIND_CALLBACK, "CallbackFixture"),
        _spec(
            "function",
            pb.EXTENSION_KIND_CALLBACK,
            "callback_function",
            {"callback_events": ["success"]},
        ),
    ]
    await _activate(host_stub, "callbacks", specs)
    events = [_callback_event("callbacks", plugin_id) for plugin_id in ("logger", "function")]
    response = await host_stub.PublishCallbackEvents(pb.PublishCallbackEventsRequest(events=events), metadata=METADATA)
    assert all(operation.ok for operation in response.operations)
    assert CallbackFixture.events == [("test-model", "response-1")]
    assert function_events == ["test-model"]


@pytest.mark.asyncio
async def test_duplex_iterator_can_emit_multiple_chunks_per_input(host_stub):
    await _activate(
        host_stub,
        "streaming",
        [_spec("stream", pb.EXTENSION_KIND_CALLBACK, "StreamFixture")],
    )

    async def frames() -> AsyncIterator[pb.StreamFrame]:
        yield pb.StreamFrame(
            kind=pb.STREAM_FRAME_KIND_OPEN,
            stream_id="stream-1",
            open=pb.StreamOpen(
                context=_context("streaming"),
                plugin_id="stream",
                request_json=b'{"model":"test-model"}',
                auth=pb.AuthContext(key_hash="hashed"),
                iterator_hook=True,
            ),
        )
        yield pb.StreamFrame(
            kind=pb.STREAM_FRAME_KIND_INPUT_CHUNK,
            stream_id="stream-1",
            chunk_json=b'{"value":"hello"}',
        )
        yield pb.StreamFrame(kind=pb.STREAM_FRAME_KIND_END, stream_id="stream-1")

    output = [frame async for frame in host_stub.TransformStream(frames(), metadata=METADATA)]
    assert [json.loads(frame.chunk_json) for frame in output[:-1]] == [
        {"value": "HELLO"},
        {"value": "!"},
    ], [(frame.kind, frame.error.message) for frame in output]
    assert output[-1].kind == pb.STREAM_FRAME_KIND_END


@pytest.mark.asyncio
async def test_duplex_chunk_hook_transforms_each_input(host_stub):
    await _activate(
        host_stub,
        "chunk-streaming",
        [_spec("chunk-stream", pb.EXTENSION_KIND_CALLBACK, "ChunkStreamFixture")],
    )

    async def frames() -> AsyncIterator[pb.StreamFrame]:
        yield pb.StreamFrame(
            kind=pb.STREAM_FRAME_KIND_OPEN,
            stream_id="stream-2",
            open=pb.StreamOpen(
                context=_context("chunk-streaming"),
                plugin_id="chunk-stream",
                request_json=b'{"model":"test-model"}',
                auth=pb.AuthContext(key_hash="hashed"),
                iterator_hook=False,
            ),
        )
        yield pb.StreamFrame(
            kind=pb.STREAM_FRAME_KIND_INPUT_CHUNK,
            stream_id="stream-2",
            chunk_json=b'{"value":"hello"}',
        )
        yield pb.StreamFrame(kind=pb.STREAM_FRAME_KIND_END, stream_id="stream-2")

    output = [frame async for frame in host_stub.TransformStream(frames(), metadata=METADATA)]
    assert json.loads(output[0].chunk_json) == {"value": "HELLO"}
    assert output[1].kind == pb.STREAM_FRAME_KIND_END


@pytest.mark.asyncio
async def test_invocation_scoped_reverse_cache_access_is_revoked():
    registry = InvocationCacheRegistry()
    cache = FakeCache()
    cache_ref = registry.register("invocation-1", cache)
    assert cache_ref is not None

    gateway_server = grpc.aio.server()
    pb_grpc.add_GatewayServicesServicer_to_server(GatewayServices(TOKEN, registry), gateway_server)
    gateway_port = gateway_server.add_insecure_port("127.0.0.1:0")
    await gateway_server.start()
    gateway_channel = grpc.aio.insecure_channel(f"127.0.0.1:{gateway_port}")
    gateway_stub = pb_grpc.GatewayServicesStub(gateway_channel)

    host_server = grpc.aio.server()
    pb_grpc.add_PythonExtensionHostServicer_to_server(
        PythonExtensionHostService(TOKEN, gateway_stub=gateway_stub), host_server
    )
    host_port = host_server.add_insecure_port("127.0.0.1:0")
    await host_server.start()
    host_channel = grpc.aio.insecure_channel(f"127.0.0.1:{host_port}")
    host = pb_grpc.PythonExtensionHostStub(host_channel)
    try:
        await _activate(
            host,
            "cache-revision",
            [_spec("cache", pb.EXTENSION_KIND_GUARDRAIL, "CacheGuardrailFixture")],
        )
        request = _guardrail_request("cache-revision", plugin_id="cache")
        request.cache.CopyFrom(cache_ref)
        result = await host.ExecuteGuardrail(request, metadata=METADATA)
        assert json.loads(result.request_json)["cached"] == {"value": 7}
        assert cache.values["written"] == {"ok": True}

        registry.revoke(cache_ref)
        revoked = await gateway_stub.CacheGet(pb.CacheGetRequest(cache=cache_ref, key="seed"), metadata=METADATA)
        assert revoked.operation.error_code == pb.ERROR_CODE_NOT_FOUND
    finally:
        await host_channel.close()
        await host_server.stop(grace=0)
        await gateway_channel.close()
        await gateway_server.stop(grace=0)


def test_manifest_build_does_not_import_customer_modules():
    module_name = "customer_extension_that_proxy_must_not_import"
    config = {
        "litellm_settings": {
            "callbacks": [f"{module_name}.logger"],
            "success_callback": [f"{module_name}.success"],
            "failure_callback": [f"{module_name}.failure"],
        },
        "guardrails": [
            {
                "guardrail_name": "customer-guardrail",
                "litellm_params": {
                    "guardrail": f"{module_name}.Guardrail",
                    "mode": "pre_call",
                },
            }
        ],
    }

    manifest = build_manifest(config)

    assert len(manifest.specs) == 4
    assert module_name not in sys.modules
    assert settings_from_config(config) is None


@pytest.mark.asyncio
async def test_python_client_fails_open_when_host_is_unavailable():
    port = _unused_port()
    manifest = build_manifest({"litellm_settings": {"callbacks": [f"{MODULE}.CallbackFixture"]}})
    client = PythonExtensionClient(
        ExtensionHostSettings(
            endpoint=f"http://127.0.0.1:{port}",
            token=TOKEN,
            connect_timeout_seconds=0.05,
            hook_timeout_seconds=0.05,
        ),
        manifest,
    )
    try:
        assert await client.start() == ()
        result = await client.execute_guardrail(_guardrail_request(manifest.revision_id))
        assert result.decision == pb.GUARDRAIL_DECISION_ALLOW
        assert client.health.healthy is False
        assert client.bypass_counts
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_proxy_and_sidecar_share_contract_without_proxy_import(tmp_path: Path):
    module_name = "customer_sidecar_only_plugin"
    import_marker = tmp_path / "import.pid"
    event_marker = tmp_path / "event.pid"
    (tmp_path / f"{module_name}.py").write_text(
        """
import os
from pathlib import Path
from litellm.integrations.custom_logger import CustomLogger

Path(os.environ["EXTENSION_IMPORT_MARKER"]).write_text(str(os.getpid()))

class HostedLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        Path(os.environ["EXTENSION_EVENT_MARKER"]).write_text(str(os.getpid()))
""".lstrip(),
        encoding="utf-8",
    )
    port = _unused_port()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), os.getcwd()))
    environment["LITELLM_EXTENSION_HOST_TOKEN"] = TOKEN
    environment["EXTENSION_IMPORT_MARKER"] = str(import_marker)
    environment["EXTENSION_EVENT_MARKER"] = str(event_marker)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "litellm.python_extension_host.server",
        "--listen",
        f"127.0.0.1:{port}",
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        await asyncio.wait_for(channel.channel_ready(), timeout=10)
        config = {
            "general_settings": {
                "python_extension_host": {
                    "endpoint": f"http://127.0.0.1:{port}",
                    "token": TOKEN,
                    "connect_timeout_seconds": 2,
                    "hook_timeout_seconds": 2,
                }
            },
            "litellm_settings": {
                "success_callback": [f"{module_name}.HostedLogger"],
            },
        }
        runtime = await configure_extension_runtime(config)
        assert runtime is not None
        assert module_name not in sys.modules
        await _wait_for_file(import_marker)
        assert int(import_marker.read_text()) == process.pid

        callback = runtime.callback(f"{module_name}.HostedLogger")
        assert callback is not None
        await callback.async_log_success_event(
            {"model": "hosted-model"},
            {"id": "response-1"},
            datetime.now(),
            datetime.now(),
        )
        await _wait_for_file(event_marker)
        assert int(event_marker.read_text()) == process.pid
        assert process.pid != os.getpid()
    finally:
        await configure_extension_runtime({})
        await channel.close()
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()


def _unused_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


async def _wait_for_file(path: Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        await asyncio.sleep(0.025)
    raise AssertionError(f"timed out waiting for {path}")


def _spec(
    plugin_id: str,
    kind: int,
    target: str,
    constructor: dict[str, object] | None = None,
) -> pb.ExtensionSpec:
    return pb.ExtensionSpec(
        id=plugin_id,
        kind=kind,
        entrypoint=f"{MODULE}.{target}",
        constructor_json=json.dumps(constructor or {}).encode(),
    )


async def _activate(host_stub, revision: str, specs: list[pb.ExtensionSpec]) -> None:
    prepared = await host_stub.PrepareRevision(
        pb.PrepareRevisionRequest(revision_id=revision, extensions=specs), metadata=METADATA
    )
    assert prepared.operation.ok, prepared.operation.error_message
    committed = await host_stub.CommitRevision(pb.CommitRevisionRequest(revision_id=revision), metadata=METADATA)
    assert committed.ok


def _context(revision: str) -> pb.InvocationContext:
    return pb.InvocationContext(
        request_id="request-1",
        invocation_id="invocation-1",
        active_revision=revision,
        api_surface="chat.completions",
        call_type="acompletion",
    )


def _guardrail_request(revision: str, plugin_id: str = "guardrail") -> pb.GuardrailInvocation:
    return pb.GuardrailInvocation(
        context=_context(revision),
        plugin_id=plugin_id,
        hook_phase=pb.HOOK_PHASE_PRE_CALL,
        request_json=b'{"model":"test-model"}',
        auth=pb.AuthContext(key_hash="sha256-only", user_id="user-1", team_id="team-1"),
    )


def _callback_event(revision: str, plugin_id: str) -> pb.CallbackEvent:
    return pb.CallbackEvent(
        context=_context(revision),
        plugin_id=plugin_id,
        kind=pb.CALLBACK_EVENT_KIND_SUCCESS,
        standard_logging_payload_json=b'{"model":"test-model"}',
        response_json=b'{"id":"response-1"}',
        start_time_seconds=datetime(2026, 1, 1).timestamp(),
        end_time_seconds=datetime(2026, 1, 1, 0, 0, 1).timestamp(),
    )
