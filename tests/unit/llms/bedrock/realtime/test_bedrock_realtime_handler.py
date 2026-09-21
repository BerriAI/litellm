import asyncio
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.constants import (
    BEDROCK_REALTIME_SDK_SUPPORTED_RANGE,
    REALTIME_SESSION_SUCCESS_LOGGED_KEY,
    WEBSOCKET_CLOSE_REASON_MAX_BYTES,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.realtime.handler import BedrockRealtime
from litellm.llms.bedrock.realtime.transformation import BedrockRealtimeConfig


@pytest.fixture(autouse=True)
def _isolate_host_aws_config(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for env_var in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_REGION_NAME",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(env_var, raising=False)


class FakePayloadPart:
    def __init__(self, bytes_):
        self.bytes_ = bytes_


class FakeInputChunk:
    def __init__(self, value):
        self.value = value


class FakeInputStream:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, event):
        self.sent.append(event)

    async def close(self):
        self.closed = True


class SendFailingInputStream(FakeInputStream):
    async def send(self, event):
        raise RuntimeError("bedrock send failed")


class FailOnPromptEndStream(FakeInputStream):
    async def send(self, event):
        payload = json.loads(event.value.bytes_.decode("utf-8"))
        if "promptEnd" in payload.get("event", {}):
            raise RuntimeError("bedrock rejected promptEnd")
        self.sent.append(event)


class FakeBedrockStream:
    def __init__(self, input_stream=None):
        self.input_stream = input_stream if input_stream is not None else FakeInputStream()

    async def await_output(self):
        return (None, EndedBedrockReceiver())


class ServiceUnavailableException(Exception):
    """Named like the modeled AWS SDK error so the handler maps it to HTTP 503"""


class ModelStreamErrorException(Exception):
    """Named like the modeled AWS SDK error so the handler maps it to HTTP 424"""


class UnavailableBedrockStream:
    """Lazy duplex stream whose HTTP response only fails once the output is awaited"""

    def __init__(self):
        self.input_stream = FakeInputStream()

    async def await_output(self):
        raise ServiceUnavailableException("fault injected: Bedrock realtime unavailable")


class FakeLogging:
    def __init__(self, trace_id="trace-nova-sonic"):
        self.litellm_trace_id = trace_id
        self.model_call_details = {}


class DisconnectingClientWS:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_to_client = []
        self.scope = {}

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("client disconnected")

    async def send_text(self, message):
        self.sent_to_client.append(message)


class ClosableClientWS:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class EndedBedrockReceiver:
    async def receive(self):
        return None


class EndedBedrockStream:
    async def await_output(self):
        return (None, EndedBedrockReceiver())


class RealtimeClientWS:
    def __init__(self):
        self.closed = False
        self.sent_to_client = []
        self.scope = {}

    async def receive_text(self):
        raise RuntimeError("client disconnected")

    async def send_text(self, message):
        self.sent_to_client.append(message)

    async def close(self, code=None, reason=None):
        self.closed = True


class ConnectedClientWS(RealtimeClientWS):
    """Client that sends its scripted messages and then stays connected until the server closes it"""

    def __init__(self, messages):
        super().__init__()
        self._messages = list(messages)
        self._closed_event = asyncio.Event()

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        await self._closed_event.wait()
        raise RuntimeError("client disconnected")

    async def close(self, code=None, reason=None):
        self.closed = True
        self._closed_event.set()


class ScriptedBedrockReceiver:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    async def receive(self):
        if not self._payloads:
            return None
        payload = self._payloads.pop(0)
        return SimpleNamespace(value=SimpleNamespace(bytes_=payload.encode("utf-8")))


class BreakingBedrockReceiver(ScriptedBedrockReceiver):
    """Delivers its payloads, then the provider stream breaks instead of ending normally"""

    async def receive(self):
        if not self._payloads:
            await asyncio.sleep(0)
            raise ModelStreamErrorException("Nova Sonic stream broke")
        return await super().receive()


class DrainedThenOpenBedrockReceiver(ScriptedBedrockReceiver):
    """Delivers its payloads, flags `drained`, then stays open like a live Nova Sonic turn"""

    def __init__(self, payloads):
        super().__init__(payloads)
        self.drained = asyncio.Event()

    async def receive(self):
        if not self._payloads:
            self.drained.set()
            await asyncio.Event().wait()
        return await super().receive()


class ResetOnAudioInputStream(FakeInputStream):
    """Accepts session setup, then the provider resets the input side once the first response was delivered"""

    def __init__(self, drained):
        super().__init__()
        self._drained = drained

    async def send(self, event):
        if "audioInput" in json.loads(event.value.bytes_.decode("utf-8")).get("event", {}):
            await self._drained.wait()
            raise RuntimeError("bedrock input stream reset")
        self.sent.append(event)


class ScriptedBedrockStream:
    def __init__(self, payloads, receiver_type=ScriptedBedrockReceiver):
        self.input_stream = FakeInputStream()
        self._receiver = receiver_type(payloads)

    async def await_output(self):
        return (None, self._receiver)


class FakeAWSCredentialsIdentity:
    def __init__(self, access_key_id, secret_access_key, session_token=None):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token


class FakeStaticCredentialsResolver:
    def __init__(self, identity=None):
        self.identity = identity


class FakeAWSCRTHTTPClient:
    pass


class NoCredentialsBedrockRealtime(BedrockRealtime):
    def get_credentials(self, **kwargs):
        return None


class StubCredentialsBedrockRealtime(BedrockRealtime):
    def __init__(self, frozen_credentials):
        super().__init__()
        self.frozen_credentials = frozen_credentials
        self.get_credentials_kwargs = None

    def get_credentials(self, **kwargs):
        self.get_credentials_kwargs = kwargs
        return SimpleNamespace(get_frozen_credentials=lambda: self.frozen_credentials)


class FakeOperationInput:
    def __init__(self, model_id):
        self.model_id = model_id


def _install_fake_sdk_modules(monkeypatch, client_module, config_module):
    """Wire fake aws_sdk_bedrock_runtime / smithy packages into sys.modules for the handler's lazy imports."""
    package = types.ModuleType("aws_sdk_bedrock_runtime")
    models_module = types.ModuleType("aws_sdk_bedrock_runtime.models")
    models_module.BidirectionalInputPayloadPart = FakePayloadPart
    models_module.InvokeModelWithBidirectionalStreamInputChunk = FakeInputChunk
    models_module.InvokeModelWithBidirectionalStreamOperationInput = FakeOperationInput
    package.client = client_module
    package.config = config_module
    package.models = models_module
    smithy_package = types.ModuleType("smithy_aws_core")
    identity_module = types.ModuleType("smithy_aws_core.identity")
    identity_module.AWSCredentialsIdentity = FakeAWSCredentialsIdentity
    identity_module.StaticCredentialsResolver = FakeStaticCredentialsResolver
    smithy_package.identity = identity_module
    smithy_http_package = types.ModuleType("smithy_http")
    smithy_http_aio = types.ModuleType("smithy_http.aio")
    crt_module = types.ModuleType("smithy_http.aio.crt")
    crt_module.AWSCRTHTTPClient = FakeAWSCRTHTTPClient
    smithy_http_aio.crt = crt_module
    smithy_http_package.aio = smithy_http_aio

    stubbed_modules = {
        "aws_sdk_bedrock_runtime": package,
        "aws_sdk_bedrock_runtime.client": client_module,
        "aws_sdk_bedrock_runtime.config": config_module,
        "aws_sdk_bedrock_runtime.models": models_module,
        "smithy_aws_core": smithy_package,
        "smithy_aws_core.identity": identity_module,
        "smithy_http": smithy_http_package,
        "smithy_http.aio": smithy_http_aio,
        "smithy_http.aio.crt": crt_module,
    }
    for module_name, module in stubbed_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)


@pytest.fixture
def stub_aws_sdk_client(monkeypatch):
    """Fake of the aws-sdk-bedrock-runtime 0.10/0.11 surface: async config resolve, async client with close()"""
    captured = {}

    class FakeAsyncBedrockRuntimeConfig:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        @classmethod
        async def resolve(cls, **kwargs):
            captured["config_kwargs"] = kwargs
            return cls(kwargs)

    class FakeAsyncBedrockRuntimeClient:
        def __init__(self, config):
            captured["client_config"] = config
            captured["client_closed"] = False

        async def invoke_model_with_bidirectional_stream(self, operation_input):
            captured["operation_input"] = operation_input
            if captured.get("streams"):
                stream = captured["streams"].pop(0)
                if isinstance(stream, Exception):
                    raise stream
                captured["open_stream"] = stream
                return stream
            stream = ScriptedBedrockStream(captured.get("scripted_payloads", []))
            captured["open_stream"] = stream
            return stream

        async def close(self):
            open_stream = captured.get("open_stream")
            captured["input_closed_before_client_close"] = open_stream is None or open_stream.input_stream.closed
            captured["client_closed"] = True

    client_module = types.ModuleType("aws_sdk_bedrock_runtime.client")
    client_module.AsyncBedrockRuntimeClient = FakeAsyncBedrockRuntimeClient
    config_module = types.ModuleType("aws_sdk_bedrock_runtime.config")
    config_module.AsyncBedrockRuntimeConfig = FakeAsyncBedrockRuntimeConfig
    _install_fake_sdk_modules(monkeypatch, client_module, config_module)

    for env_var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_REGION_NAME",
        "AWS_SESSION_NAME",
        "AWS_PROFILE_NAME",
        "AWS_ROLE_NAME",
        "AWS_WEB_IDENTITY_TOKEN",
        "AWS_STS_ENDPOINT",
        "AWS_EXTERNAL_ID",
    ):
        monkeypatch.delenv(env_var, raising=False)

    return captured


@pytest.fixture
def stub_aws_models(monkeypatch):
    package = types.ModuleType("aws_sdk_bedrock_runtime")
    models = types.ModuleType("aws_sdk_bedrock_runtime.models")
    models.BidirectionalInputPayloadPart = FakePayloadPart
    models.InvokeModelWithBidirectionalStreamInputChunk = FakeInputChunk
    package.models = models
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", package)
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.models", models)


class TestBedrockRealtimeHandler:
    """Client disconnect must close the Bedrock session gracefully (LIT-2239 regression)"""

    @pytest.mark.asyncio
    async def test_client_disconnect_flushes_session_close_messages(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream()
        client_ws = DisconnectingClientWS(
            [json.dumps({"type": "session.update", "session": {"instructions": "You are helpful."}})]
        )

        await handler._forward_client_to_bedrock(client_ws, stream, config, "amazon.nova-sonic-v1:0", {})

        sent_events = [json.loads(chunk.value.bytes_.decode("utf-8")) for chunk in stream.input_stream.sent]
        event_names = [next(iter(event["event"])) for event in sent_events]
        assert event_names[0] == "sessionStart"
        assert event_names[-2:] == ["promptEnd", "sessionEnd"]
        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_client_disconnect_before_session_update_sends_nothing(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream()

        await handler._forward_client_to_bedrock(
            DisconnectingClientWS([]), stream, config, "amazon.nova-sonic-v1:0", {}
        )

        assert stream.input_stream.sent == []
        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_input_stream_closed_even_when_close_flush_fails(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream(input_stream=SendFailingInputStream())
        client_ws = DisconnectingClientWS(
            [json.dumps({"type": "session.update", "session": {"instructions": "You are helpful."}})]
        )

        with pytest.raises(RuntimeError, match="bedrock send failed"):
            await handler._forward_client_to_bedrock(client_ws, stream, config, "amazon.nova-sonic-v1:0", {})

        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_close_flush_continues_after_partial_send_failure(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream(input_stream=FailOnPromptEndStream())
        client_ws = DisconnectingClientWS(
            [json.dumps({"type": "session.update", "session": {"instructions": "You are helpful."}})]
        )

        await handler._forward_client_to_bedrock(client_ws, stream, config, "amazon.nova-sonic-v1:0", {})

        sent_events = [json.loads(chunk.value.bytes_.decode("utf-8")) for chunk in stream.input_stream.sent]
        event_names = [next(iter(event["event"])) for event in sent_events]
        assert "sessionEnd" in event_names
        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_forwarded_events_are_filtered_to_logged_types_for_spend_logging(self):
        handler = BedrockRealtime()
        stream = ScriptedBedrockStream(
            [
                json.dumps({"event": {"userSpeechStart": {}}}),
                json.dumps({"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}),
                json.dumps({"event": {"textOutput": {"content": "Hi"}}}),
                json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN"}}}),
            ]
        )
        client_ws = RealtimeClientWS()

        logged_events = [
            event
            async for event in handler._forward_bedrock_to_client(
                stream,
                client_ws,
                BedrockRealtimeConfig(),
                "amazon.nova-sonic-v1:0",
                FakeLogging(),
                {},
            )
        ]

        assert [event["type"] for event in logged_events] == ["response.done"]
        sent_types = [json.loads(message)["type"] for message in client_ws.sent_to_client]
        assert "input_audio_buffer.speech_started" in sent_types
        assert "response.text.delta" in sent_types
        assert "response.done" in sent_types
        assert client_ws.closed

    @pytest.mark.asyncio
    async def test_logged_event_types_star_collects_every_forwarded_event(self, monkeypatch):
        monkeypatch.setattr(litellm, "logged_real_time_event_types", "*")
        handler = BedrockRealtime()
        stream = ScriptedBedrockStream(
            [
                json.dumps({"event": {"userSpeechStart": {}}}),
                json.dumps({"event": {"userSpeechEnd": {}}}),
            ]
        )
        client_ws = RealtimeClientWS()

        logged_events = [
            event
            async for event in handler._forward_bedrock_to_client(
                stream,
                client_ws,
                BedrockRealtimeConfig(),
                "amazon.nova-sonic-v1:0",
                FakeLogging(),
                {},
            )
        ]

        assert [event["type"] for event in logged_events] == [
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        ]

    @pytest.mark.asyncio
    async def test_trailing_usage_after_last_done_is_dispatched_for_spend(self, stub_aws_sdk_client, monkeypatch):
        import litellm.llms.bedrock.realtime.handler as handler_module

        dispatched = {}

        class RecordingLogging(FakeLogging):
            async def dispatch_success_handlers(self, result=None, prefer_async_handlers=False, **kwargs):
                dispatched["events"] = result

        class RecordingLoggingWorker:
            def ensure_initialized_and_enqueue(self, coro):
                dispatched["coro"] = coro

        monkeypatch.setattr(handler_module, "GLOBAL_LOGGING_WORKER", RecordingLoggingWorker())
        stub_aws_sdk_client["scripted_payloads"] = [
            json.dumps(
                {
                    "event": {
                        "usageEvent": {
                            "totalInputTokens": 3,
                            "totalOutputTokens": 6,
                            "totalTokens": 9,
                            "details": {
                                "total": {
                                    "input": {"speechTokens": 3, "textTokens": 0},
                                    "output": {"speechTokens": 0, "textTokens": 6},
                                }
                            },
                        }
                    }
                }
            )
        ]

        await BedrockRealtime().async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=RealtimeClientWS(),
            logging_obj=RecordingLogging(),
            aws_region_name="us-east-1",
            aws_access_key_id="k",
            aws_secret_access_key="s",
        )
        await dispatched["coro"]

        assert [event["type"] for event in dispatched["events"]] == ["response.done"]
        usage = dispatched["events"][0]["response"]["usage"]
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (3, 6, 9)
        assert usage["input_token_details"] == {"audio_tokens": 3, "text_tokens": 0, "cached_tokens": 0}
        assert usage["output_token_details"] == {"audio_tokens": 0, "text_tokens": 6}

    @pytest.mark.asyncio
    async def test_bedrock_stream_end_closes_client_websocket(self):
        handler = BedrockRealtime()
        client_ws = ClosableClientWS()

        async for _ in handler._forward_bedrock_to_client(
            EndedBedrockStream(),
            client_ws,
            BedrockRealtimeConfig(),
            "amazon.nova-sonic-v1:0",
            MagicMock(),
            {},
        ):
            pass

        assert client_ws.closed


class TestBedrockRealtimeSessionLifecycle:
    """Server must emit session.created on connect and session.updated on session.update (LIT-4655 regression)"""

    @pytest.mark.asyncio
    async def test_session_created_sent_on_connect_before_any_client_input(self, stub_aws_sdk_client):
        handler = BedrockRealtime()
        websocket = RealtimeClientWS()

        await handler.async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=websocket,
            logging_obj=FakeLogging(),
            aws_region_name="us-east-1",
            aws_access_key_id="k",
            aws_secret_access_key="s",
        )

        assert websocket.sent_to_client, "server sent nothing on connect: spec-conformant clients deadlock"
        first_event = json.loads(websocket.sent_to_client[0])
        assert first_event["type"] == "session.created"
        assert first_event["session"]["id"] == "trace-nova-sonic"
        assert first_event["session"]["model"] == "amazon.nova-sonic-v1:0"

    @pytest.mark.asyncio
    async def test_session_update_is_acked_with_session_updated(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream()
        client_ws = DisconnectingClientWS(
            [json.dumps({"type": "session.update", "session": {"instructions": "hi", "modalities": ["text"]}})]
        )

        await handler._forward_client_to_bedrock(client_ws, stream, config, "amazon.nova-sonic-v1:0", {}, FakeLogging())

        acked = [json.loads(message) for message in client_ws.sent_to_client]
        updated = [event for event in acked if event["type"] == "session.updated"]
        assert updated, "session.update was not acked"
        assert updated[0]["session"]["modalities"] == ["text"], "ack must reflect the requested modalities"

    @pytest.mark.asyncio
    async def test_no_session_updated_without_logging_obj(self, stub_aws_models):
        handler = BedrockRealtime()
        config = BedrockRealtimeConfig()
        stream = FakeBedrockStream()
        client_ws = DisconnectingClientWS([json.dumps({"type": "session.update", "session": {"instructions": "hi"}})])

        await handler._forward_client_to_bedrock(client_ws, stream, config, "amazon.nova-sonic-v1:0", {})

        assert client_ws.sent_to_client == []


class TestBedrockRealtimeProviderFailurePropagation:
    """Deferred Nova Sonic failures must escape async_realtime so the router can fall back / cool down (LIT-6484)"""

    SESSION_UPDATE = json.dumps({"type": "session.update", "session": {"instructions": "hi", "modalities": ["text"]}})
    AWS_PARAMS = {"aws_region_name": "us-east-1", "aws_access_key_id": "k", "aws_secret_access_key": "s"}

    @pytest.mark.asyncio
    async def test_readiness_failure_escapes_and_fallback_replays_session_update(self, stub_aws_sdk_client):
        handler = BedrockRealtime()
        websocket = ConnectedClientWS([self.SESSION_UPDATE])
        healthy_stream = ScriptedBedrockStream([])
        eager_failure = ServiceUnavailableException("fault injected before the stream was returned")
        stub_aws_sdk_client["streams"] = [UnavailableBedrockStream(), eager_failure, healthy_stream]

        with pytest.raises(BedrockError) as failure:
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0", websocket=websocket, logging_obj=FakeLogging(), **self.AWS_PARAMS
            )

        assert failure.value.status_code == 503
        assert [json.loads(m)["type"] for m in websocket.sent_to_client] == ["session.created"]
        assert not websocket.closed, "the proxy route owns the client-facing error event and 1011 close"

        with pytest.raises(ServiceUnavailableException):
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0", websocket=websocket, logging_obj=FakeLogging(), **self.AWS_PARAMS
            )

        await handler.async_realtime(
            model="amazon.nova-sonic-v1:0", websocket=websocket, logging_obj=FakeLogging(), **self.AWS_PARAMS
        )

        assert [json.loads(m)["type"] for m in websocket.sent_to_client] == ["session.created", "session.updated"]
        replayed = [json.loads(chunk.value.bytes_.decode("utf-8")) for chunk in healthy_stream.input_stream.sent]
        assert [next(iter(event["event"])) for event in replayed][:2] == ["sessionStart", "promptStart"]
        assert websocket.closed

    TEXT_TURN = (
        json.dumps({"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}),
        json.dumps({"event": {"textOutput": {"content": "Hi"}}}),
        json.dumps({"event": {"contentEnd": {"stopReason": "END_TURN"}}}),
    )

    @pytest.fixture
    def spend_dispatch(self, monkeypatch):
        import litellm.llms.bedrock.realtime.handler as handler_module

        dispatched = {}

        class RecordingLogging(FakeLogging):
            async def dispatch_success_handlers(self, result=None, prefer_async_handlers=False, **kwargs):
                dispatched["events"] = result

        class RecordingLoggingWorker:
            def ensure_initialized_and_enqueue(self, coro):
                dispatched["coro"] = coro

        monkeypatch.setattr(handler_module, "GLOBAL_LOGGING_WORKER", RecordingLoggingWorker())
        dispatched["logging_obj"] = RecordingLogging()
        return dispatched

    @pytest.mark.asyncio
    async def test_mid_stream_failure_escapes_keeps_partial_spend_and_blocks_replay(
        self, stub_aws_sdk_client, spend_dispatch
    ):
        handler = BedrockRealtime()
        websocket = ConnectedClientWS([self.SESSION_UPDATE])
        stream = ScriptedBedrockStream(self.TEXT_TURN, receiver_type=BreakingBedrockReceiver)
        stub_aws_sdk_client["streams"] = [stream]

        with pytest.raises(BedrockError) as failure:
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=websocket,
                logging_obj=spend_dispatch["logging_obj"],
                **self.AWS_PARAMS,
            )

        assert failure.value.status_code == 424
        await spend_dispatch["coro"]
        assert [event["type"] for event in spend_dispatch["events"]] == ["response.done"]
        assert "response.done" in [json.loads(m)["type"] for m in websocket.sent_to_client]
        flushed = [json.loads(chunk.value.bytes_.decode("utf-8")) for chunk in stream.input_stream.sent]
        assert [next(iter(event["event"])) for event in flushed][-2:] == ["promptEnd", "sessionEnd"]
        assert stream.input_stream.closed

        with pytest.raises(BedrockError) as replay:
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=websocket,
                logging_obj=spend_dispatch["logging_obj"],
                **self.AWS_PARAMS,
            )

        assert replay.value.status_code == 400, "a committed session must not be silently restarted on a fallback"
        assert not litellm._should_retry(replay.value.status_code), "the router must not retry the replay refusal"
        assert "Nova Sonic stream broke" in replay.value.message, "the router surfaces the last attempt's error"

    @pytest.mark.asyncio
    async def test_input_side_failure_keeps_spend_for_responses_already_delivered(
        self, stub_aws_sdk_client, spend_dispatch
    ):
        receiver = DrainedThenOpenBedrockReceiver(self.TEXT_TURN)
        stream = ScriptedBedrockStream(self.TEXT_TURN, receiver_type=lambda _payloads: receiver)
        stream.input_stream = ResetOnAudioInputStream(receiver.drained)
        stub_aws_sdk_client["streams"] = [stream]
        websocket = ConnectedClientWS(
            [self.SESSION_UPDATE, json.dumps({"type": "input_audio_buffer.append", "audio": "AAAA"})]
        )

        with pytest.raises(RuntimeError, match="bedrock input stream reset"):
            await BedrockRealtime().async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=websocket,
                logging_obj=spend_dispatch["logging_obj"],
                **self.AWS_PARAMS,
            )

        assert "response.done" in [json.loads(m)["type"] for m in websocket.sent_to_client]
        await spend_dispatch["coro"]
        assert [event["type"] for event in spend_dispatch["events"]] == ["response.done"]

    @pytest.mark.asyncio
    async def test_success_dispatch_stamps_the_ownership_marker_only_when_spend_was_logged(
        self, stub_aws_sdk_client, spend_dispatch
    ):
        stub_aws_sdk_client["streams"] = [ScriptedBedrockStream(self.TEXT_TURN)]
        await BedrockRealtime().async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=ConnectedClientWS([self.SESSION_UPDATE]),
            logging_obj=spend_dispatch["logging_obj"],
            **self.AWS_PARAMS,
        )
        await spend_dispatch["coro"]
        assert [event["type"] for event in spend_dispatch["events"]] == ["response.done"]
        assert spend_dispatch["logging_obj"].model_call_details.get(REALTIME_SESSION_SUCCESS_LOGGED_KEY) is True

        idle_logging = FakeLogging()
        stub_aws_sdk_client["streams"] = [ScriptedBedrockStream([])]
        await BedrockRealtime().async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=ConnectedClientWS([self.SESSION_UPDATE]),
            logging_obj=idle_logging,
            **self.AWS_PARAMS,
        )
        assert REALTIME_SESSION_SUCCESS_LOGGED_KEY not in idle_logging.model_call_details

    @pytest.mark.asyncio
    async def test_stream_failure_after_client_disconnect_is_not_a_provider_failure(self, stub_aws_sdk_client):
        stream = ScriptedBedrockStream([], receiver_type=BreakingBedrockReceiver)
        stub_aws_sdk_client["streams"] = [stream]

        await BedrockRealtime().async_realtime(
            model="amazon.nova-sonic-v1:0", websocket=RealtimeClientWS(), logging_obj=FakeLogging(), **self.AWS_PARAMS
        )

        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_client_disconnect_ends_the_session_while_bedrock_output_stays_open(self, stub_aws_sdk_client):
        receiver = DrainedThenOpenBedrockReceiver([])
        stream = ScriptedBedrockStream([], receiver_type=lambda _payloads: receiver)
        stub_aws_sdk_client["streams"] = [stream]

        await asyncio.wait_for(
            BedrockRealtime().async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=RealtimeClientWS(),
                logging_obj=FakeLogging(),
                **self.AWS_PARAMS,
            ),
            timeout=1,
        )

        assert receiver.drained.is_set(), "the handler must have been waiting on the open provider stream"
        assert stream.input_stream.closed

    @pytest.mark.asyncio
    async def test_session_updated_is_not_sent_before_bedrock_is_ready(self, stub_aws_models):
        handler = BedrockRealtime()
        stream = UnavailableBedrockStream()
        client_ws = DisconnectingClientWS([self.SESSION_UPDATE])

        with pytest.raises(ServiceUnavailableException):
            await handler._forward_client_to_bedrock(
                client_ws, stream, BedrockRealtimeConfig(), "amazon.nova-sonic-v1:0", {}, FakeLogging()
            )

        assert client_ws.sent_to_client == []
        assert stream.input_stream.closed


class TestBedrockRealtimeAwsAuth:
    """AWS auth params passed via litellm_params must reach the Smithy client config (LIT-3923 regression)"""

    @pytest.mark.asyncio
    async def test_static_credentials_from_litellm_params_reach_smithy_config(self, stub_aws_sdk_client):
        handler = BedrockRealtime()
        websocket = RealtimeClientWS()

        await handler.async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=websocket,
            logging_obj=FakeLogging(),
            aws_region_name="us-east-1",
            aws_access_key_id="litellm-params-access-key",
            aws_secret_access_key="litellm-params-secret-key",
            aws_session_token="litellm-params-session-token",
        )

        config_kwargs = stub_aws_sdk_client["config_kwargs"]
        resolver = config_kwargs["aws_credentials_identity_resolver"]
        assert isinstance(resolver, FakeStaticCredentialsResolver)
        assert resolver.identity.access_key_id == "litellm-params-access-key"
        assert resolver.identity.secret_access_key == "litellm-params-secret-key"
        assert resolver.identity.session_token == "litellm-params-session-token"
        assert config_kwargs["region"] == "us-east-1"
        assert config_kwargs["endpoint_uri"] == "https://bedrock-runtime.us-east-1.amazonaws.com"
        assert isinstance(config_kwargs["transport"], FakeAWSCRTHTTPClient)
        assert stub_aws_sdk_client["client_config"].kwargs is config_kwargs
        assert stub_aws_sdk_client["operation_input"].model_id == "amazon.nova-sonic-v1:0"
        assert websocket.closed

    @pytest.mark.asyncio
    async def test_api_base_overrides_default_endpoint(self, stub_aws_sdk_client):
        await BedrockRealtime().async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=RealtimeClientWS(),
            logging_obj=FakeLogging(),
            aws_region_name="us-east-1",
            aws_access_key_id="k",
            aws_secret_access_key="s",
            api_base="https://vpce-bedrock.example.internal",
            aws_bedrock_runtime_endpoint="https://ignored.example.internal",
        )

        assert stub_aws_sdk_client["config_kwargs"]["endpoint_uri"] == "https://vpce-bedrock.example.internal"

    @pytest.mark.asyncio
    async def test_role_assumption_params_forwarded_to_get_credentials(self, stub_aws_sdk_client):
        handler = StubCredentialsBedrockRealtime(
            SimpleNamespace(
                access_key="assumed-access-key",
                secret_key="assumed-secret-key",
                token="assumed-session-token",
            )
        )

        await handler.async_realtime(
            model="amazon.nova-sonic-v1:0",
            websocket=RealtimeClientWS(),
            logging_obj=FakeLogging(),
            aws_region_name="eu-west-1",
            aws_role_name="arn:aws:iam::123456789012:role/nova-sonic",
            aws_session_name="realtime-session",
            aws_external_id="realtime-external-id",
            aws_session_tags=[{"Key": "team", "Value": "realtime"}],
        )

        assert handler.get_credentials_kwargs == {
            "aws_access_key_id": None,
            "aws_secret_access_key": None,
            "aws_session_token": None,
            "aws_region_name": "eu-west-1",
            "aws_session_name": "realtime-session",
            "aws_profile_name": None,
            "aws_role_name": "arn:aws:iam::123456789012:role/nova-sonic",
            "aws_web_identity_token": None,
            "aws_sts_endpoint": None,
            "aws_external_id": "realtime-external-id",
            "aws_session_tags": ({"Key": "team", "Value": "realtime"},),
        }
        resolver = stub_aws_sdk_client["config_kwargs"]["aws_credentials_identity_resolver"]
        assert isinstance(resolver, FakeStaticCredentialsResolver)
        assert resolver.identity.access_key_id == "assumed-access-key"
        assert resolver.identity.secret_access_key == "assumed-secret-key"
        assert resolver.identity.session_token == "assumed-session-token"

    @pytest.mark.asyncio
    async def test_unresolvable_credentials_raise_clear_auth_error(self, stub_aws_sdk_client):
        handler = NoCredentialsBedrockRealtime()

        with pytest.raises(BedrockError, match="No AWS credentials found for Bedrock realtime"):
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=RealtimeClientWS(),
                logging_obj=MagicMock(),
                aws_region_name="us-east-1",
            )

        assert "config_kwargs" not in stub_aws_sdk_client


class TestBedrockRealtimeSdkLifecycle:
    """aws-sdk-bedrock-runtime 0.10/0.11: async config, async client, CRT transport, close() (LIT-7938 regression)"""

    AWS_ARGS = {
        "model": "amazon.nova-sonic-v1:0",
        "aws_region_name": "us-east-1",
        "aws_access_key_id": "k",
        "aws_secret_access_key": "s",
    }

    @pytest.mark.asyncio
    async def test_client_closed_after_input_stream_on_normal_completion(self, stub_aws_sdk_client):
        await BedrockRealtime().async_realtime(websocket=RealtimeClientWS(), logging_obj=FakeLogging(), **self.AWS_ARGS)

        assert stub_aws_sdk_client["client_closed"]
        assert stub_aws_sdk_client["input_closed_before_client_close"]

    @pytest.mark.asyncio
    async def test_client_closed_when_stream_open_fails(self, stub_aws_sdk_client):
        stub_aws_sdk_client["streams"] = [ServiceUnavailableException("bedrock unavailable")]

        with pytest.raises(ServiceUnavailableException):
            await BedrockRealtime().async_realtime(
                websocket=RealtimeClientWS(), logging_obj=FakeLogging(), **self.AWS_ARGS
            )

        assert stub_aws_sdk_client["client_closed"]

    @pytest.mark.asyncio
    async def test_client_closed_when_provider_stream_fails_mid_session(self, stub_aws_sdk_client):
        stub_aws_sdk_client["streams"] = [ScriptedBedrockStream([], receiver_type=BreakingBedrockReceiver)]

        with pytest.raises(BedrockError):
            await BedrockRealtime().async_realtime(
                websocket=ConnectedClientWS([]), logging_obj=FakeLogging(), **self.AWS_ARGS
            )

        assert stub_aws_sdk_client["client_closed"]
        assert stub_aws_sdk_client["input_closed_before_client_close"]

    @pytest.mark.asyncio
    async def test_client_without_close_completes_session(self, monkeypatch):
        class ClientWithoutClose:
            def __init__(self, config):
                pass

            async def invoke_model_with_bidirectional_stream(self, operation_input):
                return ScriptedBedrockStream([])

        class ConfigWithoutCapture:
            @classmethod
            async def resolve(cls, **kwargs):
                return cls()

        client_module = types.ModuleType("aws_sdk_bedrock_runtime.client")
        client_module.AsyncBedrockRuntimeClient = ClientWithoutClose
        config_module = types.ModuleType("aws_sdk_bedrock_runtime.config")
        config_module.AsyncBedrockRuntimeConfig = ConfigWithoutCapture
        _install_fake_sdk_modules(monkeypatch, client_module, config_module)
        websocket = RealtimeClientWS()

        await BedrockRealtime().async_realtime(websocket=websocket, logging_obj=FakeLogging(), **self.AWS_ARGS)

        assert websocket.closed


class TestBedrockRealtimeSdkImportErrors:
    """Init errors must tell 'SDK not installed' apart from 'SDK installed but unsupported version' (LIT-7938)"""

    @pytest.mark.asyncio
    async def test_absent_sdk_names_install_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", None)
        handler = BedrockRealtime(sdk_version_lookup=lambda: None)

        with pytest.raises(ImportError) as exc_info:
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0", websocket=RealtimeClientWS(), logging_obj=FakeLogging()
            )

        message = str(exc_info.value)
        assert message.startswith("Missing aws_sdk_bedrock_runtime")
        assert "litellm[bedrock-realtime]" in message
        assert "is installed but" not in message
        close_reason = message.encode()[:WEBSOCKET_CLOSE_REASON_MAX_BYTES].decode()
        assert BEDROCK_REALTIME_SDK_SUPPORTED_RANGE in close_reason
        assert "pip install 'litellm[bedrock-realtime]'" in close_reason

    @pytest.mark.asyncio
    async def test_incompatible_sdk_names_installed_version_and_supported_range(self, monkeypatch):
        legacy_client_module = types.ModuleType("aws_sdk_bedrock_runtime.client")
        legacy_client_module.BedrockRuntimeClient = object
        legacy_config_module = types.ModuleType("aws_sdk_bedrock_runtime.config")
        legacy_config_module.Config = object
        _install_fake_sdk_modules(monkeypatch, legacy_client_module, legacy_config_module)
        handler = BedrockRealtime(sdk_version_lookup=lambda: "0.7.0")

        with pytest.raises(ImportError) as exc_info:
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0", websocket=RealtimeClientWS(), logging_obj=FakeLogging()
            )

        message = str(exc_info.value)
        assert "aws-sdk-bedrock-runtime 0.7.0 is installed but" in message
        assert ">=0.10.0,<0.12.0" in message
        assert not message.startswith("Missing aws_sdk_bedrock_runtime")
        assert isinstance(exc_info.value.__cause__, ImportError)
        assert str(exc_info.value.__cause__) not in message
        assert "cannot import name" not in message
        close_reason = message.encode()[:WEBSOCKET_CLOSE_REASON_MAX_BYTES].decode()
        assert "0.7.0 is installed" in close_reason
        assert BEDROCK_REALTIME_SDK_SUPPORTED_RANGE in close_reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
