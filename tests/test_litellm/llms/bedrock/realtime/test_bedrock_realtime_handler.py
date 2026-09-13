import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.realtime.handler import BedrockRealtime
from litellm.llms.bedrock.realtime.transformation import BedrockRealtimeConfig


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


class FakeLogging:
    def __init__(self, trace_id="trace-nova-sonic"):
        self.litellm_trace_id = trace_id


class DisconnectingClientWS:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_to_client = []

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

    async def receive_text(self):
        raise RuntimeError("client disconnected")

    async def send_text(self, message):
        self.sent_to_client.append(message)

    async def close(self, code=None, reason=None):
        self.closed = True


class ScriptedBedrockReceiver:
    def __init__(self, payloads):
        self._payloads = list(payloads)

    async def receive(self):
        if not self._payloads:
            return None
        payload = self._payloads.pop(0)
        return SimpleNamespace(value=SimpleNamespace(bytes_=payload.encode("utf-8")))


class ScriptedBedrockStream:
    def __init__(self, payloads):
        self.input_stream = FakeInputStream()
        self._receiver = ScriptedBedrockReceiver(payloads)

    async def await_output(self):
        return (None, self._receiver)


class FakeStaticCredentialsResolver:
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


@pytest.fixture
def stub_aws_sdk_client(monkeypatch):
    captured = {}

    class CapturingConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs
            self.kwargs = kwargs

    class FakeOperationInput:
        def __init__(self, model_id):
            self.model_id = model_id

    class FakeBedrockRuntimeClient:
        def __init__(self, config):
            captured["client_config"] = config

        async def invoke_model_with_bidirectional_stream(self, operation_input):
            captured["operation_input"] = operation_input
            return ScriptedBedrockStream(captured.get("scripted_payloads", []))

    package = types.ModuleType("aws_sdk_bedrock_runtime")
    client_module = types.ModuleType("aws_sdk_bedrock_runtime.client")
    client_module.BedrockRuntimeClient = FakeBedrockRuntimeClient
    client_module.InvokeModelWithBidirectionalStreamOperationInput = FakeOperationInput
    config_module = types.ModuleType("aws_sdk_bedrock_runtime.config")
    config_module.Config = CapturingConfig
    models_module = types.ModuleType("aws_sdk_bedrock_runtime.models")
    models_module.BidirectionalInputPayloadPart = FakePayloadPart
    models_module.InvokeModelWithBidirectionalStreamInputChunk = FakeInputChunk
    package.client = client_module
    package.config = config_module
    package.models = models_module
    smithy_package = types.ModuleType("smithy_aws_core")
    identity_module = types.ModuleType("smithy_aws_core.identity")
    identity_module.StaticCredentialsResolver = FakeStaticCredentialsResolver
    smithy_package.identity = identity_module

    stubbed_modules = {
        "aws_sdk_bedrock_runtime": package,
        "aws_sdk_bedrock_runtime.client": client_module,
        "aws_sdk_bedrock_runtime.config": config_module,
        "aws_sdk_bedrock_runtime.models": models_module,
        "smithy_aws_core": smithy_package,
        "smithy_aws_core.identity": identity_module,
    }
    for module_name, module in stubbed_modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)

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
        assert config_kwargs["aws_access_key_id"] == "litellm-params-access-key"
        assert config_kwargs["aws_secret_access_key"] == "litellm-params-secret-key"
        assert config_kwargs["aws_session_token"] == "litellm-params-session-token"
        assert isinstance(config_kwargs["aws_credentials_identity_resolver"], FakeStaticCredentialsResolver)
        assert config_kwargs["region"] == "us-east-1"
        assert stub_aws_sdk_client["client_config"].kwargs is config_kwargs
        assert stub_aws_sdk_client["operation_input"].model_id == "amazon.nova-sonic-v1:0"
        assert websocket.closed

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
        }
        config_kwargs = stub_aws_sdk_client["config_kwargs"]
        assert config_kwargs["aws_access_key_id"] == "assumed-access-key"
        assert config_kwargs["aws_secret_access_key"] == "assumed-secret-key"
        assert config_kwargs["aws_session_token"] == "assumed-session-token"
        assert isinstance(config_kwargs["aws_credentials_identity_resolver"], FakeStaticCredentialsResolver)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
