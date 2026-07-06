import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.bedrock.realtime.handler import BedrockRealtime, BotoCredentialsResolver
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


class DisconnectingClientWS:
    def __init__(self, messages):
        self._messages = list(messages)

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        raise RuntimeError("client disconnected")


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


@pytest.fixture
def stub_aws_models(monkeypatch):
    package = types.ModuleType("aws_sdk_bedrock_runtime")
    models = types.ModuleType("aws_sdk_bedrock_runtime.models")
    models.BidirectionalInputPayloadPart = FakePayloadPart
    models.InvokeModelWithBidirectionalStreamInputChunk = FakeInputChunk
    package.models = models
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime", package)
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.models", models)


class FakeAWSCredentialsIdentity:
    def __init__(self, access_key_id, secret_access_key, session_token=None, account_id=None):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.account_id = account_id


class FakeFrozenCredentials:
    def __init__(self, access_key, secret_key, token):
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token


class FakeBotoCredentials:
    def __init__(self, access_key, secret_key, token=None):
        self._frozen = FakeFrozenCredentials(access_key, secret_key, token)

    def get_frozen_credentials(self):
        return self._frozen


@pytest.fixture
def stub_smithy_identity(monkeypatch):
    package = types.ModuleType("smithy_aws_core")
    identity = types.ModuleType("smithy_aws_core.identity")
    identity.AWSCredentialsIdentity = FakeAWSCredentialsIdentity
    package.identity = identity
    monkeypatch.setitem(sys.modules, "smithy_aws_core", package)
    monkeypatch.setitem(sys.modules, "smithy_aws_core.identity", identity)


@pytest.fixture
def stub_aws_client(monkeypatch, stub_aws_models):
    created_configs = []

    class FakeConfig:
        def __init__(self, **kwargs):
            created_configs.append(kwargs)

    class FakeOperationInput:
        def __init__(self, model_id):
            self.model_id = model_id

    class FakeBedrockRuntimeClient:
        def __init__(self, config):
            self.config = config

        async def invoke_model_with_bidirectional_stream(self, operation_input):
            raise RuntimeError("stop before streaming")

    client_module = types.ModuleType("aws_sdk_bedrock_runtime.client")
    client_module.BedrockRuntimeClient = FakeBedrockRuntimeClient
    client_module.InvokeModelWithBidirectionalStreamOperationInput = FakeOperationInput
    config_module = types.ModuleType("aws_sdk_bedrock_runtime.config")
    config_module.Config = FakeConfig
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.client", client_module)
    monkeypatch.setitem(sys.modules, "aws_sdk_bedrock_runtime.config", config_module)
    return created_configs


class TestBedrockRealtimeCredentials:
    """Config-provided AWS credentials must reach the Bedrock realtime client, not just env vars"""

    @pytest.mark.asyncio
    async def test_resolver_returns_identity_from_boto_credentials(self, stub_smithy_identity):
        resolver = BotoCredentialsResolver(FakeBotoCredentials("config-key", "config-secret", "config-token"))

        identity = await resolver.get_identity(properties={})

        assert identity.access_key_id == "config-key"
        assert identity.secret_access_key == "config-secret"
        assert identity.session_token == "config-token"

    @pytest.mark.asyncio
    async def test_async_realtime_uses_config_credentials(self, stub_aws_client, monkeypatch):
        handler = BedrockRealtime()
        boto_credentials = FakeBotoCredentials("config-key", "config-secret")
        received_kwargs = {}

        def fake_get_credentials(**kwargs):
            received_kwargs.update(kwargs)
            return boto_credentials

        monkeypatch.setattr(handler, "get_credentials", fake_get_credentials)

        with pytest.raises(RuntimeError, match="stop before streaming"):
            await handler.async_realtime(
                model="amazon.nova-sonic-v1:0",
                websocket=ClosableClientWS(),
                logging_obj=MagicMock(),
                aws_region_name="us-east-1",
                aws_access_key_id="config-key",
                aws_secret_access_key="config-secret",
                aws_role_name="arn:aws:iam::123456789012:role/realtime",
            )

        assert received_kwargs["aws_access_key_id"] == "config-key"
        assert received_kwargs["aws_secret_access_key"] == "config-secret"
        assert received_kwargs["aws_role_name"] == "arn:aws:iam::123456789012:role/realtime"
        resolver = stub_aws_client[0]["aws_credentials_identity_resolver"]
        assert isinstance(resolver, BotoCredentialsResolver)
        assert resolver._credentials is boto_credentials


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
    async def test_bedrock_stream_end_closes_client_websocket(self):
        handler = BedrockRealtime()
        client_ws = ClosableClientWS()

        await handler._forward_bedrock_to_client(
            EndedBedrockStream(),
            client_ws,
            BedrockRealtimeConfig(),
            "amazon.nova-sonic-v1:0",
            MagicMock(),
            {},
        )

        assert client_ws.closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
