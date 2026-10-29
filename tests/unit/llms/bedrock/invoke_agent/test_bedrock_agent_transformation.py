import asyncio, base64, importlib, json, litellm, litellm.types, re, struct, zlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import httpx
import openai
import pytest
import respx
from botocore.loaders import Loader
from botocore.model import ServiceModel


from litellm.llms.bedrock.chat.invoke_agent.transformation import (
    AmazonInvokeAgentConfig,
)
from litellm.types.utils import ModelResponse
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


class TestAmazonInvokeAgentConfig:
    """Test suite for AmazonInvokeAgentConfig methods"""

    @pytest.fixture
    def config(self):
        """Create a test instance of AmazonInvokeAgentConfig"""
        return AmazonInvokeAgentConfig()

    @pytest.fixture
    def sample_messages(self):
        """Sample messages for testing"""
        return [
            {"role": "user", "content": "Hello, how can you help me?"},
            {"role": "assistant", "content": "I can help with various tasks."},
            {"role": "user", "content": "What is the weather like?"},
        ]

    @pytest.fixture
    def sample_events(self):
        """Sample events for testing event parsing"""
        return [
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("Hello ".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "chunk"},
                "payload": {
                    "bytes": base64.b64encode("world!".encode("utf-8")).decode("utf-8")
                },
            },
            {
                "headers": {"event_type": "trace"},
                "payload": {
                    "trace": {
                        "preProcessingTrace": {
                            "modelInvocationOutput": {
                                "metadata": {
                                    "usage": {"inputTokens": 10, "outputTokens": 20}
                                }
                            }
                        },
                        "orchestrationTrace": {
                            "modelInvocationInput": {
                                "foundationModel": "anthropic.claude-v2"
                            }
                        },
                    }
                },
            },
        ]

    def test_get_agent_id_and_alias_id_valid(self, config):
        """Test parsing valid agent model string"""
        model = "agent/L1RT58GYRW/MFPSBCXYTW"
        agent_id, agent_alias_id = config._get_agent_id_and_alias_id(model)

        assert agent_id == "L1RT58GYRW"
        assert agent_alias_id == "MFPSBCXYTW"

    def test_get_agent_id_and_alias_id_invalid_format(self, config):
        """Test parsing invalid agent model string"""
        invalid_models = [
            "invalid/L1RT58GYRW/MFPSBCXYTW",  # Wrong prefix
            "agent/L1RT58GYRW",  # Missing alias
            "agent/L1RT58GYRW/MFPSBCXYTW/extra",  # Too many parts
            "L1RT58GYRW/MFPSBCXYTW",  # Missing prefix
        ]

        for invalid_model in invalid_models:
            with pytest.raises(ValueError, match="Invalid model format"):
                config._get_agent_id_and_alias_id(invalid_model)

    @patch(
        "litellm.llms.bedrock.chat.invoke_agent.transformation.convert_content_list_to_str"
    )
    def test_transform_request(self, mock_convert, config, sample_messages):
        """Test transform_request method"""
        mock_convert.return_value = "What is the weather like?"

        model = "agent/TEST123/ALIAS456"
        optional_params = {}
        litellm_params = {}
        headers = {}

        result = config.transform_request(model, sample_messages, optional_params, litellm_params, headers)

        expected = {
            "inputText": "What is the weather like?",
            "enableTrace": True,
        }
        assert result == expected
        mock_convert.assert_called_once_with(sample_messages[-1])

    def test_extract_response_content(self, config, sample_events):
        """Test _extract_response_content method"""
        result = config._extract_response_content(sample_events)
        assert result == "Hello world!"

    def test_extract_response_content_empty_events(self, config):
        """Test _extract_response_content with empty events"""
        result = config._extract_response_content([])
        assert result == ""

    def test_extract_response_content_no_chunk_events(self, config):
        """Test _extract_response_content with no chunk events"""
        events = [{"headers": {"event_type": "trace"}, "payload": {"some": "data"}}]
        result = config._extract_response_content(events)
        assert result == ""

    def test_is_trace_event(self, config):
        """Test _is_trace_event method"""
        trace_event = {"headers": {"event_type": "trace"}, "payload": {"some": "data"}}
        chunk_event = {"headers": {"event_type": "chunk"}, "payload": {"bytes": "data"}}
        invalid_event = {"headers": {"event_type": "trace"}, "payload": None}

        assert config._is_trace_event(trace_event) is True
        assert config._is_trace_event(chunk_event) is False
        assert config._is_trace_event(invalid_event) is False

    def test_get_trace_data(self, config):
        """Test _get_trace_data method"""
        event = {"payload": {"trace": {"preProcessingTrace": {"some": "data"}}}}
        result = config._get_trace_data(event)
        assert result == {"preProcessingTrace": {"some": "data"}}

    def test_get_trace_data_no_payload(self, config):
        """Test _get_trace_data with no payload"""
        event = {"payload": None}
        result = config._get_trace_data(event)
        assert result is None

    def test_extract_usage_info(self, config, sample_events):
        """Test _extract_usage_info method"""
        result = config._extract_usage_info(sample_events)

        assert result["inputTokens"] == 10
        assert result["outputTokens"] == 20
        assert result["model"] == "anthropic.claude-v2"

    def test_extract_usage_info_empty_events(self, config):
        """Test _extract_usage_info with empty events"""
        result = config._extract_usage_info([])

        assert result["inputTokens"] == 0
        assert result["outputTokens"] == 0
        assert result["model"] is None

    def test_extract_and_update_preprocessing_usage(self, config):
        """Test _extract_and_update_preprocessing_usage method"""
        trace_data = {
            "preProcessingTrace": {
                "modelInvocationOutput": {
                    "metadata": {"usage": {"inputTokens": 15, "outputTokens": 25}}
                }
            }
        }
        usage_info = {"inputTokens": 5, "outputTokens": 10, "model": None}

        config._extract_and_update_preprocessing_usage(trace_data, usage_info)

        assert usage_info["inputTokens"] == 20  # 5 + 15
        assert usage_info["outputTokens"] == 35  # 10 + 25

    def test_extract_and_update_preprocessing_usage_no_data(self, config):
        """Test _extract_and_update_preprocessing_usage with missing data"""
        trace_data = {}
        usage_info = {"inputTokens": 5, "outputTokens": 10, "model": None}

        config._extract_and_update_preprocessing_usage(trace_data, usage_info)

        # Should remain unchanged
        assert usage_info["inputTokens"] == 5
        assert usage_info["outputTokens"] == 10

    def test_extract_orchestration_model(self, config):
        """Test _extract_orchestration_model method"""
        trace_data = {
            "orchestrationTrace": {
                "modelInvocationInput": {"foundationModel": "anthropic.claude-v2"}
            }
        }
        result = config._extract_orchestration_model(trace_data)
        assert result == "anthropic.claude-v2"

    def test_extract_orchestration_model_no_data(self, config):
        """Test _extract_orchestration_model with missing data"""
        trace_data = {}
        result = config._extract_orchestration_model(trace_data)
        assert result is None

    def test_build_model_response(self, config):
        """Test _build_model_response method"""
        content = "Hello, world!"
        model = "agent/TEST123/ALIAS456"
        usage_info = {
            "inputTokens": 10,
            "outputTokens": 20,
            "model": "anthropic.claude-v2",
        }
        model_response = ModelResponse()

        result = config._build_model_response(
            content, model, usage_info, model_response
        )

        assert len(result.choices) == 1
        assert result.choices[0].message.content == content
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.model == "anthropic.claude-v2"
        assert hasattr(result, "usage")
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30

    @patch(
        "litellm.llms.bedrock.chat.invoke_agent.transformation.convert_content_list_to_str"
    )
    @patch.object(AmazonInvokeAgentConfig, "get_runtime_endpoint")
    @patch.object(AmazonInvokeAgentConfig, "_get_aws_region_name")
    def test_get_complete_url(self, mock_region, mock_endpoint, mock_convert, config):
        """Test get_complete_url method"""
        mock_endpoint.return_value = (
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            None,
        )
        mock_region.return_value = "us-east-1"

        api_base = None
        api_key = None
        model = "agent/L1RT58GYRW/MFPSBCXYTW"
        optional_params = {}
        litellm_params = {}

        result = config.get_complete_url(
            api_base, api_key, model, optional_params, litellm_params
        )

        assert (
            "https://bedrock-runtime.us-east-1.amazonaws.com/agents/L1RT58GYRW/agentAliases/MFPSBCXYTW/sessions"
            in result
        )

    @patch(
        "litellm.llms.bedrock.chat.invoke_agent.transformation.convert_content_list_to_str"
    )
    @patch.object(AmazonInvokeAgentConfig, "get_runtime_endpoint")
    @patch.object(AmazonInvokeAgentConfig, "_get_aws_region_name")
    def test_get_complete_url_encodes_session_id(
        self, mock_region, mock_endpoint, mock_convert, config
    ):
        """Test get_complete_url encodes session ID path segment."""
        mock_endpoint.return_value = (
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            None,
        )
        mock_region.return_value = "us-east-1"

        result = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="agent/L1RT58GYRW/MFPSBCXYTW",
            optional_params={"sessionID": "../../sessions/other?x=1#frag"},
            litellm_params={},
        )

        assert "sessions/..%2F..%2Fsessions%2Fother%3Fx%3D1%23frag/text" in result


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}

@pytest.fixture
def _pr4_bedrock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "pr4-test-aws-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "pr4-test-aws-secret-key")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")

@pytest.mark.usefixtures("_pr4_bedrock_env", "_vcr_outcome_gate", "setup_and_teardown")
def test_bedrock_agents_with_custom_params():
    litellm.turn_on_debug()
    from unittest.mock import MagicMock

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agent/L1RT58GYRW/MFPSBCXYTW",
                messages=[
                    {
                        "role": "user",
                        "content": "Hi who is ishaan cto of litellm, tell me 10 things about him",
                    }
                ],
                invocationId="my-test-invocation-id",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        print(f"mock_post.call_args.kwargs: {mock_post.call_args.kwargs}")


def _agent_frame_header(name: str, value: str) -> bytes:
    return struct.pack("!B", len(name)) + name.encode() + b"\x07" + struct.pack("!H", len(value)) + value.encode()


def _agent_frame(headers: Mapping[str, str], payload: bytes) -> bytes:
    encoded_headers: Final = b"".join(_agent_frame_header(name, value) for name, value in headers.items())
    prelude: Final = struct.pack("!II", 16 + len(encoded_headers) + len(payload), len(encoded_headers))
    message: Final = prelude + struct.pack("!I", zlib.crc32(prelude)) + encoded_headers + payload
    return message + struct.pack("!I", zlib.crc32(message))


def _agent_chunk(text: str) -> bytes:
    return _agent_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
        json.dumps({"bytes": base64.b64encode(text.encode()).decode()}).encode(),
    )


def _agent_exception(exception_type: str, message: str) -> bytes:
    return _agent_frame(
        {":exception-type": exception_type, ":content-type": "application/json", ":message-type": "exception"},
        json.dumps({"message": message}).encode(),
    )


_AGENT_URL: Final = re.compile(r"https://bedrock-agent-runtime\.us-east-1\.amazonaws\.com/.*")
_AGENT_RESPONSE_HEADERS: Final = MappingProxyType(
    {
        "content-type": "application/vnd.amazon.eventstream",
        "x-amzn-RequestId": "req-agent-1",
        "x-agent-secret": "s3cr3t",
    }
)
_UNDECODABLE_AGENT_BODIES: Final = (
    pytest.param(b"", id="empty"),
    pytest.param(b"\x00\x00\x00\x05", id="shorter-than-a-prelude"),
    pytest.param(_agent_chunk("Hello")[:-8], id="truncated-first-message"),
)


@pytest.fixture
def _agent_http(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "agent-test-aws-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "agent-test-aws-secret-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


async def _invoke_agent(respx_mock: respx.MockRouter, body: bytes, sync_mode: bool, stream: bool) -> str:
    respx_mock.post(_AGENT_URL).mock(return_value=httpx.Response(200, content=body, headers=_AGENT_RESPONSE_HEADERS))
    request: Final = {
        "model": "bedrock/agent/AGENT1234/ALIAS1234",
        "messages": [{"role": "user", "content": "hi"}],
        "aws_region_name": "us-east-1",
        "num_retries": 0,
        "stream": stream,
    }
    if sync_mode:
        response: Final = litellm.completion(**request)
        if stream:
            return "".join(chunk.choices[0].delta.content or "" for chunk in response)
        return response.choices[0].message.content
    async_response: Final = await litellm.acompletion(**request)
    if stream:
        return "".join([chunk.choices[0].delta.content or "" async for chunk in async_response])
    return async_response.choices[0].message.content


@pytest.mark.asyncio
@pytest.mark.usefixtures("_agent_http")
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("body", _UNDECODABLE_AGENT_BODIES)
async def test_invoke_agent_200_body_that_decodes_to_no_events_raises_bad_gateway(
    respx_mock: respx.MockRouter, body: bytes, sync_mode: bool, stream: bool
) -> None:
    with pytest.raises(litellm.BadGatewayError) as exc_info:
        await _invoke_agent(respx_mock, body, sync_mode, stream)

    assert exc_info.value.status_code == 502
    assert "HTTP 200" in exc_info.value.message
    assert "decoded to no events" in exc_info.value.message
    assert f"{len(body)} bytes received" in exc_info.value.message
    assert "application/vnd.amazon.eventstream" in exc_info.value.message
    assert "req-agent-1" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.usefixtures("_agent_http")
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_invoke_agent_body_cut_inside_a_message_raises_bad_gateway(
    respx_mock: respx.MockRouter, sync_mode: bool
) -> None:
    cut_message: Final = _agent_chunk(" world")[:-8]

    with pytest.raises(litellm.BadGatewayError) as exc_info:
        await _invoke_agent(respx_mock, _agent_chunk("Hello") + cut_message, sync_mode, stream=False)

    assert exc_info.value.status_code == 502
    assert f"{len(cut_message)} undecoded bytes after 1 events" in exc_info.value.message
    assert "req-agent-1" in exc_info.value.message


def _modeled_agent_stream_status(exception_type: str) -> int:
    service_model: Final = ServiceModel(Loader().load_service_model("bedrock-agent-runtime", "service-2"))
    return service_model.shape_for("ResponseStream").members[exception_type].metadata["error"]["httpStatusCode"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_agent_http")
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "exception_type", ["throttlingException", "dependencyFailedException", "accessDeniedException"]
)
async def test_invoke_agent_exception_event_raises_its_modeled_status_instead_of_an_empty_turn(
    respx_mock: respx.MockRouter, exception_type: str, sync_mode: bool, stream: bool
) -> None:
    body: Final = _agent_chunk("Hello") + _agent_exception(exception_type, "agent call failed")

    with pytest.raises(openai.APIStatusError) as exc_info:
        await _invoke_agent(respx_mock, body, sync_mode, stream)

    assert exc_info.value.status_code == _modeled_agent_stream_status(exception_type)
    assert exception_type in exc_info.value.message
    assert "agent call failed" in exc_info.value.message
    assert exc_info.value.response.headers.get("x-amzn-requestid") == "req-agent-1"
    assert "x-agent-secret" not in exc_info.value.response.headers


@pytest.mark.asyncio
@pytest.mark.usefixtures("_agent_http")
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_invoke_agent_error_event_raises_with_its_error_code_and_message(
    respx_mock: respx.MockRouter, sync_mode: bool
) -> None:
    error_event: Final = _agent_frame(
        {":message-type": "error", ":error-code": "InternalFailure", ":error-message": "agent runtime failed"}, b""
    )

    with pytest.raises(openai.APIStatusError) as exc_info:
        await _invoke_agent(respx_mock, _agent_chunk("Hello") + error_event, sync_mode, stream=False)

    assert "InternalFailure agent runtime failed" in exc_info.value.message


@pytest.mark.asyncio
@pytest.mark.usefixtures("_agent_http")
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_invoke_agent_complete_body_returns_its_text(
    respx_mock: respx.MockRouter, sync_mode: bool, stream: bool
) -> None:
    content: Final = await _invoke_agent(respx_mock, _agent_chunk("Hello") + _agent_chunk(" world"), sync_mode, stream)

    assert content == "Hello world"
