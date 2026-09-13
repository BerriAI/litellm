import base64
import json
import struct
import tracemalloc
from binascii import crc32
from datetime import datetime
from unittest.mock import patch

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import PassthroughStreamCollector
from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig
from litellm.types.utils import ModelResponse

CONVERSE_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
CONVERSE_STREAM_ENDPOINT = f"/model/{CONVERSE_MODEL}/converse-stream"
INVOKE_STREAM_ENDPOINT = f"/model/{CONVERSE_MODEL}/invoke-with-response-stream"


def test_bedrock_passthrough_get_complete_url_default_endpoint():
    """Test get_complete_url with default AWS endpoint (no override)"""
    config = BedrockPassthroughConfig()

    # Mock the methods following the pattern from test_base_aws_llm.py
    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ) as mock_get_runtime,
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={},
        )

        # Verify get_runtime_endpoint was called with correct parameters
        mock_get_runtime.assert_called_once_with(
            api_base=None,
            aws_bedrock_runtime_endpoint=None,
            aws_region_name="us-east-1",
            endpoint_type="runtime",
        )

        # Verify URL construction
        assert (
            str(url)
            == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke"
        )
        assert api_base == "https://bedrock-runtime.us-east-1.amazonaws.com"


def test_bedrock_passthrough_get_complete_url_custom_endpoint_no_path():
    """Test get_complete_url with custom endpoint (no base path)"""
    config = BedrockPassthroughConfig()

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-west-2"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=("http://proxy.com", "http://proxy.com"),
        ) as mock_get_runtime,
    ):
        url, api_base = config.get_complete_url(
            api_base="http://proxy.com",
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={},
        )

        # Verify get_runtime_endpoint was called with the api_base
        mock_get_runtime.assert_called_once_with(
            api_base="http://proxy.com",
            aws_bedrock_runtime_endpoint=None,
            aws_region_name="us-west-2",
            endpoint_type="runtime",
        )

        # Verify URL construction
        assert str(url) == "http://proxy.com/model/anthropic.claude-3-sonnet/invoke"
        assert api_base == "http://proxy.com"


def test_bedrock_passthrough_get_complete_url_custom_endpoint_with_path():
    """Test get_complete_url with custom endpoint that has a base path"""
    config = BedrockPassthroughConfig()

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-west-2"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "http://proxy.com/bedrockproxy",
                "http://proxy.com/bedrockproxy",
            ),
        ) as mock_get_runtime,
    ):
        url, api_base = config.get_complete_url(
            api_base="http://proxy.com/bedrockproxy",
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={
                "aws_bedrock_runtime_endpoint": "http://proxy.com/bedrockproxy"
            },
        )

        # Verify get_runtime_endpoint was called with correct parameters
        mock_get_runtime.assert_called_once_with(
            api_base="http://proxy.com/bedrockproxy",
            aws_bedrock_runtime_endpoint="http://proxy.com/bedrockproxy",
            aws_region_name="us-west-2",
            endpoint_type="runtime",
        )

        # Verify URL construction preserves the proxy path
        assert (
            str(url)
            == "http://proxy.com/bedrockproxy/model/anthropic.claude-3-sonnet/invoke"
        )
        assert api_base == "http://proxy.com/bedrockproxy"


def test_format_url_simple_joining():
    """Test format_url with simple URL joining"""
    config = BedrockPassthroughConfig()

    result = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="https://api.example.com",
        request_query_params={},
    )

    assert str(result) == "https://api.example.com/model/test/invoke"


def test_format_url_preserves_proxy_paths():
    """Test format_url preserves proxy paths in base URL"""
    config = BedrockPassthroughConfig()

    result = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={},
    )

    # This is the key test - proxy path should be preserved
    assert str(result) == "http://proxy.com/bedrockproxy/model/test/invoke"


def test_format_url_with_query_parameters():
    """Test format_url properly handles query parameters"""
    config = BedrockPassthroughConfig()

    result = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={"param1": "value1", "param2": "value2"},
    )

    # Should preserve proxy path and add query params
    result_str = str(result)
    assert "http://proxy.com/bedrockproxy/model/test/invoke" in result_str
    assert "param1=value1" in result_str
    assert "param2=value2" in result_str


def test_format_url_handles_trailing_slash_normalization():
    """Test format_url properly handles base URLs with and without trailing slashes"""
    config = BedrockPassthroughConfig()

    # Test with trailing slash
    result_with_slash = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy/",
        request_query_params={},
    )

    # Test without trailing slash
    result_without_slash = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={},
    )

    # Both should produce the same result
    assert str(result_with_slash) == str(result_without_slash)
    assert str(result_with_slash) == "http://proxy.com/bedrockproxy/model/test/invoke"


def test_bedrock_passthrough_with_application_inference_profile():
    """
    Test get_complete_url with Application Inference Profile ARN as model_id.

    This test verifies the fix for GitHub issue #18761 where Bedrock passthrough
    was not working with Application Inference Profiles. The model_id (ARN) should
    replace the translated model name in the endpoint URL and be properly encoded.
    """
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-20250514-v1:0"
    model_id = (
        "arn:aws:bedrock:eu-west-1:123456789:application-inference-profile/abcdefgh1234"
    )
    endpoint = f"model/{model}/invoke"

    with (
        patch.object(config, "_get_aws_region_name", return_value="eu-west-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.eu-west-1.amazonaws.com",
                "https://bedrock-runtime.eu-west-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id, "aws_region_name": "eu-west-1"},
        )

        # Verify that the URL contains the encoded model_id (ARN) instead of the model name
        url_str = str(url)
        # The ARN slash should be encoded as %2F
        assert (
            "application-inference-profile%2F" in url_str
        ), f"Expected encoded ARN in URL, but got: {url_str}"
        assert (
            model not in url_str
        ), f"Model name should be replaced by model_id, but got: {url_str}"
        assert "/invoke" in url_str, "Expected /invoke action in URL"

        # Verify the complete URL structure with encoded ARN
        encoded_model_id = "arn:aws:bedrock:eu-west-1:123456789:application-inference-profile%2Fabcdefgh1234"
        expected_url = f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{encoded_model_id}/invoke"
        assert url_str == expected_url, f"Expected {expected_url}, but got: {url_str}"


def test_bedrock_passthrough_with_inference_profile_converse_endpoint():
    """Test Application Inference Profile with converse endpoint and proper ARN encoding"""
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-20250514-v1:0"
    model_id = (
        "arn:aws:bedrock:us-east-1:123456789:application-inference-profile/xyz123"
    )
    endpoint = f"model/{model}/converse"

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id},
        )

        url_str = str(url)
        # The ARN should be encoded with %2F
        assert "application-inference-profile%2F" in url_str
        assert "/converse" in url_str
        assert model not in url_str


def test_bedrock_passthrough_without_model_id_backward_compatibility():
    """
    Test that passthrough still works without model_id (backward compatibility).

    When model_id is not provided, the system should use the model name as before.
    """
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-3-sonnet"
    endpoint = f"model/{model}/invoke"

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={},  # No model_id provided
        )

        # Verify that the URL contains the model name (not replaced)
        url_str = str(url)
        assert (
            model in url_str
        ), f"Expected model name in URL when model_id not provided, but got: {url_str}"
        expected_url = (
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model}/invoke"
        )
        assert url_str == expected_url


def test_bedrock_passthrough_region_extraction_from_inference_profile_arn():
    """Test that AWS region is correctly extracted from Application Inference Profile ARN"""
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-20250514-v1:0"
    # ARN contains us-west-2 region
    model_id = (
        "arn:aws:bedrock:us-west-2:123456789:application-inference-profile/test123"
    )
    endpoint = f"model/{model}/invoke"

    # Don't provide aws_region_name in litellm_params to test ARN extraction
    with patch.object(
        config,
        "get_runtime_endpoint",
        return_value=(
            "https://bedrock-runtime.us-west-2.amazonaws.com",
            "https://bedrock-runtime.us-west-2.amazonaws.com",
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={
                "model_id": model_id
            },  # Region should be extracted from ARN
        )

        # Verify that the region from ARN is used in the base URL
        assert (
            "us-west-2" in api_base
        ), f"Expected region 'us-west-2' from ARN in base URL, but got: {api_base}"


def test_bedrock_passthrough_model_id_arn_encoding():
    """
    Test that model_id ARNs are properly URL-encoded when used in endpoints.

    This is the critical fix for the issue where ARNs with slashes need to be encoded
    so they're treated as a single path component rather than multiple path segments.

    For example:
    arn:aws:bedrock:us-east-1:590183661440:application-inference-profile/b943q2qbl3m7
    should become:
    arn:aws:bedrock:us-east-1:590183661440:application-inference-profile%2Fb943q2qbl3m7
    """
    config = BedrockPassthroughConfig()

    model = "bedrock-claude-4-5-sonnet"
    # ARN with a slash that needs encoding
    model_id = "arn:aws:bedrock:us-east-1:590183661440:application-inference-profile/b943q2qbl3m7"
    endpoint = f"/model/{model}/converse"

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id},
        )

        url_str = str(url)

        # The slash in the ARN after application-inference-profile should be encoded as %2F
        assert (
            "application-inference-profile%2F" in url_str
        ), f"Expected encoded ARN with %2F in URL, but got: {url_str}"

        # The unencoded version should NOT be in the URL
        assert (
            "application-inference-profile/" not in url_str
        ), f"ARN slash should be encoded, but found unencoded version in: {url_str}"

        # Verify the complete expected URL structure
        expected_encoded_model_id = "arn:aws:bedrock:us-east-1:590183661440:application-inference-profile%2Fb943q2qbl3m7"
        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{expected_encoded_model_id}/converse"
        assert url_str == expected_url, f"Expected {expected_url}, but got: {url_str}"


def test_bedrock_passthrough_model_id_arn_encoding_invoke_endpoint():
    """
    Test ARN encoding with /invoke endpoint (not just /converse).
    """
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_id = (
        "arn:aws:bedrock:us-east-1:123456789:application-inference-profile/xyz789"
    )
    endpoint = f"/model/{model}/invoke"

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id},
        )

        url_str = str(url)

        # Verify encoding
        assert "application-inference-profile%2F" in url_str
        assert "/invoke" in url_str

        expected_encoded_model_id = (
            "arn:aws:bedrock:us-east-1:123456789:application-inference-profile%2Fxyz789"
        )
        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{expected_encoded_model_id}/invoke"
        assert url_str == expected_url


def test_bedrock_passthrough_model_id_without_arn():
    """
    Test that non-ARN model_ids (regular model IDs) are not affected by encoding logic.
    """
    config = BedrockPassthroughConfig()

    model = "my-model"
    # Regular model ID (not an ARN)
    model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    endpoint = f"/model/{model}/converse"

    with (
        patch.object(config, "_get_aws_region_name", return_value="us-east-1"),
        patch.object(
            config,
            "get_runtime_endpoint",
            return_value=(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "https://bedrock-runtime.us-east-1.amazonaws.com",
            ),
        ),
    ):
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id},
        )

        url_str = str(url)

        # Regular model ID should be used as-is (no encoding needed)
        assert model_id in url_str
        assert "%2F" not in url_str, "Non-ARN model IDs should not be encoded"

        expected_url = (
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        )
        assert url_str == expected_url


def _event_frame(event_type: str, payload: dict) -> bytes:
    def header(name: str, value: str) -> bytes:
        name_b, value_b = name.encode(), value.encode()
        return struct.pack("!B", len(name_b)) + name_b + struct.pack("!B", 7) + struct.pack("!H", len(value_b)) + value_b

    payload_b = json.dumps(payload, separators=(",", ":")).encode()
    headers_b = (
        header(":event-type", event_type)
        + header(":content-type", "application/json")
        + header(":message-type", "event")
    )
    prelude = struct.pack("!II", 12 + len(headers_b) + len(payload_b) + 4, len(headers_b))
    prelude_crc = crc32(prelude) & 0xFFFFFFFF
    message = struct.pack("!I", prelude_crc) + headers_b + payload_b
    return prelude + message + struct.pack("!I", crc32(message, prelude_crc) & 0xFFFFFFFF)


def _text_block(index: int, texts: list[str]) -> bytes:
    return (
        _event_frame("contentBlockStart", {"contentBlockIndex": index, "start": {}})
        + b"".join(
            _event_frame("contentBlockDelta", {"contentBlockIndex": index, "delta": {"text": text}}) for text in texts
        )
        + _event_frame("contentBlockStop", {"contentBlockIndex": index})
    )


def _stream_tail(stop_reason: str, output_tokens: int) -> bytes:
    return _event_frame("messageStop", {"stopReason": stop_reason}) + _event_frame(
        "metadata",
        {
            "metrics": {"latencyMs": 1234},
            "usage": {"inputTokens": 25, "outputTokens": output_tokens, "totalTokens": 25 + output_tokens},
        },
    )


def _invoke_chunk(payload: dict) -> bytes:
    return _event_frame("chunk", {"bytes": base64.b64encode(json.dumps(payload).encode()).decode()})


def _stream_logging_obj(endpoint: str) -> Logging:
    logging_obj = Logging(
        model=CONVERSE_MODEL,
        messages=[],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="call-1",
        function_id="fn-1",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "bedrock"
    logging_obj.model_call_details["endpoint"] = endpoint
    return logging_obj


def _converse_stream_logging_obj() -> Logging:
    return _stream_logging_obj(CONVERSE_STREAM_ENDPOINT)


def _stream_collector(endpoint: str) -> PassthroughStreamCollector:
    return BedrockPassthroughConfig().create_stream_collector(
        model=CONVERSE_MODEL, custom_llm_provider="bedrock", endpoint=endpoint
    )


def _converse_stream_collector() -> PassthroughStreamCollector:
    return _stream_collector(CONVERSE_STREAM_ENDPOINT)


def _feed(collector: PassthroughStreamCollector, stream: bytes, chunk_size: int = 16384) -> None:
    for offset in range(0, len(stream), chunk_size):
        collector.add(stream[offset : offset + chunk_size])


def test_converse_stream_collector_keeps_usage_without_retaining_the_stream():
    texts = [f"tok{i} " for i in range(4000)]
    stream = _event_frame("messageStart", {"role": "assistant"}) + _text_block(0, texts) + _stream_tail("end_turn", 4000)
    _feed(_converse_stream_collector(), stream)

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        collector = _converse_stream_collector()
        _feed(collector, stream)
        retained = tracemalloc.get_traced_memory()[0] - base
    finally:
        tracemalloc.stop()

    assert retained < len(stream) // 4

    response = collector.build_logged_response(_converse_stream_logging_obj())
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "".join(texts)
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (25, 4000)


def test_converse_stream_collector_keeps_tool_calls_between_text_runs():
    stream = (
        _event_frame("messageStart", {"role": "assistant"})
        + _text_block(0, ["Let me ", "check."])
        + _event_frame(
            "contentBlockStart",
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tool-1", "name": "get_weather"}}},
        )
        + _event_frame("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"city": '}}})
        + _event_frame("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"Paris"}'}}})
        + _event_frame("contentBlockStop", {"contentBlockIndex": 1})
        + _text_block(2, ["Done", "."])
        + _stream_tail("tool_use", 12)
    )
    collector = _converse_stream_collector()
    _feed(collector, stream, chunk_size=7)

    response = collector.build_logged_response(_converse_stream_logging_obj())
    assert isinstance(response, ModelResponse)
    message = response.choices[0].message
    assert message.content == "Let me check.Done."
    assert [(call.function.name, call.function.arguments) for call in message.tool_calls] == [
        ("get_weather", '{"city": "Paris"}')
    ]
    assert response.choices[0].finish_reason == "tool_calls"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (25, 12)


def test_invoke_stream_collector_keeps_usage_without_retaining_the_stream():
    texts = [f"tok{i} " for i in range(4000)]
    stream = (
        _invoke_chunk(
            {
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "model": CONVERSE_MODEL,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 25, "output_tokens": 1},
                },
            }
        )
        + _invoke_chunk({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        + b"".join(
            _invoke_chunk({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})
            for text in texts
        )
        + _invoke_chunk({"type": "content_block_stop", "index": 0})
        + _invoke_chunk(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4000}}
        )
        + _invoke_chunk({"type": "message_stop"})
    )
    _feed(_stream_collector(INVOKE_STREAM_ENDPOINT), stream)

    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        collector = _stream_collector(INVOKE_STREAM_ENDPOINT)
        _feed(collector, stream)
        retained = tracemalloc.get_traced_memory()[0] - base
    finally:
        tracemalloc.stop()

    assert retained < len(stream) // 4

    response = collector.build_logged_response(_stream_logging_obj(INVOKE_STREAM_ENDPOINT))
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "".join(texts)
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (25, 4000)


def test_stream_collector_logs_nothing_for_an_unrecognized_endpoint():
    collector = BedrockPassthroughConfig().create_stream_collector(
        model=CONVERSE_MODEL, custom_llm_provider="bedrock", endpoint=f"/model/{CONVERSE_MODEL}/rerank"
    )
    collector.add(_event_frame("messageStart", {"role": "assistant"}))

    assert collector.build_logged_response(_converse_stream_logging_obj()) is None
