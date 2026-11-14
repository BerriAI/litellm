import base64
import gc
import json
import struct
import sys
import types
from binascii import crc32
from collections.abc import Callable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import PassthroughStreamCollector
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.passthrough.transformation import (
    BedrockPassthroughConfig,
    is_bedrock_streaming_endpoint,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.passthrough.main import AsyncPassthroughStreamingResponse, PassthroughStreamingResponse
from litellm.types.utils import EmbeddingResponse, ModelResponse

CONVERSE_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
CONVERSE_STREAM_ENDPOINT = f"/model/{CONVERSE_MODEL}/converse-stream"
INVOKE_STREAM_ENDPOINT = f"/model/{CONVERSE_MODEL}/invoke-with-response-stream"


@pytest.mark.asyncio
async def test_bedrock_passthrough_router_replaces_alias_in_endpoint() -> None:
    deployment_model: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"generation": "ok", "stop_reason": "stop"}, request=request)

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "special-bedrock-model",
                "litellm_params": {
                    "model": f"bedrock/{deployment_model}",
                    "aws_access_key_id": "AKIAROUTERKEY",
                    "aws_secret_access_key": "router-secret",
                    "aws_region_name": "us-west-2",
                },
            }
        ]
    )
    response: Final = await router.allm_passthrough_route(
        model="special-bedrock-model",
        method="POST",
        endpoint="/model/special-bedrock-model/invoke",
        data={"prompt": "Hey"},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(respond)),
    )
    request: Final = requests[0]

    assert response.status_code == 200
    assert request.url.host == "bedrock-runtime.us-west-2.amazonaws.com"
    assert request.url.raw_path == f"/model/{deployment_model}/invoke".encode()
    request_body: Final = json.loads(request.content)

    assert request_body == {"prompt": "Hey"}


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("converse-stream", True),
        ("invoke-with-response-stream", True),
        ("converse", False),
        ("invoke", False),
        ("model/my-converse-stream-model/converse", False),
        ("model/x/converse-stream?foo=1", True),
        ("model/x/converse-stream/", True),
    ],
)
def test_is_bedrock_streaming_endpoint_matches_final_action_segment(endpoint: str, expected: bool) -> None:
    assert is_bedrock_streaming_endpoint(endpoint) is expected


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
        assert str(url) == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke"
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
            litellm_params={"aws_bedrock_runtime_endpoint": "http://proxy.com/bedrockproxy"},
        )

        # Verify get_runtime_endpoint was called with correct parameters
        mock_get_runtime.assert_called_once_with(
            api_base="http://proxy.com/bedrockproxy",
            aws_bedrock_runtime_endpoint="http://proxy.com/bedrockproxy",
            aws_region_name="us-west-2",
            endpoint_type="runtime",
        )

        # Verify URL construction preserves the proxy path
        assert str(url) == "http://proxy.com/bedrockproxy/model/anthropic.claude-3-sonnet/invoke"
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
    model_id = "arn:aws:bedrock:eu-west-1:123456789:application-inference-profile/abcdefgh1234"
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
        assert "application-inference-profile%2F" in url_str, f"Expected encoded ARN in URL, but got: {url_str}"
        assert model not in url_str, f"Model name should be replaced by model_id, but got: {url_str}"
        assert "/invoke" in url_str, "Expected /invoke action in URL"

        # Verify the complete URL structure with encoded ARN
        encoded_model_id = "arn:aws:bedrock:eu-west-1:123456789:application-inference-profile%2Fabcdefgh1234"
        expected_url = f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{encoded_model_id}/invoke"
        assert url_str == expected_url, f"Expected {expected_url}, but got: {url_str}"


def test_bedrock_passthrough_with_inference_profile_converse_endpoint():
    """Test Application Inference Profile with converse endpoint and proper ARN encoding"""
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-20250514-v1:0"
    model_id = "arn:aws:bedrock:us-east-1:123456789:application-inference-profile/xyz123"
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
        assert model in url_str, f"Expected model name in URL when model_id not provided, but got: {url_str}"
        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model}/invoke"
        assert url_str == expected_url


def test_bedrock_passthrough_region_extraction_from_inference_profile_arn():
    """Test that AWS region is correctly extracted from Application Inference Profile ARN"""
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-20250514-v1:0"
    # ARN contains us-west-2 region
    model_id = "arn:aws:bedrock:us-west-2:123456789:application-inference-profile/test123"
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
            litellm_params={"model_id": model_id},  # Region should be extracted from ARN
        )

        # Verify that the region from ARN is used in the base URL
        assert "us-west-2" in api_base, f"Expected region 'us-west-2' from ARN in base URL, but got: {api_base}"


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
        assert "application-inference-profile%2F" in url_str, (
            f"Expected encoded ARN with %2F in URL, but got: {url_str}"
        )

        # The unencoded version should NOT be in the URL
        assert "application-inference-profile/" not in url_str, (
            f"ARN slash should be encoded, but found unencoded version in: {url_str}"
        )

        # Verify the complete expected URL structure
        expected_encoded_model_id = (
            "arn:aws:bedrock:us-east-1:590183661440:application-inference-profile%2Fb943q2qbl3m7"
        )
        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{expected_encoded_model_id}/converse"
        assert url_str == expected_url, f"Expected {expected_url}, but got: {url_str}"


def test_bedrock_passthrough_model_id_arn_encoding_invoke_endpoint():
    """
    Test ARN encoding with /invoke endpoint (not just /converse).
    """
    config = BedrockPassthroughConfig()

    model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_id = "arn:aws:bedrock:us-east-1:123456789:application-inference-profile/xyz789"
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

        expected_encoded_model_id = "arn:aws:bedrock:us-east-1:123456789:application-inference-profile%2Fxyz789"
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

        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        assert url_str == expected_url


def _event_frame(event_type: str, payload: dict) -> bytes:
    def header(name: str, value: str) -> bytes:
        name_b, value_b = name.encode(), value.encode()
        return (
            struct.pack("!B", len(name_b)) + name_b + struct.pack("!B", 7) + struct.pack("!H", len(value_b)) + value_b
        )

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


def _stream_logging_obj(
    endpoint: str,
    dynamic_async_success_callbacks: list[CustomLogger] | None = None,
    model: str = CONVERSE_MODEL,
) -> Logging:
    logging_obj = Logging(
        model=f"bedrock/{model}",
        messages=[],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime(2025, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
        dynamic_async_success_callbacks=dynamic_async_success_callbacks,
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={})
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


_SHARED_OBJECT_TYPES: Final = (types.ModuleType, type, types.FunctionType, types.MethodType, types.CodeType, property)
_SMALL_TOKENS: Final = 4000
_LARGE_TOKENS: Final = 8000


def _bytes_reachable_from(root: object) -> int:
    seen: Final[set[int]] = set()
    pending: Final[list[object]] = [root]
    sizes: Final[list[int]] = []
    while pending:
        obj = pending.pop()
        if id(obj) in seen or isinstance(obj, _SHARED_OBJECT_TYPES):
            continue
        seen.add(id(obj))
        sizes.append(sys.getsizeof(obj))
        pending.extend(gc.get_referents(obj))
    return sum(sizes)


def _converse_text_stream(texts: list[str]) -> bytes:
    return (
        _event_frame("messageStart", {"role": "assistant"})
        + _text_block(0, texts)
        + _stream_tail("end_turn", len(texts))
    )


def _retained_growth(build_stream: Callable[[list[str]], bytes], endpoint: str) -> tuple[int, int, list[str]]:
    small_texts: Final = [f"tok{i} " for i in range(_SMALL_TOKENS)]
    large_texts: Final = [f"tok{i} " for i in range(_LARGE_TOKENS)]
    small_stream, large_stream = build_stream(small_texts), build_stream(large_texts)
    small, large = _stream_collector(endpoint), _stream_collector(endpoint)
    _feed(small, small_stream)
    _feed(large, large_stream)
    growth: Final = _bytes_reachable_from(large) - _bytes_reachable_from(small)
    return growth, len(large_stream) - len(small_stream), large_texts


def test_converse_stream_collector_keeps_usage_without_retaining_the_stream():
    growth, extra_stream_bytes, texts = _retained_growth(_converse_text_stream, CONVERSE_STREAM_ENDPOINT)
    extra_text_bytes = sum(len(text) for text in texts[_SMALL_TOKENS:])

    assert extra_text_bytes <= growth < extra_stream_bytes // 4, (growth, extra_text_bytes, extra_stream_bytes)

    collector = _converse_stream_collector()
    _feed(collector, _converse_text_stream(texts))
    response = collector.build_logged_response(_converse_stream_logging_obj())
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "".join(texts)
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (25, _LARGE_TOKENS)
    assert (
        litellm.completion_cost(
            model=f"bedrock/{CONVERSE_MODEL}",
            completion_response=response,
        )
        > 0
    )


def test_converse_stream_cut_before_usage_estimates_prompt_tokens_from_the_request():
    logging_obj: Final = _converse_stream_logging_obj()
    logging_obj.model_call_details["input"] = {
        "system": [{"text": "You are a terse assistant."}],
        "messages": [
            {"role": "user", "content": [{"text": "Write a story about a lighthouse keeper."}]},
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t1", "name": "lookup", "input": {"query": "harbor log"}}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t1",
                            "content": [{"text": "calm water"}, {"json": {"wind": "steady"}}],
                        }
                    }
                ],
            },
        ],
    }
    cut: Final = _converse_stream_collector()
    _feed(cut, _event_frame("messageStart", {"role": "assistant"}) + _text_block(0, ["Once ", "upon"]))
    response: Final = cut.build_logged_response(logging_obj)
    assert isinstance(response, ModelResponse)
    expected_messages: Final = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Write a story about a lighthouse keeper."},
        {"role": "user", "content": '{"query": "harbor log"}'},
        {"role": "user", "content": 'calm water\n{"wind": "steady"}'},
    ]
    assert response.usage.prompt_tokens == litellm.token_counter(model=logging_obj.model, messages=expected_messages)
    assert response.usage.prompt_tokens > 0

    for request_without_turns in ("not a Converse request", {"prompt": "a model-native invoke body"}):
        logging_obj.model_call_details["input"] = request_without_turns
        unestimated = cut.build_logged_response(logging_obj)
        assert isinstance(unestimated, ModelResponse)
        assert (unestimated.usage.prompt_tokens, unestimated.usage.completion_tokens) == (
            0,
            response.usage.completion_tokens,
        ), request_without_turns

    full: Final = _converse_stream_collector()
    _feed(full, _converse_text_stream(["Once ", "upon"]))
    full_response: Final = full.build_logged_response(logging_obj)
    assert isinstance(full_response, ModelResponse)
    assert full_response.usage.prompt_tokens == 25


class _CaptureAsyncSuccessCallback(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs: dict[str, object] | None = None

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.kwargs = kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "stream_kind"),
    [
        (CONVERSE_STREAM_ENDPOINT, "converse"),
        (INVOKE_STREAM_ENDPOINT, "invoke"),
    ],
)
async def test_bedrock_stream_passthrough_logs_cost_and_standard_metadata(endpoint: str, stream_kind: str) -> None:
    callback: Final = _CaptureAsyncSuccessCallback()
    logging_obj: Final = _stream_logging_obj(endpoint, [callback])
    collector: Final = _stream_collector(endpoint)
    stream: Final = (
        _converse_text_stream(["Converse response"])
        if stream_kind == "converse"
        else _invoke_text_stream(["Invoke response"])
    )
    _feed(collector, stream, chunk_size=7)

    await logging_obj.async_flush_passthrough_collected_chunks(collector=collector)

    callback_kwargs: Final = callback.kwargs
    assert callback_kwargs is not None
    assert isinstance(callback_kwargs["response_cost"], (int, float)), logging_obj.model_call_details.get(
        "response_cost_failure_debug_information"
    )
    assert callback_kwargs["response_cost"] > 0
    standard_logging_object: Final = callback_kwargs["standard_logging_object"]
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["model"] == f"bedrock/{CONVERSE_MODEL}"


@pytest.mark.asyncio
async def test_bedrock_stream_flush_says_whether_it_logged_the_stream() -> None:
    stream: Final = _converse_text_stream(["Converse response"])
    part_of_the_first_frame: Final = _stream_collector(CONVERSE_STREAM_ENDPOINT)
    _feed(part_of_the_first_frame, stream[:12])
    whole_stream: Final = _stream_collector(CONVERSE_STREAM_ENDPOINT)
    _feed(whole_stream, stream)
    nothing_logged: Final = _CaptureAsyncSuccessCallback()
    logged: Final = _CaptureAsyncSuccessCallback()

    assert (
        await _stream_logging_obj(CONVERSE_STREAM_ENDPOINT, [nothing_logged]).async_flush_passthrough_collected_chunks(
            collector=part_of_the_first_frame
        )
        is False
    )
    assert nothing_logged.kwargs is None
    assert (
        await _stream_logging_obj(CONVERSE_STREAM_ENDPOINT, [logged]).async_flush_passthrough_collected_chunks(
            collector=whole_stream
        )
        is True
    )
    assert logged.kwargs is not None


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


def _invoke_text_stream(texts: list[str]) -> bytes:
    return (
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
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": len(texts)}}
        )
        + _invoke_chunk({"type": "message_stop"})
    )


def test_invoke_stream_collector_keeps_usage_without_retaining_the_stream():
    growth, extra_stream_bytes, texts = _retained_growth(_invoke_text_stream, INVOKE_STREAM_ENDPOINT)
    extra_text_bytes = sum(len(text) for text in texts[_SMALL_TOKENS:])

    assert extra_text_bytes <= growth < extra_stream_bytes // 4, (growth, extra_text_bytes, extra_stream_bytes)

    collector = _stream_collector(INVOKE_STREAM_ENDPOINT)
    _feed(collector, _invoke_text_stream(texts))
    response = collector.build_logged_response(_stream_logging_obj(INVOKE_STREAM_ENDPOINT))
    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "".join(texts)
    assert response.choices[0].finish_reason == "stop"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (25, _LARGE_TOKENS)
    assert (
        litellm.completion_cost(
            model=f"bedrock/{CONVERSE_MODEL}",
            completion_response=response,
        )
        > 0
    )


def test_stream_collector_logs_nothing_for_an_unrecognized_endpoint():
    collector = BedrockPassthroughConfig().create_stream_collector(
        model=CONVERSE_MODEL, custom_llm_provider="bedrock", endpoint=f"/model/{CONVERSE_MODEL}/rerank"
    )
    collector.add(_event_frame("messageStart", {"role": "assistant"}))

    assert collector.build_logged_response(_converse_stream_logging_obj()) is None


TIERED_MODEL: Final = "amazon.nova-pro-v1:0"
TIERED_INVOKE_ENDPOINT: Final = f"/model/{TIERED_MODEL}/invoke"
TIERED_INVOKE_STREAM_ENDPOINT: Final = f"/model/{TIERED_MODEL}/invoke-with-response-stream"
SERVICE_TIER_HEADER: Final = "x-amzn-bedrock-service-tier"
TIERED_PROMPT_TOKENS: Final = 25
TIERED_COMPLETION_TOKENS: Final = 12
SERVED_TIER_PRICES: Final = (
    pytest.param({SERVICE_TIER_HEADER: "priority"}, "_priority", id="priority"),
    pytest.param({}, "", id="standard"),
)


def _tier_priced_cost(price_suffix: str) -> float:
    prices: Final = litellm.model_cost[TIERED_MODEL]
    return (
        TIERED_PROMPT_TOKENS * prices[f"input_cost_per_token{price_suffix}"]
        + TIERED_COMPLETION_TOKENS * prices[f"output_cost_per_token{price_suffix}"]
    )


def _nova_invoke_text_stream(texts: list[str]) -> bytes:
    return b"".join(
        _invoke_chunk(event)
        for event in (
            {"messageStart": {"role": "assistant"}},
            *({"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}} for text in texts),
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {
                        "inputTokens": TIERED_PROMPT_TOKENS,
                        "outputTokens": TIERED_COMPLETION_TOKENS,
                        "totalTokens": TIERED_PROMPT_TOKENS + TIERED_COMPLETION_TOKENS,
                    },
                    "metrics": {"latencyMs": 120},
                }
            },
        )
    )


def _provider_stream_reply(reply_headers: Mapping[str, str], stream: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/vnd.amazon.eventstream", **reply_headers},
        content=stream,
        request=httpx.Request(
            "POST", f"https://bedrock-runtime.us-east-1.amazonaws.com{TIERED_INVOKE_STREAM_ENDPOINT}"
        ),
    )


@pytest.mark.parametrize(("reply_headers", "price_suffix"), SERVED_TIER_PRICES)
def test_invoke_reply_is_priced_at_the_service_tier_bedrock_served(
    reply_headers: Mapping[str, str], price_suffix: str
) -> None:
    reply: Final = httpx.Response(
        200,
        json={
            "output": {"message": {"role": "assistant", "content": [{"text": "Invoke response"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": TIERED_PROMPT_TOKENS,
                "outputTokens": TIERED_COMPLETION_TOKENS,
                "totalTokens": TIERED_PROMPT_TOKENS + TIERED_COMPLETION_TOKENS,
            },
        },
        headers=reply_headers,
    )

    logged: Final = BedrockPassthroughConfig().logging_non_streaming_response(
        model=TIERED_MODEL,
        custom_llm_provider="bedrock",
        httpx_response=reply,
        request_data={},
        logging_obj=_stream_logging_obj(TIERED_INVOKE_ENDPOINT, model=TIERED_MODEL),
        endpoint=TIERED_INVOKE_ENDPOINT,
    )

    assert litellm.completion_cost(model=f"bedrock/{TIERED_MODEL}", completion_response=logged) == pytest.approx(
        _tier_priced_cost(price_suffix)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply_headers", "price_suffix"), SERVED_TIER_PRICES)
async def test_invoke_stream_relay_is_priced_at_the_service_tier_bedrock_served(
    reply_headers: Mapping[str, str], price_suffix: str
) -> None:
    callback: Final = _CaptureAsyncSuccessCallback()
    stream: Final = _nova_invoke_text_stream(["Invoke ", "response"])
    reply: Final = _provider_stream_reply(reply_headers, stream)

    async def provider_reply() -> httpx.Response:
        return reply

    relay: Final = AsyncPassthroughStreamingResponse(
        response=provider_reply(),
        litellm_logging_obj=_stream_logging_obj(TIERED_INVOKE_STREAM_ENDPOINT, [callback], model=TIERED_MODEL),
        provider_config=BedrockPassthroughConfig(),
    )
    relayed: Final = b"".join([chunk async for chunk in relay])

    assert relayed == stream
    assert await relay.wait_for_spend_flush() is True
    assert callback.kwargs is not None
    assert callback.kwargs["response_cost"] == pytest.approx(_tier_priced_cost(price_suffix))


class _ImmediateExecutor:
    def submit(self, fn: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        fn(*args, **kwargs)


@pytest.mark.parametrize(("reply_headers", "price_suffix"), SERVED_TIER_PRICES)
def test_sync_invoke_stream_relay_is_priced_at_the_service_tier_bedrock_served(
    reply_headers: Mapping[str, str], price_suffix: str
) -> None:
    logging_obj: Final = _stream_logging_obj(TIERED_INVOKE_STREAM_ENDPOINT, model=TIERED_MODEL)
    stream: Final = _nova_invoke_text_stream(["Invoke ", "response"])

    with patch("litellm.utils.executor", _ImmediateExecutor()):
        relayed: Final = b"".join(
            PassthroughStreamingResponse(
                _provider_stream_reply(reply_headers, stream), logging_obj, BedrockPassthroughConfig()
            )
        )

    assert relayed == stream
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(_tier_priced_cost(price_suffix))


REQUEST_METADATA_HEADER = "x-amzn-bedrock-request-metadata"
REQUEST_METADATA_FIELDS = ["user_api_key_alias", "user_api_key_team_alias", "spend_logs_metadata"]
PROXY_IDENTITY = {
    "user_api_key_alias": "prod-key",
    "user_api_key_team_alias": "platform",
    "spend_logs_metadata": {"agent": "triage-bot"},
}
EXPECTED_REQUEST_METADATA = {
    "user_api_key_alias": "prod-key",
    "user_api_key_team_alias": "platform",
    "agent": "triage-bot",
}
CONVERSE_BODY = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
INVOKE_BODY = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 8,
    "messages": [{"role": "user", "content": "hi"}],
}


async def _sent_passthrough_request(
    action: str,
    body: dict[str, object],
    identity: dict[str, object],
    deployment_params: Mapping[str, object] = MappingProxyType({}),
    **kwargs: object,
) -> httpx.Request:
    requests: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-pt",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "aws_access_key_id": "AKIAROUTERKEY",
                    "aws_secret_access_key": "router-secret",
                    "aws_region_name": "us-west-2",
                    **deployment_params,
                },
            }
        ]
    )
    await router.allm_passthrough_route(
        model="bedrock-pt",
        method="POST",
        endpoint=f"/model/bedrock-pt/{action}",
        data=body,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(respond)),
        litellm_metadata=identity,
        **kwargs,
    )
    return requests[0]


def _signed_header_names(request: httpx.Request) -> frozenset[str]:
    return frozenset(request.headers["authorization"].split("SignedHeaders=")[1].split(",")[0].split(";"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "query_params"),
    [("converse", None), ("converse-stream", None), ("converse/", None), ("converse", {"trace": "1"})],
)
async def test_converse_passthrough_signs_proxy_identity_into_request_metadata(
    action: str, query_params: dict[str, str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)
    caller_body: Final = {**CONVERSE_BODY, "requestMetadata": {"user_api_key_team_alias": "forged", "ticket": "T-42"}}

    request: Final = await _sent_passthrough_request(
        action,
        caller_body,
        PROXY_IDENTITY,
        request_query_params=query_params,
        request_headers={f"x-pass-{REQUEST_METADATA_HEADER.upper()}": '{"user_api_key_team_alias":"forged"}'},
    )

    assert json.loads(request.content) == {
        **CONVERSE_BODY,
        "requestMetadata": {**EXPECTED_REQUEST_METADATA, "ticket": "T-42"},
    }
    assert caller_body["requestMetadata"] == {"user_api_key_team_alias": "forged", "ticket": "T-42"}
    assert REQUEST_METADATA_HEADER not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_invoke_passthrough_sends_proxy_identity_as_a_signed_header(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    request: Final = await _sent_passthrough_request(
        action,
        INVOKE_BODY,
        PROXY_IDENTITY,
        request_headers={f"x-pass-{REQUEST_METADATA_HEADER}": '{"user_api_key_team_alias":"forged"}'},
    )

    assert json.loads(request.headers[REQUEST_METADATA_HEADER]) == EXPECTED_REQUEST_METADATA
    assert REQUEST_METADATA_HEADER in _signed_header_names(request)
    assert json.loads(request.content) == INVOKE_BODY


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_invoke_passthrough_keeps_caller_body_request_metadata_in_the_body(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)
    caller_body: Final = {**INVOKE_BODY, "requestMetadata": {"user_api_key_team_alias": "forged", "ticket": "T-42"}}

    request: Final = await _sent_passthrough_request(
        action,
        caller_body,
        PROXY_IDENTITY,
        request_headers={f"x-pass-{REQUEST_METADATA_HEADER.upper()}": '{"user_api_key_team_alias":"forged"}'},
    )

    assert json.loads(request.content) == {
        **INVOKE_BODY,
        "requestMetadata": {**EXPECTED_REQUEST_METADATA, "ticket": "T-42"},
    }
    assert REQUEST_METADATA_HEADER not in request.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, []])
async def test_passthrough_requests_are_untouched_when_forwarding_is_off(
    setting: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", setting)
    caller_body: Final = {**CONVERSE_BODY, "requestMetadata": {"user_api_key_team_alias": "forged"}}

    converse: Final = await _sent_passthrough_request("converse", caller_body, PROXY_IDENTITY)
    invoke: Final = await _sent_passthrough_request("invoke", INVOKE_BODY, PROXY_IDENTITY)

    assert json.loads(converse.content) == caller_body
    assert REQUEST_METADATA_HEADER not in invoke.headers


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "body"), [("converse", CONVERSE_BODY), ("invoke", INVOKE_BODY)])
async def test_passthrough_drops_reserved_caller_keys_when_nothing_resolves(
    action: str, body: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)
    caller_body: Final = {**body, "requestMetadata": {"user_api_key_team_alias": "forged"}}

    request: Final = await _sent_passthrough_request(action, caller_body, {})

    assert json.loads(request.content) == body
    assert REQUEST_METADATA_HEADER not in request.headers


MALFORMED_REQUEST_METADATA: Final = (
    pytest.param(
        {"user_api_key_team_alias": "forged", "bad!": "x"}, "key 'bad!' must be", "ValidationException", id="bad-key"
    ),
    pytest.param({"ticket": "a\u00a0b"}, "value for key 'ticket' must be", "ValidationException", id="bad-value"),
    pytest.param("ticket=T-42", "must be a dictionary", "SerializationException", id="not-a-dictionary"),
    pytest.param({f"key_{index}": "v" for index in range(17)}, "maximum of 16 items", "ValidationException", id="17"),
    pytest.param({"ticket": 42}, "must be strings", "SerializationException", id="non-string-value"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["converse", "converse-stream"])
@pytest.mark.parametrize(("caller_metadata", "violation", "converse_error_type"), MALFORMED_REQUEST_METADATA)
async def test_malformed_converse_request_metadata_is_rejected_with_the_error_type_bedrock_returns(
    action: str, caller_metadata: object, violation: str, converse_error_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    with pytest.raises(BedrockError, match=violation) as rejected:
        await _sent_passthrough_request(action, {**CONVERSE_BODY, "requestMetadata": caller_metadata}, PROXY_IDENTITY)

    assert rejected.value.status_code == 400
    assert httpx.Headers(rejected.value.headers).get("x-amzn-errortype") == converse_error_type


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
@pytest.mark.parametrize(("caller_metadata", "violation", "converse_error_type"), MALFORMED_REQUEST_METADATA)
async def test_malformed_invoke_request_metadata_is_rejected_as_a_bedrock_validation_error(
    action: str, caller_metadata: object, violation: str, converse_error_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    with pytest.raises(BedrockError, match=violation) as rejected:
        await _sent_passthrough_request(action, {**INVOKE_BODY, "requestMetadata": caller_metadata}, PROXY_IDENTITY)

    assert rejected.value.status_code == 400
    assert httpx.Headers(rejected.value.headers).get("x-amzn-errortype") == "ValidationException"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_invoke_null_request_metadata_is_rejected_like_bedrock_does(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    with pytest.raises(BedrockError, match="must be a dictionary") as rejected:
        await _sent_passthrough_request(action, {**INVOKE_BODY, "requestMetadata": None}, PROXY_IDENTITY)

    assert rejected.value.status_code == 400
    assert httpx.Headers(rejected.value.headers).get("x-amzn-errortype") == "ValidationException"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["converse", "converse-stream"])
async def test_converse_null_request_metadata_is_treated_as_absent_like_bedrock_does(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    request: Final = await _sent_passthrough_request(action, {**CONVERSE_BODY, "requestMetadata": None}, PROXY_IDENTITY)

    assert json.loads(request.content) == {**CONVERSE_BODY, "requestMetadata": EXPECTED_REQUEST_METADATA}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_invoke_metadata_too_long_for_the_header_sends_the_leading_pairs_that_fit(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)
    long_pairs: Final = {f"{index:02d}".ljust(256, "k"): "\n" * 256 for index in range(12)}
    resolved: Final = tuple(
        {"user_api_key_alias": "prod-key", "user_api_key_team_alias": "platform", **long_pairs}.items()
    )

    request: Final = await _sent_passthrough_request(
        action,
        INVOKE_BODY,
        {**PROXY_IDENTITY, "spend_logs_metadata": long_pairs},
        request_headers={f"x-pass-{REQUEST_METADATA_HEADER}": '{"user_api_key_team_alias":"forged"}'},
    )

    header: Final = request.headers[REQUEST_METADATA_HEADER]
    sent: Final = json.loads(header)
    one_more_pair: Final = json.dumps(dict(resolved[: len(sent) + 1]), separators=(",", ":"))
    assert sent == dict(resolved[: len(sent)])
    assert len(header) <= 8500 < len(one_more_pair)
    assert json.loads(request.content) == INVOKE_BODY


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "body"), [("converse", CONVERSE_BODY), ("invoke", INVOKE_BODY)])
@pytest.mark.parametrize("drop_globally", [True, False])
async def test_malformed_caller_request_metadata_is_dropped_when_params_are_dropped(
    action: str, body: dict[str, object], drop_globally: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)
    monkeypatch.setattr(litellm, "drop_params", drop_globally)
    caller_body: Final = {**body, "requestMetadata": {"user_api_key_team_alias": "forged", "bad!": "x"}}

    dropped: Final = await _sent_passthrough_request(
        action, caller_body, PROXY_IDENTITY, **({} if drop_globally else {"drop_params": True})
    )

    assert json.loads(dropped.content) == {**body, "requestMetadata": EXPECTED_REQUEST_METADATA}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "body", "caller_metadata"),
    [
        ("converse", CONVERSE_BODY, {"user_api_key_team_alias": "caller-team", "bad!": "x"}),
        ("invoke", INVOKE_BODY, {"user_api_key_team_alias": "caller-team", "bad!": "x"}),
        ("invoke", INVOKE_BODY, None),
    ],
)
@pytest.mark.parametrize("drop_globally", [True, False])
async def test_passthrough_body_is_sent_as_is_when_params_are_dropped_and_forwarding_is_off(
    action: str,
    body: dict[str, object],
    caller_metadata: dict[str, str] | None,
    drop_globally: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    monkeypatch.setattr(litellm, "drop_params", drop_globally)
    caller_body: Final = {**body, "requestMetadata": caller_metadata}

    sent: Final = await _sent_passthrough_request(
        action, caller_body, PROXY_IDENTITY, **({} if drop_globally else {"drop_params": True})
    )

    assert json.loads(sent.content) == caller_body
    assert REQUEST_METADATA_HEADER not in sent.headers


DEPLOYMENT_GUARDRAIL: Final[Mapping[str, str]] = {
    "guardrailIdentifier": "deployment-guardrail",
    "guardrailVersion": "3",
    "trace": "enabled",
    "streamProcessingMode": "sync",
}
DEPLOYMENT_INVOKE_GUARDRAIL_HEADERS: Final[Mapping[str, str]] = {
    "x-amzn-bedrock-guardrailidentifier": "deployment-guardrail",
    "x-amzn-bedrock-guardrailversion": "3",
    "x-amzn-bedrock-trace": "ENABLED",
}
CALLER_GUARDRAIL: Final[Mapping[str, str]] = {
    "guardrailIdentifier": "caller-guardrail",
    "guardrailVersion": "9",
    "trace": "disabled",
}
CALLER_INVOKE_GUARDRAIL_HEADERS: Final[Mapping[str, str]] = {
    "x-pass-X-Amzn-Bedrock-GuardrailIdentifier": "caller-guardrail",
    "x-pass-X-Amzn-Bedrock-GuardrailVersion": "9",
    "x-pass-X-Amzn-Bedrock-Trace": "DISABLED",
}


def _invoke_guardrail_headers(request: httpx.Request) -> dict[str, str]:
    return {
        name: value
        for name, value in request.headers.items()
        if name.lower().startswith("x-amzn-bedrock-guardrail") or name.lower() == "x-amzn-bedrock-trace"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "sent_guardrail"),
    [
        ("converse", {"guardrailIdentifier": "deployment-guardrail", "guardrailVersion": "3", "trace": "enabled"}),
        ("converse-stream", DEPLOYMENT_GUARDRAIL),
    ],
)
async def test_converse_passthrough_sends_the_deployment_guardrail_in_the_body(
    action: str, sent_guardrail: Mapping[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    request: Final = await _sent_passthrough_request(
        action, CONVERSE_BODY, {}, deployment_params={"guardrailConfig": DEPLOYMENT_GUARDRAIL}
    )

    assert json.loads(request.content) == {**CONVERSE_BODY, "guardrailConfig": sent_guardrail}
    assert _invoke_guardrail_headers(request) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_invoke_passthrough_sends_the_deployment_guardrail_as_signed_headers(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", REQUEST_METADATA_FIELDS)

    request: Final = await _sent_passthrough_request(
        action, INVOKE_BODY, PROXY_IDENTITY, deployment_params={"guardrailConfig": DEPLOYMENT_GUARDRAIL}
    )

    assert _invoke_guardrail_headers(request) == DEPLOYMENT_INVOKE_GUARDRAIL_HEADERS
    assert {*DEPLOYMENT_INVOKE_GUARDRAIL_HEADERS, REQUEST_METADATA_HEADER} <= _signed_header_names(request)
    assert json.loads(request.headers[REQUEST_METADATA_HEADER]) == EXPECTED_REQUEST_METADATA
    assert json.loads(request.content) == INVOKE_BODY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "caller_guardrail"),
    [("converse", CALLER_GUARDRAIL), ("converse-stream", {**CALLER_GUARDRAIL, "streamProcessingMode": "async"})],
)
async def test_deployment_guardrail_replaces_the_caller_converse_guardrail(
    action: str, caller_guardrail: Mapping[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    request: Final = await _sent_passthrough_request(
        action,
        {**CONVERSE_BODY, "guardrailConfig": caller_guardrail},
        {},
        deployment_params={"guardrailConfig": {"guardrailIdentifier": "deployment-guardrail"}},
        request_headers=CALLER_INVOKE_GUARDRAIL_HEADERS,
    )

    assert json.loads(request.content) == {
        **CONVERSE_BODY,
        "guardrailConfig": {"guardrailIdentifier": "deployment-guardrail"},
    }
    assert _invoke_guardrail_headers(request) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
async def test_deployment_guardrail_replaces_every_invoke_guardrail_header_the_caller_sent(
    action: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    request: Final = await _sent_passthrough_request(
        action,
        INVOKE_BODY,
        {},
        deployment_params={"guardrailConfig": {"guardrailIdentifier": "deployment-guardrail"}},
        request_headers=CALLER_INVOKE_GUARDRAIL_HEADERS,
    )

    assert _invoke_guardrail_headers(request) == {"x-amzn-bedrock-guardrailidentifier": "deployment-guardrail"}
    assert json.loads(request.content) == INVOKE_BODY


@pytest.mark.asyncio
async def test_passthrough_without_a_deployment_guardrail_sends_the_caller_guardrail_as_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    caller_body: Final = {**CONVERSE_BODY, "guardrailConfig": CALLER_GUARDRAIL}

    converse: Final = await _sent_passthrough_request("converse", caller_body, {})
    invoke: Final = await _sent_passthrough_request(
        "invoke", INVOKE_BODY, {}, request_headers=CALLER_INVOKE_GUARDRAIL_HEADERS
    )

    assert json.loads(converse.content) == caller_body
    assert _invoke_guardrail_headers(invoke) == {
        "x-amzn-bedrock-guardrailidentifier": "caller-guardrail",
        "x-amzn-bedrock-guardrailversion": "9",
        "x-amzn-bedrock-trace": "DISABLED",
    }


CALLER_GUARDRAIL_IDENTITY_HEADERS: Final[Mapping[str, str]] = {
    "x-pass-X-Amzn-Bedrock-GuardrailIdentifier": "caller-guardrail",
    "x-pass-X-Amzn-Bedrock-GuardrailVersion": "9",
}
SENT_CALLER_GUARDRAIL_IDENTITY: Final[Mapping[str, str]] = {
    "x-amzn-bedrock-guardrailidentifier": "caller-guardrail",
    "x-amzn-bedrock-guardrailversion": "9",
}


SDK_INVOKE_MODEL: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _sdk_passthrough_call(
    model: str, action: str, request_headers: Mapping[str, str], reply_body: Mapping[str, object]
) -> tuple[httpx.Request, object]:
    sent: Final[list[httpx.Request]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=reply_body, request=request)

    response: Final = litellm.llm_passthrough_route(
        model=f"bedrock/{model}",
        method="POST",
        endpoint=f"/model/{model}/{action}",
        json=INVOKE_BODY,
        request_headers=dict(request_headers),
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        aws_access_key_id="AKIASDKKEY",
        aws_secret_access_key="sdk-secret",
        aws_region_name="us-east-1",
    )
    return sent[0], response


@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
@pytest.mark.parametrize("caller_trace", [None, "DISABLED", "enabled"])
def test_sdk_invoke_passthrough_sends_the_caller_guardrail_trace_exactly_as_sent(
    action: str, caller_trace: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    caller_trace_header: Final = {} if caller_trace is None else {"x-pass-X-Amzn-Bedrock-Trace": caller_trace}
    sent_trace_header: Final = {} if caller_trace is None else {"x-amzn-bedrock-trace": caller_trace}

    request, _ = _sdk_passthrough_call(
        SDK_INVOKE_MODEL, action, {**CALLER_GUARDRAIL_IDENTITY_HEADERS, **caller_trace_header}, {"ok": True}
    )

    assert _invoke_guardrail_headers(request) == {**SENT_CALLER_GUARDRAIL_IDENTITY, **sent_trace_header}
    assert json.loads(request.content) == INVOKE_BODY


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["invoke", "invoke-with-response-stream"])
@pytest.mark.parametrize("guardrail_config", ["deployment-guardrail", {"guardrailVersion": "3"}])
async def test_invoke_passthrough_refuses_a_deployment_guardrail_it_cannot_send(
    action: str, guardrail_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    with pytest.raises(BedrockError, match="guardrailConfig") as rejected:
        await _sent_passthrough_request(
            action, INVOKE_BODY, {}, deployment_params={"guardrailConfig": guardrail_config}
        )

    assert rejected.value.status_code == 400


@pytest.mark.asyncio
async def test_converse_passthrough_forwards_a_non_object_deployment_guardrail_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)

    request: Final = await _sent_passthrough_request(
        "converse", CONVERSE_BODY, {}, deployment_params={"guardrailConfig": "deployment-guardrail"}
    )

    assert json.loads(request.content) == {**CONVERSE_BODY, "guardrailConfig": "deployment-guardrail"}


@pytest.mark.asyncio
async def test_passthrough_leaves_actions_without_a_guardrail_field_as_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    count_tokens_body: Final = {"input": {"converse": CONVERSE_BODY}}

    request: Final = await _sent_passthrough_request(
        "count-tokens", count_tokens_body, {}, deployment_params={"guardrailConfig": DEPLOYMENT_GUARDRAIL}
    )

    assert json.loads(request.content) == count_tokens_body
    assert _invoke_guardrail_headers(request) == {}


GUARDRAIL_UNIT_PRICE: Final = 0.00015
GUARDRAIL_PRICING: Final = {
    "litellm_provider": "bedrock",
    "mode": "guardrail",
    "guardrail_cost_per_unit": {"topicPolicyUnits": GUARDRAIL_UNIT_PRICE, "contentPolicyUnits": GUARDRAIL_UNIT_PRICE},
}
_ASSESSMENT_USAGE: Final = {"invocationMetrics": {"usage": {"topicPolicyUnits": 1, "contentPolicyUnits": 1}}}
INVOKE_GUARDRAIL_TRACE: Final = {
    "guardrail": {
        "input": {"caller-guardrail": _ASSESSMENT_USAGE},
        "outputs": [{"caller-guardrail": _ASSESSMENT_USAGE}],
        "actionReason": "No action.",
    }
}
BILLED_GUARDRAIL_USAGE: Final = {"topicPolicyUnits": 2, "contentPolicyUnits": 2}
BILLED_GUARDRAIL_COST: Final = 4 * GUARDRAIL_UNIT_PRICE
NOVA_INVOKE_MODEL: Final = "us.amazon.nova-2-lite-v1:0"
LLAMA_INVOKE_MODEL: Final = "meta.llama3-8b-instruct-v1:0"
INVOKE_METRICS: Final = {
    "inputTokenCount": 25,
    "outputTokenCount": 12,
    "invocationLatency": 300,
    "firstByteLatency": 90,
}

INVOKE_REPLIES_WITH_A_GUARDRAIL_TRACE: Final = (
    pytest.param(
        NOVA_INVOKE_MODEL,
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 25, "outputTokens": 12, "totalTokens": 37},
            "amazon-bedrock-guardrailAction": "NONE",
        },
        id="nova",
    ),
    pytest.param(
        LLAMA_INVOKE_MODEL,
        {"generation": "hi", "prompt_token_count": 25, "generation_token_count": 12, "stop_reason": "stop"},
        id="llama",
    ),
    pytest.param(
        "mistral.mistral-7b-instruct-v0:2", {"outputs": [{"text": "hi", "stop_reason": "stop"}]}, id="mistral"
    ),
    pytest.param(
        CONVERSE_MODEL,
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": CONVERSE_MODEL,
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 25, "output_tokens": 12},
        },
        id="claude",
    ),
)

INVOKE_STREAMS_WITH_A_GUARDRAIL_TRACE: Final = (
    pytest.param(
        NOVA_INVOKE_MODEL,
        (
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 25, "outputTokens": 12, "totalTokens": 37},
                    "metrics": {},
                    "trace": {},
                },
                "amazon-bedrock-invocationMetrics": INVOKE_METRICS,
                "amazon-bedrock-guardrailAction": "NONE",
                "amazon-bedrock-trace": INVOKE_GUARDRAIL_TRACE,
            },
        ),
        id="nova",
    ),
    pytest.param(
        LLAMA_INVOKE_MODEL,
        (
            {"generation": "hi", "prompt_token_count": 25, "generation_token_count": 1, "stop_reason": None},
            {
                "generation": "",
                "prompt_token_count": None,
                "generation_token_count": 12,
                "stop_reason": "stop",
                "amazon-bedrock-invocationMetrics": INVOKE_METRICS,
                "amazon-bedrock-trace": INVOKE_GUARDRAIL_TRACE,
            },
        ),
        id="llama",
    ),
    pytest.param(
        CONVERSE_MODEL,
        (
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
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}},
            {
                "type": "message_stop",
                "amazon-bedrock-invocationMetrics": INVOKE_METRICS,
                "amazon-bedrock-trace": INVOKE_GUARDRAIL_TRACE,
            },
        ),
        id="claude",
    ),
)


def _billed_guardrails(standard_logging_object: object) -> list[tuple[object, object]]:
    assert isinstance(standard_logging_object, dict)
    return [
        (entry["guardrail_usage"], entry["guardrail_cost"])
        for entry in standard_logging_object["guardrail_information"] or []
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "reply_body"), INVOKE_REPLIES_WITH_A_GUARDRAIL_TRACE)
async def test_invoke_reply_bills_the_guardrail_bedrock_traced(
    model: str, reply_body: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(litellm.model_cost, "bedrock/guardrails", GUARDRAIL_PRICING)
    endpoint: Final = f"/model/{model}/invoke"
    callback: Final = _CaptureAsyncSuccessCallback()
    logging_obj: Final = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="allm_passthrough_route",
        start_time=datetime(2025, 1, 1),
        litellm_call_id="call-1",
        function_id="fn-1",
        dynamic_async_success_callbacks=[callback],
    )
    logging_obj.update_environment_variables(litellm_params={}, optional_params={})
    logging_obj.model_call_details["custom_llm_provider"] = "bedrock"
    logging_obj.model_call_details["endpoint"] = endpoint
    reply: Final = httpx.Response(
        200,
        json={**reply_body, "amazon-bedrock-trace": INVOKE_GUARDRAIL_TRACE},
        request=httpx.Request("POST", f"https://bedrock-runtime.us-east-1.amazonaws.com{endpoint}"),
    )

    await logging_obj.async_success_handler(
        result=reply, start_time=datetime(2025, 1, 1), end_time=datetime(2025, 1, 1, 0, 0, 1)
    )

    assert callback.kwargs is not None
    assert _billed_guardrails(callback.kwargs["standard_logging_object"]) == [
        (BILLED_GUARDRAIL_USAGE, pytest.approx(BILLED_GUARDRAIL_COST))
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "events"), INVOKE_STREAMS_WITH_A_GUARDRAIL_TRACE)
async def test_invoke_stream_relay_bills_the_guardrail_bedrock_traced(
    model: str, events: tuple[dict[str, object], ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(litellm.model_cost, "bedrock/guardrails", GUARDRAIL_PRICING)
    endpoint: Final = f"/model/{model}/invoke-with-response-stream"
    callback: Final = _CaptureAsyncSuccessCallback()
    stream: Final = b"".join(_invoke_chunk(event) for event in events)

    async def provider_reply() -> httpx.Response:
        return _provider_stream_reply({}, stream)

    relay: Final = AsyncPassthroughStreamingResponse(
        response=provider_reply(),
        litellm_logging_obj=_stream_logging_obj(endpoint, [callback], model=model),
        provider_config=BedrockPassthroughConfig(),
    )
    relayed: Final = b"".join([chunk async for chunk in relay])

    assert relayed == stream
    assert await relay.wait_for_spend_flush() is True
    assert callback.kwargs is not None
    assert _billed_guardrails(callback.kwargs["standard_logging_object"]) == [
        (BILLED_GUARDRAIL_USAGE, pytest.approx(BILLED_GUARDRAIL_COST))
    ]


class _CaptureSyncSuccessCallback(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_objects: list[object] = []

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.standard_logging_objects.append(kwargs.get("standard_logging_object"))


@pytest.mark.parametrize(
    ("model", "reply_body"),
    [param for param in INVOKE_REPLIES_WITH_A_GUARDRAIL_TRACE if param.id in {"nova", "claude"}],
)
def test_sdk_invoke_passthrough_bills_the_caller_guardrail_bedrock_traced_once(
    model: str, reply_body: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", None)
    monkeypatch.setitem(litellm.model_cost, "bedrock/guardrails", GUARDRAIL_PRICING)
    callback: Final = _CaptureSyncSuccessCallback()
    monkeypatch.setattr(litellm, "callbacks", [callback])
    traced_reply: Final = {**reply_body, "amazon-bedrock-trace": INVOKE_GUARDRAIL_TRACE}

    with patch("litellm.utils.executor", _ImmediateExecutor()):
        request, response = _sdk_passthrough_call(
            model,
            "invoke",
            {**CALLER_GUARDRAIL_IDENTITY_HEADERS, "x-pass-X-Amzn-Bedrock-Trace": "ENABLED"},
            traced_reply,
        )

    assert isinstance(response, httpx.Response)
    assert response.json() == traced_reply
    assert _invoke_guardrail_headers(request) == {
        **SENT_CALLER_GUARDRAIL_IDENTITY,
        "x-amzn-bedrock-trace": "ENABLED",
    }
    assert [_billed_guardrails(logged) for logged in callback.standard_logging_objects] == [
        [(BILLED_GUARDRAIL_USAGE, pytest.approx(BILLED_GUARDRAIL_COST))]
    ]


TITAN_EMBEDDING_VECTOR: Final = (0.1, 0.2, 0.3)
TITAN_EMBEDDING_INPUT_TOKENS: Final = 20


def _log_titan_invoke_reply(
    model: str,
    body: Mapping[str, object],
    request_data: Mapping[str, object] = MappingProxyType({"inputText": "example text"}),
) -> object:
    endpoint: Final = f"/model/{model}/invoke"
    logging_obj: Final = _stream_logging_obj(endpoint, model=model)
    return BedrockPassthroughConfig().logging_non_streaming_response(
        model=logging_obj.model,
        custom_llm_provider="bedrock",
        httpx_response=httpx.Response(200, json=body),
        request_data=dict(request_data),
        logging_obj=logging_obj,
        endpoint=endpoint,
    )


@pytest.mark.parametrize(
    "model",
    [
        "amazon.titan-embed-text-v2:0",
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-g1-text-02",
        "amazon.titan-embed-image-v1",
    ],
)
def test_invoke_titan_embedding_reply_is_logged_and_priced_as_an_embedding(model: str) -> None:
    logged: Final = _log_titan_invoke_reply(
        model, {"embedding": list(TITAN_EMBEDDING_VECTOR), "inputTextTokenCount": TITAN_EMBEDDING_INPUT_TOKENS}
    )

    assert isinstance(logged, EmbeddingResponse)
    assert tuple(logged.data[0]["embedding"]) == TITAN_EMBEDDING_VECTOR
    assert logged.usage is not None
    assert (logged.usage.prompt_tokens, logged.usage.completion_tokens) == (TITAN_EMBEDDING_INPUT_TOKENS, 0)
    assert litellm.completion_cost(
        model=f"bedrock/{model}", completion_response=logged, call_type="aembedding"
    ) == pytest.approx(TITAN_EMBEDDING_INPUT_TOKENS * litellm.model_cost[model]["input_cost_per_token"])


@pytest.mark.parametrize("encoding", ["binary", "float"])
def test_invoke_titan_reply_carrying_only_embeddings_by_type_is_logged_as_an_embedding(encoding: str) -> None:
    vector: Final = [1, 0, 1]
    logged: Final = _log_titan_invoke_reply(
        "amazon.titan-embed-text-v2:0",
        {"embeddingsByType": {encoding: vector}, "inputTextTokenCount": TITAN_EMBEDDING_INPUT_TOKENS},
    )

    assert isinstance(logged, EmbeddingResponse)
    assert logged.data[0]["embedding"] == vector
    assert logged.usage is not None
    assert logged.usage.prompt_tokens == TITAN_EMBEDDING_INPUT_TOKENS


@pytest.mark.parametrize("text_tokens", [0, TITAN_EMBEDDING_INPUT_TOKENS])
def test_invoke_titan_image_embedding_reply_bills_the_input_image(text_tokens: int) -> None:
    model: Final = "amazon.titan-embed-image-v1"
    logged: Final = _log_titan_invoke_reply(
        model,
        {"embedding": list(TITAN_EMBEDDING_VECTOR), "inputTextTokenCount": text_tokens},
        request_data={"inputImage": "aGVsbG8="},
    )

    assert isinstance(logged, EmbeddingResponse)
    assert logged.usage is not None
    assert logged.usage.prompt_tokens == text_tokens
    assert logged.usage.prompt_tokens_details is not None
    assert logged.usage.prompt_tokens_details.image_count == 1
    prices: Final = litellm.model_cost[model]
    assert litellm.completion_cost(
        model=f"bedrock/{model}", completion_response=logged, call_type="aembedding"
    ) == pytest.approx(text_tokens * prices["input_cost_per_token"] + prices["input_cost_per_image"])


def test_invoke_titan_text_reply_is_still_logged_as_a_chat_completion() -> None:
    logged: Final = _log_titan_invoke_reply(
        "amazon.titan-text-express-v1",
        {"inputTextTokenCount": 3, "results": [{"tokenCount": 2, "outputText": "hi", "completionReason": "FINISH"}]},
    )

    assert isinstance(logged, ModelResponse)
    assert logged.choices[0].message.content == "hi"
