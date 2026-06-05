import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.passthrough.transformation import (
    BedrockPassthroughConfig,
    _BedrockEventStreamChunkProcessor,
)


def test_bedrock_passthrough_get_complete_url_default_endpoint():
    """Test get_complete_url with default AWS endpoint (no override)"""
    config = BedrockPassthroughConfig()

    # Mock the methods following the pattern from test_base_aws_llm.py
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )) as mock_get_runtime:

        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={}
        )

        # Verify get_runtime_endpoint was called with correct parameters
        mock_get_runtime.assert_called_once_with(
            api_base=None,
            aws_bedrock_runtime_endpoint=None,
            aws_region_name="us-east-1",
            endpoint_type="runtime"
        )
        
        # Verify URL construction
        assert str(url) == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet/invoke"
        assert api_base == "https://bedrock-runtime.us-east-1.amazonaws.com"


def test_bedrock_passthrough_get_complete_url_custom_endpoint_no_path():
    """Test get_complete_url with custom endpoint (no base path)"""
    config = BedrockPassthroughConfig()

    with patch.object(config, '_get_aws_region_name', return_value="us-west-2"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "http://proxy.com",
             "http://proxy.com"
         )) as mock_get_runtime:

        url, api_base = config.get_complete_url(
            api_base="http://proxy.com",
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={}
        )

        # Verify get_runtime_endpoint was called with the api_base
        mock_get_runtime.assert_called_once_with(
            api_base="http://proxy.com",
            aws_bedrock_runtime_endpoint=None,
            aws_region_name="us-west-2",
            endpoint_type="runtime"
        )
        
        # Verify URL construction
        assert str(url) == "http://proxy.com/model/anthropic.claude-3-sonnet/invoke"
        assert api_base == "http://proxy.com"


def test_bedrock_passthrough_get_complete_url_custom_endpoint_with_path():
    """Test get_complete_url with custom endpoint that has a base path"""
    config = BedrockPassthroughConfig()

    with patch.object(config, '_get_aws_region_name', return_value="us-west-2"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "http://proxy.com/bedrockproxy",
             "http://proxy.com/bedrockproxy"
         )) as mock_get_runtime:

        url, api_base = config.get_complete_url(
            api_base="http://proxy.com/bedrockproxy",
            api_key=None,
            model="anthropic.claude-3-sonnet",
            endpoint="/model/anthropic.claude-3-sonnet/invoke",
            request_query_params=None,
            litellm_params={
                "aws_bedrock_runtime_endpoint": "http://proxy.com/bedrockproxy"
            }
        )

        # Verify get_runtime_endpoint was called with correct parameters
        mock_get_runtime.assert_called_once_with(
            api_base="http://proxy.com/bedrockproxy",
            aws_bedrock_runtime_endpoint="http://proxy.com/bedrockproxy",
            aws_region_name="us-west-2",
            endpoint_type="runtime"
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
        request_query_params={}
    )
    
    assert str(result) == "https://api.example.com/model/test/invoke"


def test_format_url_preserves_proxy_paths():
    """Test format_url preserves proxy paths in base URL"""
    config = BedrockPassthroughConfig()
    
    result = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={}
    )
    
    # This is the key test - proxy path should be preserved
    assert str(result) == "http://proxy.com/bedrockproxy/model/test/invoke"


def test_format_url_with_query_parameters():
    """Test format_url properly handles query parameters"""
    config = BedrockPassthroughConfig()
    
    result = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={"param1": "value1", "param2": "value2"}
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
        request_query_params={}
    )
    
    # Test without trailing slash
    result_without_slash = config.format_url(
        endpoint="model/test/invoke",
        base_target_url="http://proxy.com/bedrockproxy",
        request_query_params={}
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
    
    with patch.object(config, '_get_aws_region_name', return_value="eu-west-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.eu-west-1.amazonaws.com",
             "https://bedrock-runtime.eu-west-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id, "aws_region_name": "eu-west-1"}
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
    
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id}
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
    
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={}  # No model_id provided
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
    with patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-west-2.amazonaws.com",
             "https://bedrock-runtime.us-west-2.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id}  # Region should be extracted from ARN
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
    
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id}
        )
        
        url_str = str(url)
        
        # The slash in the ARN after application-inference-profile should be encoded as %2F
        assert "application-inference-profile%2F" in url_str, \
            f"Expected encoded ARN with %2F in URL, but got: {url_str}"
        
        # The unencoded version should NOT be in the URL
        assert "application-inference-profile/" not in url_str, \
            f"ARN slash should be encoded, but found unencoded version in: {url_str}"
        
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
    model_id = "arn:aws:bedrock:us-east-1:123456789:application-inference-profile/xyz789"
    endpoint = f"/model/{model}/invoke"
    
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id}
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
    model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    endpoint = f"/model/{model}/converse"
    
    with patch.object(config, '_get_aws_region_name', return_value="us-east-1"), \
         patch.object(config, 'get_runtime_endpoint', return_value=(
             "https://bedrock-runtime.us-east-1.amazonaws.com",
             "https://bedrock-runtime.us-east-1.amazonaws.com"
         )):
        
        url, api_base = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            endpoint=endpoint,
            request_query_params=None,
            litellm_params={"model_id": model_id}
        )
        
        url_str = str(url)
        
        # Regular model ID should be used as-is (no encoding needed)
        assert model_id in url_str
        assert "%2F" not in url_str, "Non-ARN model IDs should not be encoded"

        expected_url = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse"
        assert url_str == expected_url


def test_bedrock_chunk_processor_parses_incrementally():
    """
    Verify that _BedrockEventStreamChunkProcessor feeds each chunk to the
    EventStreamBuffer as it arrives (not all at once at the end) and emits
    parsed messages incrementally — so peak memory is not O(full stream).
    """
    # One MagicMock event per "chunk arrival". Identity-based parser maps
    # each event to a known message string.
    event_a = MagicMock(name="event_a")
    event_b = MagicMock(name="event_b")
    event_c = MagicMock(name="event_c")

    parsed_by_event = {
        id(event_a): '{"token": "Hello"}',
        id(event_b): '{"token": "World"}',
        id(event_c): '{"token": "!"}',
    }

    def parse_message(event):
        return parsed_by_event.get(id(event))

    # Track add_data calls and yield the corresponding event each time.
    add_data_calls = []
    pending_events = [[event_a], [event_b], [event_c]]

    mock_buffer = MagicMock()

    def fake_add_data(chunk):
        add_data_calls.append(chunk)

    mock_buffer.add_data.side_effect = fake_add_data
    # Each iteration of the buffer yields the events queued for the most recent add_data.
    mock_buffer.__iter__.side_effect = lambda: iter(
        pending_events.pop(0) if pending_events else []
    )

    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        processor = _BedrockEventStreamChunkProcessor(parse_message=parse_message)

        # Feed three chunks one at a time
        result_1 = processor.process(b"chunk-1-bytes")
        result_2 = processor.process(b"chunk-2-bytes")
        result_3 = processor.process(b"chunk-3-bytes")

    # Each chunk produced its event's parsed message — incrementally.
    assert result_1 == ['{"token": "Hello"}']
    assert result_2 == ['{"token": "World"}']
    assert result_3 == ['{"token": "!"}']

    # Each chunk was added to the buffer separately (no end-of-stream batch).
    assert add_data_calls == [b"chunk-1-bytes", b"chunk-2-bytes", b"chunk-3-bytes"]


def test_bedrock_chunk_processor_skips_unparseable_events():
    """Events that parse_message returns None for are dropped, not retained."""
    event_real = MagicMock(name="real")
    event_skip = MagicMock(name="skip")

    def parse_message(event):
        if event is event_real:
            return '{"token": "ok"}'
        return None

    mock_buffer = MagicMock()
    mock_buffer.__iter__.return_value = [event_real, event_skip]

    with patch("botocore.eventstream.EventStreamBuffer", return_value=mock_buffer):
        processor = _BedrockEventStreamChunkProcessor(parse_message=parse_message)
        result = processor.process(b"some-bytes")

    assert result == ['{"token": "ok"}']


def test_bedrock_passthrough_config_create_streaming_chunk_processor():
    """BedrockPassthroughConfig wires up the event-stream processor, not the default."""
    config = BedrockPassthroughConfig()
    processor = config.create_streaming_chunk_processor()
    assert isinstance(processor, _BedrockEventStreamChunkProcessor)

