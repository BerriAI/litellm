import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig


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


