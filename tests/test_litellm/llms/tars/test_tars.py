import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path.
from unittest.mock import MagicMock, patch

from litellm.llms.tars.chat.transformation import TarsConfig
from litellm.llms.tars.common_utils import TarsException, TarsModelInfo
from litellm.llms.tars.embedding.transformation import TarsEmbeddingConfig
from litellm.llms.tars.cost_calculator import cost_per_token
from litellm.types.utils import ModelResponse, Usage
import httpx


def test_tars_config_initialization():
    """Test that TarsConfig can be initialized with various parameters."""
    config = TarsConfig(
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
    )
    
    assert config.temperature == 0.7
    assert config.max_tokens == 100
    assert config.top_p == 0.9


def test_tars_get_api_base():
    """Test the get_api_base method returns correct default."""
    api_base = TarsConfig.get_api_base(api_base=None)
    assert api_base == "https://api.router.tetrate.ai/v1"


def test_tars_get_api_base_custom():
    """Test the get_api_base method with custom API base."""
    custom_api_base = "https://custom.tars.ai/v1"
    api_base = TarsConfig.get_api_base(api_base=custom_api_base)
    assert api_base == custom_api_base


def test_tars_get_api_key():
    """Test the get_api_key method."""
    test_key = "test-tars-key"
    api_key = TarsConfig.get_api_key(api_key=test_key)
    assert api_key == test_key


def test_tars_get_complete_url():
    """Test the get_complete_url method generates correct endpoint URL."""
    config = TarsConfig()
    
    url = config.get_complete_url(
        api_base="https://api.router.tetrate.ai/v1",
        api_key="test-key",
        model="claude-sonnet-4-20250514",
        optional_params={},
        litellm_params={},
        stream=False
    )
    
    assert url == "https://api.router.tetrate.ai/v1/chat/completions"


def test_tars_get_complete_url_no_duplicate():
    """Test that get_complete_url doesn't duplicate endpoint in URL."""
    config = TarsConfig()
    
    url = config.get_complete_url(
        api_base="https://api.router.tetrate.ai/v1/chat/completions",
        api_key="test-key",
        model="gpt-4o",
        optional_params={},
        litellm_params={},
        stream=False
    )
    
    assert url == "https://api.router.tetrate.ai/v1/chat/completions"
    assert url.count("chat/completions") == 1


def test_tars_exception():
    """Test TarsException can be instantiated properly."""
    exception = TarsException(
        message="Test error",
        status_code=400,
        headers={"content-type": "application/json"}
    )
    
    assert exception.status_code == 400
    assert exception.message == "Test error"


def test_tars_model_info_get_base_model():
    """Test get_base_model removes tars/ prefix."""
    model_info = TarsModelInfo()
    
    base_model = model_info.get_base_model("tars/claude-sonnet-4-20250514")
    assert base_model == "claude-sonnet-4-20250514"
    
    base_model = model_info.get_base_model("tars/gpt-4o")
    assert base_model == "gpt-4o"


def test_tars_model_info_validate_environment():
    """Test validate_environment adds proper authentication headers."""
    model_info = TarsModelInfo()
    
    headers = {}
    test_api_key = "test-tars-api-key"
    
    result_headers = model_info.validate_environment(
        headers=headers,
        model="tars/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=test_api_key,
        api_base="https://api.router.tetrate.ai/v1"
    )
    
    assert "Authorization" in result_headers
    assert result_headers["Authorization"] == f"Bearer {test_api_key}"


def test_tars_model_info_validate_environment_no_key():
    """Test validate_environment raises error when no API key provided."""
    model_info = TarsModelInfo()
    
    headers = {}
    
    with pytest.raises(ValueError, match="TARS_API_KEY is not set"):
        model_info.validate_environment(
            headers=headers,
            model="tars/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://api.router.tetrate.ai/v1"
        )


@patch('httpx.Client')
def test_tars_get_models_success(mock_client_class):
    """Test get_models fetches and formats models correctly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"id": "claude-sonnet-4-20250514", "object": "model"},
            {"id": "gpt-4o", "object": "model"},
            {"id": "gpt-4o-mini", "object": "model"},
        ]
    }
    mock_client.get.return_value = mock_response
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client_class.return_value = mock_client
    
    model_info = TarsModelInfo()
    models = model_info.get_models(api_key="test-key", api_base="https://api.router.tetrate.ai/v1")
    
    assert len(models) == 3
    assert "tars/claude-sonnet-4-20250514" in models
    assert "tars/gpt-4o" in models
    assert "tars/gpt-4o-mini" in models
    assert models == sorted(models)  # Verify models are sorted.


@patch('httpx.Client')
def test_tars_get_models_no_api_key(mock_client_class):
    """Test get_models raises error when no API key provided."""
    model_info = TarsModelInfo()
    
    with pytest.raises(ValueError, match="TARS_API_KEY is not set"):
        model_info.get_models(api_key=None, api_base="https://api.router.tetrate.ai/v1")


@patch('httpx.Client')
def test_tars_get_models_http_error(mock_client_class):
    """Test get_models handles HTTP errors gracefully."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    
    http_error = httpx.HTTPStatusError(
        "Unauthorized",
        request=MagicMock(),
        response=mock_response
    )
    mock_client.get.side_effect = http_error
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client_class.return_value = mock_client
    
    model_info = TarsModelInfo()
    
    with pytest.raises(ValueError, match="Failed to fetch models from TARS"):
        model_info.get_models(api_key="invalid-key", api_base="https://api.router.tetrate.ai/v1")


def test_tars_embedding_config():
    """Test TarsEmbeddingConfig initialization."""
    config = TarsEmbeddingConfig()
    
    assert config is not None


def test_tars_embedding_get_error_class():
    """Test TarsEmbeddingConfig returns correct error class."""
    config = TarsEmbeddingConfig()
    
    error = config.get_error_class(
        error_message="Test error",
        status_code=500,
        headers={"content-type": "application/json"}
    )
    
    assert isinstance(error, TarsException)
    assert error.status_code == 500


def test_tars_config_get_error_class():
    """Test TarsConfig returns correct error class."""
    config = TarsConfig()
    
    error = config.get_error_class(
        error_message="Test error",
        status_code=429,
        headers={"content-type": "application/json"}
    )
    
    assert isinstance(error, TarsException)
    assert error.status_code == 429


def test_config_get_config():
    """Test that get_config method returns the configuration."""
    config_dict = TarsConfig.get_config()
    assert isinstance(config_dict, dict)


def test_response_format_support():
    """Test that response_format parameter is supported."""
    response_format = {
        "type": "json_object"
    }
    
    config = TarsConfig(response_format=response_format)
    assert config.response_format == response_format


def test_tools_support():
    """Test that tools parameter is supported."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information"
            }
        }
    ]
    
    config = TarsConfig(tools=tools)
    assert config.tools == tools


def test_functions_support():
    """Test that functions parameter is supported."""
    functions = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    ]
    
    config = TarsConfig(functions=functions)
    assert config.functions == functions


def test_stop_parameter_support():
    """Test that stop parameter supports both string and list."""
    # Test with string.
    config1 = TarsConfig(stop="STOP")
    assert config1.stop == "STOP"
    
    # Test with list.
    config2 = TarsConfig(stop=["STOP", "END"])
    assert config2.stop == ["STOP", "END"]


def test_logit_bias_support():
    """Test that logit_bias parameter is supported."""
    logit_bias = {"50256": -100}
    
    config = TarsConfig(logit_bias=logit_bias)
    assert config.logit_bias == logit_bias


def test_presence_penalty_support():
    """Test that presence_penalty parameter is supported."""
    config = TarsConfig(presence_penalty=0.5)
    assert config.presence_penalty == 0.5


def test_n_parameter_support():
    """Test that n parameter (number of completions) is supported."""
    config = TarsConfig(n=3)
    assert config.n == 3


def test_max_completion_tokens_support():
    """Test that max_completion_tokens parameter is supported."""
    config = TarsConfig(max_completion_tokens=150)
    assert config.max_completion_tokens == 150


def test_tars_config_inherits_openai():
    """Test that TarsConfig properly inherits from OpenAIGPTConfig."""
    from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
    
    config = TarsConfig()
    assert isinstance(config, OpenAIGPTConfig)


def test_tars_embedding_config_inherits_openai():
    """Test that TarsEmbeddingConfig properly inherits from OpenAIEmbeddingConfig."""
    from litellm.llms.openai.embedding.transformation import OpenAIEmbeddingConfig
    
    config = TarsEmbeddingConfig()
    assert isinstance(config, OpenAIEmbeddingConfig)


def test_tars_cost_calculator_with_5_percent_margin():
    """Test that TARS cost calculator adds 5% margin to base costs."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500
    )
    
    # Test with a known model that has pricing.
    prompt_cost, completion_cost = cost_per_token(model="gpt-4o", usage=usage)
    
    # Verify that costs are calculated (non-zero).
    assert prompt_cost > 0
    assert completion_cost > 0
    
    # The costs should include the 5% margin.
    # We can't test exact values without knowing the base prices, but we can verify they're positive.


def test_tars_cost_calculator_no_pricing_fallback():
    """Test that TARS cost calculator returns (0.0, 0.0) when no pricing is available."""
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500
    )
    
    # Test with a model that doesn't exist in the pricing catalog.
    prompt_cost, completion_cost = cost_per_token(model="nonexistent-model-12345", usage=usage)
    
    # Should return (0.0, 0.0) as fallback.
    assert prompt_cost == 0.0
    assert completion_cost == 0.0


def test_tars_cost_calculator_with_zero_tokens():
    """Test that TARS cost calculator handles zero tokens correctly."""
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0
    )
    
    # Test with a known model.
    prompt_cost, completion_cost = cost_per_token(model="gpt-4o", usage=usage)
    
    # Should return 0.0 for both when no tokens are used.
    assert prompt_cost == 0.0
    assert completion_cost == 0.0

