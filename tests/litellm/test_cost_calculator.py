import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

import litellm
from litellm.cost_calculator import (
    handle_realtime_stream_cost_calculation,
    response_cost_calculator,
)
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage, ImageResponse, ImageData


# Mock data for model costs
mock_model_cost_data = {
    "vertex_ai/imagen-3.0-generate-002": {
        "output_cost_per_image": 0.04,
        "litellm_provider": "vertex_ai-image-models",
    },
    "vertex_ai/imagen-3.0-generate-001": {
        "output_cost_per_image": 0.04,
        "litellm_provider": "vertex_ai-image-models",
    },
    "vertex_ai/imagen-3.0-fast-generate-001": {
        "output_cost_per_image": 0.02,
        "litellm_provider": "vertex_ai-image-models",
    },
    "gemini-2.0-flash-001": { # Added for existing test_cost_calculator_with_usage
        "input_cost_per_audio_token": 0.0001,
        "input_cost_per_token": 0.0002,
        "output_cost_per_token": 0.0003,
        "litellm_provider": "vertex_ai",
    },
     "gpt-3.5-turbo": { # Added for existing test_handle_realtime_stream_cost_calculation
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
    },
    "gpt-4": { # Added for existing test_handle_realtime_stream_cost_calculation
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
    },
    "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": { # Added for existing test_default_image_cost_calculator
        "litellm_provider": "azure",
        "input_cost_per_pixel": 10,
    }
}


def _create_mock_image_response(num_images: int) -> ImageResponse:
    """Helper function to create a mock ImageResponse."""
    return ImageResponse(
        data=[ImageData(url="http://example.com/image.png")] * num_images,
        model="vertex_ai/imagen-3.0-generate-002", # model name doesn't affect cost calc here due to override
        created=1677652288 # dummy timestamp
    )


def _create_mock_model_response() -> ModelResponse:
    """Helper function to create a mock ModelResponse."""
    return ModelResponse(
        choices=[MagicMock()],
        model="vertex_ai/imagen-3.0-generate-002", # model name doesn't affect cost calc here
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )


def test_cost_calculator_with_response_cost_in_additional_headers():
    class MockResponse(BaseModel):
        _hidden_params = {
            "additional_headers": {"llm_provider-x-litellm-response-cost": 1000}
        }

    result = response_cost_calculator(
        response_object=MockResponse(),
        model="",
        custom_llm_provider=None,
        call_type="",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    assert result == 1000


def test_cost_calculator_with_usage(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    # from litellm import get_model_info # Not needed as we mock model_cost

    # os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True" # Not needed as we mock
    # litellm.model_cost = litellm.get_model_cost_map(url="") # Not needed

    usage = Usage(
        prompt_tokens=100,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10, audio_tokens=90
        ),
    )
    mr = ModelResponse(usage=usage, model="gemini-2.0-flash-001")

    result = response_cost_calculator(
        response_object=mr,
        model="gemini-2.0-flash-001", # Pass model explicitly
        custom_llm_provider="vertex_ai",
        call_type="acompletion",
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    model_info = litellm.model_cost["gemini-2.0-flash-001"]

    expected_cost = (
        usage.prompt_tokens_details.audio_tokens # type: ignore
        * model_info["input_cost_per_audio_token"] 
        + usage.prompt_tokens_details.text_tokens # type: ignore
        * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_token"]
    )

    assert result == expected_cost, f"Got {result}, Expected {expected_cost}"


def test_handle_realtime_stream_cost_calculation(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    from litellm.cost_calculator import RealtimeAPITokenUsageProcessor

    # Setup test data
    results: OpenAIRealtimeStreamList = [
        {"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}, # type: ignore
        {
            "type": "response.done", # type: ignore
            "response": { 
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
            },
        },
        {
            "type": "response.done", # type: ignore
            "response": {
                "usage": {
                    "input_tokens": 200,
                    "output_tokens": 100,
                    "total_tokens": 300,
                }
            },
        },
    ]

    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )

    # Test with explicit model name
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    # Calculate expected cost
    # gpt-3.5-turbo costs: $0.0015/1K tokens input, $0.002/1K tokens output
    expected_cost = (300 * mock_model_cost_data["gpt-3.5-turbo"]["input_cost_per_token"] ) + ( 
        150 * mock_model_cost_data["gpt-3.5-turbo"]["output_cost_per_token"]
    ) 
    assert (
        abs(cost - expected_cost) <= 0.00075
    )  # Allow small floating point differences

    # Test with different model name in session
    results[0]["session"]["model"] = "gpt-4" # type: ignore

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo", # litellm_model_name is fallback
    )

    # Calculate expected cost using gpt-4 rates
    expected_cost = (300 * mock_model_cost_data["gpt-4"]["input_cost_per_token"]) + ( 
        150 * mock_model_cost_data["gpt-4"]["output_cost_per_token"]
    )  
    assert abs(cost - expected_cost) < 0.00076

    # Test with no response.done events
    results = [{"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}] # type: ignore
    combined_usage_object = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results,
    )
    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )
    assert cost == 0.0  # No usage, no cost


def test_custom_pricing_with_router_model_id(monkeypatch):
    # This test doesn't rely on the global litellm.model_cost, 
    # it defines costs within the Router's model_list.
    # So, no monkeypatch for litellm.model_cost is strictly needed here for its core logic,
    # but we keep it for consistency if any underlying functions indirectly access it.
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "prod/claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "id": "my-unique-model-id",
                    "input_cost_per_token": 0.000006,
                    "output_cost_per_token": 0.00003,
                    "cache_creation_input_token_cost": 0.0000075,
                    "cache_read_input_token_cost": 0.0000006,
                },
            },
            {
                "model_name": "claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-20240620",
                    "api_key": "test_api_key",
                },
                "model_info": {
                    "input_cost_per_token": 100, # Intentionally high for test
                    "output_cost_per_token": 200, # Intentionally high for test
                },
            },
        ]
    )

    # Mock response needs to include usage for cost calculation
    result = router.completion(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
        specific_mock_response = ModelResponse(id="foo", choices=[], created=0, model="claude-3-5-sonnet-20240620", usage=Usage(prompt_tokens=10, completion_tokens=10))
    )

    result_2 = router.completion(
        model="prod/claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response=True,
        specific_mock_response = ModelResponse(id="foo", choices=[], created=0, model="anthropic/claude-3-5-sonnet-20240620", usage=Usage(prompt_tokens=10, completion_tokens=10))

    )
    
    assert result._hidden_params is not None and "response_cost" in result._hidden_params
    assert result_2._hidden_params is not None and "response_cost" in result_2._hidden_params
    assert (
        result._hidden_params["response_cost"]
        > result_2._hidden_params["response_cost"]
    )

    model_info = router.get_deployment_model_info(
        model_id="my-unique-model-id", model_name="anthropic/claude-3-5-sonnet-20240620"
    )
    assert model_info is not None
    assert model_info["input_cost_per_token"] == 0.000006
    assert model_info["output_cost_per_token"] == 0.00003
    assert model_info["cache_creation_input_token_cost"] == 0.0000075
    assert model_info["cache_read_input_token_cost"] == 0.0000006


def test_azure_realtime_cost_calculator(monkeypatch):
    # from litellm import get_model_info # Not needed as we mock model_cost
    # os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True" # Not needed
    # litellm.model_cost = litellm.get_model_cost_map(url="") # Not needed

    # Add a specific model for this test to mock_model_cost_data if it's not covered
    # For example, if "gpt-4o-realtime-preview-2024-12-17" needs specific costs
    mock_model_cost_data["gpt-4o-realtime-preview-2024-12-17"] = {
        "input_cost_per_token": 0.005, 
        "output_cost_per_token": 0.015,
        "input_cost_per_audio_token": 0.0001, # Assuming some audio cost
        "litellm_provider": "azure"
    }
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)


    cost = handle_realtime_stream_cost_calculation(
        results=[
            {
                "type": "session.created", # type: ignore
                "session": {"model": "gpt-4o-realtime-preview-2024-12-17"}, 
            },
        ],
        combined_usage_object=Usage(
            prompt_tokens=100,
            completion_tokens=100, # Not used by current handle_realtime_stream_cost_calculation
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=10, audio_tokens=90
            ),
        ),
        custom_llm_provider="azure",
        litellm_model_name="my-custom-azure-deployment", # Fallback if session model not in cost map
    )

    assert cost > 0


def test_default_image_cost_calculator(monkeypatch):
    from litellm.cost_calculator import default_image_cost_calculator
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)

    # temp_object = { # This is defined in mock_model_cost_data now
    #     "litellm_provider": "azure",
    #     "input_cost_per_pixel": 10,
    # }

    # monkeypatch.setattr(
    #     litellm,
    #     "model_cost",
    #     {
    #         "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": temp_object
    #     },
    # )

    args = {
        "model": "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b",
        "custom_llm_provider": "azure",
        "quality": "standard",
        "n": 1,
        "size": "1024-x-1024",
        "optional_params": {},
    }
    cost = default_image_cost_calculator(**args)
    assert cost == 10485760


# New Vertex AI Imagen Test Cases
def test_vertex_imagen_3_0_generate_002_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    mock_response = _create_mock_image_response(num_images=1)
    cost = litellm.completion_cost(
        completion_response=mock_response,
        model="vertex_ai/imagen-3.0-generate-002",
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models"
    )
    assert cost == 0.04

def test_vertex_imagen_3_0_generate_002_multiple_images(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    mock_response = _create_mock_image_response(num_images=3)
    # Manually set the model in hidden_params as completion_cost would expect it for provider routing
    mock_response._hidden_params = {"custom_llm_provider": "vertex_ai-image-models", "optional_params": {"n": 3}}

    cost = litellm.completion_cost(
        completion_response=mock_response,
        model="vertex_ai/imagen-3.0-generate-002",
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models",
        n=3 # Pass n explicitly
    )
    assert cost == 0.04 * 3

def test_vertex_imagen_3_0_generate_001_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    mock_response = _create_mock_image_response(num_images=1)
    cost = litellm.completion_cost(
        completion_response=mock_response,
        model="vertex_ai/imagen-3.0-generate-001",
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models"
    )
    assert cost == 0.04

def test_vertex_imagen_3_0_fast_generate_001_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    mock_response = _create_mock_image_response(num_images=1)
    cost = litellm.completion_cost(
        completion_response=mock_response,
        model="vertex_ai/imagen-3.0-fast-generate-001",
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models"
    )
    assert cost == 0.02

def test_imagen_wrong_call_type_returns_zero(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    mock_response = _create_mock_model_response()
    cost = litellm.completion_cost(
        completion_response=mock_response,
        model="vertex_ai/imagen-3.0-generate-002", # This model in mock_model_cost_data only has image pricing
        call_type="completion", # Text completion call type
        custom_llm_provider="vertex_ai-image-models" # Provider indicates image model group
    )
    assert cost == 0.0
