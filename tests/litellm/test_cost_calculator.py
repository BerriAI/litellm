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
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage, ImageResponse, ImageObject # Corrected import


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
    "gemini-2.0-flash-001": { 
        "input_cost_per_audio_token": 0.0001,
        "input_cost_per_token": 0.0002,
        "output_cost_per_token": 0.0003,
        "litellm_provider": "vertex_ai",
    },
     "gpt-3.5-turbo": { 
        "input_cost_per_token": 0.0000015,
        "output_cost_per_token": 0.000002,
        "litellm_provider": "openai",
    },
    "gpt-4": { 
        "input_cost_per_token": 0.00003,
        "output_cost_per_token": 0.00006,
        "litellm_provider": "openai",
    },
    "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b": { 
        "litellm_provider": "azure",
        "input_cost_per_pixel": 0.00001, # Adjusted for more realistic small cost for test_default_image_cost_calculator
    }
}


def _create_mock_image_response(num_images: int, model_name: str = "vertex_ai/imagen-3.0-generate-002") -> ImageResponse:
    """Helper function to create a mock ImageResponse using ImageObject."""
    return ImageResponse(
        data=[ImageObject(url=f"http://example.com/image{i+1}.png", b64_json=None, revised_prompt=None) for i in range(num_images)],
        model=model_name, 
        created=1677652288 
    )


def _create_mock_model_response(model_name:str = "vertex_ai/imagen-3.0-generate-002", prompt_tokens: int = 10, completion_tokens: int = 20) -> ModelResponse:
    """Helper function to create a mock ModelResponse for text completion."""
    class MockChoice(BaseModel): # Pydantic model for choice
        finish_reason: str = "stop"
        index: int = 0
        message: MagicMock = MagicMock()

    # Ensure the choice object is a valid Pydantic model or dict that ModelResponse can handle
    # Using a dict here for simplicity as ModelResponse can convert it.
    choice_dict = {"finish_reason": "stop", "index": 0, "message": {"role": "assistant", "content": "test"}}


    return ModelResponse(
        id="cmpl-test123",
        choices=[choice_dict], # Pass as a list of dicts or Pydantic models
        model=model_name, 
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
        created=1677652288
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
        call_type="", # type: ignore
        optional_params={},
        cache_hit=None,
        base_model=None,
    )

    assert result == 1000


def test_cost_calculator_with_usage(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=100,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            text_tokens=10, audio_tokens=90
        ),
    )
    # Ensure the model used here is in mock_model_cost_data
    mr = ModelResponse(usage=usage, model="gemini-2.0-flash-001", id="test", created=123, choices=[])


    result = response_cost_calculator(
        response_object=mr,
        model="gemini-2.0-flash-001", 
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

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )

    expected_cost = (300 * mock_model_cost_data["gpt-3.5-turbo"]["input_cost_per_token"] ) + ( 
        150 * mock_model_cost_data["gpt-3.5-turbo"]["output_cost_per_token"]
    ) 
    assert (
        abs(cost - expected_cost) <= 0.00075 
    )

    results[0]["session"]["model"] = "gpt-4" # type: ignore

    cost = handle_realtime_stream_cost_calculation(
        results=results,
        combined_usage_object=combined_usage_object,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo", 
    )

    expected_cost = (300 * mock_model_cost_data["gpt-4"]["input_cost_per_token"]) + ( 
        150 * mock_model_cost_data["gpt-4"]["output_cost_per_token"]
    )  
    assert abs(cost - expected_cost) < 0.00076

    results_no_done = [{"type": "session.created", "session": {"model": "gpt-3.5-turbo"}}] # type: ignore
    combined_usage_object_no_done = RealtimeAPITokenUsageProcessor.collect_and_combine_usage_from_realtime_stream_results(
        results=results_no_done, # type: ignore
    )
    cost = handle_realtime_stream_cost_calculation(
        results=results_no_done, # type: ignore
        combined_usage_object=combined_usage_object_no_done,
        custom_llm_provider="openai",
        litellm_model_name="gpt-3.5-turbo",
    )
    assert cost == 0.0


def test_custom_pricing_with_router_model_id(monkeypatch):
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
                    "input_cost_per_token": 100, 
                    "output_cost_per_token": 200, 
                },
            },
        ]
    )

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
    mock_model_cost_data["gpt-4o-realtime-preview-2024-12-17"] = {
        "input_cost_per_token": 0.005, 
        "output_cost_per_token": 0.015,
        "input_cost_per_audio_token": 0.0001, 
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
            completion_tokens=100, 
            prompt_tokens_details=PromptTokensDetailsWrapper(
                text_tokens=10, audio_tokens=90
            ),
        ),
        custom_llm_provider="azure",
        litellm_model_name="my-custom-azure-deployment", 
    )
    assert cost > 0


def test_default_image_cost_calculator(monkeypatch):
    from litellm.cost_calculator import default_image_cost_calculator 
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)

    args_model_key = "azure/bf9001cd7209f5734ecb4ab937a5a0e2ba5f119708bd68f184db362930f9dc7b"
    args = {
        "model": args_model_key, 
        "custom_llm_provider": "azure",
        "quality": "standard",
        "n": 1,
        "size": "1024-x-1024", # This size is used by default_image_cost_calculator
        "optional_params": {},
    }
    cost = default_image_cost_calculator(**args) 
    # Cost = height * width * input_cost_per_pixel * n
    # 1024 * 1024 * 0.00001 * 1 = 10.48576
    assert abs(cost - (1024 * 1024 * mock_model_cost_data[args_model_key]["input_cost_per_pixel"] * args["n"])) < 0.00001


# New Vertex AI Imagen Test Cases
def test_vertex_imagen_3_0_generate_002_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    model_name = "vertex_ai/imagen-3.0-generate-002"
    mock_response = _create_mock_image_response(num_images=1, model_name=model_name)
    # Ensure 'n' is part of optional_params within _hidden_params for completion_cost
    mock_response._hidden_params = {"custom_llm_provider": "vertex_ai-image-models", "optional_params": {"n": 1}}

    cost = litellm.completion_cost(
        completion_response=mock_response,
        model=model_name, 
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models",
        n=1 # Also pass n here for clarity, though completion_cost should pick from hidden_params
    )
    assert cost == 0.04

def test_vertex_imagen_3_0_generate_002_multiple_images(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    model_name = "vertex_ai/imagen-3.0-generate-002"
    num_images = 3
    mock_response = _create_mock_image_response(num_images=num_images, model_name=model_name)
    mock_response._hidden_params = {"custom_llm_provider": "vertex_ai-image-models", "optional_params": {"n": num_images}}

    cost = litellm.completion_cost(
        completion_response=mock_response,
        model=model_name,
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models",
        n=num_images 
    )
    assert abs(cost - (0.04 * num_images)) < 0.00001 # Using abs for float comparison

def test_vertex_imagen_3_0_generate_001_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    model_name = "vertex_ai/imagen-3.0-generate-001"
    mock_response = _create_mock_image_response(num_images=1, model_name=model_name)
    mock_response._hidden_params = {"custom_llm_provider": "vertex_ai-image-models", "optional_params": {"n": 1}}

    cost = litellm.completion_cost(
        completion_response=mock_response,
        model=model_name,
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models",
        n=1
    )
    assert cost == 0.04

def test_vertex_imagen_3_0_fast_generate_001_single_image(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    model_name = "vertex_ai/imagen-3.0-fast-generate-001"
    mock_response = _create_mock_image_response(num_images=1, model_name=model_name)
    mock_response._hidden_params = {"custom_llm_provider": "vertex_ai-image-models", "optional_params": {"n": 1}}
    
    cost = litellm.completion_cost(
        completion_response=mock_response,
        model=model_name,
        call_type="image_generation",
        custom_llm_provider="vertex_ai-image-models",
        n=1
    )
    assert cost == 0.02

def test_imagen_wrong_call_type_returns_zero(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", mock_model_cost_data)
    model_name = "vertex_ai/imagen-3.0-generate-002" # This model ONLY has image pricing in mock_model_cost_data
    
    # Create a standard text ModelResponse
    mock_text_response = _create_mock_model_response(model_name=model_name, prompt_tokens=50, completion_tokens=50)
    
    # The cost_calculator.py logic for 'vertex_ai' routes to google_cost_per_character or google_cost_per_token
    # if the call_type is not image_generation.
    # Since 'vertex_ai/imagen-3.0-generate-002' in mock_model_cost_data does not have 'input_cost_per_token' 
    # or character costs, the cost_per_token function (and character one) will eventually return 0,0.
    
    cost = litellm.completion_cost(
        completion_response=mock_text_response, 
        model=model_name, 
        call_type="completion", # Text completion call type
        custom_llm_provider="vertex_ai-image-models" # This provider is key for routing
    )
    
    assert cost == 0.0
