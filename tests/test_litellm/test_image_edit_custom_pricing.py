"""
Regression test for #22244: aimage_edit cost calculation breaks when the
deployment has custom per-image pricing and the shared backend key has its
pricing fields stripped (PR #20679).

The root cause was that ``_response_cost_calculator`` never extracted
``router_model_id`` from the deployment metadata set by the Router.  Without
the model-id, ``default_image_cost_calculator`` fell back to the shared key
whose ``input_cost_per_image`` had been intentionally stripped by PR #20679 to
prevent cross-deployment pricing pollution.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import ImageResponse, Router
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging


CUSTOM_COST_PER_IMAGE = 0.00676128
DEPLOYMENT_ID = "image-edit-deploy-test"


@pytest.fixture(autouse=True)
def _setup_router():
    """Create a Router with a single deployment that has custom per-image pricing."""
    Router(
        model_list=[
            {
                "model_name": "test-image-model",
                "litellm_params": {
                    "model": "openai/flux-2-klein-4b",
                    "api_base": "http://localhost:8080/v1",
                    "api_key": "placeholder",
                },
                "model_info": {
                    "id": DEPLOYMENT_ID,
                    "mode": "image_generation",
                    "input_cost_per_image": CUSTOM_COST_PER_IMAGE,
                },
            }
        ],
    )
    yield


def test_shared_key_strips_input_cost_per_image():
    """Verify the shared backend key does NOT contain input_cost_per_image (PR #20679 behaviour)."""
    shared = litellm.model_cost.get("openai/flux-2-klein-4b", {})
    assert "input_cost_per_image" not in shared or shared["input_cost_per_image"] is None


def test_model_id_key_retains_input_cost_per_image():
    """Verify the deployment model-id key still retains input_cost_per_image."""
    entry = litellm.model_cost.get(DEPLOYMENT_ID, {})
    assert entry.get("input_cost_per_image") == CUSTOM_COST_PER_IMAGE


def test_aimage_edit_cost_with_router_model_id():
    """Cost calc succeeds when router_model_id is provided explicitly."""
    resp = ImageResponse(created=1, data=[{"url": "http://example.com/img.png"}])
    cost = litellm.completion_cost(
        completion_response=resp,
        model="openai/flux-2-klein-4b",
        call_type="aimage_edit",
        custom_llm_provider="openai",
        custom_pricing=True,
        router_model_id=DEPLOYMENT_ID,
    )
    assert cost == pytest.approx(CUSTOM_COST_PER_IMAGE)


def test_aimage_generation_cost_with_router_model_id():
    """Cost calc succeeds for aimage_generation when router_model_id is provided."""
    resp = ImageResponse(created=1, data=[{"url": "http://example.com/img.png"}])
    cost = litellm.completion_cost(
        completion_response=resp,
        model="openai/flux-2-klein-4b",
        call_type="aimage_generation",
        custom_llm_provider="openai",
        custom_pricing=True,
        router_model_id=DEPLOYMENT_ID,
    )
    assert cost == pytest.approx(CUSTOM_COST_PER_IMAGE)


def test_response_cost_calculator_extracts_model_id_from_metadata():
    """
    The logging object's _response_cost_calculator should extract
    router_model_id from litellm_params metadata when hidden_params
    does not contain model_id.  This is the core regression in #22244.
    """
    resp = ImageResponse(created=1, data=[{"url": "http://example.com/img.png"}])

    logging_obj = LitellmLogging(
        model="openai/flux-2-klein-4b",
        messages=[],
        stream=False,
        call_type="aimage_edit",
        litellm_call_id="test-call-id",
        start_time=None,
        function_id="test",
    )

    # Simulate what the Router sets on litellm_params
    logging_obj.litellm_params = {
        "metadata": {
            "model_info": {"id": DEPLOYMENT_ID},
        },
        "input_cost_per_image": CUSTOM_COST_PER_IMAGE,
    }
    logging_obj.model_call_details = {
        "custom_llm_provider": "openai",
    }
    logging_obj.optional_params = {}
    logging_obj.standard_built_in_tools_params = None

    cost = logging_obj._response_cost_calculator(result=resp)
    assert cost is not None
    assert cost == pytest.approx(CUSTOM_COST_PER_IMAGE)


def test_response_cost_calculator_extracts_model_id_from_litellm_metadata():
    """
    Same as above but with litellm_metadata (used by generic API calls
    like aimage_edit routed through _ageneric_api_call_with_fallbacks).
    """
    resp = ImageResponse(created=1, data=[{"url": "http://example.com/img.png"}])

    logging_obj = LitellmLogging(
        model="openai/flux-2-klein-4b",
        messages=[],
        stream=False,
        call_type="aimage_edit",
        litellm_call_id="test-call-id-2",
        start_time=None,
        function_id="test",
    )

    # Generic API calls store model_info under litellm_metadata
    logging_obj.litellm_params = {
        "litellm_metadata": {
            "model_info": {"id": DEPLOYMENT_ID},
        },
        "input_cost_per_image": CUSTOM_COST_PER_IMAGE,
    }
    logging_obj.model_call_details = {
        "custom_llm_provider": "openai",
    }
    logging_obj.optional_params = {}
    logging_obj.standard_built_in_tools_params = None

    cost = logging_obj._response_cost_calculator(result=resp)
    assert cost is not None
    assert cost == pytest.approx(CUSTOM_COST_PER_IMAGE)
