# What is this?
## This tests the llm guard integration

# What is this?
## Unit test for presidio pii masking
import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm.proxy.enterprise.enterprise_hooks.openai_moderation import (
    _ENTERPRISE_OpenAI_Moderation,
)
from litellm import Router, mock_completion
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.caching import DualCache

### UNIT TESTS FOR OpenAI Moderation ###


@pytest.mark.asyncio
async def test_openai_moderation_error_raising(monkeypatch):
    """
    Tests to see OpenAI Moderation raises an error for a flagged response
    """
    from unittest.mock import AsyncMock, MagicMock
    from litellm.types.llms.openai import OpenAIModerationResponse
    
    litellm.openai_moderations_model_name = "text-moderation-latest"
    openai_mod = _ENTERPRISE_OpenAI_Moderation()
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key)
    local_cache = DualCache()

    from litellm.proxy.proxy_server import llm_router

    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "text-moderation-latest",
                "litellm_params": {
                    "model": "text-moderation-latest",
                    "api_key": os.environ.get("OPENAI_API_KEY", "fake-key"),
                },
            }
        ]
    )

    # Mock the amoderation call to return a flagged response
    mock_response = MagicMock(spec=OpenAIModerationResponse)
    mock_response.results = [MagicMock(flagged=True)]
    
    async def mock_amoderation(*args, **kwargs):
        return mock_response
    
    llm_router.amoderation = mock_amoderation

    setattr(litellm.proxy.proxy_server, "llm_router", llm_router)

    try:
        await openai_mod.async_moderation_hook(
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "fuck off you're the worst",
                    }
                ]
            },
            user_api_key_dict=user_api_key_dict,
            call_type="completion",
        )
        pytest.fail(f"Should have failed")
    except Exception as e:
        print("Got exception: ", e)
        assert "Violated content safety policy" in str(e)
        pass


@pytest.mark.asyncio
async def test_openai_moderation_responses_api_input_field():
    """
    Tests that OpenAI Moderation works with Responses API input field via apply_guardrail.
    
    This test verifies that the unified guardrail interface (apply_guardrail) correctly
    handles different input types: plain text strings, structured messages, and lists.
    """
    from unittest.mock import patch
    from litellm.types.llms.openai import (
        OpenAIModerationResponse,
        OpenAIModerationResult,
    )
    from litellm.proxy.guardrails.guardrail_hooks.openai.moderations import (
        OpenAIModerationGuardrail,
    )
    from litellm.types.utils import GenericGuardrailAPIInputs
    
    # Initialize the open-source OpenAI Moderation guardrail
    openai_mod = OpenAIModerationGuardrail(
        guardrail_name="openai-moderation-test",
        api_key="fake-key-for-testing",
        model="omni-moderation-latest",
    )
    
    # Mock the async_make_request to return a flagged response
    mock_moderation_response = OpenAIModerationResponse(
        id="modr-123",
        model="omni-moderation-latest",
        results=[
            OpenAIModerationResult(
                flagged=True,
                categories={"violence": True, "hate": False},
                category_scores={"violence": 0.95, "hate": 0.1},
                category_applied_input_types=None,
            )
        ],
    )
    
    with patch.object(
        openai_mod, "async_make_request", return_value=mock_moderation_response
    ):
        # Test 1: Responses API / Embeddings with texts (string input)
        try:
            inputs = GenericGuardrailAPIInputs(texts=["I want to hurt people"])
            await openai_mod.apply_guardrail(
                inputs=inputs,
                request_data={"model": "gpt-4o", "input": "I want to hurt people"},
                input_type="request",
            )
            pytest.fail("Should have raised HTTPException for flagged content")
        except Exception as e:
            print("Got exception for texts input: ", e)
            assert "Violated OpenAI moderation policy" in str(e)
        
        # Test 2: Responses API with structured_messages (list of message objects)
        try:
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "I want to hurt people"}]
            )
            await openai_mod.apply_guardrail(
                inputs=inputs,
                request_data={"model": "gpt-4o", "input": [{"role": "user", "content": "I want to hurt people"}]},
                input_type="request",
            )
            pytest.fail("Should have raised HTTPException for flagged content")
        except Exception as e:
            print("Got exception for structured_messages input: ", e)
            assert "Violated OpenAI moderation policy" in str(e)
        
        # Test 3: Chat Completions with structured_messages
        try:
            inputs = GenericGuardrailAPIInputs(
                structured_messages=[{"role": "user", "content": "I want to hurt people"}]
            )
            await openai_mod.apply_guardrail(
                inputs=inputs,
                request_data={"model": "gpt-4o", "messages": [{"role": "user", "content": "I want to hurt people"}]},
                input_type="request",
            )
            pytest.fail("Should have raised HTTPException for flagged content")
        except Exception as e:
            print("Got exception for chat completions input: ", e)
            assert "Violated OpenAI moderation policy" in str(e)
    
    print("✓ All Responses API moderation tests passed!")
