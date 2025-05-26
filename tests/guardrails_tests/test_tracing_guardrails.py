import sys
import os
import io, asyncio
import json
import pytest
import time
from litellm import mock_completion
from unittest.mock import MagicMock, AsyncMock, patch
sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking, PresidioPerRequestConfig
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload, StandardLoggingGuardrailInformation
from litellm.types.guardrails import GuardrailEventHooks
from typing import Optional


class TestCustomLogger(CustomLogger):
    def __init__(self, *args, **kwargs):
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")
        pass

@pytest.mark.asyncio
async def test_standard_logging_payload_includes_guardrail_information():
    """
    Test that the standard logging payload includes the guardrail information when a guardrail is applied
    """
    test_custom_logger = TestCustomLogger()
    litellm.callbacks = [test_custom_logger]
    presidio_guard = _OPTIONAL_PresidioPIIMasking(
        guardrail_name="presidio_guard",
        event_hook=GuardrailEventHooks.pre_call,
        presidio_analyzer_api_base=os.getenv("PRESIDIO_ANALYZER_API_BASE"),
        presidio_anonymizer_api_base=os.getenv("PRESIDIO_ANONYMIZER_API_BASE"),
    )
    # 1. call the pre call hook with guardrail
    request_data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
        ],
        "mock_response": "Hello",
        "guardrails": ["presidio_guard"],
        "metadata": {},
    }
    await presidio_guard.async_pre_call_hook(
        user_api_key_dict={},
        cache=None,
        data=request_data,
        call_type="acompletion"
    )

    # 2. call litellm.acompletion
    response = await litellm.acompletion(**request_data)

    # 3. assert that the standard logging payload includes the guardrail information
    await asyncio.sleep(1)
    print("got standard logging payload=", json.dumps(test_custom_logger.standard_logging_payload, indent=4, default=str))
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["guardrail_name"] == "presidio_guard"
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["guardrail_mode"] == GuardrailEventHooks.pre_call

    # assert that the guardrail_response is a response from presidio analyze
    presidio_response = test_custom_logger.standard_logging_payload["guardrail_information"]["guardrail_response"]
    assert isinstance(presidio_response, list)
    for response_item in presidio_response:
        assert "analysis_explanation" in response_item
        assert "start" in response_item
        assert "end" in response_item
        assert "score" in response_item
        assert "entity_type" in response_item
        assert "recognition_metadata" in response_item
    

    # assert that the duration is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["duration"] is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["duration"] > 0

    # assert that we get the count of masked entities
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["masked_entity_count"] is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"]["masked_entity_count"]["PHONE_NUMBER"] == 1




@pytest.mark.asyncio
@pytest.mark.skip(reason="Local only test")
async def test_langfuse_trace_includes_guardrail_information():
    """
    Test that the langfuse trace includes the guardrail information when a guardrail is applied
    """
    import httpx
    from unittest.mock import AsyncMock, patch
    from litellm.integrations.langfuse.langfuse_prompt_management import LangfusePromptManagement 
    callback = LangfusePromptManagement(flush_interval=3)
    import json
    
    # Create a mock Response object
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    
    # Create mock for httpx.Client.post
    mock_post = AsyncMock()
    mock_post.return_value = mock_response
    
    with patch("httpx.Client.post", mock_post):
        litellm._turn_on_debug()
        litellm.callbacks = [callback]
        presidio_guard = _OPTIONAL_PresidioPIIMasking(
            guardrail_name="presidio_guard",
            event_hook=GuardrailEventHooks.pre_call,
            presidio_analyzer_api_base=os.getenv("PRESIDIO_ANALYZER_API_BASE"),
            presidio_anonymizer_api_base=os.getenv("PRESIDIO_ANONYMIZER_API_BASE"),
        )
        # 1. call the pre call hook with guardrail
        request_data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, my phone number is +1 412 555 1212"},
            ],
            "mock_response": "Hello",
            "guardrails": ["presidio_guard"],
            "metadata": {},
        }
        await presidio_guard.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=request_data,
            call_type="acompletion"
        )

        # 2. call litellm.acompletion
        response = await litellm.acompletion(**request_data)

        # 3. Wait for async logging operations to complete
        await asyncio.sleep(5)
        
        # 4. Verify the Langfuse payload
        assert mock_post.call_count >= 1
        url = mock_post.call_args[0][0]
        request_body = mock_post.call_args[1].get("content")
        
        # Parse the JSON body
        actual_payload = json.loads(request_body)
        print("\nLangfuse payload:", json.dumps(actual_payload, indent=2))
        
        # Look for the guardrail span in the payload
        guardrail_span = None
        for item in actual_payload["batch"]:
            if (item["type"] == "span-create" and 
                item["body"].get("name") == "guardrail"):
                guardrail_span = item
                break
        
        # Assert that the guardrail span exists
        assert guardrail_span is not None, "No guardrail span found in Langfuse payload"
        
        # Validate the structure of the guardrail span
        assert guardrail_span["body"]["name"] == "guardrail"
        assert "metadata" in guardrail_span["body"]
        assert guardrail_span["body"]["metadata"]["guardrail_name"] == "presidio_guard"
        assert guardrail_span["body"]["metadata"]["guardrail_mode"] == GuardrailEventHooks.pre_call
        assert "guardrail_masked_entity_count" in guardrail_span["body"]["metadata"]
        assert guardrail_span["body"]["metadata"]["guardrail_masked_entity_count"]["PHONE_NUMBER"] == 1
        
        # Validate the output format matches the expected structure
        assert "output" in guardrail_span["body"]
        assert isinstance(guardrail_span["body"]["output"], list)
        assert len(guardrail_span["body"]["output"]) > 0
        
        # Validate the first output item has the expected structure
        output_item = guardrail_span["body"]["output"][0]
        assert "entity_type" in output_item
        assert output_item["entity_type"] == "PHONE_NUMBER"
        assert "score" in output_item
        assert "start" in output_item
        assert "end" in output_item
        assert "recognition_metadata" in output_item
        assert "recognizer_name" in output_item["recognition_metadata"]
