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


class CustomLoggerForTesting(CustomLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.standard_logging_payload = kwargs.get("standard_logging_object")
        pass

@pytest.mark.asyncio
async def test_standard_logging_payload_includes_guardrail_information():
    """
    Test that the standard logging payload includes the guardrail information when a guardrail is applied
    """
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    presidio_guard = _OPTIONAL_PresidioPIIMasking(
        guardrail_name="presidio_guard",
        event_hook=GuardrailEventHooks.pre_call,
        presidio_analyzer_api_base="https://mock-presidio-analyzer.com/",
        presidio_anonymizer_api_base="https://mock-presidio-anonymizer.com/",
    )
    
    # Mock the Presidio API responses
    mock_analyze_response = [
        {
            "analysis_explanation": {"recognizer": "PhoneRecognizer", "pattern": "phone"},
            "start": 26,
            "end": 40,
            "score": 0.75,
            "entity_type": "PHONE_NUMBER"
        }
    ]
    
    mock_anonymize_response = {
        "text": "Hello, my phone number is <PHONE_NUMBER>",
        "items": [
            {
                "start": 26,
                "end": 40,
                "entity_type": "PHONE_NUMBER",
                "text": "<PHONE_NUMBER>",
                "operator": "replace"
            }
        ]
    }
    
    # Create mock response objects
    mock_analyze_resp = MagicMock()
    mock_analyze_resp.json = AsyncMock(return_value=mock_analyze_response)
    
    mock_anonymize_resp = MagicMock()
    mock_anonymize_resp.json = AsyncMock(return_value=mock_anonymize_response)
    
    # Mock the aiohttp ClientSession with global call tracking
    call_counter = {"count": 0}
    
    class MockClientSession:
        def __init__(self):
            self.closed = False
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        async def close(self):
            self.closed = True
            
        def post(self, url, json=None):
            class MockResponse:
                def __init__(self, response_obj):
                    self.response_obj = response_obj
                    
                async def __aenter__(self):
                    return self.response_obj
                    
                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass
            
            # Return analyze response first, then anonymize response
            call_counter["count"] += 1
            if "analyze" in url:
                return MockResponse(mock_analyze_resp)
            else:
                return MockResponse(mock_anonymize_resp)
    
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
    
    with patch("aiohttp.ClientSession", MockClientSession):
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
    
    # guardrail_information is now a list
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_name"] == "presidio_guard"
    assert guardrail_info["guardrail_mode"] == GuardrailEventHooks.pre_call

    # assert that the guardrail_response is a response from presidio analyze
    presidio_response = guardrail_info["guardrail_response"]
    assert isinstance(presidio_response, list)
    for response_item in presidio_response:
        assert "analysis_explanation" in response_item
        assert "start" in response_item
        assert "end" in response_item
        assert "score" in response_item
        assert "entity_type" in response_item

    # assert that the duration is not None
    assert guardrail_info["duration"] is not None
    assert guardrail_info["duration"] > 0

    # assert that we get the count of masked entities
    assert guardrail_info["masked_entity_count"] is not None
    assert guardrail_info["masked_entity_count"]["PHONE_NUMBER"] == 1




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


@pytest.mark.asyncio
async def test_bedrock_guardrail_status_blocked():
    """
    Test that Bedrock guardrail sets correct status fields when blocking content.
    
    This test verifies that when Bedrock guardrail blocks content:
    1. The guardrail_information contains guardrail_status="blocked" 
    2. The status_fields.guardrail_status is set to "guardrail_intervened"
    3. The status_fields.llm_api_status remains "success" (mock LLM call succeeds)
    """
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import AsyncMock, MagicMock, patch
    litellm._turn_on_debug()
    
    # Setup custom logger to capture standard logging payload
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    
    # Create Bedrock guardrail with mock AWS credentials
    bedrock_guard = BedrockGuardrail(
        guardrail_name="bedrock_guard",
        event_hook=GuardrailEventHooks.pre_call,
        guardrailIdentifier="test-id",
        guardrailVersion="1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_region_name="us-east-1",
    )
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "action": "GUARDRAIL_INTERVENED",
        "outputs": [{"text": "Blocked"}],
        "assessments": [{
            "topicPolicy": {
                "topics": [{"name": "harmful", "action": "BLOCKED"}]
            }
        }]
    }
    with patch.object(bedrock_guard.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "harmful content"}],
            "mock_response": "Hello",
            "metadata": {}
        }
        
        # Mock should_run_guardrail to ensure guardrail logic executes
        with patch.object(bedrock_guard, 'should_run_guardrail', return_value=True):
            # Call guardrail pre_call hook - this will raise an exception when content is blocked
            try:
                await bedrock_guard.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=None,
                    data=request_data,
                    call_type="completion"
                )
            except Exception:
                # Expected exception when guardrail blocks content
                pass
        
        # Call litellm.acompletion to trigger logging callbacks
        # This populates the standard_logging_payload in our custom logger
        response = await litellm.acompletion(**request_data)
        await asyncio.sleep(1)
    
    # Verify the standard logging payload was captured
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    # Verify guardrail information fields (guardrail_information is now a list)
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_status"] == "guardrail_intervened"
    assert guardrail_info["guardrail_provider"] == "bedrock"
    
    # Verify the new typed status fields
    # guardrail_status should be "guardrail_intervened" when content is blocked
    # llm_api_status should be "success" since the mock LLM call itself succeeded
    status_fields = test_custom_logger.standard_logging_payload.get("status_fields", {})
    assert status_fields.get("llm_api_status") == "success"
    assert status_fields.get("guardrail_status") == "guardrail_intervened"


@pytest.mark.asyncio
async def test_bedrock_guardrail_status_success():
    """
    Test that Bedrock guardrail sets correct status fields when allowing content.
    
    This test verifies that when Bedrock guardrail allows content through:
    1. The guardrail_information contains guardrail_status="success"
    2. The status_fields.guardrail_status is set to "success" 
    3. The status_fields.llm_api_status is "success"
    """
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import AsyncMock, MagicMock, patch
    
    # Reset callbacks completely to avoid event loop conflicts
    litellm.callbacks = []
    await asyncio.sleep(0.1)  # Let previous callbacks finish
    
    # Setup custom logger to capture standard logging payload
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    
    # Create Bedrock guardrail
    bedrock_guard = BedrockGuardrail(
        guardrail_name="bedrock_guard",
        event_hook=GuardrailEventHooks.pre_call,
        guardrailIdentifier="test-id",
        guardrailVersion="1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_region_name="us-east-1",
    )
    
    # Mock success response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "action": "NONE",
        "outputs": [{"text": "Safe content"}],
        "assessments": []
    }
    with patch.object(bedrock_guard.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "safe content"}],
            "mock_response": "Hello",
            "metadata": {}
        }
        
        # Mock should_run_guardrail to return True
        with patch.object(bedrock_guard, 'should_run_guardrail', return_value=True):
            await bedrock_guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=None,
                data=request_data,
                call_type="completion"
            )
        
        # Call litellm.acompletion to trigger logging
        response = await litellm.acompletion(**request_data)
        await asyncio.sleep(1)
    
    # Check standard logging payload status fields
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_status"] == "success"
    assert guardrail_info["guardrail_provider"] == "bedrock"
    
    # Check status fields
    status_fields = test_custom_logger.standard_logging_payload.get("status_fields", {})
    assert status_fields.get("llm_api_status") == "success"
    assert status_fields.get("guardrail_status") == "success"


@pytest.mark.asyncio
async def test_bedrock_guardrail_status_failure():
    """
    Test that Bedrock guardrail sets correct status fields when the API endpoint fails.
    
    This test verifies that when Bedrock guardrail API is down/fails:
    1. The guardrail_information contains guardrail_status="failure"
    2. The status_fields.guardrail_status is set to "guardrail_failed_to_respond"
    3. The exception is still raised (maintaining existing behavior)
    """
    from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import AsyncMock, MagicMock, patch
    import httpx
    
    # Reset callbacks completely to avoid event loop conflicts
    litellm.callbacks = []
    await asyncio.sleep(0.1)
    
    # Setup custom logger to capture standard logging payload
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    
    # Create Bedrock guardrail
    bedrock_guard = BedrockGuardrail(
        guardrail_name="bedrock_guard",
        event_hook=GuardrailEventHooks.pre_call,
        guardrailIdentifier="test-id",
        guardrailVersion="1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
        aws_region_name="us-east-1",
    )
    
    # Mock network failure (endpoint down)
    with patch.object(bedrock_guard.async_handler, "post", AsyncMock(side_effect=httpx.ConnectError("Connection failed"))):
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "test content"}],
            "mock_response": "Hello",
            "metadata": {}
        }
        
        # Mock should_run_guardrail to return True
        with patch.object(bedrock_guard, 'should_run_guardrail', return_value=True):
            # Call guardrail (will raise exception on network failure)
            try:
                await bedrock_guard.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=None,
                    data=request_data,
                    call_type="completion"
                )
            except Exception:
                # Expected exception when endpoint is down
                pass
        
        # Call litellm.acompletion to trigger logging
        response = await litellm.acompletion(**request_data)
        await asyncio.sleep(1)
    
    # Check standard logging payload status fields
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_status"] == "guardrail_failed_to_respond"
    assert guardrail_info["guardrail_provider"] == "bedrock"
    
    # Check status fields
    status_fields = test_custom_logger.standard_logging_payload.get("status_fields", {})
    assert status_fields.get("llm_api_status") == "success"
    assert status_fields.get("guardrail_status") == "guardrail_failed_to_respond"


@pytest.mark.asyncio
async def test_noma_guardrail_status_blocked():
    """
    Test that Noma guardrail sets correct status fields when blocking content.
    
    This test verifies that when Noma guardrail blocks content (verdict=False):
    1. The guardrail_information contains guardrail_status="blocked"
    2. The status_fields.guardrail_status is set to "guardrail_intervened"
    3. The status_fields.llm_api_status remains "success"
    """
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import AsyncMock, MagicMock, patch
    
    # Reset callbacks completely to avoid event loop conflicts
    litellm.callbacks = []
    await asyncio.sleep(0.1)  # Let previous callbacks finish
    
    # Setup custom logger to capture standard logging payload
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    
    # Create Noma guardrail
    noma_guard = NomaGuardrail(
        guardrail_name="noma_guard",
        event_hook=GuardrailEventHooks.pre_call,
        api_key="test-key",
        monitor_mode=False,
    )
    
    # Mock blocked response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "verdict": False,
        "aggregatedScanResult": True,
        "originalResponse": {
            "prompt": {
                "topicDetector": {"harmful": {"result": True}}
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(noma_guard.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "harmful content"}],
            "mock_response": "Hello",
            "metadata": {}
        }
        
        # Mock should_run_guardrail to return True
        with patch.object(noma_guard, 'should_run_guardrail', return_value=True):
            # Call guardrail (will raise exception on block)
            try:
                await noma_guard.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=None,
                    data=request_data,
                    call_type="completion"
                )
            except Exception:
                pass
        
        # Call litellm.acompletion to trigger logging
        response = await litellm.acompletion(**request_data)
        await asyncio.sleep(1)
    
    # Check standard logging payload status fields
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_status"] == "guardrail_intervened"
    assert guardrail_info["guardrail_provider"] == "noma"
    
    # Check status fields
    status_fields = test_custom_logger.standard_logging_payload.get("status_fields", {})
    assert status_fields.get("llm_api_status") == "success"
    assert status_fields.get("guardrail_status") == "guardrail_intervened"


@pytest.mark.asyncio
async def test_noma_guardrail_status_success():
    """
    Test that Noma guardrail sets correct status fields when allowing content.
    
    This test verifies that when Noma guardrail allows content (verdict=True):
    1. The guardrail_information contains guardrail_status="success"
    2. The status_fields.guardrail_status is set to "success"
    3. The status_fields.llm_api_status is "success"
    """
    from litellm.proxy.guardrails.guardrail_hooks.noma.noma import NomaGuardrail
    from litellm.proxy._types import UserAPIKeyAuth
    from unittest.mock import AsyncMock, MagicMock, patch
    
    # Reset callbacks completely to avoid event loop conflicts
    litellm.callbacks = []
    await asyncio.sleep(0.1)  # Let previous callbacks finish
    
    # Setup custom logger to capture standard logging payload
    test_custom_logger = CustomLoggerForTesting()
    litellm.callbacks = [test_custom_logger]
    
    # Create Noma guardrail
    noma_guard = NomaGuardrail(
        guardrail_name="noma_guard",
        event_hook=GuardrailEventHooks.pre_call,
        api_key="test-key",
        monitor_mode=False,
    )
    
    # Mock success response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "verdict": True,
        "aggregatedScanResult": False,
        "originalResponse": {"prompt": {}}
    }
    mock_response.raise_for_status = MagicMock()
    with patch.object(noma_guard.async_handler, "post", AsyncMock(return_value=mock_response)):
        request_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "safe content"}],
            "mock_response": "Hello",
            "metadata": {}
        }
        
        # Mock should_run_guardrail to return True
        with patch.object(noma_guard, 'should_run_guardrail', return_value=True):
            await noma_guard.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=None,
                data=request_data,
                call_type="completion"
            )
        
        # Call litellm.acompletion to trigger logging
        response = await litellm.acompletion(**request_data)
        await asyncio.sleep(1)
    
    # Check standard logging payload status fields
    assert test_custom_logger.standard_logging_payload is not None
    assert test_custom_logger.standard_logging_payload["guardrail_information"] is not None
    assert isinstance(test_custom_logger.standard_logging_payload["guardrail_information"], list)
    assert len(test_custom_logger.standard_logging_payload["guardrail_information"]) > 0
    
    guardrail_info = test_custom_logger.standard_logging_payload["guardrail_information"][0]
    assert guardrail_info["guardrail_status"] == "success"
    assert guardrail_info["guardrail_provider"] == "noma"
    
    # Check status fields
    status_fields = test_custom_logger.standard_logging_payload.get("status_fields", {})
    assert status_fields.get("llm_api_status") == "success"
    assert status_fields.get("guardrail_status") == "success"


def test_guardrail_status_fields_computation():
    """
    Test that status fields are computed correctly from guardrail information.
    
    This unit test verifies the _get_status_fields function correctly maps:
    - guardrail_status="blocked" -> status_fields.guardrail_status="guardrail_intervened" (legacy)
    - guardrail_status="guardrail_intervened" -> status_fields.guardrail_status="guardrail_intervened"
    - guardrail_status="success" -> status_fields.guardrail_status="success"
    - guardrail_status="failure" -> status_fields.guardrail_status="guardrail_failed_to_respond" (legacy)
    - guardrail_status="guardrail_failed_to_respond" -> status_fields.guardrail_status="guardrail_failed_to_respond"
    - no guardrail -> status_fields.guardrail_status="not_run"
    """
    from litellm.litellm_core_utils.litellm_logging import _get_status_fields
    
    # Test guardrail_intervened status (content was blocked by guardrail)
    # guardrail_information is now a list
    intervened_info = [{"guardrail_status": "guardrail_intervened"}]
    status_fields_intervened = _get_status_fields(
        status="success",
        guardrail_information=intervened_info,
        error_str=None
    )
    assert status_fields_intervened["llm_api_status"] == "success"
    assert status_fields_intervened["guardrail_status"] == "guardrail_intervened"
    
    # Test legacy blocked status (for backward compatibility)
    blocked_info = [{"guardrail_status": "blocked"}]
    status_fields_blocked = _get_status_fields(
        status="success",
        guardrail_information=blocked_info,
        error_str=None
    )
    assert status_fields_blocked["llm_api_status"] == "success"
    assert status_fields_blocked["guardrail_status"] == "guardrail_intervened"
    
    # Test success status
    success_info = [{"guardrail_status": "success"}]
    status_fields_success = _get_status_fields(
        status="success",
        guardrail_information=success_info,
        error_str=None
    )
    assert status_fields_success["llm_api_status"] == "success"
    assert status_fields_success["guardrail_status"] == "success"
    
    # Test guardrail_failed_to_respond status
    failed_info = [{"guardrail_status": "guardrail_failed_to_respond"}]
    status_fields_failed = _get_status_fields(
        status="failure",
        guardrail_information=failed_info,
        error_str=None
    )
    assert status_fields_failed["llm_api_status"] == "failure"
    assert status_fields_failed["guardrail_status"] == "guardrail_failed_to_respond"
    
    # Test legacy failure status (for backward compatibility)
    failure_info = [{"guardrail_status": "failure"}]
    status_fields_failure = _get_status_fields(
        status="failure",
        guardrail_information=failure_info,
        error_str=None
    )
    assert status_fields_failure["llm_api_status"] == "failure"
    assert status_fields_failure["guardrail_status"] == "guardrail_failed_to_respond"
    
    # Test no guardrail run
    no_guardrail = None
    status_fields_no_guardrail = _get_status_fields(
        status="success",
        guardrail_information=no_guardrail,
        error_str=None
    )
    assert status_fields_no_guardrail["llm_api_status"] == "success"
    assert status_fields_no_guardrail["guardrail_status"] == "not_run"