import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from litellm.proxy.batches_endpoints.batch_guardrail_utils import run_pre_call_guardrails_on_batch_file, _get_call_type_from_endpoint
from litellm.proxy._types import UserAPIKeyAuth

@pytest.fixture
def mock_proxy_logging_obj():
    mock_obj = MagicMock()
    mock_obj.pre_call_hook = AsyncMock()
    return mock_obj

@pytest.fixture
def mock_user_api_key_dict():
    return UserAPIKeyAuth()

@pytest.mark.asyncio
async def test_guardrail_runs_on_batch_file_messages(mock_proxy_logging_obj, mock_user_api_key_dict):
    # Create a batch JSONL file with messages
    req1 = {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "system", "content": "You are a helpful assistant."}]}}
    req2 = {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}]}}
    
    file_content = f"{json.dumps(req1)}\n{json.dumps(req2)}\n".encode("utf-8")
    
    await run_pre_call_guardrails_on_batch_file(
        file_content=file_content,
        proxy_logging_obj=mock_proxy_logging_obj,
        user_api_key_dict=mock_user_api_key_dict
    )
    
    # Verify pre_call_hook is called for each line
    assert mock_proxy_logging_obj.pre_call_hook.call_count == 2
    
    # Check args of the first call
    call_args_1 = mock_proxy_logging_obj.pre_call_hook.call_args_list[0].kwargs
    assert call_args_1["call_type"] == "acompletion"
    assert "messages" in call_args_1["data"]
    assert call_args_1["data"]["messages"][0]["content"] == "You are a helpful assistant."
    
    # Check args of the second call
    call_args_2 = mock_proxy_logging_obj.pre_call_hook.call_args_list[1].kwargs
    assert call_args_2["call_type"] == "acompletion"
    assert "messages" in call_args_2["data"]
    assert call_args_2["data"]["messages"][0]["content"] == "Hello!"

@pytest.mark.asyncio
async def test_guardrail_rejection_raises_exception(mock_proxy_logging_obj, mock_user_api_key_dict):
    # Configure mock to raise an exception simulating a guardrail block
    mock_proxy_logging_obj.pre_call_hook.side_effect = Exception("Content blocked by policy")
    
    req = {"custom_id": "bad-request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": [{"role": "user", "content": "bad stuff"}]}}
    file_content = f"{json.dumps(req)}\n".encode("utf-8")
    
    # Verify rejection propagates and includes the custom_id and line info
    with pytest.raises(Exception) as exc_info:
        await run_pre_call_guardrails_on_batch_file(
            file_content=file_content,
            proxy_logging_obj=mock_proxy_logging_obj,
            user_api_key_dict=mock_user_api_key_dict
        )
        
    assert "bad-request-1" in str(exc_info.value)
    assert "line 1" in str(exc_info.value)
    assert "Content blocked by policy" in str(exc_info.value)

@pytest.mark.asyncio
async def test_endpoint_to_call_type_mapping():
    assert _get_call_type_from_endpoint("/v1/chat/completions") == "acompletion"
    assert _get_call_type_from_endpoint("/v1/embeddings") == "aembedding"
    assert _get_call_type_from_endpoint("/v1/audio/transcriptions") == "acompletion" # Default fallback
    assert _get_call_type_from_endpoint("/custom/endpoint") == "acompletion"

@pytest.mark.asyncio
async def test_malformed_jsonl_handling(mock_proxy_logging_obj, mock_user_api_key_dict):
    # Test file with some valid lines and some malformed/empty lines
    req1 = {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5", "messages": [{"role": "user", "content": "Hi"}]}}
    
    file_content = f"{json.dumps(req1)}\nthis is not json\n\n{{\"not_complete\": true \n".encode("utf-8")
    
    # Should complete without error, skipping malformed lines
    await run_pre_call_guardrails_on_batch_file(
        file_content=file_content,
        proxy_logging_obj=mock_proxy_logging_obj,
        user_api_key_dict=mock_user_api_key_dict
    )
    
    # Only the first valid line should have triggered a call
    assert mock_proxy_logging_obj.pre_call_hook.call_count == 1

@pytest.mark.asyncio
async def test_skips_lines_without_messages(mock_proxy_logging_obj, mock_user_api_key_dict):
    # Request without messages
    req1 = {"custom_id": "req-1", "method": "POST", "url": "/v1/models", "body": {}}
    # Request with messages
    req2 = {"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4", "messages": []}}
    
    file_content = f"{json.dumps(req1)}\n{json.dumps(req2)}\n".encode("utf-8")
    
    await run_pre_call_guardrails_on_batch_file(
        file_content=file_content,
        proxy_logging_obj=mock_proxy_logging_obj,
        user_api_key_dict=mock_user_api_key_dict
    )
    
    # Only req2 should trigger the hook since req1 has no messages
    assert mock_proxy_logging_obj.pre_call_hook.call_count == 1
    
