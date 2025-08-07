import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.responses.litellm_completion_transformation import session_handler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
)


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_for_previous_response_id():
    """
    Test get_chat_completion_message_history_for_previous_response_id with mock data
    """
    # Mock data based on the provided spend logs (simplified version)
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb",
            "call_type": "aresponses",
            "api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "spend": 0.004803,
            "total_tokens": 329,
            "prompt_tokens": 11,
            "completion_tokens": 318,
            "startTime": "2025-05-30T03:17:06.703+00:00",
            "endTime": "2025-05-30T03:17:11.894+00:00",
            "model": "claude-3-5-sonnet-latest",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "who is Michael Jordan",
                "model": "anthropic/claude-3-5-sonnet-latest",
            },
            "response": {
                "id": "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb",
                "model": "claude-3-5-sonnet-20241022",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Michael Jordan (born February 17, 1963) is widely considered the greatest basketball player of all time. Here are some key points about him...",
                            "tool_calls": None,
                            "function_call": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "created": 1748575031,
                "usage": {
                    "total_tokens": 329,
                    "prompt_tokens": 11,
                    "completion_tokens": 318,
                },
            },
            "status": "success",
        },
        {
            "request_id": "chatcmpl-370760c9-39fa-4db7-b034-d1f8d933c935",
            "call_type": "aresponses",
            "api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "spend": 0.010437,
            "total_tokens": 967,
            "prompt_tokens": 339,
            "completion_tokens": 628,
            "startTime": "2025-05-30T03:17:28.600+00:00",
            "endTime": "2025-05-30T03:17:39.921+00:00",
            "model": "claude-3-5-sonnet-latest",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "can you tell me more about him",
                "model": "anthropic/claude-3-5-sonnet-latest",
                "previous_response_id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmFudGhyb3BpYzttb2RlbF9pZDplMGYzMDJhMTQxMmU3ODQ3MGViYjI4Y2JlZDAxZmZmNWY4OGMwZDMzMWM2NjdlOWYyYmE0YjQxM2M2ZmJkMjgyO3Jlc3BvbnNlX2lkOmNoYXRjbXBsLTkzNWI4ZGFkLWZkYzItNDY2ZS1hOGNhLWUyNmU1YThhMjFiYg==",
            },
            "response": {
                "id": "chatcmpl-370760c9-39fa-4db7-b034-d1f8d933c935",
                "model": "claude-3-5-sonnet-20241022",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Here's more detailed information about Michael Jordan...",
                            "tool_calls": None,
                            "function_call": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "created": 1748575059,
                "usage": {
                    "total_tokens": 967,
                    "prompt_tokens": 339,
                    "completion_tokens": 628,
                },
            },
            "status": "success",
        },
    ]

    # Mock the get_all_spend_logs_for_previous_response_id method
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        # Test the function
        previous_response_id = "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb"
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id
        )

        # Verify the mock was called with correct parameters
        mock_get_spend_logs.assert_called_once_with(previous_response_id)

        # Verify the returned ChatCompletionSession structure
        assert "messages" in result
        assert "litellm_session_id" in result

        # Verify session_id is extracted correctly
        assert result["litellm_session_id"] == "a96757c4-c6dc-4c76-b37e-e7dfa526b701"

        # Verify messages structure
        messages = result["messages"]
        assert len(messages) == 4  # 2 user messages + 2 assistant messages

        # Check the message sequence
        # First user message
        assert messages[0].get("role") == "user"
        assert messages[0].get("content") == "who is Michael Jordan"

        # First assistant response
        assert messages[1].get("role") == "assistant"
        content_1 = messages[1].get("content", "")
        if isinstance(content_1, str):
            assert "Michael Jordan" in content_1
            assert content_1.startswith("Michael Jordan (born February 17, 1963)")

        # Second user message
        assert messages[2].get("role") == "user"
        assert messages[2].get("content") == "can you tell me more about him"

        # Second assistant response
        assert messages[3].get("role") == "assistant"
        content_3 = messages[3].get("content", "")
        if isinstance(content_3, str):
            assert "Here's more detailed information about Michael Jordan" in content_3


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_empty_spend_logs():
    """
    Test get_chat_completion_message_history_for_previous_response_id with empty spend logs
    """
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = []

        previous_response_id = "non-existent-id"
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id
        )

        # Verify empty result structure
        assert result.get("messages") == []
        assert result.get("litellm_session_id") is None


@pytest.mark.asyncio
async def test_e2e_cold_storage_successful_retrieval():
    """
    Test end-to-end cold storage functionality with successful retrieval of full proxy request from cold storage.
    """
    # Mock spend logs with cold storage object key in metadata
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-123",
            "session_id": "session-456",
            "metadata": '{"cold_storage_object_key": "s3://test-bucket/requests/session_456_req1.json"}',
            "proxy_server_request": '{"litellm_truncated": true}',  # Truncated payload
            "response": {
                "id": "chatcmpl-test-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I am an AI assistant."
                        }
                    }
                ]
            }
        }
    ]
    
    # Full proxy request data from cold storage
    full_proxy_request = {
        "input": "Hello, who are you?",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, who are you?"}]
    }
    
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs, \
    patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage, \
    patch("litellm.proxy.spend_tracking.cold_storage_handler.ColdStorageHandler._get_configured_cold_storage_custom_logger", return_value="s3"):
        
        # Setup mocks
        mock_get_spend_logs.return_value = mock_spend_logs
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(return_value=full_proxy_request)
        
        # Call the main function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-123"
        )
        
        # Verify cold storage was called with correct object key
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key.assert_called_once_with(
            object_key="s3://test-bucket/requests/session_456_req1.json"
        )
        
        # Verify result structure
        assert result.get("litellm_session_id") == "session-456"
        assert len(result.get("messages", [])) >= 1  # At least the assistant response


@pytest.mark.asyncio
async def test_e2e_cold_storage_fallback_to_truncated_payload():
    """
    Test end-to-end cold storage functionality when object key is missing, falling back to truncated payload.
    """
    # Mock spend logs without cold storage object key
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-789",
            "session_id": "session-999",
            "metadata": '{"user_api_key": "test-key"}',  # No cold storage object key
            "proxy_server_request": '{"input": "Truncated message", "model": "gpt-4"}',  # Regular payload
            "response": {
                "id": "chatcmpl-test-789",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "This is a response."
                        }
                    }
                ]
            }
        }
    ]
    
    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs, \
    patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage:
        
        # Setup mocks
        mock_get_spend_logs.return_value = mock_spend_logs
        
        # Call the main function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-789"
        )
        
        # Verify cold storage was NOT called since no object key in metadata
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key.assert_not_called()
        
        # Verify result structure
        assert result.get("litellm_session_id") == "session-999"
        assert len(result.get("messages", [])) >= 1  # At least the assistant response
