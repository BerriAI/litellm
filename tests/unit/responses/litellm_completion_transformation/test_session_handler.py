import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType, SimpleNamespace
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import pytest
import httpx
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, PREVIOUS_RESPONSE_IDS_METADATA_KEY
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.prompt_templates.factory import EMPTY_TEXT_PLACEHOLDER, THOUGHT_SIGNATURE_SEPARATOR
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field
from litellm.litellm_core_utils.tool_fields import chat_member_name, truncate_tool_name
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.spend_tracking.legacy_spend_log_rows import (
    _get_proxy_server_request_for_spend_logs_payload,
    _sanitize_request_body_for_spend_logs_payload,
)
from litellm.responses.litellm_completion_transformation import session_handler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
    StoredTurns,
    _normalize_redacted_tool_call_arguments,
)
from litellm.responses.litellm_completion_transformation.tool_search import CLIENT_TOOL_SEARCH_FUNCTION_NAME
from litellm.responses.litellm_completion_transformation.transformation import (
    ChatCompletionSession,
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Message


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
            "api_key": "sk-test-mock-api-key-123",
            "spend": 0.004803,
            "total_tokens": 329,
            "prompt_tokens": 11,
            "completion_tokens": 318,
            "startTime": "2025-05-30T03:17:06.703+00:00",
            "endTime": "2025-05-30T03:17:11.894+00:00",
            "model": "claude-sonnet-4-5-20250929",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "who is Michael Jordan",
                "model": "anthropic/claude-sonnet-4-5-20250929",
            },
            "response": {
                "id": "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb",
                "model": "claude-sonnet-4-5-20250929",
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
            "api_key": "sk-test-mock-api-key-123",
            "spend": 0.010437,
            "total_tokens": 967,
            "prompt_tokens": 339,
            "completion_tokens": 628,
            "startTime": "2025-05-30T03:17:28.600+00:00",
            "endTime": "2025-05-30T03:17:39.921+00:00",
            "model": "claude-sonnet-4-5-20250929",
            "session_id": "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "proxy_server_request": {
                "input": "can you tell me more about him",
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "previous_response_id": "resp_bGl0ZWxsbTpjdXN0b21fbGxtX3Byb3ZpZGVyOmFudGhyb3BpYzttb2RlbF9pZDplMGYzMDJhMTQxMmU3ODQ3MGViYjI4Y2JlZDAxZmZmNWY4OGMwZDMzMWM2NjdlOWYyYmE0YjQxM2M2ZmJkMjgyO3Jlc3BvbnNlX2lkOmNoYXRjbXBsLTkzNWI4ZGFkLWZkYzItNDY2ZS1hOGNhLWUyNmU1YThhMjFiYg==",
            },
            "response": {
                "id": "chatcmpl-370760c9-39fa-4db7-b034-d1f8d933c935",
                "model": "claude-sonnet-4-5-20250929",
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
        mock_get_spend_logs.assert_called_once_with(previous_response_id, ())

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
                            "content": "I am an AI assistant.",
                        },
                    }
                ],
            },
        }
    ]

    # Full proxy request data from cold storage
    full_proxy_request = {
        "input": "Hello, who are you?",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
    }

    with (
        patch.object(
            ResponsesSessionHandler,
            "get_all_spend_logs_for_previous_response_id",
            new_callable=AsyncMock,
        ) as mock_get_spend_logs,
        patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage,
        patch("litellm.cold_storage_custom_logger", return_value="s3"),
    ):
        # Setup mocks
        mock_get_spend_logs.return_value = mock_spend_logs
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(
            return_value=full_proxy_request
        )

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
                            "content": "This is a response.",
                        },
                    }
                ],
            },
        }
    ]

    with (
        patch.object(
            ResponsesSessionHandler,
            "get_all_spend_logs_for_previous_response_id",
            new_callable=AsyncMock,
        ) as mock_get_spend_logs,
        patch.object(session_handler, "COLD_STORAGE_HANDLER") as mock_cold_storage,
    ):
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


@pytest.mark.asyncio
async def test_should_check_cold_storage_for_full_payload():
    """
    Test _should_check_cold_storage_for_full_payload returns True for proxy server requests with truncated content
    """

    # Test case 1: Proxy server request with truncated PDF content (should return True)
    proxy_request_with_truncated_pdf = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [
                    {
                        "text": "what was datadogs largest source of operating cash ? quote the section you saw ",
                        "type": "input_text",
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:application/pdf;base64,JVBERi0xLjcKJYGBgYEKCjcgMCBvYmoKPDwKL0ZpbHRlciAvRmxhdGVEZWNvZGUKL0xlbmd0aCA1NjcxCj4+CnN0cmVhbQp4nO1dW4/cthV+31+h5wKVeb8AhoG9Bn0I0DYL9NlInQBFHKSpA+Tnl5qRNNRIn8ij4WpnbdqAsRaX90Oe23cOWyH94U/Dwt+/ttF/neKt59675sfPN/+9Ubp1MvwRjfAtN92fRkgn2+5jI5Xyre9++fdPN//6S/NrqCFax4XqvnVtn/631FLogjfd339+1xx/+P3nm3ffyebn/92ww2Bc46zRrGv/p5vWMOmb+N9Qb/4xtOEazn2oH3rjfV3fDTj+t6s7+zjU5XFdFxo9fPs8/CiaX26cYmc/svDjhlF+Pv7QNdT30/9wbI8dFjK0cfzhUO8wPjaOr/Eq/v/d8827vzfv37/7/v5vD6HKhw93D/c3755UI3jYuOb5p7Dsh53nYQtZqyUXugn71Dx/vnnPmHQfmuf/3HDdKhY2z8jwq8//broSjkrE/aHEtZIxZhQ/VbHHKqoVQhsv7KmKgyUWdcPkoUSHaYgwmmhk5liFt85w45WcDUC0RgntpTp1o44lMhCpl95eNOZ+AI/f3988Pp9tAV/dAu5VKz0Ls+SB0vstgNNZWTUDtw1vqC65oahKctUW5gl3etg1EnXSCWplzHewG7gDoq9jWu28DYuzPB3NucmZzm3UmvFcLI/aMpcxT0w1AvUi2aFEsJZLprxMF0xIQzmXQQArRwA2Fo/YCm7P13LxdIrodIa7QIojL5zekqblloeuwkiGI3o7jk+zMEJ9vqW+NWFDtTDnU7KtCpet1WZGHqyVxjLNxPlcdcudsU648xnNOxmIY95Wf3JduAidF3q+bgtVUC/s6VAgZ/TUE9q8oD+DCwM2qA/UFD/eWqY1RrFQ61QgUYFDBRYV3GOKkSPFaEQxQqqWWRcGzZ3u... (litellm_truncated 1197576 chars)",
                    },
                ],
            }
        ],
        "model": "anthropic/claude-4-sonnet-20250514",
        "stream": True,
        "litellm_trace_id": "16b86861-c120-4ecb-865b-4d2238bfd8f0",
    }

    # Test case 2: Regular proxy request without truncation (should return False)
    proxy_request_regular = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": "Hello, this is a regular message",
            }
        ],
        "model": "anthropic/claude-4-sonnet-20250514",
        "stream": True,
    }

    # Test case 3: Empty request (should return True)
    proxy_request_empty = {}

    # Test case 4: None request (should return True)
    proxy_request_none = None

    with patch("litellm.cold_storage_custom_logger", return_value="s3"):
        # Test case 1: Should return True for truncated content
        result1 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_request_with_truncated_pdf)
        assert result1 == True, "Should return True for proxy request with truncated PDF content"

        # Test case 2: Should return False for regular content
        result2 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_request_regular)
        assert result2 == False, "Should return False for regular proxy request without truncation"

        # Test case 3: Should return True for empty request
        result3 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_request_empty)
        assert result3 == True, "Should return True for empty proxy request"

        # Test case 4: Should return True for None request
        result4 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_request_none)
        assert result4 == True, "Should return True for None proxy request"

    # Test case 5: Should return False when cold storage is not configured
    with patch.object(litellm, "cold_storage_custom_logger", None):
        result5 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(proxy_request_with_truncated_pdf)
        assert result5 == False, "Should return False when cold storage is not configured, even with truncated content"


@pytest.mark.asyncio
async def test_get_chat_completion_message_history_empty_response_dict():
    """
    Test that empty response dict is handled correctly without processing.
    This tests the fix for response validation to check for empty dict responses.
    """
    from unittest.mock import AsyncMock, patch

    # Mock spend logs with empty response dict
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-test-empty-response",
            "call_type": "aresponses",
            "api_key": "test_key",
            "spend": 0.001,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "startTime": "2025-01-15T10:30:00.000+00:00",
            "endTime": "2025-01-15T10:30:01.000+00:00",
            "model": "gpt-4",
            "session_id": "test-session",
            "proxy_server_request": {"input": "test input", "model": "gpt-4"},
            "response": {},  # Empty dict - should not be processed
        }
    ]

    with patch.object(ResponsesSessionHandler, "get_all_spend_logs_for_previous_response_id") as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        # Call the function
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-test-empty-response"
        )

        # Verify that user message was added but no assistant response
        # Since response is empty dict, no assistant response should be processed
        # But user input from proxy_server_request should still be included
        messages = result["messages"]
        assert len(messages) == 1  # Only user message, no assistant response
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "test input"

        # Verify the session was still created correctly
        assert result["litellm_session_id"] == "test-session"


def _chat_completion_response(request_id: str, content: str) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": 1748575031,
        "model": "claude-haiku-4-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


class _FakePrismaDB:
    """Serves the next queued result per history read; the referenced-row read peeks at it, dropping a miss."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def query_raw(self, query, request_ids, *path):
        self.calls.append(tuple(request_ids))
        if "matching_session" in query:
            return list(self._results.pop(0)) if self._results else []
        queued = self._results[0] if self._results else []
        referenced = [row for row in queued if row.get("request_id") in request_ids]
        if not referenced and self._results:
            self._results.pop(0)
        return referenced


class _FakePrismaClient:
    def __init__(self, results):
        self.db = _FakePrismaDB(results)


def _spend_log(request_id: str, session_id: str, prompt: str, answer: str) -> dict[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": session_id,
        "proxy_server_request": {
            "input": [{"role": "user", "content": prompt}],
            "model": "claude-bridge",
        },
        "response": _chat_completion_response(request_id, answer),
    }


@pytest.fixture
def instant_session_lookup_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm.constants, "RESPONSES_SESSION_LOOKUP_RETRY_INTERVAL", 0.0)


@pytest.mark.asyncio
async def test_message_history_reconstructs_list_shaped_input():
    """
    The Responses API sends `input` as a list of items, which is what lands in the stored
    proxy_server_request. The user turns have to survive session reconstruction.
    """
    request_id = "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb"
    mock_spend_logs = [
        _spend_log(
            request_id,
            "a96757c4-c6dc-4c76-b37e-e7dfa526b701",
            "Remember this: my favorite color is chartreuse.",
            "OK",
        )
    ]

    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(request_id)

    messages = result["messages"]
    assert [(message.get("role"), message.get("content")) for message in messages] == [
        ("user", "Remember this: my favorite color is chartreuse."),
        ("assistant", "OK"),
    ]
    assert result["litellm_session_id"] == "a96757c4-c6dc-4c76-b37e-e7dfa526b701"


@pytest.mark.asyncio
async def test_message_history_retries_a_spend_log_the_batch_writer_has_not_flushed_yet(
    instant_session_lookup_retries: None,
):
    """
    A follow-up sent right after the previous turn can beat that turn's spend log to the
    DB. The lookup has to try again instead of handing back an empty conversation.
    """
    request_id = "chatcmpl-6c1f5f6c-6a2b-4c62-8d1f-0d9d4ce0a1b2"
    session_id = "b7d0a5b0-6d20-4a68-9d24-6ba0f6d1f1a3"
    spend_log = _spend_log(
        request_id,
        session_id,
        "Remember this: my favorite color is chartreuse.",
        "OK",
    )
    fake_prisma_client = _FakePrismaClient(results=[[], [spend_log]])

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(request_id)

    messages = result["messages"]
    assert [(message.get("role"), message.get("content")) for message in messages] == [
        ("user", "Remember this: my favorite color is chartreuse."),
        ("assistant", "OK"),
    ]
    assert result["litellm_session_id"] == session_id
    assert [call[0] for call in fake_prisma_client.db.calls] == [request_id, request_id, request_id]


@pytest.mark.asyncio
async def test_message_history_reconstructs_every_turn_of_the_session_in_order():
    session_id = "5c5f9a3e-1c86-4c0e-9d7c-0a54b8a0f2f1"
    first_request_id = "chatcmpl-1111"
    second_request_id = "chatcmpl-2222"
    fake_prisma_client = _FakePrismaClient(
        results=[
            [
                _spend_log(first_request_id, session_id, "My favorite color is chartreuse.", "Got it."),
                _spend_log(second_request_id, session_id, "And my favorite city is Lisbon.", "Noted."),
            ]
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            second_request_id
        )

    messages = result["messages"]
    assert [(message.get("role"), message.get("content")) for message in messages] == [
        ("user", "My favorite color is chartreuse."),
        ("assistant", "Got it."),
        ("user", "And my favorite city is Lisbon."),
        ("assistant", "Noted."),
    ]
    assert result["litellm_session_id"] == session_id


@pytest.mark.asyncio
async def test_message_history_skips_the_turns_the_provider_rejected():
    session_id: Final = "5c5f9a3e-1c86-4c0e-9d7c-0a54b8a0f2f1"
    answered_request_id: Final = "chatcmpl-1111"
    rejected_request_id: Final = "chatcmpl-2222"
    answered_turn: Final = {
        **_spend_log(answered_request_id, session_id, "What is the weather in Paris?", "Checking."),
        "status": "success",
    }
    rejected_turn: Final = {
        **_spend_log(rejected_request_id, session_id, "Never mind. Just say hello", ""),
        "status": "failure",
        "response": {},
    }
    fake_prisma_client: Final = _FakePrismaClient(results=[[answered_turn, rejected_turn]])

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            answered_request_id
        )

    assert [(message.get("role"), message.get("content")) for message in result["messages"]] == [
        ("user", "What is the weather in Paris?"),
        ("assistant", "Checking."),
    ]


@pytest.mark.asyncio
async def test_session_lookup_stops_retrying_once_the_budget_is_spent(
    instant_session_lookup_retries: None,
):
    fake_prisma_client = _FakePrismaClient(results=[])

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        spend_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            "chatcmpl-does-not-exist"
        )

    assert spend_logs == []
    assert len(fake_prisma_client.db.calls) == litellm.constants.RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_message_history_looks_up_the_decoded_chat_completion_id():
    """
    A `previous_response_id` handed back by the proxy is base64 encoded; spend logs store
    the bare chat completion id, so that is what the lookup has to query on.
    """
    request_id = "chatcmpl-935b8dad-fdc2-466e-a8ca-e26e5a8a21bb"
    encoded_response_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="anthropic",
        model_id="e0f302a1412e78470ebb28cbed01fff5f88c0d331c667e9f2ba4b413c6fbd282",
        response_id=request_id,
    )
    fake_prisma_client = _FakePrismaClient(results=[[_spend_log(request_id, "session-a", "Hello.", "Hi.")]])

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(encoded_response_id)

    assert len(fake_prisma_client.db.calls) == 2
    assert all(request_id in call for call in fake_prisma_client.db.calls)


class _StoredSpendLogs:
    def __init__(self, rows: tuple[dict, ...]) -> None:
        self.rows = rows

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [row for row in self.rows if row["session_id"] in sessions]


@pytest.mark.asyncio
async def test_message_history_replays_a_bridged_mcp_turn_stored_under_the_client_response_id():
    client_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="anthropic",
        model_id="e0f302a1412e78470ebb28cbed01fff5f88c0d331c667e9f2ba4b413c6fbd282",
        response_id="chatcmpl-mcp-final-round",
    )
    session_id: Final = "mcp-session"
    stored: Final = _StoredSpendLogs(
        (
            {
                "request_id": "list-tools-row",
                "call_type": "list_mcp_tools",
                "session_id": session_id,
                "proxy_server_request": {},
                "response": [{"name": "demo-echo"}],
            },
            {
                "request_id": client_response_id,
                "call_type": "aresponses",
                "session_id": session_id,
                "proxy_server_request": {"input": "Call echo with PING.", "model": "claude-bridge"},
                "response": {
                    "id": client_response_id,
                    "object": "response",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "rs_mcp-final-round",
                            "summary": [{"type": "summary_text", "text": "The echo tool returned PING."}],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "PING", "annotations": []}],
                        },
                        *LiteLLM_Proxy_MCP_Handler.add_mcp_output_elements_to_response(
                            ResponsesAPIResponse(id=client_response_id, created_at=0, output=[]),
                            mcp_tools_fetched=[{"name": "demo-echo"}],
                            tool_results=[{"tool_name": "demo-echo", "result": "PING"}],
                        ).output,
                    ],
                },
            },
            {
                "request_id": "call-tool-row",
                "call_type": "call_mcp_tool",
                "session_id": session_id,
                "proxy_server_request": {"name": "echo", "arguments": {"value": "PING"}},
                "response": {"content": [{"type": "text", "text": "PING"}], "is_error": False},
            },
        )
    )

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored)):
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            client_response_id
        )

    assert [(message.get("role"), _text_of(message.get("content"))) for message in result["messages"]] == [
        ("user", "Call echo with PING."),
        ("assistant", "PING"),
    ]
    assert result["messages"][-1].get("reasoning_content") == "The echo tool returned PING."
    assert result["litellm_session_id"] == session_id


def _text_of(content: object) -> object:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


@pytest.mark.asyncio
async def test_session_lookup_does_not_retry_when_spend_logs_are_disabled(
    instant_session_lookup_retries: None,
):
    """
    A deployment that writes no spend logs has nothing to wait for, so the miss path keeps
    the single query it always had.
    """
    fake_prisma_client = _FakePrismaClient(results=[])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client),
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
    ):
        spend_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            "chatcmpl-does-not-exist"
        )

    assert spend_logs == []
    assert len(fake_prisma_client.db.calls) == 1


def test_normalize_redacted_arguments_skips_custom_tool_calls():
    """Custom tool calls have no .function; the normalizer must skip them, not crash (session replay path)."""
    message = Message(
        content=None,
        tool_calls=[
            {"id": "call_c", "type": "custom", "custom": {"name": "run_code", "input": "print(1)"}},
            {
                "id": "call_f",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "redacted-by-litellm"},
            },
        ],
    )

    _normalize_redacted_tool_call_arguments(message)

    assert message.tool_calls[0].custom.input == "print(1)"
    assert message.tool_calls[1].function.arguments == "{}"


@pytest.mark.asyncio
async def test_message_history_normalizes_redacted_tool_call_arguments():
    """Sessions stored with turn_off_message_logging hold the bare sentinel
    in tool-call arguments; replay must normalize it to valid JSON."""
    mock_spend_logs = [
        {
            "request_id": "chatcmpl-redacted-1",
            "call_type": "aresponses",
            "session_id": "sess-redacted",
            "proxy_server_request": {
                "input": "what is the weather in sf",
                "model": "gpt-4o",
            },
            "response": {
                "id": "chatcmpl-redacted-1",
                "model": "gpt-4o",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "redacted-by-litellm",
                                    },
                                }
                            ],
                            "function_call": None,
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "created": 1748575031,
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            },
            "status": "success",
        }
    ]

    with (
        patch.object(  # test-quality-ok: the handler has no DI seam for the spend-log fetch; every test in this file stubs this same boundary
            ResponsesSessionHandler,
            "get_all_spend_logs_for_previous_response_id",
            new_callable=AsyncMock,
        ) as mock_get_spend_logs
    ):
        mock_get_spend_logs.return_value = mock_spend_logs

        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-redacted-1"
        )

    assistant_message = result["messages"][-1]
    tool_call = assistant_message.tool_calls[0]
    assert tool_call.function.arguments == "{}"
    assert json.loads(tool_call.function.arguments) == {}


@pytest.mark.asyncio
async def test_message_history_replays_real_key_named_tool_payloads() -> None:
    request_id: Final = "chatcmpl-tool-payload"
    function_arguments: Final = {"sort_key": "created_at", "access_level": "admin"}
    function_output: Final = {
        "status": "active",
        "token_type": "bearer",
        "partition_key": "tenant_42",
    }
    responses_request_body: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "input": [
            {"role": "user", "content": "Fetch my account settings."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_settings",
                "arguments": function_arguments,
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": function_output,
            },
            {"role": "user", "content": "Acknowledge with OK"},
        ],
        "aws_secret_access_key": "AKIAEXAMPLESECRET",
    }

    with patch(
        "litellm.proxy.spend_tracking.spend_tracking_utils.should_store_prompts_and_responses_in_spend_logs",
        return_value=True,
    ):
        proxy_server_request: Final = json.loads(
            _get_proxy_server_request_for_spend_logs_payload(
                metadata={},
                litellm_params={"proxy_server_request": {"body": responses_request_body}},
                kwargs={},
            )
        )

    spend_log: Final = {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "session-tool-payload",
        "proxy_server_request": proxy_server_request,
        "response": _chat_completion_response(request_id, "OK"),
    }

    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = [spend_log]
        result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            request_id
        )

    assistant_message: Final = result["messages"][1]
    tool_message: Final = result["messages"][2]
    assert json.loads(assistant_message["tool_calls"][0]["function"]["arguments"]) == function_arguments
    assert json.loads(tool_message["content"]) == function_output


async def test_message_history_keeps_the_joined_name_of_a_stored_self_named_namespace_member() -> None:
    request_id: Final = "chatcmpl-self-named-namespace"
    responses_request_body: Final = {
        "model": "anthropic/claude-sonnet-4-5",
        "tools": [
            {
                "type": "namespace",
                "name": "search",
                "tools": [{"type": "function", "name": "search", "parameters": {"type": "object", "properties": {}}}],
            }
        ],
        "input": [
            {"role": "user", "content": "Search for the release notes."},
            {"type": "function_call", "call_id": "call_1", "name": "search", "namespace": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "release notes found"},
        ],
    }

    with patch(
        "litellm.proxy.spend_tracking.spend_tracking_utils.should_store_prompts_and_responses_in_spend_logs",
        return_value=True,
    ):
        proxy_server_request: Final = json.loads(
            _get_proxy_server_request_for_spend_logs_payload(
                metadata={},
                litellm_params={"proxy_server_request": {"body": responses_request_body}},
                kwargs={},
            )
        )

    spend_log: Final = {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "session-self-named-namespace",
        "proxy_server_request": proxy_server_request,
        "response": _chat_completion_response(request_id, "Found them."),
    }

    with patch.object(
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = [spend_log]
        result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            request_id
        )

    assert [call["function"]["name"] for call in result["messages"][1]["tool_calls"]] == ["search__search"]


def _turn_spend_log(
    turn_input: str | list[dict[str, str]], answer: str, compaction_blocks: list[dict[str, str]] | None = None
) -> dict[str, object]:
    compaction_fields: Final = (
        {"provider_specific_fields": {"compaction_blocks": compaction_blocks}} if compaction_blocks else {}
    )
    return {
        "request_id": f"resp_{len(turn_input)}",
        "session_id": "session-compacted",
        "proxy_server_request": json.dumps({"input": turn_input}),
        "response": {
            "id": f"chatcmpl-{len(turn_input)}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer, **compaction_fields}}],
        },
    }


def _role_and_content(message: object) -> tuple[object, object]:
    if isinstance(message, dict):
        return message.get("role"), message.get("content")
    return getattr(message, "role", None), getattr(message, "content", None)


async def _history_from_stored_turns(
    spend_logs: tuple[dict[str, object], ...], previous_response_id: str
) -> ChatCompletionSession:
    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredSpendLogs(spend_logs))):
        return await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            previous_response_id
        )


@pytest.mark.asyncio
async def test_previous_response_history_starts_at_the_newest_turn_that_sent_a_compaction_item():
    first_turn: Final = [{"role": "user", "content": "Plan a Tokyo trip."}]
    compaction_item: Final = {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
    }
    second_turn: Final = [
        *first_turn,
        {"role": "assistant", "content": "Day one, Asakusa."},
        compaction_item,
        {"role": "user", "content": "And Osaka?"},
    ]
    spend_logs: Final = (
        _turn_spend_log(first_turn, "Day one, Asakusa."),
        _turn_spend_log(second_turn, "Dotonbori."),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_4")

    assert [_role_and_content(message) for message in session["messages"]] == [
        ("user", "Turn one planned Tokyo."),
        ("user", "And Osaka?"),
        ("assistant", "Dotonbori."),
    ]


@pytest.mark.parametrize(
    "first_turn_input",
    [pytest.param([{"role": "user", "content": "Hi."}], id="item_list"), pytest.param("Hi.", id="plain_string")],
)
@pytest.mark.asyncio
async def test_previous_response_history_starts_at_the_newest_answer_that_compacted(
    first_turn_input: str | list[dict[str, str]],
):
    compaction_blocks: Final = [{"type": "compaction", "content": "Turn one planned Tokyo."}]
    spend_logs: Final = (
        _turn_spend_log(first_turn_input, "Hello."),
        _turn_spend_log(
            [{"role": "developer", "content": "Answer briefly."}, {"role": "user", "content": "Plan a Tokyo trip."}],
            "Day one, Asakusa.",
            compaction_blocks,
        ),
        _turn_spend_log([{"role": "user", "content": "And Osaka?"}], "Dotonbori."),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_1")

    assert [_role_and_content(message) for message in session["messages"]] == [
        ("developer", "Answer briefly."),
        ("assistant", "Day one, Asakusa."),
        ("user", "And Osaka?"),
        ("assistant", "Dotonbori."),
    ]
    assert getattr(session["messages"][1], "provider_specific_fields", None) == {"compaction_blocks": compaction_blocks}


_COMPACTION_ITEM_SENT: Final = {
    "type": "compaction",
    "id": "cmp_1",
    "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
}


@pytest.mark.parametrize(
    "compacting_turn",
    [
        pytest.param(
            _turn_spend_log(
                [{"role": "user", "content": "Plan a Tokyo trip."}],
                "Day one, Asakusa.",
                [{"type": "compaction", "content": "Turn one planned Tokyo."}],
            ),
            id="answer_compacted",
        ),
        pytest.param(
            _turn_spend_log(
                [_COMPACTION_ITEM_SENT, {"role": "user", "content": "Plan a Tokyo trip."}], "Day one, Asakusa."
            ),
            id="input_sent_a_compaction_item",
        ),
    ],
)
@pytest.mark.asyncio
async def test_previous_response_history_keeps_instructions_from_turns_the_compaction_covers(
    compacting_turn: dict[str, object],
):
    spend_logs: Final = (
        _turn_spend_log(
            [{"role": "developer", "content": "Use metric units."}, {"role": "user", "content": "Hi."}], "Hello."
        ),
        compacting_turn,
        _turn_spend_log([{"role": "user", "content": "And Osaka?"}], "Dotonbori."),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_1")

    roles_and_contents: Final = [_role_and_content(message) for message in session["messages"]]
    assert roles_and_contents[0] == ("developer", "Use metric units.")
    assert ("user", "Hi.") not in roles_and_contents
    assert ("assistant", "Hello.") not in roles_and_contents
    assert roles_and_contents[-2:] == [("user", "And Osaka?"), ("assistant", "Dotonbori.")]


@pytest.mark.asyncio
async def test_previous_response_history_drops_a_tool_output_carrying_a_system_role_the_compaction_covers():
    spend_logs: Final = (
        _turn_spend_log(
            [
                {"role": "user", "content": "Look up the code."},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "4242", "role": "system"},
            ],
            "Found it.",
        ),
        _turn_spend_log([_COMPACTION_ITEM_SENT, {"role": "user", "content": "Plan a Tokyo trip."}], "Day one."),
        _turn_spend_log([{"role": "user", "content": "And Osaka?"}], "Dotonbori."),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_1")

    contents: Final = [str(_role_and_content(message)[1]) for message in session["messages"]]
    assert not any("4242" in content for content in contents)
    assert contents[-2:] == ["And Osaka?", "Dotonbori."]


def _output_message(text: str) -> dict[str, object]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


@pytest.mark.asyncio
async def test_previous_response_history_starts_at_a_compaction_item_in_a_stored_responses_output():
    responses_shaped_turn: Final = {
        "request_id": "resp_mcp",
        "session_id": "session-compacted",
        "proxy_server_request": json.dumps({"input": [{"role": "user", "content": "Plan a Tokyo trip."}]}),
        "response": {
            "id": "resp_mcp",
            "object": "response",
            "output": [_output_message("Checking the map."), _COMPACTION_ITEM_SENT, _output_message("Day one.")],
        },
    }
    spend_logs: Final = (
        _turn_spend_log([{"role": "user", "content": "Hi."}], "Hello."),
        responses_shaped_turn,
        _turn_spend_log([{"role": "user", "content": "And Osaka?"}], "Dotonbori."),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_1")

    replayed: Final = json.dumps([_role_and_content(message) for message in session["messages"]], default=str)
    assert "Hello." not in replayed
    assert "Plan a Tokyo trip." not in replayed
    assert "Checking the map." not in replayed
    assert "Turn one planned Tokyo." in replayed
    assert [str(_role_and_content(message)[1]) for message in session["messages"]][-2:] == [
        "And Osaka?",
        "Dotonbori.",
    ]


def _stored_response_turn(
    request_id: str, call_type: str, status: str, output: list[dict[str, object]], prompt: str | None = None
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "call_type": call_type,
        "session_id": "session-with-reads",
        "proxy_server_request": json.dumps({} if prompt is None else {"input": [{"role": "user", "content": prompt}]}),
        "response": {"id": request_id, "object": "response", "status": status, "output": output},
    }


def _identified_output_message(item_id: str, text: str) -> dict[str, object]:
    return {**_output_message(text), "id": item_id}


def _replayed_roles_and_texts(session: ChatCompletionSession) -> list[tuple[object, object]]:
    return [
        (role, _text_of(content)) for role, content in (_role_and_content(message) for message in session["messages"])
    ]


@pytest.mark.asyncio
async def test_reads_and_cancels_of_a_response_the_session_already_holds_replay_no_turns():
    answer: Final = [_identified_output_message("msg_answer", "OK")]
    spend_logs: Final = (
        _stored_response_turn("resp_create", "aresponses", "completed", answer, prompt="Reply with OK"),
        _stored_response_turn("resp_read", "aget_responses", "completed", answer),
        _stored_response_turn("resp_streamed_read", "aget_responses", "completed", answer),
        _stored_response_turn("resp_cancel", "acancel_responses", "completed", answer),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_create")

    assert _replayed_roles_and_texts(session) == [("user", "Reply with OK"), ("assistant", "OK")]


@pytest.mark.asyncio
async def test_a_polled_background_response_replays_its_finished_output_once():
    answer: Final = [_identified_output_message("msg_answer", "One, two, three")]
    spend_logs: Final = (
        _stored_response_turn("resp_queued", "aresponses", "queued", [], prompt="Count to three"),
        _stored_response_turn("resp_poll_running", "aget_responses", "in_progress", []),
        _stored_response_turn("resp_poll_done", "aget_responses", "completed", answer),
        _stored_response_turn("resp_poll_again", "aget_responses", "completed", answer),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_queued")

    assert _replayed_roles_and_texts(session) == [("user", "Count to three"), ("assistant", "One, two, three")]


@pytest.mark.asyncio
async def test_a_background_response_that_grows_between_polls_replays_each_output_item_once():
    first: Final = _identified_output_message("msg_first", "One")
    second: Final = _identified_output_message("msg_second", "Two")
    spend_logs: Final = (
        _stored_response_turn("resp_queued", "aresponses", "queued", [], prompt="Count to two"),
        _stored_response_turn("resp_poll_partial", "aget_responses", "in_progress", [first]),
        _stored_response_turn("resp_poll_done", "aget_responses", "completed", [first, second]),
        _stored_response_turn("resp_poll_again", "aget_responses", "completed", [first, second]),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_queued")

    assert _replayed_roles_and_texts(session) == [("user", "Count to two"), ("assistant", "One"), ("assistant", "Two")]


@pytest.mark.asyncio
async def test_an_item_a_poll_caught_half_written_replays_its_finished_text_once():
    spend_logs: Final = (
        _stored_response_turn("resp_queued", "aresponses", "queued", [], prompt="Count to three"),
        _stored_response_turn(
            "resp_poll_partial", "aget_responses", "in_progress", [_identified_output_message("msg_answer", "One, t")]
        ),
        _stored_response_turn(
            "resp_poll_done",
            "aget_responses",
            "completed",
            [
                _identified_output_message("msg_answer", "One, two, three"),
                _identified_output_message("msg_more", "Done"),
            ],
        ),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_queued")

    assert _replayed_roles_and_texts(session) == [
        ("user", "Count to three"),
        ("assistant", "One, two, three"),
        ("assistant", "Done"),
    ]


@pytest.mark.asyncio
async def test_a_polled_output_item_without_an_id_still_replays_in_order():
    spend_logs: Final = (
        _stored_response_turn("resp_queued", "aresponses", "queued", [], prompt="Count to two"),
        _stored_response_turn(
            "resp_poll_done",
            "aget_responses",
            "completed",
            [_output_message("One"), _identified_output_message("msg_two", "Two")],
        ),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_queued")

    assert _replayed_roles_and_texts(session) == [("user", "Count to two"), ("assistant", "One"), ("assistant", "Two")]


@pytest.mark.asyncio
async def test_a_read_that_ends_before_its_streamed_create_replays_the_answer_once_after_the_prompt():
    answer: Final = [_identified_output_message("msg_answer", "OK")]
    spend_logs: Final = (
        _stored_response_turn("resp_read_mid_stream", "aget_responses", "completed", answer),
        _stored_response_turn("resp_create", "aresponses", "completed", answer, prompt="Say OK"),
    )

    session: Final = await _history_from_stored_turns(spend_logs, "resp_create")

    assert _replayed_roles_and_texts(session) == [("user", "Say OK"), ("assistant", "OK")]


def _tool_call_ids(message: object) -> tuple[str, ...]:
    tool_calls: Final = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    return tuple(call["id"] if isinstance(call, dict) else call.id for call in tool_calls or ())


def _weather_call_turn(call_id: str) -> dict[str, object]:
    return {
        "request_id": "resp_weather",
        "session_id": "session-compacted",
        "proxy_server_request": json.dumps({"input": [{"role": "user", "content": "Weather in Paris?"}]}),
        "response": {
            "id": "chatcmpl-weather",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                }
            ],
        },
    }


_WEATHER_CALL_TURN: Final = _weather_call_turn("call_1")


def _compacted_weather_answer(request_id: str, call_id: str = "call_1") -> dict[str, object]:
    return {
        **_turn_spend_log(
            [_COMPACTION_ITEM_SENT, {"type": "function_call_output", "call_id": call_id, "output": "sunny"}],
            "It is sunny.",
        ),
        "request_id": request_id,
    }


@pytest.mark.asyncio
async def test_previous_response_history_ignores_a_compaction_item_sent_after_the_referenced_response():
    session: Final = await _history_from_stored_turns(
        (_WEATHER_CALL_TURN, _compacted_weather_answer("resp_first_send")), "resp_weather"
    )

    assert _role_and_content(session["messages"][0]) == ("user", "Weather in Paris?")
    assert tuple(chain.from_iterable(map(_tool_call_ids, session["messages"]))) == ("call_1",)


@pytest.mark.asyncio
async def test_a_turn_after_a_compaction_item_keeps_the_stored_call_its_tool_output_answers():
    session: Final = await _history_from_stored_turns(
        (_WEATHER_CALL_TURN, _compacted_weather_answer("resp_compacted")), "resp_compacted"
    )

    roles: Final = [_role_and_content(message)[0] for message in session["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert _tool_call_ids(session["messages"][1]) == ("call_1",)
    assert ("user", "Weather in Paris?") not in map(_role_and_content, session["messages"])


_SIGNED_CALL_ID: Final = f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{'S' * 3000}"
_CUT_SIGNED_CALL_ID: Final = _sanitize_request_body_for_spend_logs_payload({"call_id": _SIGNED_CALL_ID})["call_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_call_id, replayed_call_id",
    [
        pytest.param(_SIGNED_CALL_ID, _SIGNED_CALL_ID, id="call-stored-whole"),
        pytest.param(_CUT_SIGNED_CALL_ID, "call_1", id="call-stored-cut"),
    ],
)
async def test_a_turn_after_a_compaction_item_keeps_the_signed_call_its_cut_tool_output_answers(
    stored_call_id: str, replayed_call_id: str
):
    session: Final = await _history_from_stored_turns(
        (_weather_call_turn(stored_call_id), _compacted_weather_answer("resp_compacted", _CUT_SIGNED_CALL_ID)),
        "resp_compacted",
    )

    roles: Final = [_role_and_content(message)[0] for message in session["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert _tool_call_ids(session["messages"][1]) == (replayed_call_id,)


_WHOLE_THINKING_BLOCK: Final = {"type": "thinking", "thinking": "Count.", "signature": "Eq0BCkYI=="}
_LONG_SIGNATURE: Final = "Eq0BCkYICxgCKkB" * 2000


def _stored_cut(stored: dict[str, object]) -> dict[str, object]:
    return _sanitize_request_body_for_spend_logs_payload(stored)


def _answered_turn(message: dict[str, object], turn_input: object = "Count the triples.") -> dict[str, object]:
    return {
        "request_id": "resp_signed",
        "session_id": "session-signed",
        "proxy_server_request": json.dumps({"input": turn_input}),
        "response": {
            "id": "chatcmpl-signed",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
        },
    }


@pytest.mark.asyncio
async def test_a_turn_stored_with_a_cut_thinking_block_replays_without_it_and_says_so(
    caplog: pytest.LogCaptureFixture,
):
    cut_block: Final = _stored_cut({**_WHOLE_THINKING_BLOCK, "signature": _LONG_SIGNATURE})
    turn: Final = _answered_turn({"content": "954", "thinking_blocks": [_WHOLE_THINKING_BLOCK, cut_block]})

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        session: Final = await _history_from_stored_turns((turn,), "resp_signed")

    answer: Final = session["messages"][-1]
    assert _role_and_content(answer) == ("assistant", "954")
    assert message_field(answer, "thinking_blocks") == [_WHOLE_THINKING_BLOCK]
    assert [record.levelname for record in caplog.records if "resp_signed" in record.getMessage()] == ["WARNING"]


@pytest.mark.asyncio
async def test_a_turn_stored_whole_replays_its_signed_thinking_block_silently(caplog: pytest.LogCaptureFixture):
    signed_block: Final = {**_WHOLE_THINKING_BLOCK, "signature": _LONG_SIGNATURE}

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        session: Final = await _history_from_stored_turns(
            (_answered_turn({"content": "954", "thinking_blocks": [signed_block]}),), "resp_signed"
        )

    assert message_field(session["messages"][-1], "thinking_blocks") == [signed_block]
    assert not [record for record in caplog.records if "resp_signed" in record.getMessage()]


@pytest.mark.asyncio
async def test_a_turn_whose_stored_input_only_cut_the_id_its_tool_output_answers_replays_silently(
    caplog: pytest.LogCaptureFixture,
):
    signed_id: Final = f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{_LONG_SIGNATURE}"
    stored: Final = _stored_cut({"input": [{"type": "function_call_output", "call_id": signed_id, "output": "14"}]})
    turn: Final = _answered_turn({"content": "954"}, turn_input=stored["input"])

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        session: Final = await _history_from_stored_turns((turn,), "resp_signed")

    assert message_field(session["messages"][0], "tool_call_id") == "call_1"
    assert not [record for record in caplog.records if "resp_signed" in record.getMessage()]


@pytest.mark.parametrize("place", ["response", "input"])
@pytest.mark.asyncio
async def test_a_turn_whose_stored_tool_call_lost_its_signature_to_the_cut_says_so(
    place: str, caplog: pytest.LogCaptureFixture
):
    signed_id: Final = f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{_LONG_SIGNATURE}"
    function: Final = {"name": "get_weather", "arguments": "{}"}
    asked: Final = _stored_cut({"tool_calls": [{"id": signed_id, "type": "function", "function": function}]})
    resent: Final = _stored_cut({"input": [{"type": "function_call", "call_id": signed_id, **function}]})
    turn: Final = (
        _answered_turn({"content": None, **asked})
        if place == "response"
        else _answered_turn({"content": "954"}, turn_input=resent["input"])
    )

    with caplog.at_level("WARNING", logger="LiteLLM Proxy"):
        session: Final = await _history_from_stored_turns((turn,), "resp_signed")

    calls: Final = [call for message in session["messages"] for call in message_field(message, "tool_calls") or ()]
    assert [message_field(call, "id") for call in calls] == ["call_1"]
    assert [record.levelname for record in caplog.records if "resp_signed" in record.getMessage()] == ["WARNING"]


@pytest.mark.asyncio
async def test_a_tool_call_stored_with_a_cut_thought_signature_replays_unsigned():
    call: Final = _stored_cut(
        {
            "id": f"call_1{THOUGHT_SIGNATURE_SEPARATOR}{_LONG_SIGNATURE}",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
            "provider_specific_fields": {"thought_signature": _LONG_SIGNATURE},
        }
    )

    session: Final = await _history_from_stored_turns(
        (_answered_turn({"content": None, "tool_calls": [call]}),), "resp_signed"
    )

    answer: Final = session["messages"][-1]
    assert _tool_call_ids(answer) == ("call_1",)
    assert LITELLM_TRUNCATED_PAYLOAD_FIELD not in json.dumps(answer, default=lambda value: value.model_dump())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "provider"),
    [("claude-sonnet-5-5", "anthropic"), ("gemini-3.8-flash", "gemini"), ("house-model", "hosted_vllm")],
)
async def test_a_custom_tool_call_resent_in_a_stored_turn_is_made_once_on_the_turn_after(
    model: str, provider: str
) -> None:
    patch_text: Final = "*** Begin Patch " * 250
    asked: Final = _stored_cut(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": json.dumps({"content": patch_text})},
        }
    )
    resent: Final = _stored_cut(
        {
            "input": [
                {"type": "custom_tool_call", "call_id": "call_1", "name": "apply_patch", "input": patch_text},
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Patched."},
            ]
        }
    )
    stored_turns: Final = (
        {**_answered_turn({"content": None, "tool_calls": [asked]}), "request_id": "resp_asked"},
        _answered_turn({"content": "Done."}, turn_input=resent["input"]),
    )

    async def session_lookup(previous_response_id: str) -> ChatCompletionSession:
        return await _history_from_stored_turns(stored_turns, previous_response_id)

    request: Final = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
        previous_response_id="resp_signed",
        litellm_completion_request={
            "model": model,
            "custom_llm_provider": provider,
            "messages": [{"role": "user", "content": "Thanks."}],
        },
        instructions=None,
        session_lookup=session_lookup,
    )

    roles: Final = [message_field(message, "role") for message in request["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert [
        (message_field(call, "id"), message_field(message_field(call, "function"), "arguments"))
        for call in message_field(request["messages"][1], "tool_calls")
    ] == [("call_1", asked["function"]["arguments"])]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signature, replayed",
    [
        pytest.param("Eq0BCkYI==", True, id="whole-reasoning-item"),
        pytest.param(_LONG_SIGNATURE, False, id="cut-reasoning-item"),
    ],
)
async def test_reasoning_a_client_sent_back_in_a_stored_request_replays_only_when_stored_whole(
    signature: str, replayed: bool
):
    sent_block: Final = {**_WHOLE_THINKING_BLOCK, "signature": signature}
    sent_input: Final = _stored_cut(
        {
            "input": [
                {"role": "user", "content": "Count the triples."},
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": json.dumps([sent_block])},
                {"role": "assistant", "content": "954"},
                {"role": "user", "content": "And now?"},
            ]
        }
    )["input"]

    session: Final = await _history_from_stored_turns((_answered_turn({"content": "955"}, sent_input),), "resp_signed")

    blocks: Final = [message_field(message, "thinking_blocks") for message in session["messages"]]
    assert [block for block in blocks if block] == ([[sent_block]] if replayed else [])
    assert [content for _, content in map(_role_and_content, session["messages"])][-1] == "955"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spend_logs", "previous_response_id", "replayed_roles"),
    [
        ((_WEATHER_CALL_TURN, _compacted_weather_answer("resp_first_send")), "resp_weather", ["user", "assistant"]),
        (
            (_WEATHER_CALL_TURN, _compacted_weather_answer("resp_first_send"), _compacted_weather_answer("resp_retry")),
            "resp_first_send",
            ["user", "assistant", "tool", "assistant"],
        ),
    ],
)
async def test_a_compacted_session_replays_no_turn_sent_after_the_referenced_response(
    spend_logs: tuple[dict[str, object], ...], previous_response_id: str, replayed_roles: list[str]
):
    session: Final = await _history_from_stored_turns(spend_logs, previous_response_id)

    assert [_role_and_content(message)[0] for message in session["messages"]] == replayed_roles


class _SqliteBackedPrismaDB:
    def __init__(
        self,
        rows: list[tuple[str, str, str, str, str | None, str | None]],
        metadata: Mapping[str, Mapping[str, object]] = MappingProxyType({}),
    ) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" '
            '(request_id TEXT, api_key TEXT, session_id TEXT, "endTime" TEXT, "user" TEXT, team_id TEXT, metadata TEXT)'
        )
        self._connection.executemany(
            'INSERT INTO "LiteLLM_SpendLogs" VALUES (?, ?, ?, ?, ?, ?, ?)',
            [(*row, json.dumps(metadata[row[0]]) if row[0] in metadata else None) for row in rows],
        )

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        sqlite_query: Final = (
            query.replace("= ANY($1::text[])", "IN (SELECT value FROM json_each(?1))")
            .replace("= ANY($2::text[])", "IN (SELECT value FROM json_each(?2))")
            .replace("IS NOT DISTINCT FROM", "IS")
        )
        sqlite_args: Final = tuple(json.dumps(arg) if isinstance(arg, list) else arg for arg in args)
        return [dict(row) for row in self._connection.execute(sqlite_query, sqlite_args)]


@pytest.mark.asyncio
async def test_session_lookup_does_not_return_another_keys_rows_for_a_shared_session_id():
    """A caller-supplied litellm_session_id can collide with another key's session; history must stay per key."""
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("victim-1", "victim-key", "shared-session", "2026-01-01T00:00:01", "", ""),
            ("victim-2", "victim-key", "shared-session", "2026-01-01T00:00:02", "", ""),
            ("attacker-1", "attacker-key", "shared-session", "2026-01-01T00:00:03", "", ""),
            ("attacker-2", "attacker-key", "other-session", "2026-01-01T00:00:04", "", ""),
        ]
    )
    fake_prisma_client = _FakePrismaClient(results=[])
    fake_prisma_client.db = sqlite_db

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        spend_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id("attacker-1")
        victim_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id("victim-1")

    assert [row["request_id"] for row in spend_logs] == ["attacker-1"]
    assert [row["request_id"] for row in victim_logs] == ["victim-1", "victim-2"]


async def _history_request_ids(sqlite_db: _SqliteBackedPrismaDB, response_id: str) -> list[str]:
    fake_prisma_client: Final = _FakePrismaClient(results=[])
    fake_prisma_client.db = sqlite_db
    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        spend_logs: Final = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(response_id)
    return [row["request_id"] for row in spend_logs]


def _stored_path(*previous_response_ids: str) -> Mapping[str, object]:
    return {PREVIOUS_RESPONSE_IDS_METADATA_KEY: list(previous_response_ids)}


def _forked_session_db() -> _SqliteBackedPrismaDB:
    """One session: root -> a2 -> a3 on branch a, root -> b2 -> b3 on branch b, b2 sent between a2 and a3."""
    return _SqliteBackedPrismaDB(
        [
            ("root", "key", "session", "2026-01-01T00:00:01", "", ""),
            ("a2", "key", "session", "2026-01-01T00:00:02", "", ""),
            ("b2", "key", "session", "2026-01-01T00:00:03", "", ""),
            ("a3", "key", "session", "2026-01-01T00:00:04", "", ""),
            ("b3", "key", "session", "2026-01-01T00:00:05", "", ""),
        ],
        metadata={
            "root": _stored_path(),
            "a2": _stored_path("root"),
            "b2": _stored_path("root"),
            "a3": _stored_path("root", "a2"),
            "b3": _stored_path("root", "b2"),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous_response_id", "replayed"),
    [
        ("root", ["root"]),
        ("a2", ["root", "a2"]),
        ("b2", ["root", "b2"]),
        ("a3", ["root", "a2", "a3"]),
        ("b3", ["root", "b2", "b3"]),
    ],
)
async def test_a_turn_with_a_stored_path_replays_only_the_path_to_the_root(
    previous_response_id: str, replayed: list[str]
):
    """A fork shares its session with the other branch, so the stored path, not the session, picks the turns."""
    assert await _history_request_ids(_forked_session_db(), previous_response_id) == replayed


@pytest.mark.asyncio
async def test_a_stored_path_is_replayed_in_path_order_whatever_the_clock_says():
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("late-root", "key", "session", "2026-01-01T00:00:09", "", ""),
            ("child", "key", "session", "2026-01-01T00:00:01", "", ""),
        ],
        metadata={"late-root": _stored_path(), "child": _stored_path("late-root")},
    )

    assert await _history_request_ids(sqlite_db, "child") == ["late-root", "child"]


@pytest.mark.asyncio
async def test_a_turn_stored_before_paths_existed_still_replays_its_whole_session():
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("old-1", "key", "session", "2026-01-01T00:00:01", "", ""),
            ("old-2", "key", "session", "2026-01-01T00:00:02", "", ""),
            ("old-3", "key", "session", "2026-01-01T00:00:03", "", ""),
        ]
    )

    assert await _history_request_ids(sqlite_db, "old-2") == ["old-1", "old-2", "old-3"]


@pytest.mark.asyncio
async def test_a_stored_path_never_replays_another_owners_turns():
    """A path naming another key's row (a forged or reused id) replays nothing of that key's history."""
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("victim-1", "victim-key", "session", "2026-01-01T00:00:01", "", ""),
            ("attacker-1", "attacker-key", "session", "2026-01-01T00:00:02", "", ""),
        ],
        metadata={"victim-1": _stored_path(), "attacker-1": _stored_path("victim-1")},
    )

    assert await _history_request_ids(sqlite_db, "attacker-1") == ["attacker-1"]


@pytest.mark.asyncio
async def test_the_replayed_turns_become_the_next_turns_path_to_the_root():
    fake_prisma_client: Final = _FakePrismaClient(results=[])
    fake_prisma_client.db = _forked_session_db()

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        session: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "b2"
        )

    assert session["previous_response_ids"] == ("root", "b2")


@pytest.mark.asyncio
async def test_a_turn_continuing_a_session_stored_before_paths_existed_records_every_turn_it_replayed():
    """The legacy replay feeds the whole session to the model, so the new turn's path must name all of it."""
    fake_prisma_client: Final = _FakePrismaClient(
        results=[
            [
                _spend_log("old-1", "session", "My favorite number is 7.", "ok"),
                _spend_log("old-2", "session", "My second number is 42.", "ok"),
            ]
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        session: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "old-1"
        )

    assert session["previous_response_ids"] == ("old-1", "old-2")


@pytest.mark.asyncio
async def test_session_lookup_keeps_jwt_users_and_teams_apart_when_they_share_an_empty_api_key():
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("alice-1", "", "shared-session", "2026-01-01T00:00:01", "alice", "team-a"),
            ("bob-1", "", "shared-session", "2026-01-01T00:00:02", "bob", "team-a"),
            ("service-a-1", "", "shared-session", "2026-01-01T00:00:03", "", "team-a"),
            ("service-b-1", "", "shared-session", "2026-01-01T00:00:04", "", "team-b"),
            ("alice-2", "", "shared-session", "2026-01-01T00:00:05", "alice", "team-a"),
        ]
    )

    assert await _history_request_ids(sqlite_db, "alice-1") == ["alice-1", "alice-2"]
    assert await _history_request_ids(sqlite_db, "bob-1") == ["bob-1"]
    assert await _history_request_ids(sqlite_db, "service-a-1") == ["service-a-1"]
    assert await _history_request_ids(sqlite_db, "service-b-1") == ["service-b-1"]


@pytest.mark.asyncio
async def test_session_lookup_follows_the_owner_across_keys_and_keeps_ownerless_keys_apart():
    sqlite_db: Final = _SqliteBackedPrismaDB(
        [
            ("alice-old-key", "hashed-old", "shared-session", "2026-01-01T00:00:01", "alice", "team-a"),
            ("ownerless-1", "hashed-other", "shared-session", "2026-01-01T00:00:02", "", ""),
            ("alice-new-key", "hashed-new", "shared-session", "2026-01-01T00:00:03", "alice", "team-a"),
            ("ownerless-2", "hashed-another", "shared-session", "2026-01-01T00:00:04", "", ""),
        ]
    )

    assert await _history_request_ids(sqlite_db, "alice-new-key") == ["alice-old-key", "alice-new-key"]
    assert await _history_request_ids(sqlite_db, "ownerless-2") == ["ownerless-2"]


GEMINI_URL_PATTERN: Final = r"https://generativelanguage\.googleapis\.com/v1[a-z]*/models/gemini-3\.5-flash:.*"
LONG_SIGNATURE: Final = "Ab+/" * 200


def _gemini_answer() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "Sunny"}]}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
        },
    )


def _stored_function_call_turn(response_id: str) -> dict:
    return {
        "request_id": response_id,
        "call_type": "aresponses",
        "session_id": "gemini-session",
        "proxy_server_request": {"input": "What is the weather in Paris?", "model": "gemini-short-ids"},
        "response": {
            "id": response_id,
            "object": "response",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                    "status": "completed",
                    "provider_specific_fields": {"thought_signature": LONG_SIGNATURE},
                }
            ],
        },
    }


@pytest.mark.parametrize("short_ids", [False, True])
async def test_previous_response_replays_a_thought_signature_kept_out_of_the_id_only_with_short_ids(
    short_ids: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="gemini", model_id=None, response_id="chatcmpl-gemini-turn-1"
    )
    stored: Final = _StoredSpendLogs((_stored_function_call_turn(response_id),))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored)), respx.mock:
        route: Final = respx.post(url__regex=GEMINI_URL_PATTERN).mock(return_value=_gemini_answer())
        await litellm.aresponses(
            model="gemini/gemini-3.5-flash",
            input=[{"type": "function_call_output", "call_id": "call_1", "output": '{"temp_c": 21}'}],
            previous_response_id=response_id,
            api_key="test-gemini-key",
            **({"thought_signature_in_tool_call_id": False} if short_ids else {}),
        )
        sent_body: Final = json.loads(route.calls.last.request.content)

    model_turns: Final = [turn for turn in sent_body["contents"] if turn["role"] == "model"]
    model_parts: Final = list(chain.from_iterable(turn["parts"] for turn in model_turns))
    assert len(model_parts) == 1
    assert (model_parts[0]["thoughtSignature"] == LONG_SIGNATURE) is short_ids


LONG_NAMESPACE_TOOL: Final = {
    "type": "namespace",
    "name": "mcp__codex_apps__codex_document_control",
    "description": "Docs tools",
    "tools": [
        {
            "type": "function",
            "name": "_get_document_tool_schemas",
            "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}},
        }
    ],
}
LONG_MEMBER_FULL_NAME: Final = "mcp__codex_apps__codex_document_control___get_document_tool_schemas"
OTHER_LONG_NAMESPACE_TOOL: Final = {
    "type": "namespace",
    "name": "mcp__codex_apps__codex_calendar_control",
    "description": "Calendar tools",
    "tools": [{"type": "function", "name": "_get_calendar_tool_schemas", "parameters": {"type": "object"}}],
}


def _additional_tools_item(tool: Mapping[str, object]) -> dict[str, object]:
    return {"type": "additional_tools", "id": "at_1", "role": "developer", "tools": [tool]}


CLIENT_TOOL_SEARCH: Final = {"type": "tool_search", "execution": "client"}
CLIENT_TOOL_SEARCH_CALL: Final = {
    "type": "tool_search_call",
    "call_id": "search_1",
    "execution": "client",
    "arguments": {"query": "document schemas"},
}


def _client_tool_search_output(tool: Mapping[str, object], call_id: str = "search_1") -> dict[str, object]:
    return {
        "type": "tool_search_output",
        "call_id": call_id,
        "status": "completed",
        "execution": "client",
        "tools": [tool],
    }


@dataclass(frozen=True, slots=True)
class SentTools:
    tools: list[Mapping[str, object]]
    stored_input: list[Mapping[str, object]]
    next_input: list[Mapping[str, object]]


def _sent_as_tools(tool: Mapping[str, object]) -> SentTools:
    return SentTools(tools=[tool], stored_input=[], next_input=[])


def _sent_in_additional_tools(tool: Mapping[str, object]) -> SentTools:
    return SentTools(tools=[], stored_input=[_additional_tools_item(tool)], next_input=[_additional_tools_item(tool)])


def _loaded_by_client_tool_search(tool: Mapping[str, object]) -> SentTools:
    return SentTools(
        tools=[CLIENT_TOOL_SEARCH],
        stored_input=[CLIENT_TOOL_SEARCH_CALL, _client_tool_search_output(tool)],
        next_input=[],
    )


SENT_TOOL_PLACEMENTS: Final = pytest.mark.parametrize(
    "placement",
    [_sent_as_tools, _sent_in_additional_tools, _loaded_by_client_tool_search],
    ids=["tools", "additional_tools", "client_tool_search"],
)


def _stored_long_member_call_turn(
    response_id: str, placement: Callable[[Mapping[str, object]], SentTools]
) -> dict[str, object]:
    sent: Final = placement(LONG_NAMESPACE_TOOL)
    return {
        "request_id": response_id,
        "call_type": "aresponses",
        "session_id": "long-member-session",
        "proxy_server_request": {
            "input": [*sent.stored_input, {"role": "user", "content": "Fetch document N-42"}],
            "model": "gemini-flash",
            "tools": sent.tools,
        },
        "response": {
            "id": "chatcmpl-long-member",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": LONG_MEMBER_FULL_NAME, "arguments": '{"doc_id": "N-42"}'},
                            }
                        ],
                    },
                }
            ],
        },
    }


def _long_member_response_id() -> str:
    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="gemini", model_id=None, response_id="chatcmpl-long-member"
    )


LONG_MEMBER_CALL_OUTPUT: Final = {"type": "function_call_output", "call_id": "call_1", "output": "schema for N-42"}


@dataclass(frozen=True, slots=True)
class SentToolNamesOnGemini:
    declared: list[str]
    called: list[str]
    answered: list[str]


async def _tool_names_sent_to_gemini(
    stored: _StoredSpendLogs,
    previous_response_id: str,
    request_input: list[Mapping[str, object]],
    tools: list[Mapping[str, object]],
) -> SentToolNamesOnGemini:
    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored)), respx.mock:
        route: Final = respx.post(url__regex=GEMINI_URL_PATTERN).mock(return_value=_gemini_answer())
        await litellm.aresponses(
            model="gemini/gemini-3.5-flash",
            input=request_input,
            previous_response_id=previous_response_id,
            tools=tools,
            api_key="test-gemini-key",
        )
        sent_body: Final = json.loads(route.calls.last.request.content)

    declarations: Final = chain.from_iterable(tool["function_declarations"] for tool in sent_body["tools"])
    parts: Final = list(chain.from_iterable(turn["parts"] for turn in sent_body["contents"]))
    called: Final = [part["function_call"]["name"] for part in parts if "function_call" in part]
    answered: Final = [part["function_response"]["name"] for part in parts if "function_response" in part]
    return SentToolNamesOnGemini(
        declared=[declaration["name"] for declaration in declarations],
        called=[name for name in called if name != CLIENT_TOOL_SEARCH_FUNCTION_NAME],
        answered=[name for name in answered if name != CLIENT_TOOL_SEARCH_FUNCTION_NAME],
    )


async def _answer_the_long_member_call(stored: _StoredSpendLogs, response_id: str, sent: SentTools) -> None:
    names: Final = await _tool_names_sent_to_gemini(
        stored, response_id, [*sent.next_input, LONG_MEMBER_CALL_OUTPUT], sent.tools
    )

    sent_name: Final = truncate_tool_name(LONG_MEMBER_FULL_NAME)
    assert sent_name in names.declared
    assert names.called == names.answered == [sent_name]


@SENT_TOOL_PLACEMENTS
async def test_previous_response_replays_a_long_member_call_under_the_name_the_provider_was_sent(
    monkeypatch: pytest.MonkeyPatch, placement: Callable[[Mapping[str, object]], SentTools]
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_id: Final = _long_member_response_id()
    stored: Final = _StoredSpendLogs((_stored_long_member_call_turn(response_id, placement),))

    await _answer_the_long_member_call(stored, response_id, placement(LONG_NAMESPACE_TOOL))


def _searched_on_client_earlier(tool: Mapping[str, object]) -> SentTools:
    return SentTools(tools=[CLIENT_TOOL_SEARCH], stored_input=[], next_input=[])


def _client_tool_search_turn(
    request_id: str, call_id: str, tool: Mapping[str, object], answered: Sequence[Mapping[str, object]] = ()
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "long-member-session",
        "proxy_server_request": {
            "input": [
                *answered,
                {**CLIENT_TOOL_SEARCH_CALL, "call_id": call_id},
                _client_tool_search_output(tool, call_id),
            ],
            "model": "gemini-flash",
            "tools": [CLIENT_TOOL_SEARCH],
        },
        "response": {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "I can use the tools I found now."},
                }
            ],
        },
    }


async def test_previous_response_replays_a_long_member_call_whose_namespace_an_earlier_turn_searched_for(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_id: Final = _long_member_response_id()
    stored: Final = _StoredSpendLogs(
        (
            _client_tool_search_turn("client-tool-search-turn", "search_1", LONG_NAMESPACE_TOOL),
            _stored_long_member_call_turn(response_id, _searched_on_client_earlier),
        )
    )

    await _answer_the_long_member_call(stored, response_id, _searched_on_client_earlier(LONG_NAMESPACE_TOOL))


async def test_previous_response_replays_a_long_member_call_after_a_later_search_loaded_another_namespace(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_id: Final = _long_member_response_id()
    stored: Final = _StoredSpendLogs(
        (
            _client_tool_search_turn("document-search-turn", "search_1", LONG_NAMESPACE_TOOL),
            _client_tool_search_turn("calendar-search-turn", "search_2", OTHER_LONG_NAMESPACE_TOOL),
            _stored_long_member_call_turn(response_id, _searched_on_client_earlier),
        )
    )

    await _answer_the_long_member_call(stored, response_id, _searched_on_client_earlier(LONG_NAMESPACE_TOOL))


async def test_previous_response_replays_a_long_member_call_whose_namespace_a_later_search_loaded_again(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    search_again_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="gemini", model_id=None, response_id="chatcmpl-document-search-again"
    )
    stored: Final = _StoredSpendLogs(
        (
            _client_tool_search_turn("document-search-turn", "search_1", LONG_NAMESPACE_TOOL),
            _stored_long_member_call_turn(_long_member_response_id(), _searched_on_client_earlier),
            _client_tool_search_turn(search_again_id, "search_2", LONG_NAMESPACE_TOOL, (LONG_MEMBER_CALL_OUTPUT,)),
        )
    )

    names: Final = await _tool_names_sent_to_gemini(
        stored, search_again_id, [{"role": "user", "content": "Summarize it"}], [CLIENT_TOOL_SEARCH]
    )

    sent_name: Final = truncate_tool_name(LONG_MEMBER_FULL_NAME)
    assert sent_name in names.declared
    assert names.called == names.answered == [sent_name]


SELF_NAMED_NAMESPACE_TOOL: Final = {
    "type": "namespace",
    "name": "lookup_document",
    "description": "Document lookup",
    "tools": [{"type": "function", "name": "lookup_document", "parameters": {"type": "object"}}],
}


async def test_previous_response_replays_a_call_to_a_self_named_member_an_earlier_search_loaded(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="gemini", model_id=None, response_id="chatcmpl-self-named-member"
    )
    self_named_call: Final = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup_document",
        "namespace": "lookup_document",
        "arguments": '{"doc_id": "N-42"}',
    }
    stored: Final = _StoredSpendLogs(
        (
            _client_tool_search_turn("lookup-search-turn", "search_1", SELF_NAMED_NAMESPACE_TOOL),
            {
                "request_id": response_id,
                "call_type": "aresponses",
                "session_id": "long-member-session",
                "proxy_server_request": {
                    "input": [
                        self_named_call,
                        {"type": "function_call_output", "call_id": "call_1", "output": "document N-42"},
                    ],
                    "model": "gemini-flash",
                    "tools": [CLIENT_TOOL_SEARCH],
                },
                "response": {
                    "id": "chatcmpl-self-named-member",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Document N-42 is a lease."},
                        }
                    ],
                },
            },
        )
    )

    names: Final = await _tool_names_sent_to_gemini(
        stored, response_id, [{"role": "user", "content": "Summarize it"}], [CLIENT_TOOL_SEARCH]
    )

    sent_name: Final = chat_member_name("lookup_document", "lookup_document")
    assert sent_name in names.declared
    assert names.called == [sent_name]


TRUNCATED_ROW: Final = json.dumps({LITELLM_TRUNCATED_PAYLOAD_FIELD: True})
STORED_ROWS: Final = (
    pytest.param(TRUNCATED_ROW, id="truncated_row"),
    pytest.param("{}", id="prompts_not_stored"),
)


def _kept_in_cold_storage(stored_turn: Mapping[str, object], stored_request: str = TRUNCATED_ROW) -> dict[str, object]:
    return {
        **stored_turn,
        "metadata": json.dumps({"cold_storage_object_key": "cold/long-member-turn.json"}),
        "proxy_server_request": stored_request,
    }


def _cold_storage_object(
    declared_name: str, earlier_input: tuple[Mapping[str, object], ...] = ()
) -> dict[str, object]:
    return {
        "messages": [*earlier_input, {"role": "user", "content": "Fetch document N-42"}],
        "model_parameters": {
            "tools": [{"type": "function", "function": {"name": declared_name, "parameters": {"type": "object"}}}]
        },
    }


@pytest.mark.parametrize("stored_request", STORED_ROWS)
async def test_previous_response_replays_a_long_member_call_from_a_turn_kept_in_cold_storage(
    monkeypatch: pytest.MonkeyPatch, stored_request: str
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    response_id: Final = _long_member_response_id()
    stored: Final = _StoredSpendLogs(
        (_kept_in_cold_storage(_stored_long_member_call_turn(response_id, _sent_as_tools), stored_request),)
    )

    with patch.object(session_handler, "COLD_STORAGE_HANDLER") as cold_storage:
        cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(
            return_value=_cold_storage_object(truncate_tool_name(LONG_MEMBER_FULL_NAME))
        )
        await _answer_the_long_member_call(stored, response_id, _sent_as_tools(LONG_NAMESPACE_TOOL))


JOINED_MEMBER_CALL_REPLAY: Final = (
    {"type": "function_call", "call_id": "call_0", "name": LONG_MEMBER_FULL_NAME, "arguments": '{"doc_id": "N-41"}'},
    {"type": "function_call_output", "call_id": "call_0", "output": "schema for N-41"},
)


@pytest.mark.parametrize("stored_request", STORED_ROWS)
async def test_previous_response_replays_a_stored_input_call_from_a_turn_kept_in_cold_storage(
    monkeypatch: pytest.MonkeyPatch, stored_request: str
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    response_id: Final = _long_member_response_id()
    stored: Final = _StoredSpendLogs(
        (_kept_in_cold_storage(_stored_long_member_call_turn(response_id, _sent_as_tools), stored_request),)
    )
    sent: Final = _sent_as_tools(LONG_NAMESPACE_TOOL)
    sent_name: Final = truncate_tool_name(LONG_MEMBER_FULL_NAME)

    with patch.object(session_handler, "COLD_STORAGE_HANDLER") as cold_storage:
        cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(
            return_value=_cold_storage_object(sent_name, JOINED_MEMBER_CALL_REPLAY)
        )
        names: Final = await _tool_names_sent_to_gemini(
            stored, response_id, [*sent.next_input, LONG_MEMBER_CALL_OUTPUT], sent.tools
        )

    assert sent_name in names.declared
    assert names.called == names.answered == [sent_name, sent_name]


SELF_NAMED_MEMBER_CALL_REPLAY: Final = (
    {
        "type": "function_call",
        "call_id": "call_0",
        "name": "lookup_document",
        "namespace": "lookup_document",
        "arguments": '{"doc_id": "N-41"}',
    },
    {"type": "function_call_output", "call_id": "call_0", "output": "document N-41"},
)


async def test_previous_response_replays_a_stored_self_named_member_call_from_a_turn_kept_in_cold_storage(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "cold_storage_custom_logger", "s3_v2")
    response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="gemini", model_id=None, response_id="chatcmpl-self-named-cold"
    )
    stored: Final = _StoredSpendLogs(
        (
            _kept_in_cold_storage(
                {
                    "request_id": response_id,
                    "call_type": "aresponses",
                    "session_id": "self-named-cold-session",
                    "response": {
                        "id": "chatcmpl-self-named-cold",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "Document N-41 is a lease."},
                            }
                        ],
                    },
                }
            ),
        )
    )
    sent_name: Final = chat_member_name("lookup_document", "lookup_document")

    with patch.object(session_handler, "COLD_STORAGE_HANDLER") as cold_storage:
        cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = AsyncMock(
            return_value=_cold_storage_object(sent_name, SELF_NAMED_MEMBER_CALL_REPLAY)
        )
        names: Final = await _tool_names_sent_to_gemini(
            stored, response_id, [{"role": "user", "content": "Summarize it"}], [SELF_NAMED_NAMESPACE_TOOL]
        )

    assert sent_name in names.declared
    assert names.called == names.answered == [sent_name]


LONG_FUNCTION: Final = "lookup_" + "document_tool_schema_" * 4


@pytest.mark.parametrize(
    "cold_object",
    [
        pytest.param(_cold_storage_object(LONG_FUNCTION), id="sent_in_full"),
        pytest.param({"messages": [{"role": "user", "content": "Fetch document N-42"}]}, id="no_tools_recorded"),
    ],
)
async def test_cold_storage_turn_keeps_a_long_call_name_its_provider_was_not_sent_shortened(
    cold_object: dict[str, object],
):
    stored_turn: Final = _kept_in_cold_storage(_long_function_call_turn())

    assert await _replayed_call_names(stored_turn, cold_object) == [LONG_FUNCTION]


def _long_function_call_turn() -> dict[str, object]:
    return {
        "request_id": "long-function-turn",
        "call_type": "aresponses",
        "session_id": "long-function-session",
        "response": {
            "id": "chatcmpl-long-function",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": LONG_FUNCTION, "arguments": '{"doc_id": "N-42"}'},
                            }
                        ],
                    },
                }
            ],
        },
    }


async def _replayed_call_names(stored_turn: Mapping[str, object], stored_request: Mapping[str, object]) -> list[str]:
    session: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        "long-function-turn",
        stored_turns=StoredTurns(spend_logs=(stored_turn,), proxy_server_requests=(stored_request,)),
    )
    replayed: Final = session["messages"][-1]
    assert isinstance(replayed, Message)
    assert len(LONG_FUNCTION) > 64
    return [str(tool_call.function.name) for tool_call in replayed.tool_calls or []]


async def test_cold_storage_turn_replays_a_long_call_under_the_shortened_name_its_provider_was_sent():
    stored_turn: Final = _kept_in_cold_storage(_long_function_call_turn())

    assert await _replayed_call_names(stored_turn, _cold_storage_object(truncate_tool_name(LONG_FUNCTION))) == [
        truncate_tool_name(LONG_FUNCTION)
    ]


@pytest.mark.parametrize(
    "stored_request",
    [
        pytest.param(
            {
                "model": "gemini-flash",
                "input": [{"role": "user", "content": "Fetch document N-42"}],
                "tools": [{"type": "function", "name": LONG_FUNCTION, "parameters": {"type": "object"}}],
            },
            id="responses_request",
        ),
        pytest.param(
            {
                "model": "gemini-flash",
                "input": [{"role": "user", "content": "Fetch document N-42"}],
                LITELLM_TRUNCATED_PAYLOAD_FIELD: True,
            },
            id="truncated_responses_request_without_cold_storage",
        ),
        pytest.param({"messages": [{"role": "user", "content": "Fetch document N-42"}]}, id="messages_request"),
    ],
)
async def test_hot_turn_keeps_its_stored_call_name_when_the_client_sent_model_parameters(
    stored_request: dict[str, object],
):
    request_with_client_model_parameters: Final = {
        **stored_request,
        "model_parameters": _cold_storage_object(truncate_tool_name(LONG_FUNCTION))["model_parameters"],
    }
    stored_turn: Final = {
        **_long_function_call_turn(),
        "metadata": json.dumps({"cold_storage_object_key": "cold/long-function-turn.json"}),
        "proxy_server_request": json.dumps(request_with_client_model_parameters),
    }

    assert await _replayed_call_names(stored_turn, request_with_client_model_parameters) == [LONG_FUNCTION]


def test_a_stored_turn_extends_the_history_when_called_without_tool_names():
    history: Final = ResponsesSessionHandler.extend_chat_completion_message_with_spend_log_payload(
        spend_log=_long_function_call_turn(),
        proxy_server_request_dict={"input": [{"role": "user", "content": "Fetch document N-42"}]},
        chat_completion_message_history=[],
    )

    assert [_role_and_content(message)[0] for message in history] == ["user", "assistant"]
    assert [tool_call.function.name for tool_call in history[-1].tool_calls or []] == [LONG_FUNCTION]


_FILLED_REPLAY: Final = [
    ("user", "What is 2+2?"),
    ("assistant", "Four."),
    ("user", EMPTY_TEXT_PLACEHOLDER),
    ("assistant", "Eight."),
]
_DROPPED_REPLAY: Final = [("user", "What is 2+2?"), ("assistant", "Four."), ("assistant", "Eight.")]
_ROUTE_THAT_FILLS: Final = {"model": "anthropic/claude-opus-5-5", "custom_llm_provider": "anthropic"}
_ROUTE_THAT_ACCEPTS_A_PREFILL: Final = {"model": "anthropic/claude-sonnet-4-5", "custom_llm_provider": "anthropic"}
_ARN_ROUTE_THAT_FILLS: Final = {
    "model": "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
    "custom_llm_provider": "bedrock",
    "metadata": json.dumps({"model_map_information": {"model_map_key": "us.anthropic.claude-opus-5-5"}}),
}


def _stored_turns(*spend_logs: dict[str, object]) -> StoredTurns:
    return StoredTurns(
        spend_logs=cast(tuple[SpendLogsPayload, ...], spend_logs),  # cast-ok: rows hold only the fields the replay reads
        proxy_server_requests=tuple(as_str_mapping(spend_log["proxy_server_request"]) for spend_log in spend_logs),
    )


def _replayed_turns(session: ChatCompletionSession) -> list[tuple[object, object]]:
    messages: Final[Sequence[object]] = session.get("messages", [])
    return [(message_field(message, "role"), message_field(message, "content")) for message in messages]


@pytest.mark.parametrize(
    ("stored_route", "expected"),
    (
        (_ROUTE_THAT_FILLS, _FILLED_REPLAY),
        (_ARN_ROUTE_THAT_FILLS, _FILLED_REPLAY),
        (_ROUTE_THAT_ACCEPTS_A_PREFILL, _DROPPED_REPLAY),
    ),
    ids=("stored_model_refuses_a_prefill", "stored_arn_priced_as_a_model_that_refuses_it", "stored_model_accepts_it"),
)
@pytest.mark.asyncio
async def test_a_stored_follow_up_whose_lone_user_item_had_no_content_replays_what_its_own_model_saw(
    stored_route: Mapping[str, str], expected: list[tuple[object, object]]
) -> None:
    first_turn: Final = _spend_log("chatcmpl-first", "session-empty-turn", "What is 2+2?", "Four.")
    follow_up: Final = {
        **_spend_log("chatcmpl-follow-up", "session-empty-turn", "", "Eight."),
        **stored_route,
        "proxy_server_request": {"input": [{"role": "user", "content": None}], "model": "claude-bridge"},
    }

    result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        "chatcmpl-follow-up",
        stored_turns=_stored_turns(first_turn, follow_up),
        empty_user_turn_text=EMPTY_TEXT_PLACEHOLDER,
    )

    assert _replayed_turns(result) == expected


async def _replay_of_a_blank_follow_up(
    stored_route: Mapping[str, str],
    blank_input: str | list[dict[str, object]],
    empty_user_turn_text: str = EMPTY_TEXT_PLACEHOLDER,
) -> list[tuple[object, object]]:
    first_turn: Final = _spend_log("chatcmpl-first", "session-blank-turn", "What is 2+2?", "Four.")
    follow_up: Final = {
        **_spend_log("chatcmpl-follow-up", "session-blank-turn", "", "Eight."),
        **stored_route,
        "proxy_server_request": {"input": blank_input, "model": "claude-bridge"},
    }
    result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        "chatcmpl-follow-up",
        stored_turns=_stored_turns(first_turn, follow_up),
        empty_user_turn_text=empty_user_turn_text,
    )
    return _replayed_turns(result)


@pytest.mark.parametrize(
    "blank_input",
    (
        "",
        "   ",
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": [{"type": "input_text", "text": " \n"}]}],
    ),
    ids=("empty_input_string", "whitespace_input_string", "whitespace_string_item", "whitespace_text_block_item"),
)
@pytest.mark.parametrize(
    "stored_route",
    (_ROUTE_THAT_FILLS, _ARN_ROUTE_THAT_FILLS),
    ids=("stored_model_refuses_a_prefill", "stored_arn_priced_as_a_model_that_refuses_it"),
)
@pytest.mark.asyncio
async def test_a_stored_blank_user_turn_replays_the_placeholder_its_model_saw(
    stored_route: Mapping[str, str], blank_input: str | list[dict[str, object]]
) -> None:
    assert await _replay_of_a_blank_follow_up(stored_route, blank_input) == _FILLED_REPLAY


@pytest.mark.asyncio
async def test_a_stored_blank_user_turn_replays_as_sent_when_its_model_accepts_a_prefill() -> None:
    replay: Final = await _replay_of_a_blank_follow_up(_ROUTE_THAT_ACCEPTS_A_PREFILL, [{"role": "user", "content": "   "}])

    assert replay == [("user", "What is 2+2?"), ("assistant", "Four."), ("user", "   "), ("assistant", "Eight.")]


@pytest.mark.parametrize(
    ("follow_up_route", "replayed_blank"),
    (
        ({"model": "gemini-3.5-flash", "custom_llm_provider": "gemini"}, "   "),
        (_ROUTE_THAT_FILLS, EMPTY_TEXT_PLACEHOLDER),
        (_ROUTE_THAT_ACCEPTS_A_PREFILL, EMPTY_TEXT_PLACEHOLDER),
        (
            {"model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "custom_llm_provider": "bedrock"},
            EMPTY_TEXT_PLACEHOLDER,
        ),
    ),
    ids=(
        "follow_up_route_takes_blank_text",
        "follow_up_route_fills_blank_turns",
        "anthropic_follow_up_accepts_a_prefill",
        "converse_follow_up_accepts_a_prefill",
    ),
)
@pytest.mark.asyncio
async def test_a_stored_blank_user_turn_is_filled_only_when_the_follow_up_route_refuses_blank_text(
    follow_up_route: Mapping[str, str], replayed_blank: str
) -> None:
    first_turn: Final = _spend_log("chatcmpl-first", "session-blank-turn", "What is 2+2?", "Four.")
    follow_up: Final = {
        **_spend_log("chatcmpl-follow-up", "session-blank-turn", "", "Eight."),
        **_ROUTE_THAT_FILLS,
        "proxy_server_request": {"input": [{"role": "user", "content": "   "}], "model": "claude-bridge"},
    }

    request: Final = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
        previous_response_id="chatcmpl-follow-up",
        litellm_completion_request={**follow_up_route, "messages": [{"role": "user", "content": "Double it."}]},
        instructions=None,
        stored_turns=_stored_turns(first_turn, follow_up),
        request_input="Double it.",
    )

    assert _replayed_turns(ChatCompletionSession(messages=request["messages"], litellm_session_id=None)) == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        ("user", replayed_blank),
        ("assistant", "Eight."),
        ("user", "Double it."),
    ]


_STORED_REFERENCES: Final = [{"type": "tool_reference", "tool_name": "get_weather"}]


@pytest.mark.parametrize(
    ("follow_up_route", "replayed_output"),
    (
        ({"model": "claude-opus-5-5", "custom_llm_provider": "anthropic"}, _STORED_REFERENCES),
        ({"model": "gpt-5.6", "custom_llm_provider": "openai"}, json.dumps(_STORED_REFERENCES)),
        ({"model": "gemini-3.1-pro-preview", "custom_llm_provider": "vertex_ai"}, json.dumps(_STORED_REFERENCES)),
        ({"model": "my-batch-alias"}, json.dumps(_STORED_REFERENCES)),
    ),
    ids=(
        "anthropic_follow_up_replays_references",
        "openai_follow_up_replays_json",
        "vertex_follow_up_replays_json",
        "unresolved_alias_follow_up_replays_json",
    ),
)
@pytest.mark.asyncio
async def test_a_stored_tool_reference_output_replays_in_the_shape_the_follow_up_route_takes(
    follow_up_route: Mapping[str, str], replayed_output: object
) -> None:
    stored_turn: Final = {
        **_spend_log("chatcmpl-loaded-tools", "session-tool-references", "", "Loaded."),
        "proxy_server_request": {
            "input": [
                {"role": "user", "content": "Load the weather tools."},
                {"type": "function_call", "call_id": "call_replayed_refs", "name": "load_tools", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_replayed_refs", "output": _STORED_REFERENCES},
            ],
            "model": "claude-bridge",
        },
    }

    request: Final = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
        previous_response_id="chatcmpl-loaded-tools",
        litellm_completion_request={**follow_up_route, "messages": [{"role": "user", "content": "Weather in Paris?"}]},
        instructions=None,
        stored_turns=_stored_turns(stored_turn),
        request_input="Weather in Paris?",
    )

    replayed: Final = _replayed_turns(ChatCompletionSession(messages=request["messages"], litellm_session_id=None))
    assert [content for role, content in replayed if role == "tool"] == [replayed_output]


@pytest.mark.asyncio
async def test_stored_blank_user_items_that_inference_continues_replay_one_turn_each() -> None:
    replay: Final = await _replay_of_a_blank_follow_up(
        _ROUTE_THAT_FILLS, [{"role": "user", "content": ""}, {"role": "user", "content": "  "}], empty_user_turn_text=""
    )

    assert replay == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        ("user", ""),
        ("user", "  "),
        ("assistant", "Eight."),
    ]


@pytest.mark.asyncio
async def test_a_stored_turn_with_no_answer_replays_no_empty_user_turn() -> None:
    first_turn: Final = _spend_log("chatcmpl-first", "session-unanswered", "What is 2+2?", "Four.")
    unanswered: Final = {
        **_spend_log("chatcmpl-unanswered", "session-unanswered", "", ""),
        **_ROUTE_THAT_FILLS,
        "proxy_server_request": {"input": [{"role": "user", "content": None}], "model": "claude-bridge"},
        "response": {},
    }
    follow_up: Final = _spend_log("chatcmpl-follow-up", "session-unanswered", "And 3+3?", "Six.")

    result: Final = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        "chatcmpl-follow-up",
        stored_turns=_stored_turns(first_turn, unanswered, follow_up),
        empty_user_turn_text=EMPTY_TEXT_PLACEHOLDER,
    )

    assert _replayed_turns(result) == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        ("user", "And 3+3?"),
        ("assistant", "Six."),
    ]
