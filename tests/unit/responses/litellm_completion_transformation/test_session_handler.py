import json
import sqlite3
from collections.abc import Sequence
from itertools import chain
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.proxy.spend_tracking.spend_tracking_utils import (
    _get_proxy_server_request_for_spend_logs_payload,
)
from litellm.responses.litellm_completion_transformation import session_handler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
    _normalize_redacted_tool_call_arguments,
)
from litellm.responses.litellm_completion_transformation.transformation import ChatCompletionSession
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
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def query_raw(self, query, request_ids):
        self.calls.append(tuple(request_ids))
        if not self._results:
            return []
        return list(self._results.pop(0))


class _FakePrismaClient:
    def __init__(self, results):
        self.db = _FakePrismaDB(results)


def _spend_log(request_id: str, session_id: str, prompt: str, answer: str) -> dict:
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
    assert [call[0] for call in fake_prisma_client.db.calls] == [request_id, request_id]


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

    assert len(fake_prisma_client.db.calls) == 1
    assert request_id in fake_prisma_client.db.calls[0]


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


def _tool_call_ids(message: object) -> tuple[str, ...]:
    tool_calls: Final = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    return tuple(call["id"] if isinstance(call, dict) else call.id for call in tool_calls or ())


_WEATHER_CALL_TURN: Final = {
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
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            }
        ],
    },
}


def _compacted_weather_answer(request_id: str) -> dict[str, object]:
    return {
        **_turn_spend_log(
            [_COMPACTION_ITEM_SENT, {"type": "function_call_output", "call_id": "call_1", "output": "sunny"}],
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
    def __init__(self, rows: list[tuple[str, str, str, str, str | None, str | None]]) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            'CREATE TABLE "LiteLLM_SpendLogs" '
            '(request_id TEXT, api_key TEXT, session_id TEXT, "endTime" TEXT, "user" TEXT, team_id TEXT)'
        )
        self._connection.executemany('INSERT INTO "LiteLLM_SpendLogs" VALUES (?, ?, ?, ?, ?, ?)', rows)

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        sqlite_query: Final = query.replace("= ANY($1::text[])", "IN (SELECT value FROM json_each(?))").replace(
            "$1", "?"
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
