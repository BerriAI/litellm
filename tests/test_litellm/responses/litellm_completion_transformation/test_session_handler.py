import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.responses.litellm_completion_transformation import session_handler
from litellm.responses.litellm_completion_transformation.session_handler import (
    ResponsesSessionHandler,
    _normalize_redacted_tool_call_arguments,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
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
        mock_cold_storage.get_proxy_server_request_from_cold_storage_with_object_key = (
            AsyncMock(return_value=full_proxy_request)
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
        result1 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_with_truncated_pdf
        )
        assert (
            result1 == True
        ), "Should return True for proxy request with truncated PDF content"

        # Test case 2: Should return False for regular content
        result2 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_regular
        )
        assert (
            result2 == False
        ), "Should return False for regular proxy request without truncation"

        # Test case 3: Should return True for empty request
        result3 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_empty
        )
        assert result3 == True, "Should return True for empty proxy request"

        # Test case 4: Should return True for None request
        result4 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_none
        )
        assert result4 == True, "Should return True for None proxy request"

    # Test case 5: Should return False when cold storage is not configured
    with patch.object(litellm, "cold_storage_custom_logger", None):
        result5 = ResponsesSessionHandler._should_check_cold_storage_for_full_payload(
            proxy_request_with_truncated_pdf
        )
        assert (
            result5 == False
        ), "Should return False when cold storage is not configured, even with truncated content"


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

    with patch.object(
        ResponsesSessionHandler, "get_all_spend_logs_for_previous_response_id"
    ) as mock_get_spend_logs:
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

    async def query_raw(self, query, *args):
        self.calls.append(args)
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

        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            request_id
        )

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
        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            request_id
        )

    messages = result["messages"]
    assert [(message.get("role"), message.get("content")) for message in messages] == [
        ("user", "Remember this: my favorite color is chartreuse."),
        ("assistant", "OK"),
    ]
    assert result["litellm_session_id"] == session_id
    assert fake_prisma_client.db.calls == [(request_id,), (request_id,)]


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
    fake_prisma_client = _FakePrismaClient(
        results=[[_spend_log(request_id, "session-a", "Hello.", "Hi.")]]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client):
        await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            encoded_response_id
        )

    assert fake_prisma_client.db.calls == [(request_id,)]


@pytest.mark.asyncio
async def test_session_lookup_does_not_retry_when_spend_logs_are_disabled(
    instant_session_lookup_retries: None,
):
    """
    A deployment that writes no spend logs has nothing to wait for, so the miss path keeps
    the single query it always had.
    """
    fake_prisma_client = _FakePrismaClient(results=[])

    with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma_client), patch(
        "litellm.proxy.proxy_server.disable_spend_logs", True
    ):
        spend_logs = await ResponsesSessionHandler.get_all_spend_logs_for_previous_response_id(
            "chatcmpl-does-not-exist"
        )

    assert spend_logs == []
    assert fake_prisma_client.db.calls == [("chatcmpl-does-not-exist",)]


def test_normalize_redacted_arguments_skips_custom_tool_calls():
    """Custom tool calls have no .function; the normalizer must skip them, not crash (session replay path)."""
    message = Message(
        content=None,
        tool_calls=[
            {"id": "call_c", "type": "custom", "custom": {"name": "run_code", "input": "print(1)"}},
            {"id": "call_f", "type": "function", "function": {"name": "get_weather", "arguments": "redacted-by-litellm"}},
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

    with patch.object(  # test-quality-ok: the handler has no DI seam for the spend-log fetch; every test in this file stubs this same boundary
        ResponsesSessionHandler,
        "get_all_spend_logs_for_previous_response_id",
        new_callable=AsyncMock,
    ) as mock_get_spend_logs:
        mock_get_spend_logs.return_value = mock_spend_logs

        result = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
            "chatcmpl-redacted-1"
        )

    assistant_message = result["messages"][-1]
    tool_call = assistant_message.tool_calls[0]
    assert tool_call.function.arguments == "{}"
    assert json.loads(tool_call.function.arguments) == {}
