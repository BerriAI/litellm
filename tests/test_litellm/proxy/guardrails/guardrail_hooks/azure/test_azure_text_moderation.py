from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_pre_call_hook():

    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Sexual", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
                {"category": "Violence", "severity": 0},
            ],
        }
        await azure_text_moderation_guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                api_key="azure_text_moderation_api_key"
            ),
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ]
            },
            call_type="completion",
        )

        mock_async_make_request.assert_called_once()
        assert mock_async_make_request.call_args.kwargs["text"] == "Hello, how are you?"


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_violation_detected():
    """async_make_request is the single enforcement point — it raises
    HTTPException when severity thresholds are exceeded.  The caller
    (pre_call_hook) simply propagates the exception.
    """
    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.side_effect = HTTPException(
            status_code=400,
            detail={
                "error": "Azure Content Safety Guardrail: Hate crossed severity 2, Got severity: 2"
            },
        )
        with pytest.raises(HTTPException):
            await azure_text_moderation_guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="azure_text_moderation_api_key"
                ),
                cache=None,
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": "I hate you!",
                        }
                    ]
                },
                call_type="completion",
            )

        mock_async_make_request.assert_called_once()
        assert mock_async_make_request.call_args.kwargs["text"] == "I hate you!"


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_long_text_splitting():
    """Test that long text is properly split into multiple API calls."""
    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )

    # Create text longer than 10000 characters
    long_text = "This is a safe text. " * 1000  # ~20000 characters

    mock_response = Mock()
    mock_response.json.return_value = {
        "blocklistsMatch": [],
        "categoriesAnalysis": [
            {"category": "Hate", "severity": 0},
            {"category": "Sexual", "severity": 0},
            {"category": "SelfHarm", "severity": 0},
            {"category": "Violence", "severity": 0},
        ],
    }

    with patch.object(
        azure_text_moderation_guardrail.async_handler, "post",
        return_value=mock_response,
    ) as mock_post:
        await azure_text_moderation_guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                api_key="azure_text_moderation_api_key"
            ),
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": long_text,
                    }
                ]
            },
            call_type="completion",
        )

        # Should be called multiple times due to splitting
        assert mock_post.call_count > 1

        # Check that each chunk sent in the request body is <= 10000 characters
        for call in mock_post.call_args_list:
            request_body = call.kwargs["json"]
            assert len(request_body["text"]) <= 10000


@pytest.mark.asyncio
async def test_azure_text_moderation_violation_in_chunk():
    """Test that violation is detected even when it's in a chunk of long text."""
    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )

    # Create text with violation in the middle
    safe_text = "This is safe content. " * 500
    violation_text = "I hate everyone!"
    long_text = safe_text + violation_text + safe_text

    def make_mock_response(severity):
        resp = Mock()
        resp.json.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": severity},
                {"category": "Sexual", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
                {"category": "Violence", "severity": 0},
            ],
        }
        return resp

    def post_side_effect(**kwargs):
        body = kwargs.get("json", {})
        text = body.get("text", "")
        if "I hate everyone!" in text:
            return make_mock_response(severity=2)
        return make_mock_response(severity=0)

    with patch.object(
        azure_text_moderation_guardrail.async_handler, "post",
        side_effect=post_side_effect,
    ):
        with pytest.raises(HTTPException):
            await azure_text_moderation_guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="azure_text_moderation_api_key"
                ),
                cache=None,
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": long_text,
                        }
                    ]
                },
                call_type="completion",
            )


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_post_call_success_hook():

    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
            ],
        }
        result = await azure_text_moderation_guardrail.async_post_call_success_hook(
            data={},
            user_api_key_dict=UserAPIKeyAuth(
                api_key="azure_text_moderation_api_key"
            ),
            response=ModelResponse(
                choices=[
                    Choices(
                        index=0,
                        message=Message(content="Hello world"),
                    )
                ]
            ),
        )

        assert result is not None
        mock_async_make_request.assert_called_once()
        assert mock_async_make_request.call_args.kwargs["text"] == "Hello world"


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_post_call_streaming_hook():

    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
            ],
        }
        result = await azure_text_moderation_guardrail.async_post_call_streaming_hook(
            user_api_key_dict=UserAPIKeyAuth(
                api_key="azure_text_moderation_api_key"
            ),
            response="Hello world",
        )

        assert result is not None
        mock_async_make_request.assert_called_once()
        assert mock_async_make_request.call_args.kwargs["text"] == "Hello world"


def test_split_text_by_words():
    """Test the word-based text splitting functionality."""
    guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="test",
        api_key="test_key",
        api_base="test_base",
    )
    
    # Test short text (no splitting needed)
    short_text = "Hello world"
    chunks = guardrail.split_text_by_words(short_text, 100)
    assert len(chunks) == 1
    assert chunks[0] == short_text
    
    # Test text that needs splitting
    text = "word1 word2 word3 word4 word5"
    chunks = guardrail.split_text_by_words(text, 20)
    assert len(chunks) > 1
    # Verify no word is broken
    for chunk in chunks:
        assert "word1" in chunk or "word2" in chunk or "word3" in chunk or "word4" in chunk or "word5" in chunk
    
    # Test with very long single word (edge case)
    long_word = "supercalifragilisticexpialidocious" * 10
    chunks = guardrail.split_text_by_words(long_word, 50)
    assert len(chunks) > 1
    # Each chunk should be exactly 50 chars except possibly the last
    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk) == 50
    
    # Test empty string
    chunks = guardrail.split_text_by_words("", 100)
    assert chunks == [""]
    
    # Test with punctuation and special characters
    text_with_punctuation = "Hello, world! How are you? I'm fine."
    chunks = guardrail.split_text_by_words(text_with_punctuation, 30)
    # Verify no word is broken across chunks
    assert "".join(chunks) == text_with_punctuation
    for chunk in chunks:
        assert len(chunk) <= 30


def test_split_text_preserves_content():
    """Test that splitting and recombining preserves the original content exactly."""
    guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="test",
        api_key="test_key",
        api_base="test_base",
    )
    
    original_text = "The quick brown fox jumps over the lazy dog. " * 100
    chunks = guardrail.split_text_by_words(original_text, 1000)
    
    # Whitespace-preserving split: concatenation reproduces original exactly
    assert "".join(chunks) == original_text


def test_split_preserves_whitespace():
    """Test that newlines, tabs, and multiple spaces are preserved in chunks."""
    guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="test",
        api_key="test_key",
        api_base="test_base",
    )

    # Text with mixed whitespace that needs splitting
    text = "hello\n\nworld\t\tfoo   bar"
    chunks = guardrail.split_text_by_words(text, 15)
    assert len(chunks) > 1
    # Exact reconstruction
    assert "".join(chunks) == text

    # Longer text with varied whitespace
    original = ("line one\n" + "line two\t\tcol\n" + "  indented\n") * 200
    chunks = guardrail.split_text_by_words(original, 500)
    assert "".join(chunks) == original
