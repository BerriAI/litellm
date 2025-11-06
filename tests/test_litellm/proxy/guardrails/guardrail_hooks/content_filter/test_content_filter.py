"""
Tests for the Content Filter Guardrail
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

from fastapi import HTTPException

import litellm
from litellm.proxy.guardrails.guardrail_hooks.content_filter.content_filter import (
    ContentFilterGuardrail,
)
from litellm.types.guardrails import (
    BlockedWord,
    ContentFilterAction,
    ContentFilterPattern,
    GuardrailEventHooks,
)


class TestContentFilterGuardrail:
    """Test the ContentFilterGuardrail class"""

    def test_init_with_patterns(self):
        """
        Test initialization with prebuilt patterns
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_ssn",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-content-filter",
            patterns=patterns,
        )
        
        assert guardrail.guardrail_name == "test-content-filter"
        assert len(guardrail.compiled_patterns) == 1

    def test_init_with_blocked_words(self):
        """
        Test initialization with blocked words
        """
        blocked_words = [
            BlockedWord(
                keyword="secret_project",
                action=ContentFilterAction.BLOCK,
                description="Top secret project"
            ),
            BlockedWord(
                keyword="internal_api",
                action=ContentFilterAction.MASK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-content-filter",
            blocked_words=blocked_words,
        )
        
        assert len(guardrail.blocked_words) == 2
        assert "secret_project" in guardrail.blocked_words
        assert guardrail.blocked_words["secret_project"][0] == ContentFilterAction.BLOCK

    def test_check_patterns_ssn(self):
        """
        Test SSN pattern detection
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_ssn",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-ssn",
            patterns=patterns,
        )
        
        # Test with SSN
        result = guardrail._check_patterns("My SSN is 123-45-6789")
        assert result is not None
        assert result[1] == "us_ssn"
        assert result[2] == ContentFilterAction.BLOCK
        
        # Test without SSN
        result = guardrail._check_patterns("This is a normal message")
        assert result is None

    def test_check_patterns_email(self):
        """
        Test email pattern detection
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-email",
            patterns=patterns,
        )
        
        result = guardrail._check_patterns("Contact me at test@example.com")
        assert result is not None
        assert result[1] == "email"
        assert result[2] == ContentFilterAction.MASK

    def test_check_patterns_custom_regex(self):
        """
        Test custom regex pattern detection
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="regex",
                pattern=r"\b[A-Z]{3}-\d{4}\b",
                name="custom_id",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-custom",
            patterns=patterns,
        )
        
        result = guardrail._check_patterns("My ID is ABC-1234")
        assert result is not None
        assert result[1] == "custom_id"

    def test_check_blocked_words(self):
        """
        Test blocked word detection
        """
        blocked_words = [
            BlockedWord(
                keyword="confidential",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-words",
            blocked_words=blocked_words,
        )
        
        # Test with blocked word
        result = guardrail._check_blocked_words("This is CONFIDENTIAL information")
        assert result is not None
        assert result[0] == "confidential"
        assert result[1] == ContentFilterAction.BLOCK
        
        # Test without blocked word
        result = guardrail._check_blocked_words("This is normal information")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_block(self):
        """
        Test async_pre_call_hook with BLOCK action
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_ssn",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-block",
            patterns=patterns,
        )
        
        data = {
            "messages": [
                {"role": "user", "content": "My SSN is 123-45-6789"}
            ]
        }
        
        user_api_key_dict = MagicMock()
        cache = MagicMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion",
            )
        
        assert exc_info.value.status_code == 400
        assert "us_ssn" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_mask(self):
        """
        Test async_pre_call_hook with MASK action
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-mask",
            patterns=patterns,
        )
        
        data = {
            "messages": [
                {"role": "user", "content": "Contact me at test@example.com"}
            ]
        }
        
        user_api_key_dict = MagicMock()
        cache = MagicMock()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion",
        )
        
        assert result is not None
        assert "[EMAIL_REDACTED]" in result["messages"][0]["content"]
        assert "test@example.com" not in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_blocked_word_mask(self):
        """
        Test async_pre_call_hook with blocked word MASK action
        """
        blocked_words = [
            BlockedWord(
                keyword="proprietary",
                action=ContentFilterAction.MASK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-word-mask",
            blocked_words=blocked_words,
        )
        
        data = {
            "messages": [
                {"role": "user", "content": "This is PROPRIETARY information"}
            ]
        }
        
        user_api_key_dict = MagicMock()
        cache = MagicMock()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion",
        )
        
        assert result is not None
        assert "[KEYWORD_REDACTED]" in result["messages"][0]["content"]
        assert "PROPRIETARY" not in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_multimodal(self):
        """
        Test async_pre_call_hook with multimodal content
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-multimodal",
            patterns=patterns,
        )
        
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Email: user@test.com"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                    ]
                }
            ]
        }
        
        user_api_key_dict = MagicMock()
        cache = MagicMock()
        
        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type="completion",
        )
        
        assert result is not None
        assert "[EMAIL_REDACTED]" in result["messages"][0]["content"][0]["text"]
        assert "user@test.com" not in result["messages"][0]["content"][0]["text"]

    def test_mask_content(self):
        """
        Test content masking
        """
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-mask",
        )
        
        masked = guardrail._mask_content("sensitive text", "us_ssn")
        assert masked == "[US_SSN_REDACTED]"

    def test_load_blocked_words_file(self):
        """
        Test loading blocked words from a YAML file
        """
        import tempfile

        # Create a temporary blocked words file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""blocked_words:
  - keyword: "test_keyword"
    action: "BLOCK"
    description: "Test keyword"
  - keyword: "another_word"
    action: "MASK"
""")
            temp_file = f.name
        
        try:
            guardrail = ContentFilterGuardrail(
                guardrail_name="test-file-load",
                blocked_words_file=temp_file,
            )
            
            assert len(guardrail.blocked_words) == 2
            assert "test_keyword" in guardrail.blocked_words
            assert guardrail.blocked_words["test_keyword"][0] == ContentFilterAction.BLOCK
            assert guardrail.blocked_words["test_keyword"][1] == "Test keyword"
            assert "another_word" in guardrail.blocked_words
            assert guardrail.blocked_words["another_word"][0] == ContentFilterAction.MASK
        finally:
            os.unlink(temp_file)

    def test_credit_card_patterns(self):
        """
        Test credit card pattern detection
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="visa",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-cc",
            patterns=patterns,
        )
        
        # Test Visa card
        result = guardrail._check_patterns("My card is 4532-1234-5678-9010")
        assert result is not None
        assert result[1] == "visa"
        
    def test_api_key_patterns(self):
        """
        Test API key pattern detection
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="aws_access_key",
                action=ContentFilterAction.BLOCK,
            ),
        ]
        
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-api-key",
            patterns=patterns,
        )
        
        # Test AWS Access Key
        result = guardrail._check_patterns("My key is AKIAIOSFODNN7EXAMPLE")
        assert result is not None
        assert result[1] == "aws_access_key"

