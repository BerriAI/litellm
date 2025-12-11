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
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
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
                description="Top secret project",
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
    async def test_apply_guardrail_block(self):
        """
        Test apply_guardrail with BLOCK action
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

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["My SSN is 123-45-6789"]},
                request_data={},
                input_type="request",
            )

        assert exc_info.value.status_code == 400
        assert "us_ssn" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_apply_guardrail_mask(self):
        """
        Test apply_guardrail with MASK action
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

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Contact me at test@example.com"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        assert "[EMAIL_REDACTED]" in result[0]
        assert "test@example.com" not in result[0]

    @pytest.mark.asyncio
    async def test_apply_guardrail_blocked_word_mask(self):
        """
        Test apply_guardrail with blocked word MASK action
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

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["This is PROPRIETARY information"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        assert "[KEYWORD_REDACTED]" in result[0]
        assert "PROPRIETARY" not in result[0]

    @pytest.mark.asyncio
    async def test_apply_guardrail_multiple_patterns(self):
        """
        Test apply_guardrail with multiple patterns in the same text
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_ssn",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-multiple",
            patterns=patterns,
        )

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Contact user@test.com or SSN: 123-45-6789"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        # At least one pattern should be redacted (first match wins)
        assert "[EMAIL_REDACTED]" in result[0] or "[US_SSN_REDACTED]" in result[0]

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """blocked_words:
  - keyword: "test_keyword"
    action: "BLOCK"
    description: "Test keyword"
  - keyword: "another_word"
    action: "MASK"
"""
            )
            temp_file = f.name

        try:
            guardrail = ContentFilterGuardrail(
                guardrail_name="test-file-load",
                blocked_words_file=temp_file,
            )

            assert len(guardrail.blocked_words) == 2
            assert "test_keyword" in guardrail.blocked_words
            assert (
                guardrail.blocked_words["test_keyword"][0] == ContentFilterAction.BLOCK
            )
            assert guardrail.blocked_words["test_keyword"][1] == "Test keyword"
            assert "another_word" in guardrail.blocked_words
            assert (
                guardrail.blocked_words["another_word"][0] == ContentFilterAction.MASK
            )
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

    @pytest.mark.skip(
        reason="Masking in streaming responses is no longer supported after unified_guardrail.py changes. Only blocking/rejecting is supported for responses."
    )
    @pytest.mark.asyncio
    async def test_streaming_hook_mask(self):
        """
        Test streaming hook with MASK action

        Note: After changes to unified_guardrail.py, masking responses to users
        is no longer supported. This test is skipped as the feature is deprecated.
        Only BLOCK actions (test_streaming_hook_block) are supported for streaming responses.
        """
        from unittest.mock import AsyncMock

        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-streaming-mask",
            patterns=patterns,
            event_hook=GuardrailEventHooks.during_call,
        )

        # Create mock streaming chunks
        async def mock_stream():
            # Chunk 1: contains email
            chunk1 = ModelResponseStream(
                id="chunk1",
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Contact me at test@example.com"), index=0
                    )
                ],
                model="gpt-4",
            )
            yield chunk1

            # Chunk 2: normal content
            chunk2 = ModelResponseStream(
                id="chunk2",
                choices=[
                    StreamingChoices(delta=Delta(content=" for more info"), index=0)
                ],
                model="gpt-4",
            )
            yield chunk2

        user_api_key_dict = MagicMock()
        request_data = {}

        # Process streaming response - no masking expected
        result_chunks = []
        async for chunk in guardrail.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_api_key_dict,
            response=mock_stream(),
            request_data=request_data,
        ):
            result_chunks.append(chunk)

        # Chunks should pass through unchanged since masking is no longer supported
        assert len(result_chunks) == 2

    @pytest.mark.asyncio
    async def test_streaming_hook_block(self):
        """
        Test streaming hook with BLOCK action
        """
        from unittest.mock import AsyncMock

        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="us_ssn",
                action=ContentFilterAction.BLOCK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-streaming-block",
            patterns=patterns,
            event_hook=GuardrailEventHooks.during_call,
        )

        # Create mock streaming chunks with SSN
        async def mock_stream():
            chunk = ModelResponseStream(
                id="chunk1",
                choices=[
                    StreamingChoices(delta=Delta(content="SSN: 123-45-6789"), index=0)
                ],
                model="gpt-4",
            )
            yield chunk

        user_api_key_dict = MagicMock()
        request_data = {}

        # Should raise HTTPException when SSN is detected
        with pytest.raises(HTTPException) as exc_info:
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=mock_stream(),
                request_data=request_data,
            ):
                pass

        assert exc_info.value.status_code == 400
        assert "us_ssn" in str(exc_info.value.detail)

    def test_init_with_plain_dicts(self):
        """
        Test initialization with plain dicts (DB format).
        This ensures the guardrail can handle data loaded from the database
        without requiring Pydantic model conversion.
        """
        patterns = [
            {
                "pattern_type": "prebuilt",
                "pattern_name": "us_ssn",
                "action": "BLOCK",
                "name": "us_ssn",
                "pattern": None,
            },
            {
                "pattern_type": "prebuilt",
                "pattern_name": "email",
                "action": "MASK",
                "name": "email",
                "pattern": None,
            },
        ]

        blocked_words = [
            {
                "keyword": "langchain",
                "action": "BLOCK",
                "description": None,
            },
            {
                "keyword": "openai",
                "action": "MASK",
                "description": "Competitor name",
            },
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-db-format",
            patterns=patterns,
            blocked_words=blocked_words,
        )

        assert guardrail.guardrail_name == "test-db-format"
        assert len(guardrail.compiled_patterns) == 2
        assert len(guardrail.blocked_words) == 2

        # Verify blocked_words are stored as dict
        assert "langchain" in guardrail.blocked_words
        assert guardrail.blocked_words["langchain"] == ("BLOCK", None)
        assert "openai" in guardrail.blocked_words
        assert guardrail.blocked_words["openai"] == ("MASK", "Competitor name")

    @pytest.mark.asyncio
    async def test_apply_guardrail_masks_all_different_blocked_keywords(self):
        """
        Test that ALL different blocked keywords are masked, not just the first one.

        Regression test for GitHub issue #17517:
        https://github.com/BerriAI/litellm/issues/17517

        Before fix:
            Input: "Keyword01 Keyword01 Keyword02"
            Output: "[KEYWORD_REDACTED] [KEYWORD_REDACTED] Keyword02"
            (only first matching keyword type was replaced)

        After fix:
            Input: "Keyword01 Keyword01 Keyword02"
            Output: "[KEYWORD_REDACTED] [KEYWORD_REDACTED] [KEYWORD_REDACTED]"
            (all matching keywords are replaced)
        """
        blocked_words = [
            BlockedWord(
                keyword="keyword01",
                action=ContentFilterAction.MASK,
            ),
            BlockedWord(
                keyword="keyword02",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-multiple-keywords",
            blocked_words=blocked_words,
        )

        # Test case from issue #17517
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Keyword01 Keyword01 Keyword02"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        # All keywords should be redacted
        assert result[0] == "[KEYWORD_REDACTED] [KEYWORD_REDACTED] [KEYWORD_REDACTED]"
        assert "Keyword01" not in result[0]
        assert "Keyword02" not in result[0]

    @pytest.mark.asyncio
    async def test_apply_guardrail_masks_multiple_keywords_different_order(self):
        """
        Test that keyword order in input doesn't affect masking all keywords.

        Additional test for GitHub issue #17517.
        """
        blocked_words = [
            BlockedWord(
                keyword="keyword01",
                action=ContentFilterAction.MASK,
            ),
            BlockedWord(
                keyword="keyword02",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-multiple-keywords-order",
            blocked_words=blocked_words,
        )

        # Test with keyword02 appearing first
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Keyword02 Keyword01 Keyword02"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        assert result[0] == "[KEYWORD_REDACTED] [KEYWORD_REDACTED] [KEYWORD_REDACTED]"

    @pytest.mark.asyncio
    async def test_apply_guardrail_blocks_on_any_blocked_keyword(self):
        """
        Test that if any keyword has BLOCK action, it blocks even if others have MASK.
        """
        blocked_words = [
            BlockedWord(
                keyword="safe_word",
                action=ContentFilterAction.MASK,
            ),
            BlockedWord(
                keyword="danger_word",
                action=ContentFilterAction.BLOCK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-block-priority",
            blocked_words=blocked_words,
        )

        # Should block when danger_word is present, even if safe_word is also there
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"texts": ["safe_word and danger_word together"]},
                request_data={},
                input_type="request",
            )

        assert exc_info.value.status_code == 400
        assert "danger_word" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_apply_guardrail_masks_all_regex_pattern_matches(self):
        """
        Test that ALL matches of a regex pattern are masked, not just the first one.

        Regression test for GitHub issue #17687:
        https://github.com/BerriAI/litellm/issues/17687

        Before fix:
            Regex: Key\\d+
            Input: "Key1 Key1 Key2"
            Output: "[CUSTOM_REGEX_REDACTED] [CUSTOM_REGEX_REDACTED] Key2"
            (only first unique match was replaced)

        After fix:
            Input: "Key1 Key1 Key2"
            Output: "[CUSTOM_REGEX_REDACTED] [CUSTOM_REGEX_REDACTED] [CUSTOM_REGEX_REDACTED]"
            (all matches are replaced)
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="regex",
                pattern=r"Key\d+",
                name="custom_key",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-regex-all-matches",
            patterns=patterns,
        )

        # Test case from issue #17687
        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Key1 Key1 Key2"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        # All matches should be redacted
        assert result[0] == "[CUSTOM_KEY_REDACTED] [CUSTOM_KEY_REDACTED] [CUSTOM_KEY_REDACTED]"
        assert "Key1" not in result[0]
        assert "Key2" not in result[0]

    @pytest.mark.asyncio
    async def test_apply_guardrail_masks_multiple_patterns_all_matches(self):
        """
        Test that multiple different patterns each mask ALL their matches.
        """
        patterns = [
            ContentFilterPattern(
                pattern_type="prebuilt",
                pattern_name="email",
                action=ContentFilterAction.MASK,
            ),
            ContentFilterPattern(
                pattern_type="regex",
                pattern=r"Key\d+",
                name="custom_key",
                action=ContentFilterAction.MASK,
            ),
        ]

        guardrail = ContentFilterGuardrail(
            guardrail_name="test-multiple-patterns-all",
            patterns=patterns,
        )

        guardrailed_inputs = await guardrail.apply_guardrail(
            inputs={"texts": ["Key1 user@test.com Key2 admin@test.com Key1"]},
            request_data={},
            input_type="request",
        )
        result = guardrailed_inputs.get("texts", [])

        assert result is not None
        assert len(result) == 1
        # All emails should be redacted
        assert "user@test.com" not in result[0]
        assert "admin@test.com" not in result[0]
        assert result[0].count("[EMAIL_REDACTED]") == 2
        # All Key patterns should be redacted
        assert "Key1" not in result[0]
        assert "Key2" not in result[0]
        assert result[0].count("[CUSTOM_KEY_REDACTED]") == 3
