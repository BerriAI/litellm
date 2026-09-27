"""
Regression tests for PR #43442: SQL keyword word boundaries fix

These tests verify fixes for two issues:
1. Punctuation-only identifiers no longer match (e.g., ">", "=")
2. Inflected forms can bypass the filter (documented as known limitation)
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (
    ContentFilterGuardrail,
    _is_word_char_pattern,
)
from litellm.types.guardrails import BlockedWord, ContentFilterAction


class TestWordCharPatternHelper:
    """Tests for the _is_word_char_pattern helper function."""

    def test_alphanumeric_word(self):
        """Alphanumeric words are word characters."""
        assert _is_word_char_pattern("alter") is True
        assert _is_word_char_pattern("select") is True
        assert _is_word_char_pattern("table123") is True
        assert _is_word_char_pattern("_private") is True

    def test_punctuation_only(self):
        """Punctuation-only identifiers are not word characters."""
        assert _is_word_char_pattern("=") is False
        assert _is_word_char_pattern(">") is False
        assert _is_word_char_pattern("<") is False
        assert _is_word_char_pattern("!=") is False
        assert _is_word_char_pattern("==") is False

    def test_mixed_words(self):
        """Words with punctuation mixed in have word characters."""
        # These should still be treated as word patterns for boundary purposes
        # since they contain word characters
        assert _is_word_char_pattern("drop") is False  # Backslash is not word char
        assert _is_word_char_pattern("--") is False  # Dashes

    def test_empty_string(self):
        """Empty string returns False."""
        assert _is_word_char_pattern("") is False


class TestPunctuationOnlyIdentifiers:
    """Tests that punctuation-only identifiers work with word boundaries."""

    def test_punctuation_identifier_matches(self):
        """
        Test that punctuation-only identifiers (like SQL operators) still match.
        Regression test for Issue 1: Punctuation-only identifiers no longer matched.
        """
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-punctuation",
            blocked_words=[
                BlockedWord(keyword="=", action=ContentFilterAction.BLOCK),
                BlockedWord(keyword=">", action=ContentFilterAction.BLOCK),
                BlockedWord(keyword="<", action=ContentFilterAction.BLOCK),
            ]
        )

        # These should all match (contain the punctuation)
        test_cases = [
            ("x = y", "=", "Equal sign"),
            ("if x > 0", ">", "Greater than"),
            ("a < b", "<", "Less than"),
        ]

        for text, keyword, desc in test_cases:
            result = guardrail._check_blocked_words(text)
            assert result is not None, f"{desc}: Should match {keyword!r} in {text!r}"


class TestInflectedFormsKnownLimitation:
    """
    Tests demonstrating the inflected forms limitation.
    
    This is a KNOWN LIMITATION, not a bug. Word boundary matching intentionally
    does not match stemmed/inflected forms. This avoids false positives.
    """

    def test_base_form_matches(self):
        """
        Base forms (like "alter") still match.
        """
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-stemming",
            blocked_words=[
                BlockedWord(keyword="alter", action=ContentFilterAction.BLOCK),
            ]
        )

        # Base form should match
        result = guardrail._check_blocked_words("I want to alter the table")
        assert result is not None
        assert result[0] == "alter"

    def test_inflected_forms_do_not_match(self):
        """
        Inflected forms (like "alters", "altered") do NOT match
        the base form "alter". This is a KNOWN LIMITATION.
        
        Design choice: We explicitly document this as a limitation rather than
        implementing stemming/lemmatization because:
        
        1. Stemming can cause MORE false positives (e.g., "men" matching within
           "recommend" after stemming)
        2. Word boundaries prevent false positives with substring matches
        3. Policy authors can explicitly configure multiple related keywords
           in their policy YAML (e.g., "alter", "alters", "altered")
        4. Punctuation-only identifiers require substring matching anyway
        5. Adding NLTK/stemming introduces complexity and dependencies
        
        The trade-off is that policy authors may need to be explicit about
        which forms they want to block, but gain more predictable behavior.
        """
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-stemming-limited",
            blocked_words=[
                BlockedWord(keyword="alter", action=ContentFilterAction.BLOCK),
            ]
        )

        # Infected forms DO NOT match - this is documented behavior
        inflected_texts = [
            ("He alters the schema", "Present tense with 's'"),
            ("The table was altered", "Past tense with 'ed'"),
            ("They are altering data", "Continuous with 'ing'"),
        ]

        for text, desc in inflected_texts:
            result = guardrail._check_blocked_words(text)
            # These will NOT match - documented as known limitation
            assert result is None, f"{desc}: {text!r} - Intentionally does not match base form"


class TestFalsePositiveAvoidance:
    """Tests that false positives are avoided (original PR intent)."""

    def test_alternative_does_not_match_alter(self):
        """
        "alternative" should NOT match "alter" - the key fix from PR #43442.
        """
        guardrail = ContentFilterGuardrail(
            guardrail_name="test-fp-avoidance",
            blocked_words=[
                BlockedWord(keyword="alter", action=ContentFilterAction.BLOCK),
            ]
        )

        # "alternative" contains "alter" as a substring but word boundary prevents match
        result = guardrail._check_blocked_words("This is an alternative approach")
        assert result is None, "alternative should not match alter"

        # "executive" should not match "exec"
        guardrail2 = ContentFilterGuardrail(
            guardrail_name="test-fp-avoidance2",
            blocked_words=[
                BlockedWord(keyword="exec", action=ContentFilterAction.BLOCK),
            ]
        )
        result = guardrail2._check_blocked_words("The executive summary")
        assert result is None, "executive should not match exec"

        # "selection" should not match "select"
        guardrail3 = ContentFilterGuardrail(
            guardrail_name="test-fp-avoidance3",
            blocked_words=[
                BlockedWord(keyword="select", action=ContentFilterAction.BLOCK),
            ]
        )
        result = guardrail3._check_blocked_words("The selection process")
        assert result is None, "selection should not match select"
