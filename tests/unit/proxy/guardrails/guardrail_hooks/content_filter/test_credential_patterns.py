"""
Regression tests for credential guardrail patterns:
  jwt_token, openai_api_key, anthropic_api_key, gemini_api_key

Each test verifies that the pattern detects real token shapes and
does NOT false-positive on innocent strings of similar length.
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)


# ---------------------------------------------------------------------------
# jwt_token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Standard HS256 JWT
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        # RS256 JWT (longer header)
        (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        ),
        # JWT pasted mid-sentence (the common accident)
        (
            "Here is my bearer token: eyJhbGciOiJIUzI1NiJ9."
            "eyJ1c2VyIjoiYWRtaW4ifQ.abc123def456ghi789jkl012mno345pqr678 can you help debug?"
        ),
    ],
)
def test_jwt_token_detected(text: str) -> None:
    pat = get_compiled_pattern("jwt_token")
    assert pat.search(text) is not None, f"Expected JWT match in: {text[:60]}..."


@pytest.mark.parametrize(
    "text",
    [
        "eyJhbG",  # too short to be a real JWT
        "normal sentence with no token at all",
        "eyJ alone does not make a JWT without three parts",
        "some.dotted.text that is not base64url encoded properly",
    ],
)
def test_jwt_token_not_false_positive(text: str) -> None:
    pat = get_compiled_pattern("jwt_token")
    assert pat.search(text) is None, f"Unexpected JWT match in: {text}"


# ---------------------------------------------------------------------------
# openai_api_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Classic sk- format (legacy user keys)
        "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST",
        # Project-scoped key (current default)
        "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEF1234567890abcdef",
        # Service-account key
        "sk-svcacct-abcdefghijklmnopqrstuvwxyzABCDEF123456",
        # Admin key
        "sk-admin-abcdefghijklmnopqrstuvwxyzABCDEF123456",
        # Key pasted raw in a debug message
        "I keep getting 401. My key is sk-proj-TestKeyABCDEFGHIJKLMNOPQRSTUVWXYZ00 is it expired?",
    ],
)
def test_openai_api_key_detected(text: str) -> None:
    pat = get_compiled_pattern("openai_api_key")
    assert pat.search(text) is not None, f"Expected OpenAI key match in: {text[:60]}..."


@pytest.mark.parametrize(
    "text",
    [
        "sk-short",  # too short
        "not-a-key",
        "sk-ant-api03-thisisanthropicnotopenaiabc123",  # Anthropic key
        "ordinary sentence without any api key in it",
    ],
)
def test_openai_api_key_not_false_positive(text: str) -> None:
    pat = get_compiled_pattern("openai_api_key")
    assert pat.search(text) is None, f"Unexpected OpenAI key match in: {text}"


# ---------------------------------------------------------------------------
# anthropic_api_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Versioned format
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP",
        # Short form (older keys)
        "sk-ant-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU",
        # Pasted mid-sentence
        "Authorization failed with sk-ant-api03-TestKeyABCDEFGHIJKLMNOPQRSTUV what is wrong?",
    ],
)
def test_anthropic_api_key_detected(text: str) -> None:
    pat = get_compiled_pattern("anthropic_api_key")
    assert pat.search(text) is not None, f"Expected Anthropic key match in: {text[:60]}..."


@pytest.mark.parametrize(
    "text",
    [
        "sk-ant-short",  # too short
        "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEF1234567890",  # OpenAI key
        "sk-admin-abcdefghijklmnopqrstuvwxyzABCDEF123456",  # OpenAI admin key
        "sk-ant",  # just the prefix
        "plain text with nothing sensitive",
    ],
)
def test_anthropic_api_key_not_false_positive(text: str) -> None:
    pat = get_compiled_pattern("anthropic_api_key")
    assert pat.search(text) is None, f"Unexpected Anthropic key match in: {text}"


# ---------------------------------------------------------------------------
# gemini_api_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # Legacy AIza key shape (38 chars: AIza + 34)
        "AIzaSyD-9tSrke72I6_AQp2h8HI4-uE6QNKQV0",
        # Legacy variant
        "AIzaSyD-abcdefghijklmnopqrstuvwxyz123",
        # New Google AI Studio authorization key (starts with AQ. ~53 chars)
        "AQ.Ab8RN6I-7xK9pL2mQ0vW5yZ1uX3tS4rE8qD7cB6aN5mP4oI3uY2",
        # Pasted mid-sentence
        "My Gemini key AIzaSyD-9tSrke72I6_AQp2h8HI4-uE6QNKQV0 is not working",
        "Error using key AQ.Ab8RN6I-7xK9pL2mQ0vW5yZ1uX3tS4rE8qD7cB6aN5mP4oI3uY2 in prompt",
    ],
)
def test_gemini_api_key_detected(text: str) -> None:
    pat = get_compiled_pattern("gemini_api_key")
    assert pat.search(text) is not None, f"Expected Gemini key match in: {text[:60]}..."


@pytest.mark.parametrize(
    "text",
    [
        "AIza",  # just prefix
        "AIzaShort",  # too short
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",  # wrong prefix, right length
        "AQ.",  # just AQ.
        "The AQ. was reported at 42 ppm today.",  # AQ acronym ending with period + space
        "Is the question AQ. Maybe it is BQ.",  # sentence end
        "AQ.12345",  # short AQ string
        "plain text with no google key",
    ],
)
def test_gemini_api_key_not_false_positive(text: str) -> None:
    pat = get_compiled_pattern("gemini_api_key")
    assert pat.search(text) is None, f"Unexpected Gemini key match in: {text}"


# ---------------------------------------------------------------------------
# Integration: all four patterns are registered and compile cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pattern_name",
    ["jwt_token", "openai_api_key", "anthropic_api_key", "gemini_api_key"],
)
def test_new_patterns_are_registered(pattern_name: str) -> None:
    pat = get_compiled_pattern(pattern_name)
    assert pat is not None
    assert hasattr(pat, "search")
