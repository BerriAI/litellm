"""
Tests for content filter pattern loading from JSON
"""

import json
import os
import time
from typing import Final

import pytest


from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    PATTERN_CATEGORIES,
    PATTERN_DESCRIPTIONS,
    PATTERN_DISPLAY_NAMES,
    PREBUILT_PATTERNS,
    get_all_pattern_names,
    get_compiled_pattern,
    get_pattern_metadata,
)


def test_patterns_loaded_from_json():
    """
    Test that patterns are successfully loaded from JSON file
    """
    assert len(PREBUILT_PATTERNS) > 0
    assert len(PATTERN_CATEGORIES) > 0
    assert len(PATTERN_DISPLAY_NAMES) > 0
    assert len(PATTERN_DESCRIPTIONS) > 0


def test_pattern_metadata_structure():
    """
    Test that get_pattern_metadata returns correctly structured data
    """
    patterns = get_pattern_metadata()

    assert len(patterns) > 0

    for pattern in patterns:
        assert "name" in pattern
        assert "display_name" in pattern
        assert "category" in pattern
        assert "description" in pattern

        assert isinstance(pattern["name"], str)
        assert isinstance(pattern["display_name"], str)
        assert isinstance(pattern["category"], str)
        assert isinstance(pattern["description"], str)


def test_display_names_user_friendly():
    """
    Test that display names are user-friendly and different from internal names
    """
    patterns = get_pattern_metadata()

    ssn_pattern = next((p for p in patterns if p["name"] == "us_ssn"), None)
    assert ssn_pattern is not None
    assert ssn_pattern["display_name"] == "SSN (Social Security Number)"

    email_pattern = next((p for p in patterns if p["name"] == "email"), None)
    assert email_pattern is not None
    assert email_pattern["display_name"] == "Email Address"


def test_pattern_compilation():
    """
    Test that patterns can be compiled into regex objects
    """
    pattern_names = get_all_pattern_names()

    assert len(pattern_names) > 0

    for pattern_name in pattern_names:
        compiled_pattern = get_compiled_pattern(pattern_name)
        assert compiled_pattern is not None
        assert hasattr(compiled_pattern, "match")


def test_pattern_compilation_error():
    """
    Test that invalid pattern names raise ValueError
    """
    with pytest.raises(ValueError, match="Unknown pattern name"):
        get_compiled_pattern("invalid_pattern_name")


def test_categories_contain_patterns():
    """
    Test that each category contains valid pattern names
    """
    all_pattern_names = set(PREBUILT_PATTERNS.keys())

    for category, patterns in PATTERN_CATEGORIES.items():
        assert len(patterns) > 0
        for pattern_name in patterns:
            assert pattern_name in all_pattern_names


def test_json_file_exists():
    """
    Test that patterns.json file exists and is valid JSON
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json",
    )

    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)
        assert "patterns" in data
        assert len(data["patterns"]) > 0


def test_json_pattern_structure():
    """
    Test that each pattern in JSON has required fields
    """
    json_path = os.path.join(
        os.path.dirname(__file__),
        "../../../../../../litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/patterns.json",
    )

    with open(json_path, "r") as f:
        data = json.load(f)

        for pattern in data["patterns"]:
            assert "name" in pattern
            assert "display_name" in pattern
            assert "pattern" in pattern
            assert "category" in pattern
            assert "description" in pattern

            assert isinstance(pattern["name"], str)
            assert isinstance(pattern["display_name"], str)
            assert isinstance(pattern["pattern"], str)
            assert isinstance(pattern["category"], str)
            assert isinstance(pattern["description"], str)


def test_all_dictionaries_consistent():
    """
    Test that all pattern dictionaries have consistent keys
    """
    pattern_names_from_patterns = set(PREBUILT_PATTERNS.keys())
    pattern_names_from_display = set(PATTERN_DISPLAY_NAMES.keys())
    pattern_names_from_descriptions = set(PATTERN_DESCRIPTIONS.keys())

    assert pattern_names_from_patterns == pattern_names_from_display
    assert pattern_names_from_patterns == pattern_names_from_descriptions


def test_eu_patterns_loaded():
    """Verify all EU PII patterns are loaded"""
    required_patterns = [
        "fr_nir",
        "eu_iban_enhanced",
        "fr_phone",
        "eu_vat",
        "eu_passport_generic",
        "fr_postal_code",
    ]
    for pattern_name in required_patterns:
        assert pattern_name in PREBUILT_PATTERNS, f"Pattern {pattern_name} not found"


def test_eu_patterns_have_category():
    """Verify EU patterns are in correct category"""
    eu_patterns = [
        "fr_nir",
        "eu_iban_enhanced",
        "fr_phone",
        "eu_vat",
        "eu_passport_generic",
        "fr_postal_code",
    ]
    eu_category_patterns = PATTERN_CATEGORIES.get("EU PII Patterns", [])

    for pattern_name in eu_patterns:
        assert (
            pattern_name in eu_category_patterns
        ), f"Pattern {pattern_name} not in EU PII Patterns category"


def test_provider_key_patterns_are_offered_in_the_ui_as_credential_patterns() -> None:
    ui_categories: Final = {pattern["name"]: pattern["category"] for pattern in get_pattern_metadata()}
    assert [ui_categories.get(name) for name in ("openai_api_key", "anthropic_api_key", "gemini_api_key")] == [
        "Credential Patterns"
    ] * 3


@pytest.mark.parametrize(
    ("pattern_name", "key"),
    [
        pytest.param("openai_api_key", "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20, id="openai-legacy"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 74 + "T3BlbkFJ" + "b" * 74, id="openai-project"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 20 + "T3BlbkFJ" + "b" * 20, id="openai-project-shortest"),
        pytest.param("openai_api_key", "sk-svcacct-" + "a_b-" * 15 + "T3BlbkFJ" + "c-d_" * 15, id="openai-service-account"),
        pytest.param("openai_api_key", "sk-admin-" + "a" * 58 + "T3BlbkFJ" + "b" * 58, id="openai-admin"),
        pytest.param("openai_api_key", "sk-None-" + "a" * 20 + "T3BlbkFJ" + "b" * 20, id="openai-user-key"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 20 + "T3BlbkFJ" + "b" * 19 + "-", id="openai-trailing-hyphen"),
        pytest.param(
            "anthropic_api_key", "sk-ant-api03-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP", id="anthropic-versioned"
        ),
        pytest.param(
            "anthropic_api_key", "sk-ant-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU", id="anthropic-unversioned"
        ),
        pytest.param("anthropic_api_key", "sk-ant-" + "a" * 20, id="anthropic-shortest"),
        pytest.param("anthropic_api_key", "sk-ant-api03-" + "a_b-" * 23 + "AA", id="anthropic-urlsafe"),
        pytest.param(
            "anthropic_api_key", "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789-", id="anthropic-trailing-hyphen"
        ),
        pytest.param("gemini_api_key", "AIzaSyD-9tSrke72I6_AQp2h8HI4-uE6QNKQV0", id="gemini-legacy"),
        pytest.param("gemini_api_key", "AIza" + "a" * 33, id="gemini-legacy-shortest"),
        pytest.param("gemini_api_key", "AIza" + "a" * 36, id="gemini-legacy-longest"),
        pytest.param("gemini_api_key", "AIza" + "a" * 34 + "-", id="gemini-legacy-trailing-hyphen"),
        pytest.param("gemini_api_key", "AQ.Ab8RN6I-7xK9pL2mQ0vW5yZ1uX3tS4rE8qD7cB6aN5mP4oI3uY2", id="gemini-auth-key"),
        pytest.param("gemini_api_key", "AQ." + "a" * 45, id="gemini-auth-key-shortest"),
        pytest.param("gemini_api_key", "AQ." + "a" * 55, id="gemini-auth-key-longest"),
        pytest.param("gemini_api_key", "AQ." + "a" * 49 + "-", id="gemini-auth-key-trailing-hyphen"),
    ],
)
def test_credential_pattern_matches_whole_key(pattern_name: str, key: str) -> None:
    match: Final = get_compiled_pattern(pattern_name).search(f"I keep getting 401 with {key}, is it expired?")
    assert match is not None
    assert match.group(0) == key


@pytest.mark.parametrize("separator", ["\\n", "\\t", "\\u003c", "Bearer%20", "?key%3D", "?key%253D", "GOOGLE_KEY_"])
@pytest.mark.parametrize(
    ("pattern_name", "key"),
    [
        pytest.param("openai_api_key", "sk-proj-" + "a" * 20 + "T3BlbkFJ" + "b" * 20, id="openai"),
        pytest.param("anthropic_api_key", "sk-ant-api03-" + "a" * 20, id="anthropic"),
        pytest.param("gemini_api_key", "AIza" + "a" * 35, id="gemini-legacy"),
        pytest.param("gemini_api_key", "AQ." + "a" * 50, id="gemini-auth-key"),
    ],
)
def test_credential_pattern_matches_key_after_escaped_separator(pattern_name: str, key: str, separator: str) -> None:
    match: Final = get_compiled_pattern(pattern_name).search(f'{{"log": "retrying with{separator}{key} failed"}}')
    assert match is not None
    assert match.group(0) == key


@pytest.mark.parametrize(
    ("pattern_name", "text"),
    [
        pytest.param("openai_api_key", "sk-short", id="openai-too-short"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 19, id="openai-project-too-short"),
        pytest.param("openai_api_key", "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 21, id="openai-legacy-too-long"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 48, id="openai-project-without-marker"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 19 + "T3BlbkFJ" + "b" * 20, id="openai-project-marker-too-early"),
        pytest.param("openai_api_key", "sk-proj-" + "a" * 20 + "T3BlbkFJ" + "b" * 19, id="openai-project-marker-too-late"),
        pytest.param("openai_api_key", "SK-ADMIN-dashboard-redesign-phase-two", id="openai-prefixed-slug"),
        pytest.param("openai_api_key", "desk-proj-management-dashboard-v2", id="openai-inside-longer-word"),
        pytest.param(
            "openai_api_key", "sk-abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST", id="openai-legacy-without-marker"
        ),
        pytest.param("openai_api_key", "sk-0123456789abcdef0123456789abcdef", id="openai-other-vendor-sk-key"),
        pytest.param("openai_api_key", "sk-Xq3vR9tLm2Pa8KdW5yZgQw", id="openai-litellm-virtual-key"),
        pytest.param(
            "openai_api_key",
            "sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCd-_x",
            id="openai-litellm-virtual-key-raised-length",
        ),
        pytest.param("openai_api_key", "not-a-key", id="openai-no-prefix"),
        pytest.param(
            "openai_api_key", "sk-ant-api03-thisisanthropicnotopenaiabc123", id="openai-ignores-anthropic-key"
        ),
        pytest.param("openai_api_key", "ordinary sentence without any api key in it", id="openai-plain-text"),
        pytest.param("anthropic_api_key", "sk-ant-" + "a" * 19, id="anthropic-too-short"),
        pytest.param("anthropic_api_key", "flask-ant-design-system-components", id="anthropic-inside-longer-word"),
        pytest.param(
            "anthropic_api_key",
            "sk-proj-abcdefghijklmnopqrstuvwxyzABCDEF1234567890",
            id="anthropic-ignores-openai-project-key",
        ),
        pytest.param(
            "anthropic_api_key",
            "sk-admin-abcdefghijklmnopqrstuvwxyzABCDEF123456",
            id="anthropic-ignores-openai-admin-key",
        ),
        pytest.param("anthropic_api_key", "sk-ant", id="anthropic-prefix-only"),
        pytest.param("anthropic_api_key", "plain text with nothing sensitive", id="anthropic-plain-text"),
        pytest.param("gemini_api_key", "AIza", id="gemini-prefix-only"),
        pytest.param("gemini_api_key", "AIza" + "a" * 32, id="gemini-legacy-too-short"),
        pytest.param("gemini_api_key", "AIza" + "a" * 37, id="gemini-legacy-too-long"),
        pytest.param("gemini_api_key", "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", id="gemini-wrong-prefix"),
        pytest.param("gemini_api_key", "AQ.", id="gemini-auth-prefix-only"),
        pytest.param("gemini_api_key", "AQ." + "a" * 44, id="gemini-auth-key-too-short"),
        pytest.param("gemini_api_key", "AQ." + "a" * 56, id="gemini-auth-key-too-long"),
        pytest.param("gemini_api_key", "AQ-" + "a" * 50, id="gemini-auth-key-wrong-separator"),
        pytest.param("gemini_api_key", "x" + "AIza" + "a" * 35, id="gemini-inside-longer-token"),
        pytest.param("gemini_api_key", "The AQ. was reported at 42 ppm today.", id="gemini-aq-acronym"),
        pytest.param("gemini_api_key", "Is the question AQ. Maybe it is BQ.", id="gemini-aq-sentence-end"),
        pytest.param("gemini_api_key", "plain text with no google key", id="gemini-plain-text"),
        pytest.param(
            "gemini_api_key",
            "https://example.com/blog/aizawa-takeshi-interview-2024-tokyo-01",
            id="gemini-lowercase-legacy-prefix",
        ),
        pytest.param(
            "gemini_api_key",
            "ssh aq.internal-observability-metrics-collector-eu-west-01",
            id="gemini-lowercase-auth-prefix",
        ),
    ],
)
def test_credential_pattern_ignores_non_keys(pattern_name: str, text: str) -> None:
    assert get_compiled_pattern(pattern_name).search(text) is None


@pytest.mark.parametrize("suffix", ["请检查", "é", "ü der Schlüssel"])
@pytest.mark.parametrize(
    ("pattern_name", "key"),
    [
        pytest.param("openai_api_key", "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20, id="openai-legacy"),
        pytest.param("gemini_api_key", "AIza" + "a" * 35, id="gemini-legacy"),
        pytest.param("gemini_api_key", "AQ." + "a" * 50, id="gemini-auth-key"),
    ],
)
def test_credential_pattern_matches_key_followed_by_non_ascii_letters(pattern_name: str, key: str, suffix: str) -> None:
    match: Final = get_compiled_pattern(pattern_name).search(f"我的密钥是{key}{suffix}")
    assert match is not None
    assert match.group(0) == key


@pytest.mark.parametrize(
    "leading",
    [
        pytest.param("sk-proj-" + "a" * 74 + "T3BlbkFJ" + "b" * 74, id="another-key"),
        pytest.param("sk-proj-" + "x" * 20 + "T3BlbkFJ" + "y" * 192, id="key-shaped-filler"),
        pytest.param("sk-proj-" + "x" * 100 + "T3BlbkFJ" + "y" * 100, id="long-key-shaped-filler"),
    ],
)
def test_openai_key_pattern_masks_a_key_glued_to_preceding_key_shaped_text(leading: str) -> None:
    key: Final = "sk-proj-" + "c" * 74 + "T3BlbkFJ" + "d" * 74
    assert get_compiled_pattern("openai_api_key").sub("[KEY]", leading + key) == "[KEY][KEY]"


@pytest.mark.parametrize("prefix", ["sk-proj-", "sk-svcacct-", "sk-admin-", "sk-None-"])
def test_openai_key_pattern_scans_repeated_prefixes_without_stalling(prefix: str) -> None:
    prompt: Final = prefix * (128_000 // len(prefix))
    started: Final = time.perf_counter()
    match: Final = get_compiled_pattern("openai_api_key").search(prompt)
    elapsed: Final = time.perf_counter() - started
    assert match is None
    assert elapsed < 1.0
