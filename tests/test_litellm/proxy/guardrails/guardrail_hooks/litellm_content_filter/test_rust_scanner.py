import json
import re

import pytest

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.rust_scanner import (
    ContentFilterScanner,
)

try:
    import litellm_python_bridge  # noqa: F401

    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False


class _FakeScanner:
    """Records the config it was built from and returns canned matches."""

    def __init__(self, config_json, matches, compile_errors):
        self.config = json.loads(config_json)
        self.compile_errors = compile_errors
        self._matches = matches

    def scan(self, text):
        return self._matches


def _factory(matches=(), compile_errors="[]"):
    holder = {}

    def make(config_json):
        scanner = _FakeScanner(config_json, list(matches), compile_errors)
        holder["scanner"] = scanner
        return scanner

    make.holder = holder
    return make


def _identity_keyword_to_regex(keyword: str) -> str:
    return re.escape(keyword).replace(r"\*", ".?")


def _pattern_entry(pattern: str):
    return {
        "regex": re.compile(pattern),
        "pattern_name": "p",
        "keyword_regex": None,
        "allow_word_numbers": False,
    }


def test_terms_are_classified_into_literals_and_regexes():
    factory = _factory()
    ContentFilterScanner.build(
        category_keywords={
            "weapon": ("c", "high", "BLOCK"),
            "kill all": ("c", "high", "BLOCK"),
        },
        always_block_keywords={},
        blocked_words={"fu*k": ("MASK", None)},
        compiled_patterns=[_pattern_entry(r"\d{3}")],
        keyword_to_regex=_identity_keyword_to_regex,
        scanner_factory=factory,
    )
    config = factory.holder["scanner"].config
    literal_texts = {item["text"]: item["word_boundary"] for item in config["literals"]}

    # single word -> word boundary, multi word -> substring
    assert literal_texts["weapon"] is True
    assert literal_texts["kill all"] is False
    # wildcard keyword is a regex, not a literal
    assert "fu*k" not in literal_texts
    regex_patterns = [r["pattern"] for r in config["regexes"]]
    assert r"\bfu.?k\b" in regex_patterns
    assert r"\d{3}" in regex_patterns


def test_scan_maps_match_ids_back_to_their_tables():
    # ids are assigned in order: category(0), always_block, blocked, pattern.
    factory = _factory(matches=[(0, 0, 6), (1, 7, 13), (2, 20, 23)])
    scanner = ContentFilterScanner.build(
        category_keywords={"weapon": ("c", "high", "BLOCK")},  # id 0
        always_block_keywords={},
        blocked_words={"secret": ("MASK", None)},  # id 1
        compiled_patterns=[_pattern_entry(r"\d{3}")],  # id 2
        keyword_to_regex=_identity_keyword_to_regex,
        scanner_factory=factory,
    )
    result = scanner.scan("ignored")

    assert result.category_keywords_present == {"weapon"}
    assert result.blocked_words_present == {"secret"}
    assert result.pattern_spans == {0: [(20, 23)]}


def test_uncompilable_pattern_falls_back_to_python():
    # pattern is id 0 (no keywords); report it as a compile error.
    factory = _factory(compile_errors=json.dumps([{"id": 0, "message": "lookaround"}]))
    scanner = ContentFilterScanner.build(
        category_keywords={},
        always_block_keywords={},
        blocked_words={},
        compiled_patterns=[_pattern_entry(r"(?<=x)y")],
        keyword_to_regex=_identity_keyword_to_regex,
        scanner_factory=factory,
    )
    assert scanner.fallback_pattern_indexes == {0}


@pytest.mark.skipif(
    not _BRIDGE_AVAILABLE, reason="litellm_python_bridge extension not built"
)
def test_end_to_end_against_real_rust_scanner():
    scanner = ContentFilterScanner.build(
        category_keywords={"bomb": ("violence", "high", "BLOCK")},
        always_block_keywords={},
        blocked_words={},
        compiled_patterns=[_pattern_entry(r"\d{3}-\d{2}-\d{4}")],
        keyword_to_regex=_identity_keyword_to_regex,
    )
    result = scanner.scan("a bomb and ssn 123-45-6789")
    assert result.category_keywords_present == {"bomb"}
    assert result.pattern_spans  # the SSN pattern matched
