import pytest

litellm_core = pytest.importorskip(
    "litellm_core",
    reason="litellm_core Rust extension not built; run `maturin develop` in rust/",
)
build_pattern_prefilter = litellm_core.build_pattern_prefilter

SSN_PATTERN = r"\b\d{3}-\d{2}-\d{4}\b"
API_KEY_PATTERN = r"sk-[A-Za-z0-9]{20,}"
LOOKBEHIND_PATTERN = r"(?<!\d)foo"


class TestPatternPrefilter:
    def test_no_match_returns_false(self):
        prefilter, rejected = build_pattern_prefilter([SSN_PATTERN, API_KEY_PATTERN])
        assert rejected == []
        assert prefilter.any_match("hello world, nothing sensitive here") is False

    def test_match_returns_true(self):
        prefilter, rejected = build_pattern_prefilter([SSN_PATTERN, API_KEY_PATTERN])
        assert rejected == []
        assert prefilter.any_match("my ssn is 123-45-6789") is True

    def test_case_insensitive_match(self):
        prefilter, _ = build_pattern_prefilter([r"secret-token"])
        assert prefilter.any_match("here is a SECRET-TOKEN") is True

    def test_lookbehind_pattern_is_rejected_by_index(self):
        prefilter, rejected = build_pattern_prefilter(
            [SSN_PATTERN, LOOKBEHIND_PATTERN, API_KEY_PATTERN]
        )
        assert rejected == [1]

    def test_rejected_pattern_does_not_affect_others(self):
        prefilter, rejected = build_pattern_prefilter([LOOKBEHIND_PATTERN, SSN_PATTERN])
        assert rejected == [0]
        assert prefilter.any_match("hello world") is False
        assert prefilter.any_match("my ssn is 123-45-6789") is True

    def test_all_patterns_rejected_falls_back_to_true(self):
        prefilter, rejected = build_pattern_prefilter([LOOKBEHIND_PATTERN])
        assert rejected == [0]
        # No usable set was built; the safe default is to never claim "no match".
        assert prefilter.any_match("anything at all") is True

    def test_empty_pattern_list_falls_back_to_true(self):
        prefilter, rejected = build_pattern_prefilter([])
        assert rejected == []
        # No usable set was built; the safe default is to never claim "no match".
        assert prefilter.any_match("anything at all") is True
