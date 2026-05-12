"""Unit tests for the tag-authorization gate."""

from litellm.proxy.auth.tag_authorization import (
    caller_authorized_for_tag,
    filter_authorized_tags,
)


class TestCallerAuthorizedForTag:
    def test_no_metadata_sources(self):
        assert caller_authorized_for_tag("premium") is False

    def test_none_metadata_source_is_skipped(self):
        assert caller_authorized_for_tag("premium", None) is False

    def test_non_dict_metadata_source_is_skipped(self):
        assert caller_authorized_for_tag("premium", "not-a-dict") is False  # type: ignore[arg-type]

    def test_exact_match(self):
        assert caller_authorized_for_tag("premium", {"tags": ["premium"]}) is True

    def test_no_match(self):
        assert caller_authorized_for_tag("premium", {"tags": ["other"]}) is False

    def test_glob_wildcard_suffix(self):
        assert caller_authorized_for_tag("tenant:acme", {"tags": ["tenant:*"]}) is True

    def test_glob_does_not_match_across_separator_when_anchored(self):
        # `prefix-*` matches any suffix; the user picks the pattern.
        assert (
            caller_authorized_for_tag("tenant:acme:sub", {"tags": ["tenant:*"]}) is True
        )

    def test_universal_wildcard_matches_everything(self):
        assert caller_authorized_for_tag("anything", {"tags": ["*"]}) is True

    def test_matches_from_second_source(self):
        # key metadata empty, team metadata grants
        assert (
            caller_authorized_for_tag(
                "premium",
                {"tags": []},
                {"tags": ["premium"]},
            )
            is True
        )

    def test_pattern_must_be_string(self):
        assert (
            caller_authorized_for_tag(
                "premium", {"tags": ["premium", 123, None]}  # type: ignore[list-item]
            )
            is True
        )
        assert (
            caller_authorized_for_tag(
                "other", {"tags": [123, None]}  # type: ignore[list-item]
            )
            is False
        )

    def test_tags_field_not_a_list_is_skipped(self):
        assert caller_authorized_for_tag("premium", {"tags": "premium"}) is False


class TestFilterAuthorizedTags:
    def test_empty_input(self):
        assert filter_authorized_tags(None, frozenset(), {}) == []
        assert filter_authorized_tags([], frozenset(), {}) == []

    def test_non_privileged_tags_pass_through(self):
        # privileged set is empty → every caller tag passes
        result = filter_authorized_tags(["analytics", "session:42"], frozenset(), {})
        assert result == ["analytics", "session:42"]

    def test_privileged_tag_without_authorization_dropped(self):
        result = filter_authorized_tags(
            ["premium", "analytics"], frozenset({"premium"}), {"tags": []}
        )
        assert result == ["analytics"]

    def test_privileged_tag_with_authorization_kept(self):
        result = filter_authorized_tags(
            ["premium", "analytics"],
            frozenset({"premium"}),
            {"tags": ["premium"]},
        )
        assert result == ["premium", "analytics"]

    def test_glob_authorization(self):
        result = filter_authorized_tags(
            ["tenant:acme", "tenant:globex", "internal"],
            frozenset({"tenant:acme", "tenant:globex", "internal"}),
            {"tags": ["tenant:*"]},
        )
        assert result == ["tenant:acme", "tenant:globex"]

    def test_mixed_privileged_and_not(self):
        result = filter_authorized_tags(
            ["premium", "tenant:acme", "workflow:report-gen"],
            frozenset({"premium", "tenant:acme"}),
            {"tags": ["tenant:*"]},
        )
        # premium: privileged, not authorized → dropped
        # tenant:acme: privileged, glob-authorized → kept
        # workflow:report-gen: not privileged → kept
        assert result == ["tenant:acme", "workflow:report-gen"]

    def test_non_string_input_filtered(self):
        result = filter_authorized_tags(
            ["a", 1, None, "b"], frozenset(), {}  # type: ignore[list-item]
        )
        assert result == ["a", "b"]
