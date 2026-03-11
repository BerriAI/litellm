"""
Unit tests for PolicyMatcher - tests wildcard pattern matching via attachments.

Tests:
- Wildcard matching (*, prefix-*)
- Scope matching via attachments (teams, keys, models)
"""

import pytest

from litellm.proxy.policy_engine.attachment_registry import AttachmentRegistry
from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.types.proxy.policy_engine import (
    PolicyMatchContext,
    PolicyScope,
)


class TestPolicyMatcherPatternMatching:
    """Test pattern matching utilities."""

    def test_matches_pattern_exact(self):
        """Test exact pattern matching."""
        assert PolicyMatcher.matches_pattern("healthcare-team", ["healthcare-team"]) is True
        assert PolicyMatcher.matches_pattern("finance-team", ["healthcare-team"]) is False

    def test_matches_pattern_wildcard(self):
        """Test wildcard pattern matching."""
        assert PolicyMatcher.matches_pattern("any-team", ["*"]) is True
        assert PolicyMatcher.matches_pattern("dev-key-123", ["dev-key-*"]) is True
        assert PolicyMatcher.matches_pattern("prod-key-123", ["dev-key-*"]) is False

    def test_matches_pattern_none_value(self):
        """Test None value only matches '*'."""
        assert PolicyMatcher.matches_pattern(None, ["*"]) is True
        assert PolicyMatcher.matches_pattern(None, ["specific"]) is False


class TestPolicyMatcherScopeMatching:
    """Test scope matching against context."""

    def test_scope_matches_all_fields(self):
        """Test scope matches when all fields match."""
        scope = PolicyScope(teams=["healthcare-team"], keys=["*"], models=["gpt-4"])
        context = PolicyMatchContext(team_alias="healthcare-team", key_alias="any-key", model="gpt-4")
        assert PolicyMatcher.scope_matches(scope, context) is True

    def test_scope_does_not_match_team(self):
        """Test scope doesn't match when team doesn't match."""
        scope = PolicyScope(teams=["healthcare-team"], keys=["*"], models=["*"])
        context = PolicyMatchContext(team_alias="finance-team", key_alias="any-key", model="gpt-4")
        assert PolicyMatcher.scope_matches(scope, context) is False

    def test_scope_matches_with_wildcard_patterns(self):
        """Test scope matches with wildcard patterns."""
        scope = PolicyScope(teams=["*"], keys=["dev-key-*"], models=["bedrock/*"])
        context = PolicyMatchContext(team_alias="any-team", key_alias="dev-key-123", model="bedrock/claude-3")
        assert PolicyMatcher.scope_matches(scope, context) is True

    def test_scope_global_wildcard(self):
        """Test global scope with all wildcards."""
        scope = PolicyScope(teams=["*"], keys=["*"], models=["*"])
        context = PolicyMatchContext(team_alias="any-team", key_alias="any-key", model="any-model")
        assert PolicyMatcher.scope_matches(scope, context) is True


class TestPolicyMatcherScopeMatchingWithTags:
    """Test scope matching with tag patterns."""

    def test_scope_tag_matching(self):
        """Test scope tag matching: exact, wildcard, no-match, and empty context tags."""
        # Exact match
        scope = PolicyScope(teams=["*"], keys=["*"], models=["*"], tags=["healthcare"])
        context = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4",
            tags=["healthcare", "internal"],
        )
        assert PolicyMatcher.scope_matches(scope, context) is True

        # Wildcard match
        scope_wc = PolicyScope(teams=["*"], keys=["*"], models=["*"], tags=["health-*"])
        context_wc = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4",
            tags=["health-prod"],
        )
        assert PolicyMatcher.scope_matches(scope_wc, context_wc) is True

        # No match — wrong tag
        context_wrong = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4",
            tags=["finance"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong) is False

        # No match — context has no tags
        context_none = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4", tags=None,
        )
        assert PolicyMatcher.scope_matches(scope, context_none) is False

        # Scope without tags matches any context (opt-in semantics)
        scope_no_tags = PolicyScope(teams=["*"], keys=["*"], models=["*"])
        assert PolicyMatcher.scope_matches(scope_no_tags, context) is True

    def test_scope_tags_and_team_combined(self):
        """Test scope with both tags and team — both must match (AND logic)."""
        scope = PolicyScope(teams=["team-a"], keys=["*"], models=["*"], tags=["healthcare"])

        # Both match
        context_both = PolicyMatchContext(
            team_alias="team-a", key_alias="key", model="gpt-4",
            tags=["healthcare"],
        )
        assert PolicyMatcher.scope_matches(scope, context_both) is True

        # Tag matches, team doesn't
        context_wrong_team = PolicyMatchContext(
            team_alias="team-b", key_alias="key", model="gpt-4",
            tags=["healthcare"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong_team) is False

        # Team matches, tag doesn't
        context_wrong_tag = PolicyMatchContext(
            team_alias="team-a", key_alias="key", model="gpt-4",
            tags=["finance"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong_tag) is False


class TestPolicyMatcherWithAttachments:
    """Test getting matching policies via attachments."""

    def test_get_matching_policies_via_attachments(self):
        """Test matching policies through attachment registry."""
        # Create and configure attachment registry
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
            {"policy": "global-policy", "scope": "*"},
        ])

        # Test matching via the registry directly
        context = PolicyMatchContext(team_alias="healthcare-team", key_alias="k", model="gpt-4")
        attached = registry.get_attached_policies(context)

        assert "healthcare-policy" in attached
        assert "global-policy" in attached

    def test_get_matching_policies_no_match(self):
        """Test no policies match when attachments don't match context."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
        ])

        context = PolicyMatchContext(team_alias="finance-team", key_alias="k", model="gpt-4")
        attached = registry.get_attached_policies(context)

        assert "healthcare-policy" not in attached
