"""
Unit tests for AttachmentRegistry - tests policy attachment matching.

Tests the main entry point: get_attached_policies()
"""

import pytest

from litellm.proxy.policy_engine.attachment_registry import (
    AttachmentRegistry,
    get_attachment_registry,
)
from litellm.types.proxy.policy_engine import PolicyMatchContext


class TestGetAttachedPolicies:
    """Test get_attached_policies - the main entry point."""

    def test_global_scope_matches_all_requests(self):
        """Test global scope (*) matches any request context."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "global-baseline", "scope": "*"},
        ])

        # Should match any context
        context = PolicyMatchContext(
            team_alias="any-team", key_alias="any-key", model="any-model"
        )
        attached = registry.get_attached_policies(context)
        assert "global-baseline" in attached

    def test_team_specific_attachment(self):
        """Test team-specific attachment matches only that team."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
        ])

        # Match
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-4"
        )
        assert "healthcare-policy" in registry.get_attached_policies(context)

        # No match - different team
        context_other = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        assert "healthcare-policy" not in registry.get_attached_policies(context_other)

    def test_key_wildcard_pattern_attachment(self):
        """Test key pattern attachment with wildcard."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "dev-policy", "keys": ["dev-key-*"]},
        ])

        # Match - key starts with dev-key-
        context = PolicyMatchContext(
            team_alias="team", key_alias="dev-key-123", model="gpt-4"
        )
        assert "dev-policy" in registry.get_attached_policies(context)

        # No match - different prefix
        context_prod = PolicyMatchContext(
            team_alias="team", key_alias="prod-key-123", model="gpt-4"
        )
        assert "dev-policy" not in registry.get_attached_policies(context_prod)

    def test_model_specific_attachment(self):
        """Test model-specific attachment."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "gpt4-policy", "models": ["gpt-4", "gpt-4-turbo"]},
        ])

        # Match
        context = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4"
        )
        assert "gpt4-policy" in registry.get_attached_policies(context)

        # No match
        context_other = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-3.5"
        )
        assert "gpt4-policy" not in registry.get_attached_policies(context_other)

    def test_model_wildcard_pattern(self):
        """Test model wildcard pattern like bedrock/*."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "bedrock-policy", "models": ["bedrock/*"]},
        ])

        # Match
        context = PolicyMatchContext(
            team_alias="team", key_alias="key", model="bedrock/claude-3"
        )
        assert "bedrock-policy" in registry.get_attached_policies(context)

        # No match
        context_other = PolicyMatchContext(
            team_alias="team", key_alias="key", model="openai/gpt-4"
        )
        assert "bedrock-policy" not in registry.get_attached_policies(context_other)

    def test_multiple_attachments_match_same_context(self):
        """Test multiple attachments can match the same context."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "global-baseline", "scope": "*"},
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
            {"policy": "gpt4-policy", "models": ["gpt-4"]},
        ])

        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)

        # All three should match
        assert "global-baseline" in attached
        assert "healthcare-policy" in attached
        assert "gpt4-policy" in attached
        assert len(attached) == 3

    def test_same_policy_multiple_attachments_no_duplicates(self):
        """Test same policy attached multiple ways doesn't duplicate."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "multi-policy", "scope": "*"},
            {"policy": "multi-policy", "teams": ["healthcare-team"]},
        ])

        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)

        # Should only appear once
        assert attached.count("multi-policy") == 1

    def test_no_attachments_returns_empty(self):
        """Test empty attachments returns empty list."""
        registry = AttachmentRegistry()
        registry.load_attachments([])

        context = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)
        assert attached == []

    def test_no_matching_attachments_returns_empty(self):
        """Test no matching attachments returns empty list."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
        ])

        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)
        assert attached == []

    def test_combined_team_and_model_attachment(self):
        """Test attachment with both team and model constraints."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "strict-policy", "teams": ["healthcare-team"], "models": ["gpt-4"]},
        ])

        # Match - both team and model match
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-4"
        )
        assert "strict-policy" in registry.get_attached_policies(context)

        # No match - team matches but model doesn't
        context_wrong_model = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-3.5"
        )
        assert "strict-policy" not in registry.get_attached_policies(context_wrong_model)

        # No match - model matches but team doesn't
        context_wrong_team = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        assert "strict-policy" not in registry.get_attached_policies(context_wrong_team)


class TestAttachmentRegistrySingleton:
    """Test global singleton behavior."""

    def test_get_attachment_registry_returns_same_instance(self):
        """Test get_attachment_registry returns same instance."""
        registry1 = get_attachment_registry()
        registry2 = get_attachment_registry()
        assert registry1 is registry2
