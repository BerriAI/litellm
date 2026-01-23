"""
Unit tests for AttachmentRegistry - tests policy attachment management.

Tests:
- Loading attachments from config
- Getting attached policies for a context
- Global scope attachments
- Team/key/model specific attachments
"""

import pytest

from litellm.proxy.policy_engine.attachment_registry import (
    AttachmentRegistry,
    get_attachment_registry,
)
from litellm.types.proxy.policy_engine import (
    PolicyAttachment,
    PolicyMatchContext,
)


class TestAttachmentRegistryLoading:
    """Test loading attachments from configuration."""

    def test_load_attachments_simple(self):
        """Test loading simple attachments."""
        registry = AttachmentRegistry()
        config = [
            {"policy": "global-baseline", "scope": "*"},
            {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
        ]
        registry.load_attachments(config)

        assert registry.is_initialized()
        assert len(registry.get_all_attachments()) == 2

    def test_load_attachments_with_multiple_scopes(self):
        """Test loading attachments with multiple scope types."""
        registry = AttachmentRegistry()
        config = [
            {"policy": "global-baseline", "scope": "*"},
            {"policy": "team-policy", "teams": ["team-a", "team-b"]},
            {"policy": "key-policy", "keys": ["dev-key-*"]},
            {"policy": "model-policy", "models": ["gpt-4", "gpt-4-turbo"]},
        ]
        registry.load_attachments(config)

        assert len(registry.get_all_attachments()) == 4

    def test_load_attachments_empty_list(self):
        """Test loading empty attachments list."""
        registry = AttachmentRegistry()
        registry.load_attachments([])

        assert registry.is_initialized()
        assert len(registry.get_all_attachments()) == 0

    def test_clear_attachments(self):
        """Test clearing attachments."""
        registry = AttachmentRegistry()
        registry.load_attachments([{"policy": "test", "scope": "*"}])
        assert registry.is_initialized()

        registry.clear()
        assert not registry.is_initialized()
        assert len(registry.get_all_attachments()) == 0


class TestGetAttachedPolicies:
    """Test getting attached policies for a context."""

    def test_global_scope_matches_all(self):
        """Test global scope (*) matches all contexts."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "global-baseline", "scope": "*"},
        ])

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
        attached = registry.get_attached_policies(context)
        assert "healthcare-policy" in attached

        # No match
        context_other = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        attached_other = registry.get_attached_policies(context_other)
        assert "healthcare-policy" not in attached_other

    def test_key_pattern_attachment(self):
        """Test key pattern attachment matches wildcard."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "dev-policy", "keys": ["dev-key-*"]},
        ])

        # Match
        context = PolicyMatchContext(
            team_alias="team", key_alias="dev-key-123", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)
        assert "dev-policy" in attached

        # No match
        context_prod = PolicyMatchContext(
            team_alias="team", key_alias="prod-key-123", model="gpt-4"
        )
        attached_prod = registry.get_attached_policies(context_prod)
        assert "dev-policy" not in attached_prod

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
        attached = registry.get_attached_policies(context)
        assert "gpt4-policy" in attached

        # No match
        context_other = PolicyMatchContext(
            team_alias="team", key_alias="key", model="gpt-3.5"
        )
        attached_other = registry.get_attached_policies(context_other)
        assert "gpt4-policy" not in attached_other

    def test_multiple_attachments_match(self):
        """Test multiple attachments can match same context."""
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

        assert "global-baseline" in attached
        assert "healthcare-policy" in attached
        assert "gpt4-policy" in attached
        assert len(attached) == 3

    def test_no_duplicate_policies(self):
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


class TestIsPolicyAttached:
    """Test is_policy_attached method."""

    def test_policy_is_attached(self):
        """Test checking if a specific policy is attached."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "global-baseline", "scope": "*"},
        ])

        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        assert registry.is_policy_attached("global-baseline", context) is True
        assert registry.is_policy_attached("other-policy", context) is False


class TestGetAttachmentsForPolicy:
    """Test getting attachments for a specific policy."""

    def test_get_attachments_for_policy(self):
        """Test getting all attachments for a policy."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "multi-policy", "scope": "*"},
            {"policy": "multi-policy", "teams": ["team-a"]},
            {"policy": "other-policy", "teams": ["team-b"]},
        ])

        attachments = registry.get_attachments_for_policy("multi-policy")
        assert len(attachments) == 2

        attachments_other = registry.get_attachments_for_policy("other-policy")
        assert len(attachments_other) == 1


class TestAddAndRemoveAttachments:
    """Test adding and removing individual attachments."""

    def test_add_attachment(self):
        """Test adding a single attachment."""
        registry = AttachmentRegistry()
        registry.load_attachments([])

        attachment = PolicyAttachment(policy="new-policy", scope="*")
        registry.add_attachment(attachment)

        assert len(registry.get_all_attachments()) == 1

    def test_remove_attachments_for_policy(self):
        """Test removing all attachments for a policy."""
        registry = AttachmentRegistry()
        registry.load_attachments([
            {"policy": "policy-a", "scope": "*"},
            {"policy": "policy-a", "teams": ["team-a"]},
            {"policy": "policy-b", "teams": ["team-b"]},
        ])

        removed = registry.remove_attachments_for_policy("policy-a")
        assert removed == 2
        assert len(registry.get_all_attachments()) == 1
        assert len(registry.get_attachments_for_policy("policy-a")) == 0


class TestPolicyAttachmentModel:
    """Test PolicyAttachment model methods."""

    def test_is_global(self):
        """Test is_global method."""
        global_attachment = PolicyAttachment(policy="test", scope="*")
        assert global_attachment.is_global() is True

        team_attachment = PolicyAttachment(policy="test", teams=["team-a"])
        assert team_attachment.is_global() is False

    def test_to_policy_scope_global(self):
        """Test converting global attachment to PolicyScope."""
        attachment = PolicyAttachment(policy="test", scope="*")
        scope = attachment.to_policy_scope()

        assert scope.get_teams() == ["*"]
        assert scope.get_keys() == ["*"]
        assert scope.get_models() == ["*"]

    def test_to_policy_scope_specific(self):
        """Test converting specific attachment to PolicyScope."""
        attachment = PolicyAttachment(
            policy="test",
            teams=["team-a", "team-b"],
            keys=["key-*"],
            models=["gpt-4"],
        )
        scope = attachment.to_policy_scope()

        assert scope.teams == ["team-a", "team-b"]
        assert scope.keys == ["key-*"]
        assert scope.models == ["gpt-4"]


class TestGlobalSingleton:
    """Test global singleton behavior."""

    def test_get_attachment_registry_singleton(self):
        """Test get_attachment_registry returns same instance."""
        registry1 = get_attachment_registry()
        registry2 = get_attachment_registry()
        assert registry1 is registry2
