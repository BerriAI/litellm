"""
Unit tests for PolicyMatcher - tests wildcard pattern matching via attachments.

Tests:
- Wildcard matching (*, prefix-*)
- Scope matching via attachments (teams, keys, models)
"""

import logging
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import litellm.proxy.policy_engine.attachment_registry as attachment_registry_module
import litellm.proxy.policy_engine.policy_registry as policy_registry_module
from litellm.proxy.policy_engine.attachment_registry import AttachmentRegistry
from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.proxy.policy_engine.policy_registry import PolicyRegistry
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyCondition,
    PolicyGuardrails,
    PolicyMatchContext,
    PolicyScope,
)


class TestPolicyMatcherPatternMatching:
    """Test pattern matching utilities."""

    def test_matches_pattern_exact(self):
        """Test exact pattern matching."""
        assert (
            PolicyMatcher.matches_pattern("healthcare-team", ["healthcare-team"])
            is True
        )
        assert (
            PolicyMatcher.matches_pattern("finance-team", ["healthcare-team"]) is False
        )

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
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="any-key", model="gpt-4"
        )
        assert PolicyMatcher.scope_matches(scope, context) is True

    def test_scope_does_not_match_team(self):
        """Test scope doesn't match when team doesn't match."""
        scope = PolicyScope(teams=["healthcare-team"], keys=["*"], models=["*"])
        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="any-key", model="gpt-4"
        )
        assert PolicyMatcher.scope_matches(scope, context) is False

    def test_scope_matches_with_wildcard_patterns(self):
        """Test scope matches with wildcard patterns."""
        scope = PolicyScope(teams=["*"], keys=["dev-key-*"], models=["bedrock/*"])
        context = PolicyMatchContext(
            team_alias="any-team", key_alias="dev-key-123", model="bedrock/claude-3"
        )
        assert PolicyMatcher.scope_matches(scope, context) is True

    def test_scope_global_wildcard(self):
        """Test global scope with all wildcards."""
        scope = PolicyScope(teams=["*"], keys=["*"], models=["*"])
        context = PolicyMatchContext(
            team_alias="any-team", key_alias="any-key", model="any-model"
        )
        assert PolicyMatcher.scope_matches(scope, context) is True


class TestPolicyMatcherScopeMatchingWithTags:
    """Test scope matching with tag patterns."""

    def test_scope_tag_matching(self):
        """Test scope tag matching: exact, wildcard, no-match, and empty context tags."""
        # Exact match
        scope = PolicyScope(teams=["*"], keys=["*"], models=["*"], tags=["healthcare"])
        context = PolicyMatchContext(
            team_alias="team",
            key_alias="key",
            model="gpt-4",
            tags=["healthcare", "internal"],
        )
        assert PolicyMatcher.scope_matches(scope, context) is True

        # Wildcard match
        scope_wc = PolicyScope(teams=["*"], keys=["*"], models=["*"], tags=["health-*"])
        context_wc = PolicyMatchContext(
            team_alias="team",
            key_alias="key",
            model="gpt-4",
            tags=["health-prod"],
        )
        assert PolicyMatcher.scope_matches(scope_wc, context_wc) is True

        # No match — wrong tag
        context_wrong = PolicyMatchContext(
            team_alias="team",
            key_alias="key",
            model="gpt-4",
            tags=["finance"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong) is False

        # No match — context has no tags
        context_none = PolicyMatchContext(
            team_alias="team",
            key_alias="key",
            model="gpt-4",
            tags=None,
        )
        assert PolicyMatcher.scope_matches(scope, context_none) is False

        # Scope without tags matches any context (opt-in semantics)
        scope_no_tags = PolicyScope(teams=["*"], keys=["*"], models=["*"])
        assert PolicyMatcher.scope_matches(scope_no_tags, context) is True

    def test_scope_tags_and_team_combined(self):
        """Test scope with both tags and team — both must match (AND logic)."""
        scope = PolicyScope(
            teams=["team-a"], keys=["*"], models=["*"], tags=["healthcare"]
        )

        # Both match
        context_both = PolicyMatchContext(
            team_alias="team-a",
            key_alias="key",
            model="gpt-4",
            tags=["healthcare"],
        )
        assert PolicyMatcher.scope_matches(scope, context_both) is True

        # Tag matches, team doesn't
        context_wrong_team = PolicyMatchContext(
            team_alias="team-b",
            key_alias="key",
            model="gpt-4",
            tags=["healthcare"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong_team) is False

        # Team matches, tag doesn't
        context_wrong_tag = PolicyMatchContext(
            team_alias="team-a",
            key_alias="key",
            model="gpt-4",
            tags=["finance"],
        )
        assert PolicyMatcher.scope_matches(scope, context_wrong_tag) is False


class TestPolicyMatcherWithAttachments:
    """Test getting matching policies via attachments."""

    def test_get_matching_policies_via_attachments(self):
        """Test matching policies through attachment registry."""
        # Create and configure attachment registry
        registry = AttachmentRegistry()
        registry.load_attachments(
            [
                {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
                {"policy": "global-policy", "scope": "*"},
            ]
        )

        # Test matching via the registry directly
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="k", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)

        assert "healthcare-policy" in attached
        assert "global-policy" in attached

    def test_get_matching_policies_no_match(self):
        """Test no policies match when attachments don't match context."""
        registry = AttachmentRegistry()
        registry.load_attachments(
            [
                {"policy": "healthcare-policy", "teams": ["healthcare-team"]},
            ]
        )

        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="k", model="gpt-4"
        )
        attached = registry.get_attached_policies(context)

        assert "healthcare-policy" not in attached


def _global_registries(monkeypatch):
    policies = PolicyRegistry()
    policies.load_policies(
        {
            "guardrail-y": {"guardrails": {"add": ["y"]}},
            "guardrail-x": {"guardrails": {"add": ["x"]}, "condition": {"model": "claude.*"}},
        }
    )
    attachments = AttachmentRegistry()
    attachments.load_attachments(
        [
            {"policy": "guardrail-x", "tags": ["opt-in"]},
            {"policy": "guardrail-y", "scope": "*", "default": True},
        ]
    )
    monkeypatch.setattr(policy_registry_module, "get_policy_registry", lambda: policies)
    monkeypatch.setattr(attachment_registry_module, "get_attachment_registry", lambda: attachments)
    return policies


def _inherited_registries(monkeypatch, parent_condition=None):
    policies = PolicyRegistry()
    policies.load_policies(
        {
            "parent": {
                "guardrails": {"add": ["y"]},
                **({"condition": parent_condition} if parent_condition else {}),
            },
            "child": {
                "inherit": "parent",
                "guardrails": {"add": ["x"]},
                "condition": {"model": "claude.*"},
            },
            "fallback": {"guardrails": {"add": ["z"]}},
        }
    )
    attachments = AttachmentRegistry()
    attachments.load_attachments(
        [
            {"policy": "child", "scope": "*"},
            {"policy": "fallback", "scope": "*", "default": True},
        ]
    )
    monkeypatch.setattr(policy_registry_module, "get_policy_registry", lambda: policies)
    monkeypatch.setattr(attachment_registry_module, "get_attachment_registry", lambda: attachments)
    return policies


class TestGetMatchingPoliciesFallback:
    def test_condition_failing_opt_in_falls_back_to_default(self, monkeypatch):
        _global_registries(monkeypatch)
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5", tags=["opt-in"])

        assert PolicyMatcher.get_matching_policies(context=context) == ["guardrail-y"]

    def test_condition_passing_opt_in_suppresses_default(self, monkeypatch):
        _global_registries(monkeypatch)
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="claude-haiku", tags=["opt-in"])

        assert PolicyMatcher.get_matching_policies(context=context) == ["guardrail-x"]

    def test_policy_applies_reads_registry_once(self, monkeypatch):
        policies = _global_registries(monkeypatch)
        calls = []
        original = policies.get_all_policies
        monkeypatch.setattr(policies, "get_all_policies", lambda: calls.append(1) or original())
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5", tags=["opt-in"])

        PolicyMatcher.get_matching_policies(context=context)

        assert len(calls) == 1

    def test_condition_missing_child_with_unconditional_parent_still_matches(self, monkeypatch):
        _inherited_registries(monkeypatch)
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5")

        assert PolicyMatcher.get_matching_policies(context=context) == ["child"]

    def test_child_whose_whole_chain_misses_falls_back_to_default(self, monkeypatch):
        _inherited_registries(monkeypatch, parent_condition={"model": "claude.*"})
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5")

        assert PolicyMatcher.get_matching_policies(context=context) == ["fallback"]

    def test_get_policies_with_matching_conditions_keeps_missing_policy_out(self):
        policies = {
            "real": Policy(
                guardrails=PolicyGuardrails(add=["g"]),
                condition=PolicyCondition(model="claude.*"),
            ),
        }
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5")

        assert (
            PolicyMatcher.get_policies_with_matching_conditions(
                policy_names=["nope"], context=context, policies=policies
            )
            == []
        )


_MODELS: Final = ("gpt-4o", "gpt-5.5", "claude-opus-4-1")


def _policy_forest(draw: st.DrawFn) -> dict[str, Policy]:  # mutable-ok: PolicyResolver takes dict[str, Policy]
    names: Final = tuple(f"p{i}" for i in range(draw(st.integers(min_value=1, max_value=6))))
    return {  # mutable-ok: PolicyResolver takes dict[str, Policy]
        name: Policy(
            inherit=draw(st.sampled_from((None, *names[:i]))),
            guardrails=PolicyGuardrails(add=[f"g-{name}"]),  # mutable-ok: pydantic list field
            condition=draw(st.sampled_from((None, *(PolicyCondition(model=m) for m in _MODELS)))),
        )
        for i, name in enumerate(names)
    }


@st.composite
def _forest_and_request(
    draw: st.DrawFn,
) -> tuple[dict[str, Policy], tuple[str, ...], PolicyMatchContext]:  # mutable-ok: PolicyResolver takes dict
    policies: Final = _policy_forest(draw)
    attached: Final = tuple(draw(st.lists(st.sampled_from(sorted(policies)), unique=True)))
    context: Final = PolicyMatchContext(team_alias="t", key_alias="k", model=draw(st.sampled_from(_MODELS)))
    return policies, attached, context


def _own_condition_applies(policy: Policy, context: PolicyMatchContext) -> bool:
    return policy.condition is None or policy.condition.model == context.model


def _applicable_chain(
    policies: dict[str, Policy],  # mutable-ok: PolicyResolver takes dict[str, Policy]
    name: str,
    context: PolicyMatchContext,
) -> tuple[str, ...]:
    chain: Final = PolicyResolver.resolve_inheritance_chain(policy_name=name, policies=policies)
    return tuple(member for member in chain if _own_condition_applies(policies[member], context))


class TestChainMatchingProperties:
    @given(_forest_and_request())
    @settings(max_examples=400, deadline=None)
    def test_chain_matching_only_widens_to_applicable_ancestor_guardrails(
        self,
        case: tuple[dict[str, Policy], tuple[str, ...], PolicyMatchContext],  # mutable-ok: PolicyResolver takes dict
    ):
        policies, attached, context = case
        head: Final = tuple(
            PolicyMatcher.get_policies_with_matching_conditions(
                policy_names=attached, context=context, policies=policies
            )
        )
        base: Final = tuple(name for name in attached if _own_condition_applies(policies[name], context))
        expected_head: Final = tuple(name for name in attached if _applicable_chain(policies, name, context))

        assert head == expected_head, "a policy applies exactly when some chain member's own condition applies"
        assert frozenset(base) <= frozenset(head), "head must never drop a policy base applied"

        for name in head:
            resolved = PolicyResolver.resolve_policy_guardrails(policy_name=name, policies=policies, context=context)
            assert sorted(resolved.guardrails) == sorted(
                f"g-{member}" for member in _applicable_chain(policies, name, context)
            )
            if name not in base:
                assert f"g-{name}" not in resolved.guardrails, "a condition-missed child must not add its own guardrail"


class TestAncestorAdmissionLogging:
    @staticmethod
    def _chain() -> dict[str, Policy]:  # mutable-ok: PolicyResolver takes dict[str, Policy]
        return {  # mutable-ok: PolicyResolver takes dict[str, Policy]
            "parent": Policy(guardrails=PolicyGuardrails(add=["g-parent"])),  # mutable-ok: pydantic list field
            "child": Policy(
                inherit="parent",
                guardrails=PolicyGuardrails(add=["g-child"]),  # mutable-ok: pydantic list field
                condition=PolicyCondition(model="gpt-5.5"),
            ),
        }

    def test_logs_when_admitted_through_ancestor_only(self, caplog):
        context: Final = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4o")
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            result: Final = PolicyMatcher.policy_applies(context, self._chain())("child")
        records: Final = [r for r in caplog.records if "applied through ancestor" in r.getMessage()]
        assert result is True
        assert len(records) == 1
        assert "applied through ancestor 'parent'" in records[0].getMessage()
        assert "'child'" in records[0].getMessage()

    def test_no_log_when_own_condition_matches(self, caplog):
        context: Final = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-5.5")
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            result: Final = PolicyMatcher.policy_applies(context, self._chain())("child")
        assert result is True
        assert not [r for r in caplog.records if "applied through ancestor" in r.getMessage()]

    def test_no_log_when_no_chain_member_applies(self, caplog):
        policies: Final = {  # mutable-ok: PolicyResolver takes dict[str, Policy]
            "parent": Policy(
                guardrails=PolicyGuardrails(add=["g-parent"]),  # mutable-ok: pydantic list field
                condition=PolicyCondition(model="claude-opus-4-1"),
            ),
            "child": Policy(
                inherit="parent",
                guardrails=PolicyGuardrails(add=["g-child"]),  # mutable-ok: pydantic list field
                condition=PolicyCondition(model="gpt-5.5"),
            ),
        }
        context: Final = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4o")
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            result: Final = PolicyMatcher.policy_applies(context, policies)("child")
        assert result is False
        assert not [r for r in caplog.records if "applied through ancestor" in r.getMessage()]

    def test_condition_filter_logs_nothing(self, caplog):
        context: Final = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4o")
        with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
            result: Final = PolicyMatcher.get_policies_with_matching_conditions(
                policy_names=["child"], context=context, policies=self._chain()
            )
        assert result == ["child"]
        assert not [r for r in caplog.records if "applied through ancestor" in r.getMessage()]
