"""Live e2e: a policy attached to a request keeps its inherited parent guardrails
when only the child's own `condition` fails to match the request model.

The parent policy has no condition and adds a content filter. The child inherits
it, adds a second content filter, and carries a model condition. The attachment
points at the child only, so the parent is reachable through inheritance alone.
A request the child condition does not match must still be blocked by the
parent's filter; a request it does match must be blocked by both.

Uses litellm_content_filter (keyword match, no external service) so the block is
deterministic and free, with the request model routed to a real provider.
"""

from __future__ import annotations

import pytest
from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import StreamingResponse
from guardrails_client import (
    GuardrailsClient,
    PolicyConditionBody,
    PolicyCreateBody,
)
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = CHEAP_OPENAI_MODEL


def _applied_guardrails(outcome: StreamingResponse) -> frozenset[str]:
    return frozenset(
        name.strip() for name in outcome.headers.get("x-litellm-applied-guardrails", "").split(",") if name.strip()
    )


def _setup_child_policy_attached_to_tag(
    client: GuardrailsClient,
    resources: ResourceManager,
    *,
    child_condition_model: str,
    parent_banned: str,
    child_banned: str,
    tag: str,
) -> tuple[str, str]:
    """Register parent and child content filters, a parent policy adding the parent
    filter, a child policy inheriting it with `child_condition_model`, and attach
    only the child to `tag`. Returns (parent_guardrail_name, child_guardrail_name)."""
    parent_guardrail = f"e2e-parent-guard-{parent_banned}"
    child_guardrail = f"e2e-child-guard-{child_banned}"
    parent_guardrail_id = client.create_content_filter_guardrail(parent_guardrail, parent_banned, default_on=False)
    resources.defer(lambda: client.delete_guardrail(parent_guardrail_id))
    child_guardrail_id = client.create_content_filter_guardrail(child_guardrail, child_banned, default_on=False)
    resources.defer(lambda: client.delete_guardrail(child_guardrail_id))

    parent_policy = client.create_policy(
        PolicyCreateBody(policy_name=f"e2e-parent-policy-{parent_banned}", guardrails_add=[parent_guardrail])
    )
    resources.defer(lambda: client.delete_policy(parent_policy))
    child_policy = client.create_policy(
        PolicyCreateBody(
            policy_name=f"e2e-child-policy-{child_banned}",
            inherit=parent_policy,
            guardrails_add=[child_guardrail],
            condition=PolicyConditionBody(model=child_condition_model),
        )
    )
    resources.defer(lambda: client.delete_policy(child_policy))

    attachment_id = client.attach_policy_to_tags(child_policy, [tag])
    resources.defer(lambda: client.delete_policy_attachment(attachment_id))
    return parent_guardrail, child_guardrail


class TestPolicyInheritedGuardrail:
    def test_child_condition_miss_still_applies_inherited_parent_guardrail(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        parent_banned = unique_marker()
        child_banned = unique_marker()
        tag = f"e2e-policy-tag-{unique_marker()}"
        parent_guardrail, child_guardrail = _setup_child_policy_attached_to_tag(
            client,
            resources,
            child_condition_model=f"never-matches-{unique_marker()}",
            parent_banned=parent_banned,
            child_banned=child_banned,
            tag=tag,
        )

        outcome = client.chat_raw(scoped_key, MODEL, f"Reply with the single word OK. {parent_banned}", tags=[tag])

        assert outcome.status_code == 400, (
            f"the inherited parent content filter must block the banned keyword even though the child "
            f"policy's own model condition does not match {MODEL}; got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert parent_guardrail in _applied_guardrails(outcome), (
            f"x-litellm-applied-guardrails must name the inherited parent guardrail; got {outcome.headers}"
        )
        assert child_guardrail not in _applied_guardrails(outcome), (
            f"the child's own guardrail must not run when its condition fails; got {outcome.headers}"
        )

    def test_child_condition_match_applies_child_and_inherited_parent_guardrails(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        parent_banned = unique_marker()
        child_banned = unique_marker()
        tag = f"e2e-policy-tag-{unique_marker()}"
        parent_guardrail, child_guardrail = _setup_child_policy_attached_to_tag(
            client,
            resources,
            child_condition_model=MODEL,
            parent_banned=parent_banned,
            child_banned=child_banned,
            tag=tag,
        )

        outcome = client.chat_raw(scoped_key, MODEL, f"Reply with the single word OK. {child_banned}", tags=[tag])

        assert outcome.status_code == 400, (
            f"the child's own content filter must block its banned keyword when the condition matches {MODEL}; "
            f"got {outcome.status_code}: {outcome.body[:300]}"
        )
        assert {parent_guardrail, child_guardrail} <= _applied_guardrails(outcome), (
            f"both the child and inherited parent guardrails must run; got {outcome.headers}"
        )
