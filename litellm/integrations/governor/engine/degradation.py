"""The single place a tier outage becomes a verdict.

Centralizing this is the most important architectural rule: a policy raises
:class:`TierDegraded` and this module, knowing only the policy's ``fail_mode``,
produces either a reject (fail-closed) or an admit-degraded (fail-open) verdict.
Only this path sets ``degraded=True``.
"""

from litellm.integrations.governor.model.decisions import Verdict
from litellm.integrations.governor.model.errors import TierDegraded
from litellm.integrations.governor.policies.base import Policy


def verdict_for_degradation(policy: Policy, td: TierDegraded) -> Verdict:
    if policy.fail_mode == "closed":
        return Verdict.rejected(
            policy.policy_id,
            fail_mode="closed",
            reason=f"degraded_fail_closed:{td.reason}",
            degraded=True,
        )
    return Verdict.admitted_degraded(
        policy.policy_id,
        fail_mode="open",
        reason=f"degraded_fail_open:{td.reason}",
    )
