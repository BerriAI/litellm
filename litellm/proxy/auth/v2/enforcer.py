import os
from typing import Optional, Sequence

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.conf")

Rule = Sequence[str]
Rules = Sequence[Rule]


class CasbinEnforcer:
    """Wraps a casbin Enforcer built from explicit in-memory rules.

    Control-plane access is expressed as ``p`` policy rows; identity-to-role
    bridges are ``g`` grouping rows. Built per policy snapshot so swapping the
    snapshot (e.g. after a reload) is a fresh, side-effect-free object. Holds no
    litellm imports so the decision logic is testable in isolation.
    """

    def __init__(
        self,
        policies: Rules,
        groupings: Rules,
        resource_groupings: Optional[Rules] = None,
        domain_groupings: Optional[Rules] = None,
    ):
        # Imported lazily so merely importing the proxy (or this package, e.g. to
        # register the policy-admin router) never requires casbin. The dependency
        # is only needed once auth_v2 is actually enabled and builds an enforcer.
        import casbin

        self._enforcer = casbin.Enforcer(_MODEL_PATH)
        self._enforcer.enable_auto_save(False)
        for rule in policies:
            self._enforcer.add_policy(*rule)
        for rule in groupings:
            self._enforcer.add_named_grouping_policy("g", *rule)
        for rule in resource_groupings or []:
            self._enforcer.add_named_grouping_policy("g2", *rule)
        for rule in domain_groupings or []:
            self._enforcer.add_named_grouping_policy("g3", *rule)

    def enforce(self, subject: str, domain: str, obj: str, action: str) -> bool:
        return self._enforcer.enforce(subject, domain, obj, action)
