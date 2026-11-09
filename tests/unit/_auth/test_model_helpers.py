import re
from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.model_access import ModelProvider
from litellm._auth.model_helpers import ModelHelperDecision as Decision
from litellm._auth.model_helpers import ModelHelperFacts as Facts
from litellm._auth.model_helpers import model_helper


def reference(facts: Facts) -> Decision:
    if facts.kind == "Group" and (facts.grants is None or facts.model in facts.grants):
        return Decision("Complete", True)
    if facts.router_present and facts.groups is None:
        return Decision("ReadGroups")
    groups: Final = facts.groups or ()
    grants: Final = facts.grants or ()
    if facts.kind == "Group":
        return Decision("Complete", any(grant in groups for grant in grants))
    expand: Final = "all-team-models" in grants and facts.team_id is not None and facts.router_present
    if expand and facts.names is None:
        return Decision("ReadNames")
    expanded: Final = (
        (
            tuple(
                dict.fromkeys((*tuple(grant for grant in grants if grant != "all-team-models"), *(facts.names or ())))
            )
            or grants
        )
        if expand
        else grants
    )
    if any(grant in groups for grant in expanded) or facts.model in facts.team_aliases:
        return Decision("Complete", True)
    filtered: Final = tuple(grant for grant in expanded if grant not in groups)
    patterns: Final = tuple(grant.replace("*", ".*") + "$" for grant in filtered if "*" in grant)
    if any(re.match(pattern, facts.model) is not None for pattern in patterns):
        return Decision("Complete", True)
    if patterns:
        if not facts.provider_loaded:
            return Decision("ReadProvider")
        provider: Final = facts.provider
        if (
            provider is not None
            and (provider.stripped_model != facts.model or "/" not in facts.model)
            and any(
                re.match(pattern, f"{provider.provider}/{provider.stripped_model}") is not None for pattern in patterns
            )
        ):
            return Decision("Complete", True)
    return Decision(
        "Complete",
        not filtered and not expanded or "*" in filtered or "all-proxy-models" in filtered or facts.model in filtered,
    )


_GROUP: Final = Facts("Group", "model", ("model",), router_present=True)
_HELPER: Final = replace(_GROUP, kind="Helper")
_READ: Final = replace(_HELPER, groups=())


@pytest.mark.parametrize("policy", (reference, model_helper))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_GROUP, Decision("Complete", True)),
        (replace(_GROUP, grants=None), Decision("Complete", True)),
        (replace(_GROUP, grants=()), Decision("ReadGroups")),
        (replace(_GROUP, grants=(), groups=()), Decision("Complete")),
        (replace(_GROUP, grants=("group",), groups=("group",)), Decision("Complete", True)),
        (_HELPER, Decision("ReadGroups")),
        (_READ, Decision("Complete", True)),
        (replace(_READ, grants=()), Decision("Complete", True)),
        (replace(_READ, grants=("other",)), Decision("Complete")),
        (replace(_READ, grants=("other",), team_aliases=("model",)), Decision("Complete", True)),
        (replace(_READ, grants=("group",), groups=("group",)), Decision("Complete", True)),
        (replace(_READ, grants=("all-team-models",), team_id="team"), Decision("ReadNames")),
        (replace(_READ, grants=("all-team-models",), team_id="team", names=()), Decision("Complete")),
        (replace(_READ, grants=("all-team-models",), team_id="team", names=("model",)), Decision("Complete", True)),
        (replace(_READ, grants=("mod*",)), Decision("Complete", True)),
        (replace(_READ, grants=("provider/*",)), Decision("ReadProvider")),
        (replace(_READ, grants=("provider/*",), provider_loaded=True), Decision("Complete")),
        (
            replace(_READ, grants=("provider/*",), provider_loaded=True, provider=ModelProvider("model", "provider")),
            Decision("Complete", True),
        ),
        (
            replace(
                _READ,
                model="unknown/model",
                grants=("provider/*",),
                provider_loaded=True,
                provider=ModelProvider("unknown/model", "provider"),
            ),
            Decision("Complete"),
        ),
    ),
)
def test_model_helper_read_order_and_grants(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected
