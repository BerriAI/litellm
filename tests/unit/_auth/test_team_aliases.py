from collections.abc import Callable
from typing import Final

import pytest

from litellm._auth.team_aliases import TeamAliasDecision as Decision
from litellm._auth.team_aliases import TeamAliasFacts as Facts
from litellm._auth.team_aliases import team_aliases


def reference(facts: Facts) -> Decision:
    targets: Final = tuple(facts.aliases.get(name) for name in facts.requested)
    count: Final = sum(
        bool(target is not None and facts.router_present and target.startswith(f"model_name_{facts.team_id}_"))
        for target in targets
    )
    if count and facts.names is None:
        return Decision("ReadNames", count=count)
    reads: Final = iter(facts.names or ())
    return Decision(
        "Complete",
        models=tuple(
            name
            if target is None
            or (
                facts.router_present and target.startswith(f"model_name_{facts.team_id}_") and target not in next(reads)
            )
            else target
            for name, target in zip(facts.requested, targets, strict=True)
        ),
    )


@pytest.mark.parametrize("policy", (reference, team_aliases))
@pytest.mark.parametrize(
    "facts",
    (
        Facts(("alias",), {}, "team", True),
        Facts(("alias",), {"alias": "target"}, "team", True),
        Facts(("alias",), {"alias": "model_name_team_target"}, "team", False),
        Facts(("alias",), {"alias": "model_name_team_target"}, "team", True),
        Facts(("alias", "alias"), {"alias": "model_name_team_target"}, "team", True, (("model_name_team_target",), ())),
        Facts(("alias",), {"alias": "model_name_None_target"}, None, True, ((),)),
    ),
)
def test_aliases_preserve_deleted_targets_and_repeated_registry_reads(
    policy: Callable[[Facts], Decision], facts: Facts
) -> None:
    assert policy(facts) == reference(facts)
