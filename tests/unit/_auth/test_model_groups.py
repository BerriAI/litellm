from dataclasses import replace
from typing import Final, Literal

import pytest

from litellm._auth.model_groups import GroupLookup, GroupResources, ResourceGroupFacts, resource_group_decision

_BASE: Final = ResourceGroupFacts(True, True, True, ("group",), "team", "key")


@pytest.mark.parametrize(
    ("team_ids", "key_ids", "expected"),
    (
        (("team",), (), ("model",)),
        ((), ("key",), ("model",)),
        (("other",), (), ()),
        ((), ("other",), ()),
        ((), (), ()),
    ),
)
def test_group_ownership_boundary(
    team_ids: tuple[str, ...], key_ids: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    decision: Final = resource_group_decision(
        replace(_BASE, loaded=(GroupLookup("Found", GroupResources(team_ids, key_ids, ("model", "model"))),))
    )
    assert (decision.kind, decision.resources) == ("Complete", expected)


@pytest.mark.parametrize("kind", ("Missing", "Failed"))
@pytest.mark.parametrize("check_db_only", (False, True))
@pytest.mark.parametrize("strict_errors", (False, True))
def test_strict_lookup_errors_stop_before_next_group(
    kind: Literal["Missing", "Failed"], check_db_only: bool, strict_errors: bool
) -> None:
    decision: Final = resource_group_decision(
        replace(
            _BASE,
            group_ids=("first", "second"),
            check_db_only=check_db_only,
            strict_errors=strict_errors,
            loaded=(GroupLookup(kind),),
        )
    )
    assert (decision.kind, decision.index) == (("Raise", 0) if check_db_only or strict_errors else ("Read", 1))


@pytest.mark.parametrize("field", ("subject_present", "database_present", "cache_present"))
def test_unavailable_dependency_grants_nothing(field: str) -> None:
    decision: Final = resource_group_decision(replace(_BASE, **{field: False}))
    assert (decision.kind, decision.resources) == ("Complete", ())
