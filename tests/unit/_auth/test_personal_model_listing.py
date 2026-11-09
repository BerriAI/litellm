from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.owner_teams import OwnerTeamFacts, OwnerTeamLoad
from litellm._auth.personal_model_listing import PersonalModelListingDecision as Decision
from litellm._auth.personal_model_listing import PersonalModelListingFacts as Facts
from litellm._auth.personal_model_listing import personal_model_listing
from litellm._auth.personal_models import LoadedTeamModelFacts
from litellm._auth.personal_owner import PersonalOwnerFacts
from tests.unit._auth.test_personal_models import ALLOW, INITIAL
from tests.unit._auth.test_personal_models import reference as scope_reference
from tests.unit._auth.test_personal_owner import reference as owner_reference


def reference(facts: Facts) -> Decision:
    owner: Final = owner_reference(facts.owner)
    if owner.kind != "Complete":
        return Decision("ReadOwner")
    if facts.owner.teams is None or not facts.owner.teams.loads:
        return Decision("Complete")
    if facts.names is None:
        return Decision("ReadNames")
    if len(facts.models) < len(facts.names):
        return Decision("ReadModel", index=len(facts.models), teams=owner.selected)
    return Decision(
        "Complete",
        selected=tuple(
            index
            for index, model in enumerate(facts.models)
            if any(tuple(scope_reference(scope).allowed for scope in model.scopes))
        ),
    )


_BASE: Final = Facts(PersonalOwnerFacts(True, True, True, True))
_OWNER: Final = replace(
    _BASE.owner,
    owner_loaded=True,
    owner_present=True,
    teams=OwnerTeamFacts("Models", ("team",), "ui", loaded=True, loads=(OwnerTeamLoad(True),)),
)
_LOADED: Final = replace(_BASE, owner=_OWNER, names=("first", "second", "first"))
_ALLOWED: Final = LoadedTeamModelFacts((True,), (ALLOW,))
_DENIED: Final = LoadedTeamModelFacts((True,), (replace(INITIAL, io_failed=True),))


@pytest.mark.parametrize("policy", (reference, personal_model_listing))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadOwner")),
        (replace(_BASE, owner=replace(_BASE.owner, database_present=False)), Decision("Complete")),
        (replace(_BASE, owner=replace(_BASE.owner, owner_loaded=True, owner_failed=True)), Decision("Complete")),
        (replace(_BASE, owner=_OWNER), Decision("ReadNames")),
        (_LOADED, Decision("ReadModel", teams=(0,))),
        (replace(_LOADED, models=(_DENIED,)), Decision("ReadModel", index=1, teams=(0,))),
        (replace(_LOADED, models=(_ALLOWED, _DENIED, _ALLOWED)), Decision("Complete", selected=(0, 2))),
    ),
)
def test_listing_preserves_order_duplicates_and_checks_after_denial(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected
