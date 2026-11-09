from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.key_model_access import KeyAndFallbackModelAccessFacts, key_and_fallback_model_access
from litellm._auth.model_access import ModelAccessFacts

_ALLOW: Final = GroupedModelScope(ModelAccessFacts(("model",), ("model",), object_type="key"))
_DENY: Final = GroupedModelScope(replace(_ALLOW.facts, grants=("other",)))
_BASE: Final = KeyAndFallbackModelAccessFacts(False, ("model",))


@pytest.mark.parametrize(
    ("facts", "kind", "index"),
    (
        pytest.param(_BASE, "ReadPrimary", 0, id="first-read"),
        pytest.param(replace(_BASE, config_nonempty=True), "Allow", 0, id="config-before-primary"),
        pytest.param(replace(_BASE, key_models=("all-team-models",)), "Allow", 0, id="team-sentinel-before-primary"),
        pytest.param(
            replace(_BASE, primary_loaded=True, primary=_DENY), "Deny", 0, id="primary-denial-before-fallback-parsing"
        ),
        pytest.param(
            replace(_BASE, primary_loaded=True, primary=replace(_DENY, group_ids=("group",))),
            "ReadPrimaryGroups",
            0,
            id="key-group-before-fallback-parsing",
        ),
        pytest.param(
            replace(_BASE, primary_loaded=True), "ReadFallbackNames", 0, id="no-primary-still-checks-fallbacks"
        ),
        pytest.param(
            replace(_BASE, primary_loaded=True, fallback_names=("model",)),
            "ReadFallbackScope",
            0,
            id="fallback-needs-scope",
        ),
        pytest.param(
            replace(_BASE, primary_loaded=True, fallback_names=("model",), fallback_scopes=(_ALLOW,)),
            "RouteFallback",
            0,
            id="allowed-fallback-then-probe",
        ),
        pytest.param(
            replace(_BASE, primary_loaded=True, fallback_names=("model",), fallback_scopes=(_DENY,)),
            "Deny",
            0,
            id="denied-fallback-before-probe",
        ),
        pytest.param(
            replace(
                _BASE,
                primary_loaded=True,
                fallback_names=("first", "second"),
                fallback_scopes=(_ALLOW,),
                routing=("Failed",),
            ),
            "RoutingFailure",
            0,
            id="routing-failure-before-next-scope",
        ),
        pytest.param(
            replace(
                _BASE,
                primary_loaded=True,
                fallback_names=("first", "second"),
                fallback_scopes=(_ALLOW,),
                routing=("Routed",),
            ),
            "ReadFallbackScope",
            1,
            id="next-scope-after-routing",
        ),
        pytest.param(replace(_BASE, primary_loaded=True, fallback_names=()), "Allow", 0, id="empty-fallbacks"),
    ),
)
def test_key_and_fallback_access_order(facts: KeyAndFallbackModelAccessFacts, kind: str, index: int) -> None:
    decision: Final = key_and_fallback_model_access(facts)
    assert (decision.kind, decision.index) == (kind, index)
