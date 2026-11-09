from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Final, Literal
from unittest.mock import MagicMock

import pytest

from litellm._auth import model_access as native
from litellm._auth.failure import NativeAuthFailure
from litellm._auth.model_access import (
    ModelAccessFacts as Facts,
)
from litellm._auth.model_access import (
    ModelAliases as Aliases,
)
from litellm._auth.model_access import (
    ModelProvider as Provider,
)
from litellm._auth.model_access import (
    ModelRecord as Record,
)
from litellm._auth.model_access import (
    ModelRegistry as Registry,
)
from litellm.proxy._types import LiteLLM_UserTable, ModelAccessDeniedProxyException
from litellm.proxy.auth import model_access_adapter
from tests.unit._auth.support import model_reference as reference
from tests.unit._auth.support import model_registry_reference

_BASE: Final = Facts(("model",), object_type="key")
_CASES: Final = (
    pytest.param(_BASE, "Allow", id="empty-is-unrestricted"),
    pytest.param(replace(_BASE, grants=("model",)), "Allow", id="exact-grant"),
    pytest.param(replace(_BASE, grants=("other",)), "Deny", id="denied-model"),
    pytest.param(replace(_BASE, grants=("*",)), "Allow", id="global-wildcard"),
    pytest.param(replace(_BASE, grants=("all-proxy-models",)), "Allow", id="proxy-sentinel"),
    pytest.param(replace(_BASE, grants=("mo*",)), "Allow", id="prefix-wildcard"),
    pytest.param(replace(_BASE, requested=("model\n",), grants=("mo*",)), "Allow", id="python-final-newline"),
    pytest.param(replace(_BASE, requested=("mod\nel",), grants=("mo*l",)), "Deny", id="embedded-newline"),
    pytest.param(
        replace(_BASE, grants=("p/*",), registry=Registry(records={"model": Record(provider=Provider("model", "p"))})),
        "Allow",
        id="inferred-provider",
    ),
    pytest.param(
        replace(
            _BASE,
            requested=("p/model",),
            grants=("q/*",),
            registry=Registry(records={"p/model": Record(provider=Provider("model", "p"))}),
        ),
        "Deny",
        id="different-provider",
    ),
    pytest.param(
        replace(_BASE, grants=("other",), aliases=Aliases(team={"model": "target"})), "Allow", id="team-alias-exists"
    ),
    pytest.param(
        replace(_BASE, grants=("model",), aliases=Aliases(key={"model": "target"})),
        "Deny",
        id="key-alias-replaces-source",
    ),
    pytest.param(
        replace(_BASE, grants=("target",), aliases=Aliases(key={"model": "target"})), "Allow", id="key-target"
    ),
    pytest.param(
        replace(_BASE, grants=("target",), aliases=Aliases(global_={"model": "target"})), "Allow", id="global-target"
    ),
    pytest.param(
        replace(
            _BASE,
            grants=("target",),
            aliases=Aliases(
                team={"model": "middle"}, key={"middle": "third", "global": "target"}, global_={"third": "global"}
            ),
        ),
        "Allow",
        id="alias-order",
    ),
    pytest.param(
        replace(_BASE, grants=("target",), registry=Registry(present=True, aliases={"model": "target"})),
        "Allow",
        id="router-alias",
    ),
    pytest.param(
        replace(
            _BASE,
            grants=("target",),
            registry=Registry(present=True, aliases={"model": "target"}),
            aliases=Aliases(global_={"model": "wrong"}),
        ),
        "Deny",
        id="global-over-router",
    ),
    pytest.param(
        replace(_BASE, grants=("parent",), registry=Registry(compaction_parents={"model": "parent"})),
        "Allow",
        id="compaction-parent",
    ),
    pytest.param(
        replace(
            _BASE,
            grants=("group",),
            registry=Registry(present=True, records={"model": Record(access_groups=("group",))}),
        ),
        "Allow",
        id="registry-group",
    ),
    pytest.param(
        replace(
            _BASE,
            grants=("other-group",),
            registry=Registry(present=True, records={"model": Record(access_groups=("group",))}),
        ),
        "Deny",
        id="different-group",
    ),
    pytest.param(
        replace(_BASE, grants=("all-team-models",), team_id="team", registry=Registry(present=True, names=("model",))),
        "Allow",
        id="team-sentinel-expands",
    ),
    pytest.param(
        replace(_BASE, grants=("all-team-models",), team_id="team", registry=Registry(present=True)),
        "Deny",
        id="empty-registry-sentinel",
    ),
    pytest.param(
        replace(_BASE, grants=("all-team-models",), registry=Registry(present=True, names=("model",))),
        "Deny",
        id="team-sentinel-no-team",
    ),
    pytest.param(
        replace(
            _BASE,
            requested=("model_name_team_id",),
            grants=("public",),
            team_id="team",
            registry=Registry(
                present=True,
                records={"model_name_team_id": Record(deployment_team="team", deployment_public_name="public")},
            ),
        ),
        "Allow",
        id="owning-team-public-name",
    ),
    pytest.param(
        replace(
            _BASE,
            requested=("model_name_team_id",),
            grants=("public",),
            team_id="team",
            registry=Registry(
                present=True,
                records={"model_name_team_id": Record(deployment_team="other", deployment_public_name="public")},
            ),
        ),
        "Deny",
        id="public-name-wrong-team",
    ),
    pytest.param(
        replace(
            _BASE,
            requested=("model_name_team_id",),
            grants=("other",),
            team_id="team",
            aliases=Aliases(team={"public": "model_name_team_id"}),
            registry=Registry(present=True),
        ),
        "Deny",
        id="owning-alias-not-unconditional",
    ),
    pytest.param(
        replace(_BASE, requested=("model", "other"), grants=("model",), is_list=True),
        "Deny",
        id="all-list-items-required",
    ),
    pytest.param(replace(_BASE, requested=(), grants=("other",), is_list=True), "Allow", id="empty-list"),
    pytest.param(replace(_BASE, fallback_depth=5), "Depth", id="depth"),
    pytest.param(replace(_BASE, fallback_depth=4, is_list=True), "Depth", id="list-depth"),
)


@pytest.mark.parametrize("policy", (reference.model_access, native.model_access), ids=("reference", "native"))
@pytest.mark.parametrize(("facts", "kind"), _CASES)
def test_model_policy_case_table(
    policy: Callable[[Facts], native.ModelAccessDecision], facts: Facts, kind: str
) -> None:
    decision: Final = policy(facts)
    assert decision.kind == kind
    assert decision == reference.model_access(facts)


@pytest.mark.parametrize(
    ("groups", "models", "merge", "kind"),
    (
        ((), None, False, "Deny"),
        (("g",), None, False, "ReadGroups"),
        (("g",), (), False, "Deny"),
        (("g",), ("model",), False, "Allow"),
        (("g",), ("model",), True, "Allow"),
        (("g",), ("other",), False, "Deny"),
    ),
)
def test_group_fallback_waits_for_data(
    groups: tuple[str, ...], models: tuple[str, ...] | None, merge: bool, kind: str
) -> None:
    assert native.grouped_model_access(replace(_BASE, grants=("no",)), groups, models, merge=merge).kind == kind


@pytest.mark.parametrize(
    ("present", "grants", "kind"),
    (
        (False, ("no-default-models",), "Allow"),
        (True, ("no-default-models", "model"), "NoDefaultModels"),
        (True, (), "Allow"),
        (True, ("other",), "Deny"),
    ),
)
def test_user_default_scope(present: bool, grants: tuple[str, ...], kind: str) -> None:
    assert native.user_model_access(replace(_BASE, grants=grants), present=present).kind == kind


@pytest.mark.parametrize(
    "policy", (model_registry_reference.can_user_call_model, model_access_adapter.can_user_call_model)
)
@pytest.mark.asyncio
async def test_user_precheck_does_not_read_registry(
    policy: Callable[[str, MagicMock, LiteLLM_UserTable | None], Awaitable[Literal[True]]],
) -> None:
    router: Final = MagicMock()
    user: Final = LiteLLM_UserTable(user_id="user", models=["no-default-models"])
    with pytest.raises(ModelAccessDeniedProxyException, match="model"):
        await policy("model", router, user)
    assert not router.mock_calls
    assert await policy("model", router, None) is True
    assert not router.mock_calls


def test_native_model_decode_failure_is_terminal() -> None:
    with pytest.raises(NativeAuthFailure):
        native.model_access(replace(_BASE, fallback_depth=-1))


@pytest.mark.parametrize(
    ("managed", "ceilings", "kind"),
    (
        (None, (), "Allow"),
        ((), (), "NoAgentModels"),
        (("model",), ((),), "EmptyAgentCeiling"),
        (("model",), (("model",), ("other",)), "Deny"),
        (("model",), (("model",),), "Allow"),
    ),
)
def test_agent_grants_are_intersected(
    managed: tuple[str, ...] | None, ceilings: tuple[tuple[str, ...], ...], kind: str
) -> None:
    assert native.agent_model_access(_BASE, managed, ceilings).kind == kind
