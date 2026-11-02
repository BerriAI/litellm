"""
Registry reads: deployments by group, alias, wildcard and team; names and ids; name resolution;
access groups; blocked and unhealthy groups; credentials lookups.
"""

import time
from typing import Final

import pytest

from litellm.router_core import RouterCore
from tests.unit.test_router.core.registry_helpers import MakeCore, ids, names_and_ids, team_deployment, typed
from tests.unit.test_router.core.scripted_host import deployment


def _core(make_core: MakeCore) -> RouterCore:
    """
    gpt (a, b), emb (e), the openai/* wildcard (w), team TA's ``pub`` (t1) and ``openai/*`` (tw),
    team TB's ``pub`` (t2). Aliases: fast -> gpt, hid -> gpt (hidden), gpt -> emb.
    """
    return make_core(
        model_list=[
            deployment("gpt", "a", model_info={"access_groups": ["ag1", "ag2"]}),
            deployment("gpt", "b", model_info={"access_groups": ["ag1"]}),
            deployment("emb", "e", model="openai/text-embedding-3-small", model_info={"access_groups": ["ag2"]}),
            deployment("openai/*", "w", model="openai/*", model_info={"access_groups": ["wild"]}),
            team_deployment("int-a", "t1", "TA", "pub", model="openai/team-model"),
            team_deployment("int-b", "t2", "TB", "pub", model="anthropic/claude-x"),
            team_deployment("team-wild", "tw", "TA", "openai/*", model="openai/*"),
        ],
        model_group_alias={"fast": "gpt", "hid": {"model": "gpt", "hidden": True}, "gpt": "emb"},
    )


# -- registry().deployments(): the group listing ------------------------------------------------


def test_listing_every_deployment_puts_visible_alias_rows_before_the_model_list(make_core: MakeCore) -> None:
    listed = _core(make_core).registry().deployments()

    assert names_and_ids(listed) == [
        ("fast", "a"),
        ("fast", "b"),
        ("gpt", "e"),
        ("gpt", "a"),
        ("gpt", "b"),
        ("emb", "e"),
        ("openai/*", "w"),
        ("int-a", "t1"),
        ("int-b", "t2"),
        ("team-wild", "tw"),
    ]


def test_an_alias_lists_its_targets_deployments_under_the_alias_name(make_core: MakeCore) -> None:
    assert names_and_ids(_core(make_core).registry().deployments("fast")) == [("fast", "a"), ("fast", "b")]


def test_a_hidden_alias_is_listed_when_named(make_core: MakeCore) -> None:
    assert names_and_ids(_core(make_core).registry().deployments("hid")) == [("hid", "a"), ("hid", "b")]


def test_a_group_that_is_also_an_alias_lists_its_own_deployments_then_the_alias_targets(make_core: MakeCore) -> None:
    assert names_and_ids(_core(make_core).registry().deployments("gpt")) == [("gpt", "a"), ("gpt", "b"), ("gpt", "e")]


def test_an_alias_row_is_a_copy_and_leaves_the_configured_entry_named_as_configured(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.registry().deployments("fast")

    assert [entry["model_name"] for entry in core.registry().all()[:2]] == ["gpt", "gpt"]


def test_an_unknown_name_lists_nothing(make_core: MakeCore) -> None:
    assert _core(make_core).registry().deployments("nope") == []


def test_a_name_matching_only_a_wildcard_lists_the_wildcard_deployment_under_the_requested_name(
    make_core: MakeCore,
) -> None:
    assert names_and_ids(_core(make_core).registry().deployments("openai/gpt-x")) == [("openai/gpt-x", "w")]


def test_a_team_caller_gets_its_own_wildcard_deployment_after_the_global_one(make_core: MakeCore) -> None:
    listed = _core(make_core).registry().deployments("openai/gpt-x", team_id="TA")

    assert names_and_ids(listed) == [("openai/gpt-x", "w"), ("openai/gpt-x", "tw")]


@pytest.mark.parametrize(("team_id", "expected"), [("TA", [("int-a", "t1")]), ("TB", [("int-b", "t2")]), (None, [])])
def test_a_team_public_name_lists_only_that_teams_deployment(
    make_core: MakeCore, team_id: str | None, expected: list[tuple[str, str]]
) -> None:
    assert names_and_ids(_core(make_core).registry().deployments("pub", team_id=team_id)) == expected


@pytest.mark.parametrize(("team_id", "expected"), [(None, ["t1"]), ("TA", ["t1"]), ("TB", [])])
def test_a_team_deployments_internal_name_lists_it_for_no_team_or_its_own_team_only(
    make_core: MakeCore, team_id: str | None, expected: list[str]
) -> None:
    assert ids(_core(make_core).registry().deployments("int-a", team_id=team_id)) == expected


def test_all_deployments_renames_rows_to_the_given_model_alias(make_core: MakeCore) -> None:
    listed = _core(make_core).registry().deployments("gpt", source="all", model_alias="renamed")

    assert names_and_ids(listed) == [("renamed", "a"), ("renamed", "b")]


def test_all_deployments_follow_no_alias_and_no_wildcard(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.deployments("fast", source="all") == []
    assert registry.deployments("openai/gpt-x", source="all") == []


def test_all_deployments_for_a_team_public_name_reads_the_team_index(make_core: MakeCore) -> None:
    assert ids(_core(make_core).registry().deployments("pub", source="all", team_id="TA")) == ["t1"]


def test_named_deployments_are_the_group_index_entry_or_none(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert ids(registry.deployments("gpt", source="named")) == ["a", "b"]
    assert registry.deployments("fast", source="named") is None


def test_first_in_group_is_the_first_configured_deployment_of_the_group(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    first = registry.deployments("gpt", source="first_in_group")
    assert first is not None
    assert first.model_info.id == "a"
    assert registry.deployments("fast", source="first_in_group") is None


@pytest.mark.parametrize("source", ["all", "named", "wildcard", "routable", "first_in_group"])
def test_a_source_that_needs_a_model_name_raises_without_one(make_core: MakeCore, source: str) -> None:
    with pytest.raises(TypeError, match="needs a model_name"):
        _core(make_core).registry().deployments(source=source)  # pyright: ignore[reportCallIssue, reportArgumentType]  # source is one of the literals


def test_an_unknown_deployment_source_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown deployment source 'bogus'"):
        _core(make_core).registry().deployments("gpt", source="bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid source on purpose


# -- wildcards ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("team_id", "expected"), [("TA", ["tw"]), ("TB", ["w"]), (None, ["w"])])
def test_wildcard_route_deployments_prefer_the_teams_own_pattern_over_the_global_one(
    make_core: MakeCore, team_id: str | None, expected: list[str]
) -> None:
    assert ids(_core(make_core).registry().deployments("openai/gpt-x", source="wildcard", team_id=team_id)) == expected


def test_wildcard_routed_models_name_the_requested_model(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.names("openai/gpt-x", kind="wildcard") == frozenset({"openai/gpt-x"})
    assert registry.names("anthropic/x", kind="wildcard") == frozenset()


# -- names ------------------------------------------------------------------------------------------


def test_group_names_repeat_per_deployment_and_include_visible_aliases(make_core: MakeCore) -> None:
    assert _core(make_core).registry().names() == ["fast", "fast", "gpt", "gpt", "gpt", "emb", "openai/*"]


def test_group_names_for_a_team_add_its_public_names(make_core: MakeCore) -> None:
    assert _core(make_core).registry().names(team_id="TA") == [
        "fast",
        "fast",
        "gpt",
        "gpt",
        "gpt",
        "emb",
        "openai/*",
        "pub",
        "openai/*",
    ]


def test_ids_can_leave_out_team_deployments(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.names(kind="ids") == ["a", "b", "e", "w", "t1", "t2", "tw"]
    assert registry.names(kind="ids", exclude_team_models=True) == ["a", "b", "e", "w"]


def test_ids_of_a_group_read_only_the_group_index(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.names("fast", kind="ids") == []
    assert registry.names("pub", kind="ids") == []
    assert registry.names("int-a", kind="ids") == ["t1"]
    assert registry.names("int-a", kind="ids", exclude_team_models=True) == []


def test_litellm_models_follow_aliases_and_wildcards(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.names("fast", kind="litellm_models") == ("openai/a", "openai/b")
    assert registry.names("openai/gpt-x", kind="litellm_models") == ("openai/gpt-x",)
    assert registry.names("nope", kind="litellm_models") == ()


def test_litellm_base_models_use_a_deployments_base_model(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[
            deployment(
                "gpt", "a", model="azure/my-deployment", api_base="https://x", model_info={"base_model": "azure/gpt-4o"}
            ),
            deployment("gpt", "b", model="openai/gpt-4o-mini"),
        ]
    )

    assert core.registry().names("gpt", kind="litellm_base_models") == ("azure/gpt-4o", "openai/gpt-4o-mini")
    assert core.registry().names("gpt", kind="litellm_models") == ("azure/my-deployment", "openai/gpt-4o-mini")


@pytest.mark.parametrize(
    ("name", "team_id", "expected"),
    [("fast", None, {"a", "b"}), ("openai/gpt-x", None, {"w"}), ("pub", "TA", {"t1"}), ("nope", None, set())],
)
def test_route_candidates_are_the_ids_a_request_for_the_name_may_reach(
    make_core: MakeCore, name: str, team_id: str | None, expected: set[str]
) -> None:
    assert _core(make_core).registry().names(name, kind="route_candidates", team_id=team_id) == expected


@pytest.mark.parametrize("kind", ["litellm_models", "wildcard", "route_candidates"])
def test_a_name_kind_that_needs_a_model_name_raises_without_one(make_core: MakeCore, kind: str) -> None:
    with pytest.raises(TypeError, match="needs a model_name"):
        _core(make_core).registry().names(kind=kind)  # pyright: ignore[reportCallIssue, reportArgumentType]  # kind is one of the literals


# -- resolve ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("fast", "gpt"), ("hid", "gpt"), ("gpt", "emb"), ("emb", "emb"), ("nope", "nope")],
)
def test_resolving_a_group_follows_one_alias_hop(make_core: MakeCore, name: str, expected: str) -> None:
    assert _core(make_core).registry().resolve(name) == expected


@pytest.mark.parametrize(("name", "expected"), [("fast", "gpt"), ("hid", "gpt"), ("emb", None), ("pub", None)])
def test_resolving_an_alias_reads_plain_and_hidden_aliases(
    make_core: MakeCore, name: str, expected: str | None
) -> None:
    assert _core(make_core).registry().resolve(name, kind="alias") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fast", True),
        ("hid", True),
        ("gpt", True),
        ("a", True),
        ("pub", False),
        ("openai/gpt-x", False),
        ("nope", False),
    ],
)
def test_recognized_names_are_groups_aliases_and_deployment_ids(make_core: MakeCore, name: str, expected: bool) -> None:
    assert _core(make_core).registry().resolve(name, kind="recognized") is expected


@pytest.mark.parametrize(
    ("name", "team_id", "expected"),
    [
        ("fast", None, "openai/a"),
        ("gpt", None, "openai/text-embedding-3-small"),
        ("pub", "TA", "openai/team-model"),
        ("pub", None, None),
        ("openai/gpt-x", None, None),
        ("a", None, None),
    ],
)
def test_upstream_is_the_first_routable_deployments_litellm_model(
    make_core: MakeCore, name: str, team_id: str | None, expected: str | None
) -> None:
    assert _core(make_core).registry().resolve(name, kind="upstream", team_id=team_id) == expected


def test_upstream_skips_a_blocked_deployment(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("half", "x2", model_info={"blocked": True}), deployment("half", "x3")])

    assert core.registry().resolve("half", kind="upstream") == "openai/x3"


@pytest.mark.parametrize(
    ("served", "expected"), [("gpt", "gpt"), ("emb", "fast"), (None, "fast")], ids=["target", "other", "none"]
)
def test_the_served_name_is_the_served_model_only_when_the_name_routes_to_it(
    make_core: MakeCore, served: str | None, expected: str
) -> None:
    assert _core(make_core).registry().resolve("fast", kind="served", served_model=served) == expected


def test_group_of_id_lists_the_deployments_group_including_alias_rows(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert names_and_ids(registry.resolve("a", kind="group_of_id")) == [("gpt", "a"), ("gpt", "b"), ("gpt", "e")]
    assert registry.resolve("gpt", kind="group_of_id") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gpt", "gpt"),
        ("a", "gpt"),
        ("fast", "fast"),
        ("nope", None),
        (None, None),
        ("", None),
    ],
)
def test_from_id_resolves_a_group_name_or_deployment_id(
    make_core: MakeCore, name: str | None, expected: str | None
) -> None:
    assert _core(make_core).registry().resolve(name, kind="from_id") == expected


def test_from_id_matches_a_litellm_model_suffix_and_the_first_matching_row_wins(make_core: MakeCore) -> None:
    """Odd but current: the alias row ``gpt -> emb`` is listed first, so ``emb``'s model resolves to ``gpt``."""
    registry = _core(make_core).registry()

    assert registry.resolve("text-embedding-3-small", kind="from_id") == "gpt"
    assert registry.resolve("openai/text-embedding-3-small", kind="from_id") == "gpt"


def test_from_id_suffix_match_on_a_plain_core(make_core: MakeCore) -> None:
    core = make_core(
        model_list=[deployment("veo", "v", model="vertex_ai/veo-2.0", vertex_project="p", vertex_location="l")]
    )

    assert core.registry().resolve("veo-2.0", kind="from_id") == "veo"


def test_from_id_with_a_provider_returns_the_provider_prefixed_name_a_wildcard_serves(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.resolve("zzz", kind="from_id", custom_llm_provider="openai") == "openai/zzz"
    assert registry.resolve("zzz", kind="from_id", custom_llm_provider="anthropic") is None


@pytest.mark.parametrize(
    ("name", "team_id", "provider", "expected"),
    [
        ("pub", "TA", None, "pub"),
        ("gpt", "TA", None, None),
        (None, "TA", None, "pub"),
        (None, "TA", "openai", "pub"),
        (None, "TB", "openai", None),
        (None, "TB", "anthropic", "pub"),
        (None, "nobody", None, None),
    ],
)
def test_team_model_maps_a_teams_public_name_or_picks_its_first_deployment(
    make_core: MakeCore, name: str | None, team_id: str, provider: str | None, expected: str | None
) -> None:
    registry = _core(make_core).registry()

    assert registry.resolve(name, kind="team_model", team_id=team_id, custom_llm_provider=provider) == expected


def test_team_model_without_a_team_id_raises(make_core: MakeCore) -> None:
    with pytest.raises(TypeError, match="needs a team_id"):
        _core(make_core).registry().resolve("pub", kind="team_model")


def test_an_unknown_resolve_kind_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown resolve kind 'bogus'"):
        _core(make_core).registry().resolve("gpt", kind="bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid kind on purpose


# -- includes -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("team_id", "expected"), [(None, False), ("TA", True), ("TB", False)])
def test_a_team_deployment_is_usable_only_by_its_own_team(
    make_core: MakeCore, team_id: str | None, expected: bool
) -> None:
    registry = _core(make_core).registry()
    team_entry = registry.all()[4]

    assert registry.includes(team_entry, team_id=team_id) is expected


def test_a_shared_deployment_is_usable_by_any_team(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert all(registry.includes(registry.all()[0], team_id=team) for team in (None, "TA", "TB"))


@pytest.mark.parametrize(
    ("model_name", "team_id", "expected"),
    [("pub", "TA", True), ("pub", "TB", False), ("int-a", None, True), ("int-a", "TB", False), ("gpt", "TA", False)],
)
def test_includes_with_a_model_name_matches_the_public_or_internal_name(
    make_core: MakeCore, model_name: str, team_id: str | None, expected: bool
) -> None:
    registry = _core(make_core).registry()

    assert registry.includes(registry.all()[4], model_name=model_name, team_id=team_id) is expected


# -- access groups --------------------------------------------------------------------------------


def test_access_groups_map_each_group_to_the_model_names_that_carry_it(make_core: MakeCore) -> None:
    assert _core(make_core).registry().access_groups() == {
        "ag1": ["fast", "fast", "gpt", "gpt"],
        "ag2": ["fast", "gpt", "gpt", "emb"],
        "wild": ["openai/*"],
    }


def test_access_groups_of_one_model_read_its_listing(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert dict(registry.access_groups(model_name="gpt")) == {"ag1": ["gpt", "gpt"], "ag2": ["gpt", "gpt"]}
    assert dict(registry.access_groups(model_name="openai/gpt-x")) == {"wild": ["openai/gpt-x"]}
    assert dict(registry.access_groups(model_name="nope")) == {}


def test_access_groups_filtered_to_one_group(make_core: MakeCore) -> None:
    assert dict(_core(make_core).registry().access_groups(model_access_group="ag2")) == {
        "ag2": ["fast", "gpt", "gpt", "emb"]
    }


def test_access_groups_follow_a_deployment_added_later(make_core: MakeCore) -> None:
    core = _core(make_core)
    assert "late" not in core.registry().access_groups()

    core.add_deployment(  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
        typed("new", "n", model_info={"access_groups": ["late"]})
    )

    assert core.registry().access_groups()["late"] == ["new"]


@pytest.mark.parametrize(("group", "expected"), [("wild", True), ("ag1", False), ("nope", False)])
def test_an_access_group_is_a_wildcard_route_when_one_of_its_models_is_a_pattern(
    make_core: MakeCore, group: str, expected: bool
) -> None:
    assert _core(make_core).registry().resolve(group, kind="wildcard_access_group") is expected


# -- blocked and unhealthy groups -----------------------------------------------------------------


def _blocked_core(make_core: MakeCore) -> RouterCore:
    return make_core(
        model_list=[
            deployment("blk", "x1", model_info={"blocked": True}),
            deployment("half", "x2", model_info={"blocked": True}),
            deployment("half", "x3"),
            deployment("ok", "x4"),
        ],
        model_group_alias={"blk-alias": "blk"},
    )


def test_fully_blocked_names_are_groups_whose_every_deployment_is_blocked(make_core: MakeCore) -> None:
    assert _blocked_core(make_core).health().fully_blocked_model_names() == {"blk", "blk-alias"}


def test_is_blocked_and_all_blocked_read_model_info_blocked(make_core: MakeCore) -> None:
    core = _blocked_core(make_core)
    health = core.health()
    entries = core.registry().all()

    blocked = core.registry().deployment("x1")
    open_ = core.registry().deployment("x3")
    assert blocked is not None
    assert open_ is not None
    assert (health.is_blocked(blocked), health.is_blocked(open_)) == (True, False)
    assert health.all_blocked(entries[:1]) is True
    assert health.all_blocked(entries[1:3]) is False
    assert health.all_blocked([]) is False


@pytest.mark.asyncio
async def test_fully_unhealthy_names_are_groups_whose_every_deployment_failed_its_health_check(
    make_core: MakeCore,
) -> None:
    core = make_core(
        model_list=[
            deployment("sick", "s1"),
            deployment("half", "h1"),
            deployment("half", "h2"),
            team_deployment("int", "t1", "TA", "team-sick"),
        ]
    )
    now = time.time()
    core.health().health_checks.set_deployment_health_states(
        {
            "s1": {"is_healthy": False, "timestamp": now, "reason": "down"},
            "h1": {"is_healthy": False, "timestamp": now, "reason": "down"},
            "h2": {"is_healthy": True, "timestamp": now, "reason": ""},
            "t1": {"is_healthy": False, "timestamp": now, "reason": "down"},
        }
    )

    assert await core.health().fully_unhealthy_model_names() == {"sick", "int", "team-sick"}


@pytest.mark.asyncio
async def test_no_health_state_hides_nothing(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    assert await core.health().fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_a_stale_unhealthy_state_hides_nothing(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])
    stale = time.time() - 10 * core.health().health_checks.staleness_threshold
    core.health().health_checks.set_deployment_health_states(
        {"a": {"is_healthy": False, "timestamp": stale, "reason": "x"}}
    )

    assert await core.health().fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_an_allowed_fails_policy_hides_no_unhealthy_group(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")], allowed_fails_policy={"RateLimitErrorAllowedFails": 2})
    core.health().health_checks.set_deployment_health_states(
        {"a": {"is_healthy": False, "timestamp": time.time(), "reason": "x"}}
    )

    assert await core.health().fully_unhealthy_model_names() == set()


# -- credentials ------------------------------------------------------------------------------------


def _credential_core(make_core: MakeCore) -> RouterCore:
    return make_core(
        model_list=[
            deployment("gpt", "a", api_base="https://a.example", temperature=0.2),
            deployment("plain", "p", model="gpt-4o-mini"),
            deployment("claude", "c", model="claude-3-haiku", custom_llm_provider="anthropic"),
            deployment("blk", "x", model_info={"blocked": True}),
            deployment("openai/*", "w", model="openai/*"),
            team_deployment("shared", "ta", "TA", "team-name", api_key="sk-team-a"),
            team_deployment("team-wild", "tw", "TA", "openai/*", model="openai/*", api_key="sk-team-wild"),
        ]
    )


def test_credentials_are_the_credential_params_only(make_core: MakeCore) -> None:
    assert _credential_core(make_core).registry().credentials("a") == {
        "api_key": "sk-test",
        "api_base": "https://a.example",
    }


def test_credentials_of_an_unknown_or_blocked_deployment_are_none(make_core: MakeCore) -> None:
    registry = _credential_core(make_core).registry()

    assert registry.credentials("missing") is None
    assert registry.credentials("x") is None
    assert registry.credentials("x", form="with_provider") is None
    assert registry.credentials("x", form="deployment") is None


@pytest.mark.parametrize(
    ("lookup", "provider"),
    [("a", "openai"), ("p", "openai"), ("c", "anthropic")],
    ids=["prefix", "default", "explicit"],
)
def test_credentials_with_provider_add_the_model_and_its_provider(
    make_core: MakeCore, lookup: str, provider: str
) -> None:
    registry = _credential_core(make_core).registry()

    found = registry.credentials(lookup, form="with_provider")
    deployment_found = registry.deployment(lookup)
    assert found is not None
    assert deployment_found is not None
    assert (found["model"], found["custom_llm_provider"]) == (deployment_found.litellm_params.model, provider)


@pytest.mark.parametrize(
    ("lookup", "team_id", "expected"),
    [
        ("a", None, "a"),
        ("gpt", None, "a"),
        ("team-name", "TA", "ta"),
        ("team-name", None, None),
        ("shared", "TB", None),
        ("shared", "TA", "ta"),
        ("openai/gpt-x", None, "w"),
        ("openai/gpt-x", "TA", "tw"),
        ("openai/gpt-x", "TB", "w"),
        ("nope", None, None),
    ],
)
def test_the_credential_deployment_is_found_by_id_group_team_name_then_wildcard(
    make_core: MakeCore, lookup: str, team_id: str | None, expected: str | None
) -> None:
    found = _credential_core(make_core).registry().credentials(lookup, form="deployment", team_id=team_id)

    assert (None if found is None else found.model_info.id) == expected


def test_credentials_with_provider_for_a_wildcard_name_carry_the_requested_model(make_core: MakeCore) -> None:
    found = _credential_core(make_core).registry().credentials("openai/gpt-x", form="with_provider", team_id="TA")

    assert found is not None
    assert (found["api_key"], found["model"], found["custom_llm_provider"]) == (
        "sk-team-wild",
        "openai/gpt-x",
        "openai",
    )


def test_an_unknown_credential_form_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown credential form 'bogus'"):
        _credential_core(make_core).registry().credentials("a", form="bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid form on purpose


# -- single deployments and the raw list ------------------------------------------------------------


def test_deployment_dict_is_the_live_model_list_entry(make_core: MakeCore) -> None:
    registry = _core(make_core).registry()

    assert registry.deployment_dict("a") is registry.all()[0]
    assert registry.deployment_dict("missing") is None


def test_a_looked_up_deployment_is_a_copy(make_core: MakeCore) -> None:
    core = _core(make_core)
    found = core.registry().deployment("a")
    assert found is not None

    found.litellm_params.api_base = "https://mutated.example"

    assert core.registry().all()[0]["litellm_params"].get("api_base") is None


def test_a_visible_dict_alias_is_listed_like_a_plain_alias(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")], model_group_alias={"vis": {"model": "gpt", "hidden": False}})

    assert names_and_ids(core.registry().deployments()) == [("vis", "a"), ("gpt", "a")]
    assert core.registry().names() == ["vis", "gpt"]


def test_the_deprecated_generate_model_id_is_the_registry_id_hash(make_core: MakeCore) -> None:
    from litellm.router_api import deployment_id

    core = _core(make_core)

    with pytest.warns(DeprecationWarning, match="generate_model_id is deprecated"):
        generated = core.generate_model_id("g", {"model": "x"})  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public

    assert generated == deployment_id("g", {"model": "x"})


def test_assigning_model_list_swaps_and_reindexes_the_list(make_core: MakeCore) -> None:
    """The replaced deployment's id finds nothing; the new entry is found by its own id and name."""
    core = make_core(model_list=[deployment("gpt", "a")])
    version = core.registry().version

    with pytest.warns(DeprecationWarning, match=r"Router\.model_list is deprecated"):  # pyright: ignore[reportUnknownMemberType]  # untyped in the checker's view
        core.model_list = [deployment("n", "n")]  # pyright: ignore[reportAttributeAccessIssue]  # deprecated attribute, still assignable

    assert core.registry().version == version + 1
    assert core.registry().names(kind="ids") == ["n"]
    assert core.registry().deployment("a") is None
    found = core.registry().deployment("n")
    assert found is not None
    assert found.model_name == "n"
    assert ids(core.registry().deployments("n")) == ["n"]
    assert core.registry().deployments("gpt") == []


def test_an_entry_assigned_without_an_id_is_not_served(make_core: MakeCore) -> None:
    """``model_list = [...]`` keeps entries as given; one without ``model_info.id`` is never routed to."""
    core = make_core(model_list=[deployment("gpt", "a")])
    entry: Final = {"model_name": "b", "litellm_params": {"model": "openai/gpt-4o-mini"}}

    with pytest.warns(DeprecationWarning, match=r"Router\.model_list is deprecated"):  # pyright: ignore[reportUnknownMemberType]  # untyped in the checker's view
        core.model_list = [entry]  # pyright: ignore[reportAttributeAccessIssue]  # deprecated attribute, still assignable

    assert core.registry().deployments("b") == []
    assert core.registry().names(kind="ids") == []
