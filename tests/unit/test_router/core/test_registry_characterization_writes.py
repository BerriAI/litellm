"""
Registry and settings writes beyond the version bookkeeping in ``test_registry_changesets.py``:
what ``add_deployment``, ``set_model_list``, upserts and deletes leave behind in every index, how
``update_settings`` / ``ChangeSet(settings=...)`` coerce and filter values, and the ``settings()``
and ``registry()`` views.
"""

import warnings

import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import Deployment, RetryPolicy
from tests.unit.test_router.core.registry_helpers import (
    MESSAGES,
    MakeCore,
    ids,
    names_and_ids,
    team_deployment,
    typed,
    unique_id,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment


def _core(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    """gpt (a, b), emb (e), the openai/* wildcard (w), team TA's ``pub`` (t1, internal name ``int``)."""
    return make_core(
        model_list=[
            deployment("gpt", "a"),
            deployment("gpt", "b"),
            deployment("emb", "e"),
            deployment("openai/*", "w", model="openai/*"),
            team_deployment("int", "t1", "TA", "pub"),
        ],
        **arguments,
    )


# -- add_deployment ---------------------------------------------------------------------------------


def test_add_deployment_returns_the_deployment_it_serves(make_core: MakeCore) -> None:
    core = _core(make_core)

    added = core.add_deployment(typed("new", "n"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert isinstance(added, Deployment)
    assert (added.model_name, added.model_info.id, added.litellm_params.model) == ("new", "n", "openai/n")


def test_an_added_deployment_is_appended_and_names_a_new_group(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.add_deployment(typed("new", "n"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert core.registry().names(kind="ids") == ["a", "b", "e", "w", "t1", "n"]
    assert "new" in core.registry().names(kind="group_set")
    assert core.registry().index("model_name")["new"] == [5]
    assert core.registry().names(kind="deployment_names")[-1] == "openai/n"


def test_an_added_deployment_without_an_id_gets_the_id_its_model_info_generated(make_core: MakeCore) -> None:
    core = _core(make_core)
    candidate = Deployment(model_name="new", litellm_params={"model": "openai/q", "api_key": "k"})  # pyright: ignore[reportArgumentType]  # litellm_params as a dict

    added = core.add_deployment(candidate)  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert added is not None
    assert added.model_info.id == candidate.model_info.id
    assert core.registry().has_deployment(str(candidate.model_info.id))


@pytest.mark.asyncio
async def test_an_added_deployment_serves_requests_for_its_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core = _core(make_core)

    core.add_deployment(typed("new", "n"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
    await core.acompletion(model="new", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["n"]


def test_an_added_wildcard_deployment_serves_matching_names(make_core: MakeCore) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])

    core.add_deployment(typed("anthropic/*", "x", model="anthropic/*"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert names_and_ids(core.registry().deployments("anthropic/claude-x")) == [("anthropic/claude-x", "x")]
    assert core.settings().provider_default_deployment_ids == ["x"]


def test_an_added_team_deployment_is_team_indexed(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.add_deployment(Deployment(**team_deployment("int-2", "t2", "TB", "pub-b")))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]  # public on the core

    assert core.registry().index("team_model")[("TB", "pub-b")] == [5]
    assert core.registry().names(kind="team_public") == frozenset({"pub", "pub-b"})
    assert ids(core.registry().deployments("pub-b", team_id="TB")) == ["t2"]


def test_adding_a_deployment_with_an_unknown_provider_raises_and_adds_nothing(make_core: MakeCore) -> None:
    core = _core(make_core)
    version = core.registry().version

    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        core.add_deployment(typed("bad", "bad", model="notaprovider/x"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert core.registry().names(kind="ids") == ["a", "b", "e", "w", "t1"]
    assert "bad" not in core.registry().names(kind="group_set")
    assert core.registry().version == version


def test_an_added_deployment_registers_its_custom_pricing(make_core: MakeCore) -> None:
    core = _core(make_core)
    model_id = unique_id()

    core.add_deployment(typed("priced", model_id, model="openai/gpt-4o", input_cost_per_token=3e-3))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert litellm.model_cost[model_id]["input_cost_per_token"] == 3e-3


# -- upsert -----------------------------------------------------------------------------------------


def test_a_changed_upsert_moves_the_deployment_to_the_end_of_the_list(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(upsert=[typed("gpt", "a", api_base="https://changed.example")]))

    assert core.registry().names(kind="ids") == ["b", "e", "w", "t1", "a"]
    assert core.registry().index("model_name") == {"gpt": [0, 4], "emb": [1], "openai/*": [2], "int": [3]}
    assert ids(core.registry().deployments("gpt")) == ["b", "a"]


def test_an_upsert_that_renames_the_group_moves_the_deployment_between_groups(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(upsert=[typed("emb", "a")]))

    assert ids(core.registry().deployments("gpt")) == ["b"]
    assert ids(core.registry().deployments("emb")) == ["e", "a"]


def test_an_upsert_that_empties_a_group_drops_the_group_name(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(upsert=[typed("emb2", "e")]))

    assert core.registry().names(kind="group_set") == {"gpt", "openai/*", "int", "emb2"}
    assert core.registry().deployments("emb") == []


def test_a_failing_upsert_of_an_existing_deployment_raises_and_leaves_it_removed(make_core: MakeCore) -> None:
    """Odd but current: without ignore_invalid_deployments the old deployment is popped and not restored."""
    core = _core(make_core)

    with pytest.raises(litellm.BadRequestError):
        core.apply(ChangeSet(upsert=[typed("gpt", "a", model="notaprovider/x")]))

    assert core.registry().names(kind="ids") == ["b", "e", "w", "t1"]


def test_a_failing_upsert_under_ignore_invalid_deployments_restores_the_previous_deployment(
    make_core: MakeCore,
) -> None:
    core = _core(make_core, ignore_invalid_deployments=True)

    result = core.apply(ChangeSet(upsert=[typed("gpt", "a", model="notaprovider/x")]))

    assert list(result.upserted) == [None]
    restored = core.registry().deployment("a")
    assert restored is not None
    assert restored.litellm_params.model == "openai/a"
    assert core.registry().names(kind="ids") == ["b", "e", "w", "t1", "a"]


def test_the_deprecated_upsert_deployment_warns_and_returns_the_deployment(make_core: MakeCore) -> None:
    core = _core(make_core)

    with pytest.warns(DeprecationWarning, match=r"upsert_deployment is deprecated; use .*apply\(ChangeSet\(upsert="):
        upserted = core.upsert_deployment(typed("gpt", "c"))  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public

    assert upserted is not None
    assert upserted.model_info.id == "c"


# -- delete -----------------------------------------------------------------------------------------


def test_delete_returns_the_removed_model_list_entry(make_core: MakeCore) -> None:
    core = _core(make_core)

    result = core.apply(ChangeSet(delete=["e"]))

    (removed,) = result.deleted
    assert isinstance(removed, dict)
    assert (removed["model_name"], removed["model_info"]["id"]) == ("emb", "e")


def test_deleting_the_last_deployment_of_a_group_drops_the_group(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(delete=["e"]))

    assert "emb" not in core.registry().names(kind="group_set")
    assert core.registry().deployments("emb") == []
    assert core.registry().index("model_name") == {"gpt": [0, 1], "openai/*": [2], "int": [3]}


def test_deleting_a_deployment_reindexes_the_ones_after_it(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(delete=["a"]))

    for deployment_id in ("b", "e", "w", "t1"):
        found = core.registry().deployment(deployment_id)
        assert found is not None
        assert found.model_info.id == deployment_id
    assert core.registry().index("team_model") == {("TA", "pub"): [3]}


def test_deleting_a_team_deployment_drops_its_team_index_and_public_name(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(delete=["t1"]))

    assert core.registry().index("team_model") == {}
    assert core.registry().names(kind="team_public") == frozenset()
    assert core.registry().deployments("pub", team_id="TA") == []


def test_deleting_a_wildcard_deployment_stops_it_serving_matching_names(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(delete=["w"]))

    assert core.registry().deployments("openai/gpt-x") == []


def test_deleting_a_wildcard_deployment_keeps_it_among_provider_default_deployment_ids(make_core: MakeCore) -> None:
    """Odd but current: the provider default id list is append-only."""
    core = _core(make_core)

    core.apply(ChangeSet(delete=["w"]))

    assert core.settings().provider_default_deployment_ids == ["w"]


@pytest.mark.asyncio
async def test_a_deleted_deployment_is_no_longer_called(make_core: MakeCore, host: ScriptedHost) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(delete=["a"]))
    for _ in range(3):
        await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["b", "b", "b"]


def test_the_deprecated_delete_deployment_warns_and_returns_none_for_an_unknown_id(make_core: MakeCore) -> None:
    core = _core(make_core)

    with pytest.warns(DeprecationWarning, match=r"delete_deployment is deprecated; use .*apply\(ChangeSet\(delete="):
        assert core.delete_deployment("missing") is None  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public


# -- set_model_list ---------------------------------------------------------------------------------


def test_set_model_list_replaces_every_deployment_and_group(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.set_model_list([deployment("new", "n")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert core.registry().names(kind="ids") == ["n"]
    assert core.registry().names(kind="group_set") == {"new"}
    assert core.registry().index("model_name") == {"new": [0]}
    assert core.registry().index("team_model") == {}
    assert core.registry().names(kind="team_public") == frozenset()


def test_set_model_list_keeps_the_aliases(make_core: MakeCore) -> None:
    core = _core(make_core, model_group_alias={"fast": "new"})

    core.set_model_list([deployment("new", "n")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert ids(core.registry().deployments("fast")) == ["n"]


def test_set_model_list_leaves_the_given_list_unchanged(make_core: MakeCore) -> None:
    core = _core(make_core)
    given = [{"model_name": "new", "litellm_params": {"model": "openai/n", "api_key": "k"}}]

    core.set_model_list(given)  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert given == [{"model_name": "new", "litellm_params": {"model": "openai/n", "api_key": "k"}}]


def test_set_model_list_keeps_the_old_wildcard_routes(make_core: MakeCore) -> None:
    """Odd but current: the wildcard pattern router is not reset, so a replaced wildcard still matches."""
    core = _core(make_core)

    core.set_model_list([deployment("new", "n")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert names_and_ids(core.registry().deployments("openai/gpt-x")) == [("openai/gpt-x", "w")]
    assert core.settings().provider_default_deployment_ids == ["w"]


def test_set_model_list_appends_to_the_deployment_names(make_core: MakeCore) -> None:
    """Odd but current: deployment_names keep the replaced deployments' models."""
    core = make_core(model_list=[deployment("gpt", "a")])

    core.set_model_list([deployment("new", "n")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert core.registry().names(kind="deployment_names") == ["openai/a", "openai/n"]


@pytest.mark.asyncio
async def test_after_set_model_list_requests_route_to_the_new_deployments(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = _core(make_core)

    core.set_model_list([deployment("gpt", "n")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core
    await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["n"]


# -- update_settings / ChangeSet(settings=...) ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "given", "stored"),
    [
        ("timeout", "5", 5),
        ("num_retries", "3", 3),
        ("retry_after", 2.9, 2),
        ("allowed_fails", "4", 4),
        ("cooldown_time", 7.5, 7),
    ],
)
def test_integer_settings_are_cast_to_int(make_core: MakeCore, name: str, given: object, stored: int) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(settings={name: given}))

    assert core.settings().as_dict()[name] == stored


def test_a_non_integer_value_for_an_integer_setting_raises(make_core: MakeCore) -> None:
    core = _core(make_core)

    with pytest.raises(ValueError, match="invalid literal for int"):
        core.apply(ChangeSet(settings={"timeout": "abc"}))


def test_a_setting_that_is_not_runtime_updatable_is_ignored(make_core: MakeCore) -> None:
    core = _core(make_core)
    before = dict(core.settings().as_dict())

    core.apply(ChangeSet(settings={"bogus": 1, "model_list": []}))

    assert dict(core.settings().as_dict()) == before
    assert core.registry().names(kind="ids") == ["a", "b", "e", "w", "t1"]


def test_a_retry_policy_dict_is_stored_as_a_retry_policy(make_core: MakeCore) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(settings={"retry_policy": {"RateLimitErrorRetries": 2}}))

    stored = core.settings().as_dict()["retry_policy"]
    assert isinstance(stored, RetryPolicy)
    assert stored.RateLimitErrorRetries == 2


@pytest.mark.parametrize(("given", "stored"), [("true", True), (True, True), ("false", False), ("yes-ish", False)])
def test_mid_stream_fallback_continuation_reads_flag_strings(make_core: MakeCore, given: object, stored: bool) -> None:
    core = _core(make_core)

    core.apply(ChangeSet(settings={"enable_mid_stream_fallback_continuation": given}))

    assert core.settings().as_dict()["enable_mid_stream_fallback_continuation"] is stored


def test_model_group_alias_from_settings_is_visible_to_group_info(make_core: MakeCore) -> None:
    core = _core(make_core)
    assert core.registry().group("quick") is None

    core.apply(ChangeSet(settings={"model_group_alias": {"quick": "emb"}}))

    info = core.registry().group("quick")
    assert info is not None
    assert info.model_group == "quick"


def test_the_deprecated_update_settings_warns_and_applies(make_core: MakeCore) -> None:
    core = _core(make_core)

    with pytest.warns(DeprecationWarning, match=r"update_settings is deprecated"):
        core.update_settings(cooldown_time=13)  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public

    assert core.settings().cooldown_time == 13


# -- the settings() and registry() views ------------------------------------------------------------


def test_settings_as_dict_lists_the_runtime_updatable_settings(make_core: MakeCore) -> None:
    core = _core(make_core, model_group_alias={"fast": "gpt"}, num_retries=2, cooldown_time=5)

    settings = core.settings().as_dict()

    assert set(settings) == {
        "routing_strategy_args",
        "routing_strategy",
        "allowed_fails",
        "cooldown_time",
        "num_retries",
        "timeout",
        "retry_after",
        "fallbacks",
        "context_window_fallbacks",
        "model_group_retry_policy",
        "retry_policy",
        "model_group_alias",
        "enable_weighted_failover",
        "enable_mid_stream_fallback_continuation",
        "enable_tag_filtering",
        "tag_routing_prefix",
        "fallback_on_unsupported_params",
        "routing_groups",
    }
    assert (settings["routing_strategy"], settings["num_retries"], settings["cooldown_time"]) == (
        "simple-shuffle",
        2,
        5,
    )
    assert settings["model_group_alias"] == {"fast": "gpt"}
    assert settings["routing_groups"] == []


def test_settings_defaults_on_a_fresh_core(make_core: MakeCore) -> None:
    settings = _core(make_core).settings()

    assert settings.routing_strategy == "simple-shuffle"
    assert settings.allowed_fails == 3
    assert settings.fallbacks is None
    assert settings.enable_tag_filtering is False
    assert settings.provider_default_deployment_ids == ["w"]


@pytest.mark.parametrize(("name", "expected"), [("cooldown_time", True), ("bogus", False), ("num_retries", False)])
def test_settings_is_set_names_only_settings_fields(make_core: MakeCore, name: str, expected: bool) -> None:
    assert _core(make_core).settings().is_set(name) is expected


def test_registry_view_is_reused(make_core: MakeCore) -> None:
    core = _core(make_core)

    assert core.registry() is core.registry()


def test_registry_all_is_the_live_model_list(make_core: MakeCore) -> None:
    core = _core(make_core)
    listed = core.registry().all()

    core.add_deployment(typed("new", "n"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert ids(listed) == ["a", "b", "e", "w", "t1", "n"]


def test_an_unknown_index_name_raises(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="unknown index 'bogus'"):
        _core(make_core).registry().index("bogus")  # pyright: ignore[reportCallIssue, reportArgumentType]  # an invalid name on purpose


# -- aapply(refresh_model_info) ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refreshing_model_info_with_nothing_to_discover_changes_nothing(make_core: MakeCore) -> None:
    core = _core(make_core)
    version = core.registry().version

    result = await core.aapply(ChangeSet(refresh_model_info=True))  # pyright: ignore[reportAttributeAccessIssue]  # aapply is on every core

    assert (result.version, list(result.upserted), list(result.deleted)) == (version, [], [])
    assert core.registry().version == version
    assert dict(core.registry().discovered_model_info("a")) == {}


@pytest.mark.asyncio
async def test_aapply_applies_upserts_and_deletes_like_apply(make_core: MakeCore) -> None:
    core = _core(make_core)

    result = await core.aapply(ChangeSet(upsert=[typed("gpt", "c")], delete=["e"]))  # pyright: ignore[reportAttributeAccessIssue]  # aapply is on every core

    assert [None if d is None else d.model_info.id for d in result.upserted] == ["c"]
    assert core.registry().names(kind="ids") == ["a", "b", "w", "t1", "c"]


@pytest.mark.asyncio
async def test_the_deprecated_arefresh_model_info_warns(make_core: MakeCore) -> None:
    core = _core(make_core)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await core.arefresh_model_info()  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public

    assert any("arefresh_model_info is deprecated" in str(w.message) for w in caught)
