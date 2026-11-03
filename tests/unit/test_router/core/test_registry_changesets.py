"""
The core's registry and its settings, which are static after construction: ``apply(ChangeSet)``
(and ``aapply``) is the only way to change them, and a direct attribute write is routed into one.
"""

import copy
from collections.abc import Callable, Mapping

import pytest

import litellm
from litellm.router_api import ApplyResult, ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import Deployment
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    context_window,
    deployment,
    server_error,
)

MakeCore = Callable[..., RouterCore]

MESSAGES = [{"role": "user", "content": "hi"}]


def _model_list() -> list[dict[str, object]]:  # mutable-ok: a fresh model_list
    return [deployment("gpt", "a"), deployment("gpt", "b"), deployment("emb", "e")]


def _fresh(error: Callable[[], BaseException]) -> Callable[..., BaseException]:
    """A scripted outcome raising a new exception per call: the core stamps num_retries on the instance it saw."""
    return lambda **_: error()


def _ids(deployments: object) -> list[str]:  # mutable-ok: a fresh list
    return [entry["model_info"]["id"] for entry in deployments]  # pyright: ignore[reportGeneralTypeIssues, reportIndexIssue]  # deployment dicts


def _deployment(model_name: str, deployment_id: str, **params: object) -> Deployment:  # kwargs-ok: litellm_params
    return Deployment(**deployment(model_name, deployment_id, **params))  # pyright: ignore[reportArgumentType]  # a model_list entry


# -- registry lookups after construction --------------------------------------------------------


@pytest.mark.parametrize(("deployment_id", "group"), [("b", "gpt"), ("e", "emb")])
def test_registry_finds_each_configured_deployment_by_id(make_core: MakeCore, deployment_id: str, group: str) -> None:
    registry = make_core(model_list=_model_list()).registry()

    found = registry.deployment(deployment_id)

    assert found is not None
    assert (found.model_name, found.model_info.id, found.litellm_params.model) == (
        group,
        deployment_id,
        f"openai/{deployment_id}",
    )
    assert registry.has_deployment(deployment_id)


def test_registry_has_no_deployment_for_an_unknown_id(make_core: MakeCore) -> None:
    registry = make_core(model_list=_model_list()).registry()

    assert registry.deployment("missing") is None
    assert not registry.has_deployment("missing")


@pytest.mark.parametrize(("group", "ids"), [("gpt", ["a", "b"]), ("unknown", [])])
def test_registry_lists_a_groups_deployments_in_configured_order(
    make_core: MakeCore, group: str, ids: list[str]
) -> None:
    registry = make_core(model_list=_model_list()).registry()

    assert _ids(registry.deployments(group)) == ids
    assert registry.names(group, kind="ids") == ids


def test_registry_names_every_group_and_deployment_id(make_core: MakeCore) -> None:
    registry = make_core(model_list=_model_list()).registry()

    assert registry.names(kind="group_set") == {"gpt", "emb"}
    assert registry.names(kind="ids") == ["a", "b", "e"]
    assert _ids(registry.deployments()) == ["a", "b", "e"]


@pytest.mark.parametrize(
    ("name", "kind", "resolved"),
    [
        ("fast", "group", "gpt"),
        ("fast", "alias", "gpt"),
        ("gpt", "alias", None),
        ("e", "from_id", "emb"),
        ("fast", "recognized", True),
        ("nope", "recognized", False),
    ],
)
def test_registry_resolves_names_through_groups_aliases_and_ids(
    make_core: MakeCore, name: str, kind: str, resolved: object
) -> None:
    registry = make_core(model_list=_model_list(), model_group_alias={"fast": "gpt"}).registry()

    assert registry.resolve(name, kind=kind) == resolved  # pyright: ignore[reportCallIssue, reportArgumentType]  # kind is one of the literals


def test_registry_aliases_are_the_configured_model_group_alias(make_core: MakeCore) -> None:
    registry = make_core(model_list=_model_list(), model_group_alias={"fast": "gpt"}).registry()

    assert registry.aliases() == {"fast": "gpt"}
    assert _ids(registry.deployments("fast")) == ["a", "b"]


def test_registry_version_is_unchanged_by_reads(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())
    before = core.registry().version

    core.registry().deployments("gpt")
    core.registry().names()
    core.registry().resolve("gpt")
    core.settings()

    assert core.registry().version == before


# -- registry().version on every write path -----------------------------------------------------


async def _apply_upsert_new(core: RouterCore) -> None:
    core.apply(ChangeSet(upsert=[_deployment("gpt", "c")]))


async def _apply_upsert_changed(core: RouterCore) -> None:
    core.apply(ChangeSet(upsert=[_deployment("gpt", "a", api_base="https://changed.example")]))


async def _apply_delete(core: RouterCore) -> None:
    core.apply(ChangeSet(delete=["a"]))


async def _aapply_upsert_new(core: RouterCore) -> None:
    await core.aapply(ChangeSet(upsert=[_deployment("gpt", "c")]))  # pyright: ignore[reportAttributeAccessIssue]  # aapply is on every core


async def _add_deployment(core: RouterCore) -> None:
    core.add_deployment(_deployment("gpt", "c"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core


async def _set_model_list(core: RouterCore) -> None:
    core.set_model_list([deployment("gpt", "c")])  # pyright: ignore[reportAttributeAccessIssue]  # public on the core


async def _upsert_deployment(core: RouterCore) -> None:
    with pytest.deprecated_call():
        core.upsert_deployment(_deployment("gpt", "c"))  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public


async def _delete_deployment(core: RouterCore) -> None:
    with pytest.deprecated_call():
        core.delete_deployment("a")  # pyright: ignore[reportAttributeAccessIssue]  # deprecated, still public


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("write", "ids_after"),
    [
        (_apply_upsert_new, ["a", "b", "e", "c"]),
        (_apply_upsert_changed, ["a", "b", "e"]),
        (_apply_delete, ["b", "e"]),
        (_aapply_upsert_new, ["a", "b", "e", "c"]),
        (_add_deployment, ["a", "b", "e", "c"]),
        (_set_model_list, ["c"]),
        (_upsert_deployment, ["a", "b", "e", "c"]),
        (_delete_deployment, ["b", "e"]),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
async def test_every_deployment_write_advances_the_registry_version(
    make_core: MakeCore, write: Callable[[RouterCore], object], ids_after: list[str]
) -> None:
    core = make_core(model_list=_model_list())
    before = core.registry().version

    await write(core)  # pyright: ignore[reportGeneralTypeIssues]  # each write is a coroutine function

    assert core.registry().version > before
    assert sorted(core.registry().names(kind="ids")) == sorted(ids_after)


@pytest.mark.asyncio
async def test_a_changed_upsert_is_what_the_registry_returns_afterwards(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    await _apply_upsert_changed(core)

    changed = core.registry().deployment("a")
    assert changed is not None
    assert changed.litellm_params.api_base == "https://changed.example"


@pytest.mark.parametrize(
    "changes",
    [
        ChangeSet(),
        ChangeSet(upsert=[_deployment("gpt", "a")]),
        ChangeSet(delete=["missing"]),
        ChangeSet(assign={"num_retries": 4, "cooldown_time": 9}),
        ChangeSet(settings={"num_retries": 4, "allowed_fails": 2}),
    ],
    ids=["empty", "identical_upsert", "unknown_delete", "assign", "settings"],
)
def test_a_changeset_that_changes_no_deployment_keeps_the_registry_version(
    make_core: MakeCore, changes: ChangeSet
) -> None:
    core = make_core(model_list=_model_list())
    before = core.registry().version

    result = core.apply(changes)

    assert core.registry().version == before
    assert result.version == before


def test_adding_a_deployment_whose_id_exists_changes_nothing(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())
    before = core.registry().version

    added = core.add_deployment(_deployment("emb", "a"))  # pyright: ignore[reportAttributeAccessIssue]  # public on the core

    assert added is None
    assert core.registry().version == before
    assert core.registry().resolve("a", kind="from_id") == "gpt"


# -- ApplyResult --------------------------------------------------------------------------------


def test_apply_result_carries_the_new_version_and_the_upserted_deployment(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    result = core.apply(ChangeSet(upsert=[_deployment("gpt", "c")]))

    assert isinstance(result, ApplyResult)
    assert result.version == core.registry().version
    assert [None if d is None else d.model_info.id for d in result.upserted] == ["c"]
    assert list(result.deleted) == []


def test_apply_result_reports_none_for_an_upsert_that_changed_nothing(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    result = core.apply(ChangeSet(upsert=[_deployment("gpt", "a"), _deployment("gpt", "c")]))

    assert [None if d is None else d.model_info.id for d in result.upserted] == [None, "c"]


def test_apply_result_lists_each_delete_in_order_with_none_for_an_unknown_id(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    result = core.apply(ChangeSet(delete=["e", "missing", "a"]))

    assert [None if d is None else d["model_info"]["id"] for d in result.deleted] == ["e", None, "a"]  # pyright: ignore[reportIndexIssue]  # the core returns the model_list dict
    assert core.registry().names(kind="ids") == ["b"]


def test_apply_upserts_before_it_deletes(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    result = core.apply(ChangeSet(upsert=[_deployment("gpt", "c")], delete=["c"]))

    assert [d is not None for d in result.upserted] == [True]
    assert [d is not None for d in result.deleted] == [True]
    assert not core.registry().has_deployment("c")


def test_settings_and_deployments_change_together_in_one_apply(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    result = core.apply(ChangeSet(assign={"cooldown_time": 17}, upsert=[_deployment("emb", "f")], delete=["b"]))

    assert result.version == core.registry().version
    assert core.settings().cooldown_time == 17
    assert core.registry().names("emb", kind="ids") == ["e", "f"]
    assert core.registry().names("gpt", kind="ids") == ["a"]


# -- settings changed by a changeset change later calls -----------------------------------------


def _assign(**values: object) -> ChangeSet:  # kwargs-ok: settings by name
    return ChangeSet(assign=values)


def _settings(**values: object) -> ChangeSet:  # kwargs-ok: settings by name
    return ChangeSet(settings=values)


CHANGESET_KINDS = pytest.mark.parametrize("changeset", [_assign, _settings], ids=["assign", "settings"])


@pytest.mark.asyncio
@CHANGESET_KINDS
@pytest.mark.parametrize("num_retries", [0, 3])
async def test_num_retries_from_a_changeset_governs_the_next_failing_call(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet], num_retries: int
) -> None:
    core = make_core(model_list=[deployment("gpt", "a")], num_retries=2)
    host.answer("openai/a", _fresh(server_error))
    with pytest.raises(litellm.InternalServerError):
        await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method
    host.calls.clear()

    core.apply(changeset(num_retries=num_retries))
    with pytest.raises(litellm.InternalServerError):
        await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["a"] * (num_retries + 1)
    assert core.settings().as_dict()["num_retries"] == num_retries


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_fallbacks_from_a_changeset_are_used_by_the_next_call(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("backup", "f")], num_retries=0)
    host.answer("openai/a", _fresh(server_error))
    with pytest.raises(litellm.InternalServerError):
        await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method
    host.calls.clear()

    core.apply(changeset(fallbacks=[{"gpt": ["backup"]}]))
    response = await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["a", "f"]
    assert response._hidden_params["additional_headers"]["x-litellm-attempted-fallbacks"] == 1
    assert core.settings().fallbacks == [{"gpt": ["backup"]}]


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_context_window_fallbacks_from_a_changeset_are_used_by_the_next_call(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("long", "l")], num_retries=0)
    host.answer("openai/a", _fresh(context_window))
    with pytest.raises(litellm.ContextWindowExceededError):
        await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method
    host.calls.clear()

    core.apply(changeset(context_window_fallbacks=[{"gpt": ["long"]}]))
    response = await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["a", "l"]
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "long"
    assert core.settings().context_window_fallbacks == [{"gpt": ["long"]}]


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_timeout_from_a_changeset_is_passed_on_the_next_call(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a")], timeout=30)
    await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    core.apply(changeset(timeout=12))
    await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert [call.kwargs["timeout"] for call in host.calls_to("acompletion")] == [30, 12]
    assert core.settings().as_dict()["timeout"] == 12


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_model_group_alias_from_a_changeset_routes_the_next_call(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("emb", "e")], model_group_alias={"fast": "gpt"})

    core.apply(changeset(model_group_alias={"fast": "emb"}))
    await core.acompletion(model="fast", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert host.deployments_called("acompletion") == ["e"]
    assert core.registry().aliases() == {"fast": "emb"}
    assert core.registry().resolve("fast") == "emb"


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_allowed_fails_and_cooldown_time_from_a_changeset_govern_the_next_cooldown(
    make_core: MakeCore, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")])
    assert core.health().record_failure("a", server_error(), 500) is False

    core.apply(changeset(allowed_fails=0, cooldown_time=42))

    assert core.health().record_failure("b", server_error(), 500) is True
    assert [
        (i, entry["cooldown_time"]) for i, entry in core.health().cooldowns.get_active_cooldowns(["a", "b"], None)
    ] == [("b", 42)]
    assert (core.settings().allowed_fails, core.settings().cooldown_time) == (0, 42)


@pytest.mark.asyncio
async def test_a_deployment_cooled_down_under_applied_allowed_fails_is_not_picked_again(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = make_core(model_list=[deployment("gpt", "a"), deployment("gpt", "b")], num_retries=0)
    core.apply(ChangeSet(assign={"allowed_fails": 0, "cooldown_time": 60}))
    host.answer("*", _fresh(server_error))

    for _ in range(2):
        with pytest.raises(litellm.InternalServerError):
            await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert sorted(host.deployments_called("acompletion")) == ["a", "b"]


@pytest.mark.asyncio
@CHANGESET_KINDS
async def test_tag_filtering_switched_on_by_a_changeset_restricts_the_next_pick(
    make_core: MakeCore, host: ScriptedHost, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=[deployment("gpt", "a", tags=["paid"]), deployment("gpt", "b", tags=["free"])])
    tagged = {"metadata": {"tags": ["paid"]}}
    assert sorted(_ids(await core.select("gpt", dict(tagged), kind="healthy"))) == ["a", "b"]  # pyright: ignore[reportCallIssue]  # select takes kind

    core.apply(changeset(enable_tag_filtering=True))

    assert _ids(await core.select("gpt", dict(tagged), kind="healthy")) == ["a"]  # pyright: ignore[reportCallIssue]  # select takes kind
    for _ in range(3):
        await core.acompletion(model="gpt", messages=MESSAGES, metadata={"tags": ["paid"]})  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method
    assert host.deployments_called("acompletion") == ["a", "a", "a"]
    assert core.settings().enable_tag_filtering is True


@pytest.mark.asyncio
async def test_routing_strategy_from_settings_is_reported_and_still_routes(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = make_core(model_list=[deployment("gpt", "a")])
    strategy = "least-busy"

    core.apply(ChangeSet(settings={"routing_strategy": strategy}))
    await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert core.settings().routing_strategy == strategy
    assert core.settings().as_dict()["routing_strategy"] == strategy
    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_default_litellm_params_from_a_changeset_are_sent_on_the_next_call(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = make_core(model_list=[deployment("gpt", "a")], default_litellm_params={"temperature": 0.2})

    core.apply(ChangeSet(assign={"default_litellm_params": {"temperature": 0.7}}))
    await core.acompletion(model="gpt", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert [call.kwargs.get("temperature") for call in host.calls_to("acompletion")] == [0.7]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("cooldown_time", 7.5),
        ("allowed_fails", 9),
        ("disable_cooldowns", True),
        ("content_policy_fallbacks", [{"gpt": ["emb"]}]),
    ],
)
def test_settings_reflect_an_assigned_field(make_core: MakeCore, name: str, value: object) -> None:
    core = make_core(model_list=_model_list())

    core.apply(ChangeSet(assign={name: value}))

    assert getattr(core.settings(), name) == value
    assert core.settings().is_set(name)


# -- what apply refuses -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bogus", "model_list", "_core_num_retries"])
def test_assigning_an_unknown_setting_raises_and_applies_nothing(make_core: MakeCore, name: str) -> None:
    core = make_core(model_list=_model_list(), num_retries=1)
    settings_before = core.settings()

    with pytest.raises(ValueError, match=name):
        core.apply(ChangeSet(assign={"num_retries": 5, "cooldown_time": 3, name: 1}, upsert=[_deployment("gpt", "c")]))

    assert core.settings().as_dict()["num_retries"] == 1
    assert core.settings() is settings_before
    assert not core.registry().has_deployment("c")


def test_apply_refuses_refresh_model_info_and_applies_nothing(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list(), num_retries=1)
    version_before = core.registry().version

    with pytest.raises(TypeError, match="aapply"):
        core.apply(ChangeSet(refresh_model_info=True, assign={"num_retries": 5}, delete=["a"]))

    assert core.settings().as_dict()["num_retries"] == 1
    assert core.registry().version == version_before
    assert core.registry().has_deployment("a")


# -- the settings() snapshot --------------------------------------------------------------------


def test_settings_snapshot_is_reused_between_reads(make_core: MakeCore) -> None:
    core = make_core(model_list=_model_list())

    assert core.settings() is core.settings()


@CHANGESET_KINDS
def test_applying_a_setting_replaces_the_snapshot_and_leaves_the_old_one_as_it_was(
    make_core: MakeCore, changeset: Callable[..., ChangeSet]
) -> None:
    core = make_core(model_list=_model_list(), cooldown_time=5)
    before = core.settings()

    core.apply(changeset(cooldown_time=11))

    after = core.settings()
    assert after is not before
    assert (before.cooldown_time, after.cooldown_time) == (5, 11)
    assert core.settings() is after


@pytest.mark.parametrize(
    "changes",
    [ChangeSet(upsert=[_deployment("gpt", "c")]), ChangeSet(delete=["a"])],
    ids=["upsert", "delete"],
)
def test_a_changeset_without_settings_keeps_the_snapshot(make_core: MakeCore, changes: ChangeSet) -> None:
    core = make_core(model_list=_model_list())
    before = core.settings()

    core.apply(changes)

    assert core.settings() is before


# -- a direct attribute write is a changeset ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_retries", 4),
        ("timeout", 9.0),
        ("retry_after", 3),
        ("allowed_fails", 1),
        ("cooldown_time", 11.0),
        ("fallbacks", [{"gpt": ["emb"]}]),
        ("context_window_fallbacks", [{"gpt": ["emb"]}]),
        ("enable_tag_filtering", True),
        ("model_group_alias", {"fast": "emb"}),
    ],
)
def test_writing_a_setting_attribute_is_the_same_as_assigning_it_in_a_changeset(
    make_core: MakeCore, name: str, value: object
) -> None:
    written = make_core(model_list=_model_list())
    applied = make_core(model_list=_model_list())
    snapshot_before = written.settings()
    version_before = written.registry().version

    setattr(written, name, copy.deepcopy(value))
    applied.apply(ChangeSet(assign={name: copy.deepcopy(value)}))

    assert written.settings().as_dict() == applied.settings().as_dict()
    assert written.registry().aliases() == applied.registry().aliases()
    if written.settings().is_set(name):
        assert getattr(written.settings(), name) == getattr(applied.settings(), name) == value
        assert written.settings() is not snapshot_before
    assert written.registry().version == version_before


# -- constructor arguments ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("model_list", _model_list()),
        ("fallbacks", [{"gpt": ["emb"]}]),
        ("model_group_alias", {"fast": "gpt"}),
        ("model_group_retry_policy", {"gpt": {"RateLimitErrorRetries": 1}}),
        ("allowed_fails_policy", {"RateLimitErrorAllowedFails": 2}),
    ],
)
def test_construction_leaves_the_arguments_it_was_given_unchanged(
    make_core: MakeCore, name: str, value: object
) -> None:
    given = copy.deepcopy(value)
    arguments: dict[str, object] = {"model_list": _model_list(), name: given}

    make_core(**arguments)

    assert given == value


@pytest.mark.asyncio
async def test_mutating_the_model_list_after_construction_changes_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    model_list = _model_list()
    core = make_core(model_list=model_list)
    version_before = core.registry().version

    model_list.append(deployment("gpt", "late"))
    model_list[2]["model_name"] = "gpt"
    model_list[0]["litellm_params"]["model"] = "openai/elsewhere"  # pyright: ignore[reportIndexIssue]  # a model_list entry
    model_list[0]["model_info"]["id"] = "renamed"  # pyright: ignore[reportIndexIssue]  # a model_list entry
    await core.aembedding(model="emb", input=["x"])  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint method

    assert core.registry().names(kind="ids") == ["a", "b", "e"]
    assert core.registry().names(kind="group_set") == {"gpt", "emb"}
    assert core.registry().version == version_before
    assert [(call.model, call.deployment_id) for call in host.calls_to("aembedding")] == [("openai/e", "e")]


# -- retire --------------------------------------------------------------------------------------


def _registered(host: ScriptedHost) -> Mapping[str, int]:
    return {name: len(callbacks) for name, callbacks in host.lists.items() if callbacks}


def test_retire_releases_every_callback_the_core_registered_on_its_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core = make_core(model_list=_model_list())
    assert _registered(host) != {}

    core.retire()

    assert _registered(host) == {}


def test_retire_leaves_another_cores_callbacks_registered(make_core: MakeCore, host: ScriptedHost) -> None:
    kept = make_core(model_list=_model_list())
    retired = make_core(model_list=[deployment("other", "o")])
    registered_by_both = _registered(host)

    retired.retire()

    assert _registered(host) != {}
    assert sum(_registered(host).values()) < sum(registered_by_both.values())
    kept.retire()
    assert _registered(host) == {}
