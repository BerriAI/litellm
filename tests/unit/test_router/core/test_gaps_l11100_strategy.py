"""
Gap tests for the strategy-router (pre-routing) deployments of ``router_python_core.py`` (about lines
11114-11550, plus the strategy hooks of set_model_list / upsert / delete): construction-time
validation of auto, adaptive and quality router markers, the (model_name, tags) slot each marker
holds and when it is released, the licensed capability ceiling, and the adaptive router's derived
inputs and post-call hooks.
"""

from typing import Final

import pytest

from litellm.router_api import ChangeSet
from litellm.types.router import Deployment
from tests.unit.test_router.core.gaps_l11100_helpers import (
    TIERS,
    MakeCore,
    RemovingHost,
    adaptive_marker,
    complexity_marker,
    marker,
    model_ids,
    registered_tags,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment


@pytest.fixture
def host() -> ScriptedHost:
    """The adaptive router's hook sync removes callbacks from every list, so the host must support it."""
    return RemovingHost()


PLAIN: Final = [deployment("small", "s"), deployment("big", "b")]

HEURISTIC_V2: Final = {"classifier_type": "heuristic_v2", "tiers": dict(TIERS)}

HEURISTIC_V2_LIMIT_MESSAGE: Final = (
    "At most 1 auto-router(s) using the heuristic_v2 classifier can be registered but this would make 2. "
    "Choose the heuristic v1 classifier for this router or remove an existing heuristic_v2 router."
)


# --- marker validation at construction -------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {},
            "auto_router_config_path or auto_router_config is required for auto-router deployments. "
            "Please set it in the litellm_params",
        ),
        (
            {"auto_router_config": "{}"},
            "auto_router_default_model is required for auto-router deployments. Please set it in the litellm_params",
        ),
        (
            {"auto_router_config": "{}", "auto_router_default_model": "big"},
            "auto_router_embedding_model is required for auto-router deployments. Please set it in the litellm_params",
        ),
    ],
    ids=["no-config", "no-default-model", "no-embedding-model"],
)
def test_an_auto_router_marker_missing_a_required_param_is_rejected(
    make_core: MakeCore, params: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match="is required") as raised:
        make_core(model_list=[*PLAIN, marker("auto", "m", "auto_router/semantic", **params)])
    assert str(raised.value) == message


def test_an_adaptive_router_marker_without_its_config_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="is required") as raised:
        make_core(model_list=[*PLAIN, marker("ad", "m", "auto_router/adaptive_router")])
    assert str(raised.value) == "adaptive_router_config is required for adaptive-router deployments."


def test_a_quality_router_marker_without_a_default_model_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="is required") as raised:
        make_core(model_list=[*PLAIN, marker("q", "m", "auto_router/quality_router")])
    assert str(raised.value) == (
        "quality_router_default_model is required for quality-router deployments, or set default_model in "
        "quality_router_config. Please configure it in the litellm_params"
    )


def test_a_quality_router_takes_its_default_model_from_its_config(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            *PLAIN,
            marker("q", "m", "auto_router/quality_router", quality_router_config={"default_model": "big"}),
        ]
    )
    registry: Final = core.registry().plugins("quality")

    assert registered_tags(core, "quality") == {"q": [()]}
    assert registry["q"][0].strategy.config.default_model == "big"


def test_quality_router_default_model_param_beats_the_one_in_its_config(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            *PLAIN,
            marker(
                "q",
                "m",
                "auto_router/quality_router",
                quality_router_default_model="small",
                quality_router_config={"default_model": "big"},
            ),
        ]
    )

    assert core.registry().plugins("quality")["q"][0].strategy.config.default_model == "small"


# --- the (model_name, tags) slot --------------------------------------------------------------------


@pytest.mark.parametrize(("tags", "shown"), [(None, "[]"), (["x"], "['x']")], ids=["untagged", "tagged"])
def test_a_second_marker_with_the_same_name_and_tags_is_rejected(
    make_core: MakeCore, tags: list[str] | None, shown: str
) -> None:
    extra: Final = {} if tags is None else {"tags": tags}
    with pytest.raises(ValueError, match="already exists") as raised:
        make_core(
            model_list=[*PLAIN, complexity_marker("smart", "m1", **extra), complexity_marker("smart", "m2", **extra)]
        )
    assert str(raised.value) == (
        f"Complexity-router deployment smart with tags {shown} already exists. "
        "Please use a different model name or set different tags."
    )


def test_ignore_invalid_deployments_drops_the_repeated_marker_and_keeps_the_first(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1"), complexity_marker("smart", "m2")],
        ignore_invalid_deployments=True,
    )

    assert model_ids(core) == ["s", "b", "m1"]
    assert registered_tags(core, "complexity") == {"smart": [()]}


def test_markers_of_one_name_with_different_tags_each_register(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1", tags=["a"]), complexity_marker("smart", "m2", tags=["b"])]
    )

    assert registered_tags(core, "complexity") == {"smart": [("a",), ("b",)]}


def test_adding_a_marker_whose_slot_is_taken_raises_and_adds_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*PLAIN, complexity_marker("smart", "m1")])

    with pytest.raises(ValueError, match="already exists"):
        core.add_deployment(Deployment(**complexity_marker("smart", "m2")))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    assert model_ids(core) == ["s", "b", "m1"]
    assert registered_tags(core, "complexity") == {"smart": [()]}


def test_deleting_a_marker_frees_its_slot_for_a_new_marker(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*PLAIN, complexity_marker("smart", "m1")])

    core.apply(ChangeSet(delete=["m1"]))
    assert registered_tags(core, "complexity") == {}
    result: Final = core.apply(ChangeSet(upsert=[Deployment(**complexity_marker("smart", "m2"))]))  # pyright: ignore[reportArgumentType]

    assert result.upserted[0] is not None
    assert registered_tags(core, "complexity") == {"smart": [()]}
    assert model_ids(core) == ["s", "b", "m2"]


def test_deleting_one_tagged_marker_keeps_the_other_tags_of_its_name(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1", tags=["a"]), complexity_marker("smart", "m2", tags=["b"])]
    )

    core.apply(ChangeSet(delete=["m1"]))

    assert registered_tags(core, "complexity") == {"smart": [("b",)]}


def test_deleting_a_plain_deployment_sharing_a_markers_name_keeps_the_marker(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*PLAIN, complexity_marker("smart", "m1"), deployment("smart", "p")])

    core.apply(ChangeSet(delete=["p"]))

    assert registered_tags(core, "complexity") == {"smart": [()]}


def test_an_upsert_that_retags_a_marker_moves_it_to_the_new_slot(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*PLAIN, complexity_marker("smart", "m1")])

    core.apply(ChangeSet(upsert=[Deployment(**complexity_marker("smart", "m1", tags=["t"]))]))  # pyright: ignore[reportArgumentType]

    assert registered_tags(core, "complexity") == {"smart": [("t",)]}
    assert model_ids(core) == ["s", "b", "m1"]


def test_set_model_list_drops_the_strategy_routers_of_the_old_list(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("smart", "m1"),
            marker("q", "m2", "auto_router/quality_router", quality_router_default_model="big"),
        ]
    )

    core.set_model_list(list(PLAIN))  # pyright: ignore[reportAttributeAccessIssue]

    assert registered_tags(core, "complexity") == {}
    assert registered_tags(core, "quality") == {}


# --- the licensed capability ceiling -----------------------------------------------------------------


def test_a_router_beyond_the_capability_limit_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="At most 1") as raised:
        make_core(
            model_list=[
                *PLAIN,
                complexity_marker("one", "m1", config=HEURISTIC_V2),
                complexity_marker("two", "m2", config=HEURISTIC_V2),
            ],
            auto_router_capability_limit=lambda: 1,
        )
    assert str(raised.value) == HEURISTIC_V2_LIMIT_MESSAGE


def test_routers_within_the_capability_limit_or_without_the_capability_register(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("one", "m1", config=HEURISTIC_V2),
            complexity_marker("two", "m2"),
            complexity_marker("three", "m3"),
        ],
        auto_router_capability_limit=lambda: 1,
    )

    assert registered_tags(core, "complexity") == {"one": [()], "two": [()], "three": [()]}


def test_without_a_capability_limit_any_number_of_gated_routers_register(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("one", "m1", config=HEURISTIC_V2),
            complexity_marker("two", "m2", config=HEURISTIC_V2),
        ]
    )

    assert sorted(registered_tags(core, "complexity")) == ["one", "two"]


def test_the_capability_limit_is_read_again_on_every_registration(make_core: MakeCore) -> None:
    limit: Final = {"value": 2}
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("one", "m1", config=HEURISTIC_V2),
            complexity_marker("two", "m2", config=HEURISTIC_V2),
        ],
        auto_router_capability_limit=lambda: limit["value"],
    )
    limit["value"] = 1

    with pytest.raises(ValueError, match="At most 1") as raised:
        core.add_deployment(Deployment(**complexity_marker("three", "m3", config=HEURISTIC_V2)))  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    assert str(raised.value) == HEURISTIC_V2_LIMIT_MESSAGE.replace("would make 2", "would make 3")


def test_a_failed_edit_rolls_back_to_the_serving_router_even_above_a_tightened_limit(make_core: MakeCore) -> None:
    """The edited router is popped first, so the edit itself counts 1 other + itself = 2 > 1 and fails;
    the rollback re-adds the previous router with the ceiling switched off."""
    limit: Final = {"value": 2}
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("one", "m1", config=HEURISTIC_V2),
            complexity_marker("two", "m2", config=HEURISTIC_V2),
        ],
        auto_router_capability_limit=lambda: limit["value"],
        ignore_invalid_deployments=True,
    )
    limit["value"] = 1
    edited: Final = complexity_marker("two", "m2", config=HEURISTIC_V2, tags=["edited"])

    result: Final = core.apply(ChangeSet(upsert=[Deployment(**edited)]))  # pyright: ignore[reportArgumentType]

    assert list(result.upserted) == [None]
    assert model_ids(core) == ["s", "b", "m1", "m2"]
    assert registered_tags(core, "complexity") == {"one": [()], "two": [()]}
    restored: Final = core.registry().deployment("m2")
    assert restored is not None
    assert restored.litellm_params.tags is None


def test_a_failed_edit_without_ignore_invalid_deployments_raises_and_restores_the_router(
    make_core: MakeCore,
) -> None:
    limit: Final = {"value": 2}
    core: Final = make_core(
        model_list=[
            *PLAIN,
            complexity_marker("one", "m1", config=HEURISTIC_V2),
            complexity_marker("two", "m2", config=HEURISTIC_V2),
        ],
        auto_router_capability_limit=lambda: limit["value"],
    )
    limit["value"] = 1

    with pytest.raises(ValueError, match="At most 1") as raised:
        core.apply(
            ChangeSet(upsert=[Deployment(**complexity_marker("two", "m2", config=HEURISTIC_V2, tags=["edited"]))])  # pyright: ignore[reportArgumentType]
        )
    assert str(raised.value) == HEURISTIC_V2_LIMIT_MESSAGE
    assert model_ids(core) == ["s", "b", "m1", "m2"]
    assert registered_tags(core, "complexity") == {"one": [()], "two": [()]}


# --- the adaptive router -------------------------------------------------------------------------------


def _adaptive_list() -> list[dict[str, object]]:  # mutable-ok: a model_list
    return [
        adaptive_marker("ar", "ar1", ["x", "y", "z"]),
        deployment(
            "x",
            "x",
            input_cost_per_token=0.5,
            model_info={"input_cost_per_token": 0.9, "adaptive_router_preferences": {"quality_tier": 2}},
        ),
        deployment("y", "y", model_info={"input_cost_per_token": 0.25}),
    ]


def _hook_counts(host: ScriptedHost) -> tuple[int, int]:
    from litellm.router_strategy.adaptive_router.hooks import (
        AdaptiveRouterPostCallHook,
        AdaptiveRouterResponseHeaderHook,
    )

    callbacks: Final = host.lists["callbacks"]
    return (
        sum(isinstance(callback, AdaptiveRouterPostCallHook) for callback in callbacks),
        sum(isinstance(callback, AdaptiveRouterResponseHeaderHook) for callback in callbacks),
    )


def test_an_adaptive_router_is_built_after_the_whole_list_from_the_models_it_names(make_core: MakeCore) -> None:
    """The marker is listed before x and y, yet sees them: litellm_params pricing wins over model_info
    pricing, preferences come from model_info, and z (not configured) gets neither."""
    core: Final = make_core(model_list=_adaptive_list())
    registry: Final = core.registry().plugins("adaptive")

    assert registered_tags(core, "adaptive") == {"ar": [()]}
    strategy: Final = registry["ar"][0].strategy
    assert strategy.model_to_cost == {"x": 0.5, "y": 0.25}
    assert {name: prefs.quality_tier for name, prefs in strategy.model_to_prefs.items()} == {"x": 2}


def test_an_adaptive_router_registers_one_post_call_hook_and_a_header_hook(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_adaptive_list())
    if not registered_tags(core, "adaptive"):
        pytest.fail("no adaptive router was registered")

    post_call, header = _hook_counts(host)
    assert post_call == 1
    assert header == 1


def test_resyncing_an_adaptive_router_keeps_a_single_header_hook(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_adaptive_list())

    core.apply(ChangeSet(upsert=[Deployment(**adaptive_marker("ar", "ar1", ["y"]))]))  # pyright: ignore[reportArgumentType]
    core.apply(ChangeSet(upsert=[Deployment(**adaptive_marker("ar", "ar1", ["x"]))]))  # pyright: ignore[reportArgumentType]

    assert _hook_counts(host) == (1, 1)


def test_deleting_an_adaptive_router_releases_it_its_post_call_hook_and_the_header_hook(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_adaptive_list())

    core.apply(ChangeSet(delete=["ar1"]))

    assert registered_tags(core, "adaptive") == {}
    assert _hook_counts(host) == (0, 0)


def test_retiring_the_last_adaptive_router_on_the_host_removes_the_header_hook(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_adaptive_list())

    core.retire()

    assert _hook_counts(host) == (0, 0)


def test_retiring_one_adaptive_router_keeps_the_header_hook_for_another_on_the_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    first: Final = make_core(model_list=_adaptive_list())
    second: Final = make_core(model_list=_adaptive_list())

    first.retire()

    assert _hook_counts(host)[1] == 1
    second.retire()
    assert _hook_counts(host) == (0, 0)


def test_a_complexity_router_with_adaptive_enabled_also_registers_an_adaptive_router(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1", config={"tiers": TIERS, "adaptive": True})]
    )

    assert registered_tags(core, "complexity") == {"smart": [()]}
    assert registered_tags(core, "adaptive") == {"smart": [()]}


def test_deleting_an_adaptive_complexity_router_releases_both_registrations(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1", config={"tiers": TIERS, "adaptive": True})]
    )

    core.apply(ChangeSet(delete=["m1"]))

    assert registered_tags(core, "complexity") == {}
    assert registered_tags(core, "adaptive") == {}


def test_upserting_an_adaptive_complexity_router_registers_its_adaptive_companion(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PLAIN))

    core.apply(
        ChangeSet(upsert=[Deployment(**complexity_marker("smart", "m1", config={"tiers": TIERS, "adaptive": True}))])  # pyright: ignore[reportArgumentType]
    )

    assert registered_tags(core, "adaptive") == {"smart": [()]}


def test_an_upsert_that_turns_adaptive_off_releases_the_adaptive_companion(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*PLAIN, complexity_marker("smart", "m1", config={"tiers": TIERS, "adaptive": True})]
    )

    core.apply(ChangeSet(upsert=[Deployment(**complexity_marker("smart", "m1"))]))  # pyright: ignore[reportArgumentType]

    assert registered_tags(core, "complexity") == {"smart": [()]}
    assert registered_tags(core, "adaptive") == {}


def test_editing_an_adaptive_router_rebuilds_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_adaptive_list())

    core.apply(ChangeSet(upsert=[Deployment(**adaptive_marker("ar", "ar1", ["y"]))]))  # pyright: ignore[reportArgumentType]

    registry: Final = core.registry().plugins("adaptive")
    assert registered_tags(core, "adaptive") == {"ar": [()]}
    assert registry["ar"][0].strategy.model_to_cost == {"y": 0.25}
