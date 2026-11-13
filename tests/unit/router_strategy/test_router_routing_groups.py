"""
Tests for `routing_groups` — assigns named subsets of `model_name`s to a
per-group routing strategy. Models not claimed by an explicit group fall into
the implicit `"default"` group driven by the router's top-level
`routing_strategy` / `routing_strategy_args`.
"""

import asyncio
import datetime
import uuid
from collections.abc import Mapping
from typing import Final, Literal
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import litellm
from litellm import Router
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_api import RoutingContext
from litellm.router_strategy.least_busy import picks_reserve
from litellm.types.router import DeploymentTypedDict, FallbackAccessCheck, RoutingGroup
from litellm.utils import Rules, function_setup


def _model_list():
    return [
        {
            "model_name": "filtered-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "sk-test-1",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-1"},
        },
        {
            "model_name": "filtered-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "sk-test-2",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-2"},
        },
        {
            "model_name": "other-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "sk-test-3",
                "api_base": "https://example.invalid",
            },
            "model_info": {"id": "deploy-3"},
        },
    ]


def _build_router(routing_strategy="simple-shuffle", routing_groups=None):
    return Router(
        model_list=_model_list(),
        routing_strategy=routing_strategy,
        routing_groups=routing_groups,
    )


def test_overlapping_models_across_groups_raises():
    with pytest.raises(ValueError, match="appears in"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                },
                {
                    "group_name": "g2",
                    "models": ["filtered-model"],
                    "routing_strategy": "least-busy",
                },
            ],
        )


def test_reserved_default_group_name_raises():
    with pytest.raises(ValueError, match="reserved"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "default",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                }
            ],
        )


def test_invalid_strategy_in_group_raises():
    with pytest.raises(ValueError, match="Invalid routing_strategy"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "not-a-real-strategy",
                }
            ],
        )


def test_duplicate_group_name_raises():
    with pytest.raises(ValueError, match="duplicate"):
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model"],
                    "routing_strategy": "latency-based-routing",
                },
                {
                    "group_name": "g1",
                    "models": ["other-model"],
                    "routing_strategy": "least-busy",
                },
            ],
        )


def test_two_least_busy_groups_count_a_request_once(monkeypatch):
    """
    Least-busy counts a request up from the pre-call hooks on `litellm.input_callback` and
    back down from the success hooks on `litellm.callbacks`. The success list drops a second
    selector of the same class, so a pre-call list that kept both counted every request twice
    and released it once, and the deployment's in-flight count climbed until it looked pinned.
    """
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])

    router = _build_router(
        routing_strategy="least-busy",
        routing_groups=[
            {
                "group_name": "fast",
                "models": ["filtered-model"],
                "routing_strategy": "least-busy",
            }
        ],
    )
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "filtered-model"},
            "model_info": {"id": "deploy-1"},
        }
    }

    for callback in litellm.input_callback:
        if isinstance(callback, CustomLogger):
            callback.log_pre_api_call(model="filtered-model", messages=[], kwargs=kwargs)
    for callback in litellm.callbacks:
        if isinstance(callback, CustomLogger):
            callback.log_success_event(kwargs, None, None, None)

    assert router.cache.get_cache("filtered-model_request_count:deploy-1") == 0


def _filtered_model_pick(router: Router) -> str:
    deployment: Final = router.get_available_deployment(
        "filtered-model", request_kwargs=_proxy_shaped_request(model="filtered-model")
    )
    return deployment["model_info"]["id"]


def test_a_least_busy_group_pick_stops_counting_once_its_request_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router: Final = _build_router(
        routing_strategy="least-busy",
        routing_groups=[{"group_name": "fast", "models": ["filtered-model"], "routing_strategy": "least-busy"}],
    )

    request: Final = _proxy_shaped_request(model="filtered-model")
    with picks_reserve(True):
        first: Final = router.get_available_deployment("filtered-model", request_kwargs=request)["model_info"]["id"]
    kwargs: Final = {
        "litellm_params": {"metadata": {"model_group": "filtered-model"}, "model_info": {"id": first}},
        "litellm_call_id": request["litellm_call_id"],
    }
    for callback in litellm.input_callback:
        if isinstance(callback, CustomLogger):
            callback.log_pre_api_call(model="filtered-model", messages=[], kwargs=kwargs)
    for callback in litellm.callbacks:
        if isinstance(callback, CustomLogger):
            callback.log_success_event(kwargs, None, None, None)

    assert _filtered_model_pick(router) == first


def test_two_routers_in_one_process_each_count_their_own_requests(monkeypatch):
    """
    Least-busy hangs its counting off litellm's global callback lists, and those lists keep one
    logger per class unless the instances differ in a plain attribute. Two routers in one process
    (a second Router, or a per-request `user_config` one) therefore have to register separately:
    a second router whose selector is dropped counts nothing, reads zero for every deployment,
    and sends every request to whichever one is listed first.
    """
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])

    first = _build_router(routing_strategy="least-busy")
    second = _build_router(routing_strategy="least-busy")
    kwargs = {
        "litellm_params": {
            "metadata": {"model_group": "filtered-model"},
            "model_info": {"id": "deploy-1"},
        }
    }

    for callback in litellm.input_callback:
        if isinstance(callback, CustomLogger):
            callback.log_pre_api_call(model="filtered-model", messages=[], kwargs=kwargs)

    assert second.cache.get_cache("filtered-model_request_count:deploy-1") == 1
    assert (
        second.get_available_deployment(model="filtered-model", messages=[])["model_info"]["id"]
        == "deploy-2"
    )

    for callback in litellm.callbacks:
        if isinstance(callback, CustomLogger):
            callback.log_success_event(kwargs, None, None, None)

    assert first.cache.get_cache("filtered-model_request_count:deploy-1") == 0
    assert second.cache.get_cache("filtered-model_request_count:deploy-1") == 0


# ---------------------------------------------------------------------------
# Direct helper coverage
# ---------------------------------------------------------------------------


def _single_latency_group():
    return [{"group_name": "g1", "models": ["filtered-model"], "routing_strategy": "latency-based-routing"}]


def test_failed_routing_groups_update_does_not_poison_later_strategy_changes(monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router = _build_router(routing_groups=_single_latency_group())

    with pytest.raises(ValueError, match="appears in"):
        router.update_settings(
            routing_groups=[
                *_single_latency_group(),
                {"group_name": "g2", "models": ["filtered-model"], "routing_strategy": "least-busy"},
            ],
        )

    router.update_settings(routing_strategy="least-busy")

    assert list(router.routing_groups) == ["g1"]
    assert [g["group_name"] for g in router.get_settings()["routing_groups"]] == ["g1"]


def test_overlap_error_names_every_conflicting_model():
    with pytest.raises(ValueError, match="appears in") as exc_info:
        _build_router(
            routing_groups=[
                {
                    "group_name": "g1",
                    "models": ["filtered-model", "other-model"],
                    "routing_strategy": "latency-based-routing",
                },
                {
                    "group_name": "g2",
                    "models": ["filtered-model", "other-model"],
                    "routing_strategy": "least-busy",
                },
            ],
        )
    message = str(exc_info.value)
    assert "'filtered-model' appears in 'g1' and 'g2'" in message
    assert "'other-model' appears in 'g1' and 'g2'" in message


def _rpm_limited_model_list():
    return [
        {
            "model_name": "other-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "sk-test-3",
                "api_base": "https://example.invalid",
                "rpm": 1,
            },
            "model_info": {"id": "deploy-3"},
        },
    ]


@pytest.mark.asyncio
async def test_usage_based_v2_override_enforces_rpm_when_a_specific_deployment_is_requested():
    router = Router(model_list=_rpm_limited_model_list(), routing_strategy="simple-shuffle", num_retries=0)
    kwargs = {"model": "deploy-3", "messages": [{"role": "user", "content": "hi"}], "mock_response": "ok"}

    first = await router.acompletion(**kwargs, routing_strategy="usage-based-routing-v2")
    assert first.choices[0].message.content == "ok"
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(**kwargs, routing_strategy="usage-based-routing-v2")
    assert (await router.acompletion(**kwargs)).choices[0].message.content == "ok"


def test_sync_usage_based_v2_override_enforces_rpm_when_a_specific_deployment_is_requested():
    router = Router(model_list=_rpm_limited_model_list(), routing_strategy="simple-shuffle", num_retries=0)
    kwargs = {"model": "deploy-3", "messages": [{"role": "user", "content": "hi"}], "mock_response": "ok"}

    first = router.completion(**kwargs, routing_strategy="usage-based-routing-v2")
    assert first.choices[0].message.content == "ok"
    with pytest.raises(litellm.RateLimitError):
        router.completion(**kwargs, routing_strategy="usage-based-routing-v2")
    assert router.completion(**kwargs).choices[0].message.content == "ok"


def _pass_through_rpm_limited_model_list():
    deployment = _rpm_limited_model_list()[0]
    return [{**deployment, "litellm_params": {**deployment["litellm_params"], "use_in_pass_through": True}}]


@pytest.mark.asyncio
async def test_async_early_return_paths_run_the_override_pre_call_check():
    router = Router(model_list=_pass_through_rpm_limited_model_list(), routing_strategy="simple-shuffle")
    override = {"routing_strategy": "usage-based-routing-v2"}

    pinned = await router.async_get_available_deployment(
        model="other-model", request_kwargs={**override, "_encrypted_content_affinity_pinned": True}
    )
    assert pinned["model_info"]["id"] == "deploy-3"
    with pytest.raises(litellm.RateLimitError):
        await router.async_get_available_deployment_for_pass_through(model="deploy-3", request_kwargs=override)
    plain = await router.async_get_available_deployment_for_pass_through(model="deploy-3", request_kwargs={})
    assert plain["model_info"]["id"] == "deploy-3"


def test_sync_pass_through_specific_deployment_runs_the_override_pre_call_check():
    router = Router(model_list=_pass_through_rpm_limited_model_list(), routing_strategy="simple-shuffle")
    override = {"routing_strategy": "usage-based-routing-v2"}

    first = router.get_available_deployment_for_pass_through(model="deploy-3", request_kwargs=override)
    assert first["model_info"]["id"] == "deploy-3"
    with pytest.raises(litellm.RateLimitError):
        router.get_available_deployment_for_pass_through(model="deploy-3", request_kwargs=override)
    plain = router.get_available_deployment_for_pass_through(model="deploy-3", request_kwargs={})
    assert plain["model_info"]["id"] == "deploy-3"


def _two_deployment_model_list(**d1_params: object) -> list[dict[str, object]]:
    return [
        {
            "model_name": "grp",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test-1", "mock_response": "ok", **d1_params},
            "model_info": {"id": "d1"},
        },
        {
            "model_name": "grp",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test-2", "mock_response": "ok"},
            "model_info": {"id": "d2"},
        },
    ]


def _proxy_shaped_request(**data: object) -> dict[str, object]:
    """The proxy builds the request's `Logging` object before it hands the call to the router."""
    logging_obj, kwargs = function_setup(
        "acompletion",
        Rules(),
        datetime.datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        messages=[{"role": "user", "content": "hi"}],
        **data,
    )
    return {**kwargs, "litellm_logging_obj": logging_obj}


def test_override_matching_the_router_strategy_is_not_bound_twice():
    router = Router(model_list=_two_deployment_model_list(), routing_strategy="least-busy")
    request = _proxy_shaped_request(model="grp", routing_strategy="least-busy")

    router.get_available_deployment("grp", request_kwargs=request)

    assert request["litellm_logging_obj"].dynamic_input_callbacks is None


def _quality_group(strategy="latency-based-routing"):
    return [{"group_name": "quality", "models": ["filtered-model", "other-model"], "routing_strategy": strategy}]


async def _candidate_ids(router: Router, model: str) -> list[str]:
    deployments: Final = await router.async_get_healthy_deployments(model=model, request_kwargs={})
    assert isinstance(deployments, list)
    return sorted(d["model_info"]["id"] for d in deployments)


@pytest.mark.asyncio
async def test_group_name_is_callable_and_unions_member_deployments():
    router = _build_router(routing_groups=_quality_group())
    assert await _candidate_ids(router, "quality") == ["deploy-1", "deploy-2", "deploy-3"]


def test_group_name_appears_in_model_names_and_model_list():
    router = _build_router(routing_groups=_quality_group())
    assert "quality" in router.get_model_names()
    rows = router.get_model_list(model_name="quality")
    assert {r["model_name"] for r in rows} == {"quality"}
    assert sorted(r["model_info"]["id"] for r in rows) == ["deploy-1", "deploy-2", "deploy-3"]


@pytest.mark.asyncio
async def test_group_name_colliding_with_model_name_is_shadowed_with_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = _build_router(
            routing_groups=[
                {"group_name": "filtered-model", "models": ["other-model"], "routing_strategy": "latency-based-routing"}
            ]
        )
    assert any("shadowed" in record.getMessage() for record in caplog.records)
    assert router.get_routing_group("filtered-model") is None
    assert await _candidate_ids(router, "filtered-model") == ["deploy-1", "deploy-2"]


@pytest.mark.asyncio
async def test_group_name_colliding_with_model_group_alias_is_shadowed_with_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        router = Router(
            model_list=_model_list(),
            model_group_alias={"quality": "filtered-model"},
            routing_groups=_quality_group(),
        )
    assert any("shadowed" in record.getMessage() for record in caplog.records)
    assert router.get_routing_group("quality") is None
    assert await _candidate_ids(router, "quality") == ["deploy-1", "deploy-2"]


@pytest.mark.asyncio
async def test_real_model_added_later_shadows_group():
    router = _build_router(routing_groups=_quality_group())
    assert router.get_routing_group("quality") is not None

    from litellm.types.router import Deployment

    router.add_deployment(
        Deployment(
            model_name="quality",
            litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test-4", "api_base": "https://example.invalid"},
            model_info={"id": "deploy-shadow"},
        )
    )
    assert router.get_routing_group("quality") is None
    assert await _candidate_ids(router, "quality") == ["deploy-shadow"]

    router.delete_deployment(id="deploy-shadow")
    assert "quality" not in router.model_names
    assert router.get_routing_group("quality") is not None
    assert await _candidate_ids(router, "quality") == ["deploy-1", "deploy-2", "deploy-3"]


@pytest.mark.asyncio
async def test_group_with_no_member_deployments_raises_no_healthy():
    router = Router(
        model_list=_model_list(),
        routing_groups=[{"group_name": "empty-group", "models": ["ghost-model"], "routing_strategy": "simple-shuffle"}],
    )
    with pytest.raises(litellm.BadRequestError):
        await router.async_get_healthy_deployments(model="empty-group", request_kwargs={})


@pytest.mark.asyncio
async def test_alias_pointing_at_group_composes():
    router = Router(
        model_list=_model_list(),
        model_group_alias={"quality-alias": "quality"},
        routing_groups=_quality_group(),
    )
    assert await _candidate_ids(router, "quality-alias") == ["deploy-1", "deploy-2", "deploy-3"]


def test_model_group_info_reports_group():
    router = _build_router(routing_groups=_quality_group())
    info = router.get_model_group_info("quality")
    assert info is not None
    assert info.model_group == "quality"
    assert "openai" in info.providers


def test_routing_group_has_alternatives():
    router = _build_router(routing_groups=_quality_group())
    assert router.routing_group_has_alternatives("quality") is True
    assert router.routing_group_has_alternatives("filtered-model") is False
    assert router.routing_group_has_alternatives(None) is False

    solo_router = Router(
        model_list=_model_list(),
        routing_groups=[{"group_name": "solo-group", "models": ["other-model"], "routing_strategy": "simple-shuffle"}],
    )
    assert solo_router.routing_group_has_alternatives("solo-group") is False


@pytest.mark.asyncio
async def test_member_direct_call_unchanged_by_callable_groups():
    router = _build_router(routing_groups=_quality_group())
    assert await _candidate_ids(router, "other-model") == ["deploy-3"]


def test_update_settings_group_change_invalidates_model_group_info():
    router = _build_router(routing_groups=_quality_group())
    assert router.get_model_group_info("quality") is not None
    assert router.get_model_group_info("renamed-group") is None

    router.update_settings(
        routing_groups=[
            {"group_name": "renamed-group", "models": ["filtered-model"], "routing_strategy": "simple-shuffle"}
        ]
    )
    assert router.get_model_group_info("quality") is None
    info = router.get_model_group_info("renamed-group")
    assert info is not None
    assert info.model_group == "renamed-group"


def test_is_recognized_model_covers_every_virtual_model_kind():
    router = Router(
        model_list=_model_list(),
        model_group_alias={"my-alias": "filtered-model"},
        routing_groups=_quality_group(),
    )
    assert router.is_recognized_model("filtered-model") is True
    assert router.is_recognized_model("deploy-1") is True
    assert router.is_recognized_model("my-alias") is True
    assert router.is_recognized_model("quality") is True
    assert router.is_recognized_model("ghost") is False


def test_routing_group_has_alternatives_resolves_aliases():
    router = Router(
        model_list=_model_list(),
        model_group_alias={"quality-alias": "quality"},
        routing_groups=_quality_group(),
    )
    assert router.routing_group_has_alternatives("quality-alias") is True
    assert router.routing_group_has_alternatives("quality") is True


def test_group_rows_cache_invalidated_on_model_list_change():
    from litellm.types.router import Deployment

    router = _build_router(routing_groups=_quality_group())
    assert sum(1 for row in router.get_model_list() if row["model_name"] == "quality") == 3

    router.add_deployment(
        Deployment(
            model_name="filtered-model",
            litellm_params={"model": "openai/gpt-4o", "api_key": "sk-test-5", "api_base": "https://example.invalid"},
            model_info={"id": "deploy-4"},
        )
    )
    assert sum(1 for row in router.get_model_list() if row["model_name"] == "quality") == 4


async def _call_and_get_cooldowns(router, model):
    """Cool down the other members first, so the call has to land on deploy-3."""
    for model_id in ("deploy-1", "deploy-2"):
        router.cooldown_cache.add_deployment_to_cooldown(
            model_id=model_id, original_exception=RuntimeError("unavailable"), exception_status=500, cooldown_time=60
        )
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            mock_response="litellm.RateLimitError",
        )
    return [model_id for model_id, _ in await router.cooldown_cache.async_get_active_cooldowns(["deploy-3"], None)]


@pytest.mark.asyncio
async def test_group_call_429_registers_cooldown_end_to_end():
    router = Router(
        model_list=_model_list(),
        routing_groups=_quality_group("simple-shuffle"),
        num_retries=0,
        cooldown_time=60,
    )
    cooldown_ids = await _call_and_get_cooldowns(router, "quality")
    assert "deploy-3" in cooldown_ids


@pytest.mark.asyncio
async def test_alias_to_group_429_registers_cooldown_end_to_end():
    router = Router(
        model_list=_model_list(),
        model_group_alias={"quality-alias": "quality"},
        routing_groups=_quality_group("simple-shuffle"),
        num_retries=0,
        cooldown_time=60,
    )
    cooldown_ids = await _call_and_get_cooldowns(router, "quality-alias")
    assert "deploy-3" in cooldown_ids


@pytest.mark.asyncio
async def test_direct_single_deployment_member_429_keeps_exemption_end_to_end():
    router = Router(
        model_list=_model_list(),
        routing_groups=_quality_group("simple-shuffle"),
        num_retries=0,
        cooldown_time=60,
    )
    cooldown_ids = await _call_and_get_cooldowns(router, "other-model")
    assert "deploy-3" not in cooldown_ids


def test_group_rows_do_not_inherit_member_access_groups():
    router = Router(
        model_list=[
            {
                "model_name": "gated-member",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
                "model_info": {"id": "gated-1", "access_groups": ["restricted-team"]},
            }
        ],
        routing_groups=[
            {"group_name": "gated-group", "models": ["gated-member"], "routing_strategy": "simple-shuffle"}
        ],
    )
    access_groups = router.get_model_access_groups()
    assert "gated-group" not in access_groups.get("restricted-team", [])
    assert all("access_groups" not in (row.get("model_info") or {}) for row in router.get_model_list(model_name="gated-group"))
    assert "access_groups" in router.get_model_list(model_name="gated-member")[0]["model_info"]


def test_get_model_list_from_routing_groups_materializes_rows():
    router = _build_router(routing_groups=_quality_group())
    rows = router.get_model_list_from_routing_groups()
    assert {row["model_name"] for row in rows} == {"quality"}
    assert router.get_model_list_from_routing_groups() is rows

    named = router.get_model_list_from_routing_groups(model_name="quality")
    assert sorted(row["model_info"]["id"] for row in named) == ["deploy-1", "deploy-2", "deploy-3"]
    assert router.get_model_list_from_routing_groups(model_name="filtered-model") == ()


@pytest.mark.asyncio
async def test_group_call_429_cools_down_member_across_retries():
    router = Router(
        model_list=_model_list(),
        routing_groups=_quality_group("simple-shuffle"),
        num_retries=1,
        cooldown_time=60,
    )
    cooldown_ids = await _call_and_get_cooldowns(router, "quality")
    assert "deploy-3" in cooldown_ids


def _priority_group(
    name: str = "priority-group", primary: int = 1, backup: int = 2
) -> RoutingGroup:
    return RoutingGroup.model_validate({
        "group_name": name,
        "models": ["filtered-model", "other-model"],
        "routing_strategy": "priority",
        "model_priorities": {"filtered-model": primary, "other-model": backup},
    })


def _priority_deployments(
    primary_response: str = "primary", primary_blocked: bool = False, backup_response: str = "backup"
) -> list[DeploymentTypedDict]:
    return [
        {
            **deployment,
            "litellm_params": {
                **deployment["litellm_params"],
                "mock_response": (
                    primary_response if deployment["model_name"] == "filtered-model" else backup_response
                ),
                "order": 10 if deployment["model_name"] == "filtered-model" else 1,
            },
            "model_info": {
                **deployment["model_info"],
                "blocked": primary_blocked and deployment["model_name"] == "filtered-model",
            },
        }
        for deployment in _model_list()
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("requested_model", ["priority-group", "priority-alias"])
@pytest.mark.asyncio
async def test_priority_group_always_uses_primary_when_healthy(
    asynchronous: bool, requested_model: str
) -> None:
    router: Final = Router(
        model_list=_priority_deployments(),
        routing_groups=[_priority_group()],
        model_group_alias={"priority-alias": "priority-group"},
        num_retries=0,
    )
    request: Final = {"model": requested_model, "messages": [{"role": "user", "content": "hi"}]}
    response: Final = (
        await router.acompletion(**request) if asynchronous else router.completion(**request)
    )
    assert response.choices[0].message.content == "primary"


@pytest.mark.parametrize("requested_model", ["priority-group", "priority-alias"])
@pytest.mark.asyncio
async def test_priority_group_fails_over_without_retries_and_leaves_direct_calls_unchanged(
    requested_model: str,
) -> None:
    router: Final = Router(
        model_list=_priority_deployments(primary_response="litellm.RateLimitError"),
        routing_groups=[_priority_group()],
        model_group_alias={"priority-alias": "priority-group"},
        num_retries=0,
        disable_cooldowns=True,
    )
    response: Final = await router.acompletion(
        model=requested_model, messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "backup"
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(model="filtered-model", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_opposite_priority_groups_preserve_deployments_and_legacy_member_policy() -> None:
    router: Final = Router(
        model_list=_priority_deployments(),
        routing_strategy="latency-based-routing",
        routing_groups=[
            RoutingGroup(group_name="legacy", models=["filtered-model"], routing_strategy="least-busy"),
            _priority_group(),
            _priority_group("reverse-group", primary=2, backup=1),
        ],
    )
    before: Final = router.get_model_list(model_name="filtered-model")
    forward: Final = await router.acompletion(
        model="priority-group", messages=[{"role": "user", "content": "hi"}]
    )
    reverse: Final = await router.acompletion(
        model="reverse-group", messages=[{"role": "user", "content": "hi"}]
    )
    assert (forward.choices[0].message.content, reverse.choices[0].message.content) == ("primary", "backup")
    assert router.get_model_list(model_name="filtered-model") == before
    assert all(deployment["litellm_params"]["order"] == 10 for deployment in before)


@pytest.mark.parametrize("backup_priority, expected", [(1, "backup"), (2, "primary")])
@pytest.mark.asyncio
async def test_priority_group_weights_select_only_within_the_first_eligible_level(
    backup_priority: int, expected: str
) -> None:
    router: Final = Router(
        model_list=_priority_deployments(), routing_groups=[_priority_group(backup=backup_priority)]
    )
    response: Final = await router.acompletion(
        model="priority-group",
        messages=[{"role": "user", "content": "hi"}],
        _router_weights={"priority-group": {"deploy-1": 0, "deploy-2": 0, "deploy-3": 1}},
    )
    assert response.choices[0].message.content == expected


@pytest.mark.asyncio
async def test_priority_group_skips_paused_primary_and_returns_to_it_after_recovery() -> None:
    router: Final = Router(
        model_list=_priority_deployments(primary_blocked=True), routing_groups=[_priority_group()]
    )
    paused: Final = await router.acompletion(
        model="priority-group", messages=[{"role": "user", "content": "hi"}]
    )
    router.set_model_list(_priority_deployments())
    recovered: Final = await router.acompletion(
        model="priority-group", messages=[{"role": "user", "content": "hi"}]
    )
    assert (paused.choices[0].message.content, recovered.choices[0].message.content) == ("backup", "primary")


@pytest.mark.parametrize("controls", [{"disable_fallbacks": True}, {"max_fallbacks": 0}])
@pytest.mark.asyncio
async def test_priority_group_respects_request_fallback_controls(controls: dict[str, object]) -> None:
    router: Final = Router(
        model_list=_priority_deployments(primary_response="litellm.RateLimitError"),
        routing_groups=[_priority_group()],
        num_retries=0,
        disable_cooldowns=True,
    )
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(
            model="priority-group", messages=[{"role": "user", "content": "hi"}], **controls
        )


@pytest.mark.parametrize(
    "priorities",
    [None, {}, {"filtered-model": 1}, {"filtered-model": 1, "other-model": 2, "extra": 3}],
)
def test_priority_group_requires_exact_member_priorities(priorities: object) -> None:
    with pytest.raises(ValidationError):
        RoutingGroup.model_validate({
            "group_name": "priority-group",
            "models": ["filtered-model", "other-model"],
            "routing_strategy": "priority",
            "model_priorities": priorities,
        })


@pytest.mark.parametrize("priority", [True, 0, -1, 1.5, "1", 9007199254740992])
def test_priority_group_rejects_invalid_priority_values(priority: object) -> None:
    with pytest.raises(ValidationError):
        RoutingGroup.model_validate({
            "group_name": "priority-group",
            "models": ["filtered-model"],
            "routing_strategy": "priority",
            "model_priorities": {"filtered-model": priority},
        })


@pytest.mark.parametrize("models", [[], ["filtered-model", "filtered-model"]])
def test_priority_group_requires_nonempty_unique_members(models: list[str]) -> None:
    with pytest.raises(ValidationError):
        RoutingGroup.model_validate({
            "group_name": "priority-group",
            "models": models,
            "routing_strategy": "priority",
            "model_priorities": {model: 1 for model in models},
        })


@pytest.mark.parametrize(
    "changes", [{"routing_strategy": "simple-shuffle"}, {"routing_strategy_args": {"ttl": 60}}]
)
def test_priority_group_rejects_conflicting_strategy_settings(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RoutingGroup.model_validate({
            "group_name": "priority-group",
            "models": ["filtered-model"],
            "routing_strategy": "priority",
            "model_priorities": {"filtered-model": 1},
            **changes,
        })


@pytest.mark.parametrize("group_name", ["filtered-model", "priority-alias"])
def test_priority_group_rejects_names_shadowed_by_a_model_or_alias(group_name: str) -> None:
    with pytest.raises(ValueError, match=r"shadow|collid|conflict"):
        Router(
            model_list=_priority_deployments(),
            routing_groups=[_priority_group(name=group_name)],
            model_group_alias={"priority-alias": "filtered-model"},
        )


def test_priority_is_rejected_as_a_top_level_strategy() -> None:
    with pytest.raises(ValueError, match="routing_strategy"):
        _build_router(routing_strategy="priority")


@pytest.mark.asyncio
async def test_priority_group_settings_roundtrip_replaces_the_order() -> None:
    router: Final = Router(model_list=_priority_deployments(), routing_groups=[_priority_group()])
    replacement: Final = _priority_group(primary=9007199254740991, backup=1)
    router.update_settings(routing_groups=[replacement.model_dump()])
    stored: Final = router.get_settings()["routing_groups"]
    assert stored == [replacement.model_dump()]
    response: Final = await router.acompletion(
        model="priority-group", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "backup"


def _priority_auto_router_deployment() -> DeploymentTypedDict:
    return {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "classifier_type": "heuristic",
                "adaptive": False,
                "tiers": {
                    tier: "priority-group"
                    for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
                },
            },
            "complexity_router_default_model": "filtered-model",
        },
    }


@pytest.mark.asyncio
async def test_auto_router_selected_priority_group_fails_over_inside_the_group() -> None:
    router: Final = Router(
        model_list=[
            *_priority_deployments(primary_response="litellm.RateLimitError"),
            _priority_auto_router_deployment(),
        ],
        routing_groups=[_priority_group()],
        num_retries=0,
        disable_cooldowns=True,
    )
    response: Final = await router.acompletion(
        model="smart-router", messages=[{"role": "user", "content": "hi"}]
    )
    assert response.choices[0].message.content == "backup"


@pytest.mark.asyncio
async def test_priority_group_ignores_cached_backup_affinity() -> None:
    from litellm.router_utils.prompt_caching_cache import PromptCachingCache

    router: Final = Router(
        model_list=_priority_deployments(),
        routing_groups=[_priority_group()],
        optional_pre_call_checks=["prompt_caching"],
    )
    messages: Final = [{
        "role": "user",
        "content": [{"type": "text", "text": "word " * 5000, "cache_control": {"type": "ephemeral"}}],
    }]
    cache: Final = PromptCachingCache(cache=router.prompt_caching_pin_cache)
    await cache.async_add_model_id(model_id="deploy-3", messages=messages, tools=None)
    assert await cache.async_get_model_id(messages=messages, tools=None) == {"model_id": "deploy-3"}
    response: Final = await router.acompletion(model="priority-group", messages=messages)
    assert response.choices[0].message.content == "primary"


@pytest.mark.asyncio
async def test_priority_group_preserves_responses_continuity_on_a_backup() -> None:
    from litellm.responses.utils import ResponsesAPIRequestUtils

    router: Final = Router(
        model_list=_priority_deployments(),
        routing_groups=[_priority_group()],
        optional_pre_call_checks=["responses_api_deployment_check"],
    )
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="deploy-3", response_id="resp-prior"
    )
    deployment: Final = await router.async_get_available_deployment(
        model="priority-group", input="continue", request_kwargs={"previous_response_id": previous_response_id}
    )
    assert deployment["model_info"]["id"] == "deploy-3"


def _recording_fallback_gate(allowed: frozenset[str], checked: list[str]) -> FallbackAccessCheck:
    async def check(*, model: str, request_kwargs: Mapping[str, object], ctx: RoutingContext) -> bool:
        checked.append(model)
        return model in allowed

    return check


@pytest.mark.parametrize("budget_admits_group", [False, True])
@pytest.mark.asyncio
async def test_auto_priority_advance_preserves_access_scope_and_checks_paid_group_budget(
    budget_admits_group: bool,
) -> None:
    access_checks: Final[list[str]] = []
    budget_checks: Final[list[str]] = []
    router: Final = Router(
        model_list=[
            *_priority_deployments(primary_response="litellm.RateLimitError"),
            _priority_auto_router_deployment(),
        ],
        routing_groups=[_priority_group()],
        num_retries=0,
        disable_cooldowns=True,
        fallback_access_check=_recording_fallback_gate(frozenset({"smart-router"}), access_checks),
        fallback_budget_check=_recording_fallback_gate(
            frozenset({"priority-group"}) if budget_admits_group else frozenset(), budget_checks
        ),
    )
    if budget_admits_group:
        response: Final = await router.acompletion(
            model="smart-router", messages=[{"role": "user", "content": "hi"}]
        )
        assert response.choices[0].message.content == "backup"
    else:
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(model="smart-router", messages=[{"role": "user", "content": "hi"}])
    assert access_checks == []
    assert budget_checks == ["priority-group"]


@pytest.mark.asyncio
async def test_auto_priority_group_exhaustion_still_checks_external_fallback_access() -> None:
    access_checks: Final[list[str]] = []
    router: Final = Router(
        model_list=[
            *_priority_deployments(
                primary_response="litellm.RateLimitError", backup_response="litellm.RateLimitError"
            ),
            _priority_auto_router_deployment(),
            {
                **_model_list()[2],
                "model_name": "external",
                "litellm_params": {**_model_list()[2]["litellm_params"], "mock_response": "external"},
                "model_info": {"id": "external-deployment"},
            },
        ],
        routing_groups=[_priority_group()],
        fallbacks=[{"priority-group": ["external"]}],
        num_retries=0,
        disable_cooldowns=True,
        fallback_access_check=_recording_fallback_gate(frozenset({"smart-router"}), access_checks),
    )
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(model="smart-router", messages=[{"role": "user", "content": "hi"}])
    assert access_checks and set(access_checks) == {"external"}


@pytest.mark.parametrize("check_kind", ["access", "budget"])
@pytest.mark.parametrize("metadata_bucket", ["metadata", "litellm_metadata"])
@pytest.mark.asyncio
async def test_caller_cannot_spoof_a_priority_group_to_bypass_fallback_gates(
    check_kind: Literal["access", "budget"], metadata_bucket: str,
) -> None:
    checked: Final[list[str]] = []
    check: Final = _recording_fallback_gate(frozenset({"filtered-model"}), checked)
    router: Final = Router(
        model_list=_priority_deployments(primary_response="litellm.RateLimitError"),
        routing_groups=[_priority_group()],
        fallbacks=[{"filtered-model": ["priority-group"]}],
        num_retries=0,
        disable_cooldowns=True,
        fallback_access_check=check if check_kind == "access" else None,
        fallback_budget_check=check if check_kind == "budget" else None,
    )
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(
            model="filtered-model",
            messages=[{"role": "user", "content": "hi"}],
            **{metadata_bucket: {"pre_routing_selected_model": "priority-group"}},
        )
    assert checked == ["priority-group"]


def _sync_task_count() -> int:
    return sum(
        1
        for t in asyncio.all_tasks()
        if t.get_coro() is not None
        and t.get_coro().__qualname__ == "BaseRoutingStrategy.periodic_sync_in_memory_spend_with_redis"
    )


@pytest.mark.asyncio
async def test_update_settings_same_routing_strategy_args_does_not_leak_sync_tasks(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router: Final = Router(
        model_list=_model_list(),
        routing_strategy="usage-based-routing-v2",
        routing_strategy_args={"ttl": 60},
    )
    try:
        assert _sync_task_count() == 1
        selector_before: Final = router.lowesttpm_logger_v2

        for _ in range(5):
            router.update_settings(routing_strategy_args={"ttl": 60})
        await asyncio.sleep(0)
        assert _sync_task_count() == 1
        assert router.lowesttpm_logger_v2 is selector_before, "same routing_strategy_args must not rebuild the selector"

        router.update_settings(routing_strategy_args={"ttl": 120})
        await asyncio.sleep(0)
        assert _sync_task_count() == 1
        assert router.lowesttpm_logger_v2.routing_args.ttl == 120
    finally:
        for t in [
            t
            for t in asyncio.all_tasks()
            if t.get_coro() is not None
            and t.get_coro().__qualname__ == "BaseRoutingStrategy.periodic_sync_in_memory_spend_with_redis"
        ]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass


class _RecordingRedisCache:
    def __init__(self) -> None:
        self.increment_lists: list[list[RedisPipelineIncrementOperation]] = []

    async def async_increment_pipeline(self, increment_list: list[RedisPipelineIncrementOperation]) -> list[float]:
        self.increment_lists.append(list(increment_list))
        return [float(op["increment_value"]) for op in increment_list]


@pytest.mark.asyncio
async def test_update_settings_changed_routing_strategy_args_flushes_replaced_selector_queue(
    monkeypatch,
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    monkeypatch.setattr(litellm, "input_callback", [])
    router: Final = Router(
        model_list=_model_list(),
        routing_strategy="usage-based-routing-v2",
        routing_strategy_args={"ttl": 60},
    )
    try:
        redis_cache: Final = _RecordingRedisCache()
        replaced: Final = router.lowesttpm_logger_v2
        replaced.dual_cache.redis_cache = redis_cache
        replaced.redis_increment_operation_queue.append(
            RedisPipelineIncrementOperation(key="rpm-key", increment_value=3, ttl=60)
        )

        router.update_settings(routing_strategy_args={"ttl": 120})
        await asyncio.sleep(0)
        await asyncio.gather(
            *(
                t
                for t in asyncio.all_tasks()
                if t.get_coro() is not None
                and t.get_coro().__qualname__ == "BaseRoutingStrategy._push_in_memory_increments_to_redis"
            )
        )

        assert router.lowesttpm_logger_v2 is not replaced
        assert redis_cache.increment_lists == [
            [RedisPipelineIncrementOperation(key="rpm-key", increment_value=3, ttl=60)]
        ]
        assert replaced.redis_increment_operation_queue == []
    finally:
        for t in [
            t
            for t in asyncio.all_tasks()
            if t.get_coro() is not None
            and t.get_coro().__qualname__ == "BaseRoutingStrategy.periodic_sync_in_memory_spend_with_redis"
        ]:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
