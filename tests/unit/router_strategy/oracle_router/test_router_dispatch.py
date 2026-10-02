"""Router-level wiring of `auto_router/oracle_router`: registration, deferred init, dispatch, hooks, removal."""

import pytest

import litellm
from litellm import Router
from litellm.router_strategy.oracle_router.config import PROGRAM_ID_METADATA_KEY
from litellm.router_strategy.oracle_router.decision import PreRoutingDecisionMaker, ProgramContext
from litellm.router_strategy.oracle_router.hooks import OracleRouterPostCallHook
from litellm.types.router import LiteLLM_Params, RequestType


def _oracle(router: Router, name: str):
    return router.oracle_routers[name][0].strategy


def _model_list(decision_maker=None, verifier=None, extra=()):
    config = {"available_models": ["smart", "fast"]}
    if decision_maker:
        config["decision_maker"] = decision_maker
    if verifier:
        config["verifier"] = verifier
    return [
        {
            "model_name": "oracle",
            "litellm_params": {"model": "auto_router/oracle_router", "oracle_router_config": config},
        },
        {"model_name": "smart", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"}},
        {"model_name": "fast", "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}},
        *extra,
    ]


def _oracle_hooks():
    return litellm.logging_callback_manager.get_custom_loggers_for_type(OracleRouterPostCallHook)


def test_prefix_is_classified_as_oracle_and_not_as_a_semantic_router():
    router = Router(model_list=[])
    params = LiteLLM_Params(model="auto_router/oracle_router")
    assert router._is_oracle_router_deployment(litellm_params=params) is True
    assert router._is_auto_router_deployment(litellm_params=params) is False
    assert router._is_adaptive_router_deployment(litellm_params=params) is False
    assert router._is_oracle_router_deployment(litellm_params=LiteLLM_Params(model="openai/gpt-4o")) is False


def test_init_registers_the_router_and_exactly_one_hook():
    router = Router(model_list=_model_list())
    assert "oracle" in router.oracle_routers
    assert _oracle(router, "oracle").models == ("smart", "fast")
    hooks = [hook for hook in _oracle_hooks() if hook.oracle_router is _oracle(router, "oracle")]
    assert len(hooks) == 1


def test_init_without_config_raises():
    with pytest.raises(ValueError, match="oracle_router_config is required"):
        Router(model_list=[{"model_name": "oracle", "litellm_params": {"model": "auto_router/oracle_router"}}])


@pytest.mark.asyncio
async def test_pre_routing_hook_dispatches_to_the_oracle_router_and_binds_the_program():
    router = Router(model_list=_model_list(decision_maker={"type": "fixed", "model": "fast"}))
    kwargs = {"metadata": {"program_id": "task-1"}}
    response = await router.async_pre_routing_hook(
        model="oracle", request_kwargs=kwargs, messages=[{"role": "user", "content": "fix the bug"}]
    )
    assert response is not None and response.model == "fast"
    assert response.routing_decision["router_type"] == "oracle"
    assert kwargs["metadata"][PROGRAM_ID_METADATA_KEY] == "task-1"
    assert _oracle(router, "oracle").binding("task-1").model == "fast"


def test_pre_routing_decision_maker_resolves_a_strategy_router_listed_after_it():
    quality = {
        "model_name": "tiered",
        "litellm_params": {
            "model": "auto_router/quality_router",
            "quality_router_default_model": "fast",
            "quality_router_config": {"available_models": ["smart", "fast"]},
        },
    }
    model_list = _model_list(decision_maker={"type": "pre_routing", "router": "tiered"}, extra=(quality,))
    model_list[1]["model_info"] = {"litellm_routing_preferences": {"quality_tier": 3}}
    model_list[2]["model_info"] = {"litellm_routing_preferences": {"quality_tier": 1}}
    router = Router(model_list=model_list)
    assert isinstance(_oracle(router, "oracle").decision_maker, PreRoutingDecisionMaker)


def test_init_reads_adaptive_router_preferences_as_bandit_priors():
    model_list = _model_list()
    model_list[1]["model_info"] = {"adaptive_router_preferences": {"quality_tier": 3, "strengths": []}}
    router = Router(model_list=model_list)
    maker = _oracle(router, "oracle").decision_maker
    ctx = ProgramContext(program_id="p", prompt="x", request_type=RequestType.GENERAL)
    assert maker.estimate(ctx, "smart") > maker.estimate(ctx, "fast")


def test_pre_routing_decision_maker_with_unknown_router_fails_init():
    with pytest.raises(ValueError, match="not a configured strategy router"):
        Router(model_list=_model_list(decision_maker={"type": "pre_routing", "router": "ghost"}))


def test_hot_reload_keeps_the_existing_router_and_its_learned_state():
    router = Router(model_list=_model_list())
    before = _oracle(router, "oracle")
    router.set_model_list(_model_list())
    assert _oracle(router, "oracle") is before
    assert len([hook for hook in _oracle_hooks() if hook.oracle_router is before]) == 1


def test_deleting_the_deployment_unregisters_the_router_and_its_hook():
    router = Router(model_list=_model_list())
    before = _oracle(router, "oracle")
    router.delete_deployment(id=router.get_model_ids(model_name="oracle")[0])
    assert "oracle" not in router.oracle_routers
    assert all(hook.oracle_router is not before for hook in _oracle_hooks())


def test_strategy_router_dependencies_cover_available_models():
    from litellm.router_utils.auto_router_model_naming import strategy_router_dependencies

    deps = strategy_router_dependencies(
        {"model": "auto_router/oracle_router", "oracle_router_config": {"available_models": ["smart", "fast"]}}
    )
    assert {(dep.model_name, dep.role) for dep in deps} == {("smart", "tier"), ("fast", "tier")}
