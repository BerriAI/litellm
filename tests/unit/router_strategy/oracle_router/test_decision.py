"""Decision makers: each learns or proposes the way its contract says."""

import pytest

from litellm.router_strategy.oracle_router.decision import (
    DecisionMaker,
    FixedDecisionMaker,
    PreRoutingDecisionMaker,
    ProgramContext,
    ThompsonDecisionMaker,
    build_decision_maker,
    load_object,
)
from litellm.types.router import AdaptiveRouterWeights, OracleDecisionMakerConfig, PreRoutingHookResponse, RequestType

MODELS = ("smart", "fast")
COSTS = {"smart": 1e-5, "fast": 1e-6}


def _ctx(prompt: str = "fix the failing test", request_type: RequestType = RequestType.GENERAL) -> ProgramContext:
    return ProgramContext(program_id="p", prompt=prompt, request_type=request_type)


@pytest.mark.asyncio
async def test_thompson_learns_a_different_winner_per_request_type():
    maker = ThompsonDecisionMaker(MODELS, COSTS, weights=AdaptiveRouterWeights(quality=1.0, cost=0.0), seed=0)
    for _ in range(80):
        for request_type, good in ((RequestType.CODE_GENERATION, "smart"), (RequestType.WRITING, "fast")):
            ctx = _ctx(request_type=request_type)
            chosen = await maker.select(ctx)
            maker.update(ctx, chosen, 1.0 if chosen == good else 0.0)
    code, writing = _ctx(request_type=RequestType.CODE_GENERATION), _ctx(request_type=RequestType.WRITING)
    assert maker.estimate(code, "smart") > maker.estimate(code, "fast")
    assert maker.estimate(writing, "fast") > maker.estimate(writing, "smart")
    picks = [await maker.select(code) for _ in range(30)]
    assert picks.count("smart") > 24
    assert maker.snapshot()["cells"]["code_generation/smart"]["samples"] > 0


@pytest.mark.asyncio
async def test_thompson_cost_weight_prefers_the_cheap_model_when_quality_ties():
    maker = ThompsonDecisionMaker(MODELS, COSTS, weights=AdaptiveRouterWeights(quality=0.3, cost=0.7), seed=1)
    ctx = _ctx()
    for _ in range(40):
        maker.update(ctx, "smart", 0.8)
        maker.update(ctx, "fast", 0.8)
    picks = [await maker.select(ctx) for _ in range(30)]
    assert picks.count("fast") > 24


@pytest.mark.asyncio
async def test_thompson_priors_come_from_adaptive_router_preferences():
    from litellm.types.router import AdaptiveRouterPreferences

    prefs = {
        "smart": AdaptiveRouterPreferences(quality_tier=3, strengths=[]),
        "fast": AdaptiveRouterPreferences(quality_tier=1, strengths=[]),
    }
    maker = ThompsonDecisionMaker(
        MODELS, COSTS, weights=AdaptiveRouterWeights(quality=1.0, cost=0.0), model_prefs=prefs, seed=0
    )
    ctx = _ctx()
    assert maker.estimate(ctx, "smart") > maker.estimate(ctx, "fast")
    picks = [await maker.select(ctx) for _ in range(30)]
    assert picks.count("smart") > 24


def test_thompson_rejects_unknown_model_and_empty_pool():
    maker = ThompsonDecisionMaker(MODELS, COSTS)
    with pytest.raises(KeyError):
        maker.update(_ctx(), "other", 1.0)
    with pytest.raises(ValueError, match="at least one model"):
        ThompsonDecisionMaker((), COSTS)


@pytest.mark.asyncio
async def test_fixed_always_returns_its_model_and_counts_updates():
    maker = FixedDecisionMaker(MODELS, model="fast")
    assert await maker.select(_ctx()) == "fast"
    maker.update(_ctx(), "fast", 1.0)
    assert maker.snapshot() == {"type": "fixed", "model": "fast", "updates": 1}
    with pytest.raises(KeyError):
        FixedDecisionMaker(MODELS, model="nope")


class _FakeStrategy:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.calls: list[dict] = []

    async def async_pre_routing_hook(self, model, request_kwargs, messages=None, input=None, specific_deployment=False):
        self.calls.append({"model": model, "messages": messages})
        return None if self.answer is None else PreRoutingHookResponse(model=self.answer, messages=messages)


@pytest.mark.asyncio
async def test_pre_routing_uses_the_wrapped_strategy_proposal_once_per_program():
    strategy = _FakeStrategy("fast")
    maker = PreRoutingDecisionMaker(MODELS, router_name="tiered", strategy=strategy)
    assert await maker.select(_ctx("write a haiku")) == "fast"
    assert strategy.calls == [{"model": "tiered", "messages": [{"role": "user", "content": "write a haiku"}]}]
    assert maker.snapshot()["proposals"] == {"fast": 1}


@pytest.mark.asyncio
async def test_pre_routing_falls_back_to_the_first_model_when_the_proposal_is_unusable():
    assert await PreRoutingDecisionMaker(MODELS, "r", _FakeStrategy(None)).select(_ctx()) == "smart"
    maker = PreRoutingDecisionMaker(MODELS, "r", _FakeStrategy("not-configured"))
    assert await maker.select(_ctx()) == "smart"
    assert maker.snapshot()["fallbacks"] == 1


class CustomMaker:
    def __init__(self, models: tuple[str, ...]) -> None:
        self._models = models

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    async def select(self, context: ProgramContext) -> str:
        return self._models[-1]

    def update(self, context: ProgramContext, model: str, score: float) -> None:
        pass

    def snapshot(self):
        return {"type": "custom"}


class NotAMaker:
    def __init__(self, models: tuple[str, ...]) -> None:
        pass


@pytest.mark.asyncio
async def test_build_decision_maker_by_type():
    resolve = {"tiered": _FakeStrategy("smart")}.get
    thompson = build_decision_maker(OracleDecisionMakerConfig(), MODELS, COSTS, resolve)
    assert isinstance(thompson, ThompsonDecisionMaker) and await thompson.select(_ctx()) in MODELS
    assert isinstance(
        build_decision_maker(OracleDecisionMakerConfig(type="fixed", model="fast"), MODELS, COSTS, resolve),
        FixedDecisionMaker,
    )
    pre = build_decision_maker(OracleDecisionMakerConfig(type="pre_routing", router="tiered"), MODELS, COSTS, resolve)
    assert isinstance(pre, PreRoutingDecisionMaker) and await pre.select(_ctx()) == "smart"
    with pytest.raises(ValueError, match="not a configured strategy router"):
        build_decision_maker(OracleDecisionMakerConfig(type="pre_routing", router="missing"), MODELS, COSTS, resolve)
    custom = build_decision_maker(
        OracleDecisionMakerConfig(type="custom", path=f"{__name__}:CustomMaker"), MODELS, COSTS, resolve
    )
    assert isinstance(custom, DecisionMaker) and await custom.select(_ctx()) == "fast"
    with pytest.raises(TypeError):
        build_decision_maker(
            OracleDecisionMakerConfig(type="custom", path=f"{__name__}:NotAMaker"), MODELS, COSTS, resolve
        )


def test_load_object_requires_module_colon_attribute():
    with pytest.raises(ValueError, match=r"package\.module:Attribute"):
        load_object("no_colon_here")
    assert load_object(f"{__name__}:CustomMaker") is CustomMaker


def test_config_requires_the_fields_each_type_reads():
    with pytest.raises(ValueError, match=r"decision_maker\.model"):
        OracleDecisionMakerConfig(type="fixed")
    with pytest.raises(ValueError, match=r"decision_maker\.router"):
        OracleDecisionMakerConfig(type="pre_routing")
    with pytest.raises(ValueError, match=r"decision_maker\.path"):
        OracleDecisionMakerConfig(type="custom")
