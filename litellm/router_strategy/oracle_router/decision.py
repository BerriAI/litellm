"""Decision makers: the pluggable policy that binds a program to a model.

``DecisionMaker`` is the whole contract. The default, ``thompson``, is LiteLLM's own adaptive-router
bandit (a Beta posterior per request type and model, Thompson sampling, cost-weighted pick) trained on
verified program outcomes instead of per-turn regex signals. ``pre_routing`` lets any LiteLLM strategy
router (complexity, quality, adaptive, semantic) make the proposal once per program, ``fixed`` pins one
model, and ``custom`` imports an operator's class.
"""

import importlib
import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from litellm.router_strategy.adaptive_router.bandit import BanditCell, apply_delta, initial_cell, pick_best
from litellm.types.router import (
    AdaptiveRouterPreferences,
    AdaptiveRouterWeights,
    OracleDecisionMakerConfig,
    PreRoutingStrategy,
    RequestType,
)


@dataclass(frozen=True, slots=True)
class ProgramContext:
    """What a decision maker sees when a program arrives: its initial request and LiteLLM's request type."""

    program_id: str
    prompt: str
    request_type: RequestType


@runtime_checkable
class DecisionMaker(Protocol):
    @property
    def models(self) -> tuple[str, ...]: ...

    async def select(self, context: ProgramContext) -> str: ...

    def update(self, context: ProgramContext, model: str, score: float) -> None: ...

    def snapshot(self) -> Mapping[str, object]: ...


def _check_model(models: Sequence[str], model: str) -> None:
    if model not in models:
        raise KeyError(f"unknown model {model!r}; configured models: {list(models)}")


_UNIFORM_PRIOR: Final = AdaptiveRouterPreferences(quality_tier=2, strengths=[])


class ThompsonDecisionMaker:
    """LiteLLM's adaptive-router bandit, keyed by (request type, model) and updated with verified scores.

    Cells start from the same priors the adaptive router uses, read from each model's
    ``model_info.adaptive_router_preferences`` (tier 2 when a model declares none). A verified score ``s``
    in [0, 1] adds ``s`` successes and ``1 - s`` failures to the cell of the model that served the program.
    Selection samples every cell once and picks the best quality/cost score with the adaptive router's
    weights, so ``weights.cost`` is the only cost knob.
    """

    def __init__(
        self,
        models: Sequence[str],
        model_costs: Mapping[str, float],
        weights: AdaptiveRouterWeights | None = None,
        model_prefs: Mapping[str, AdaptiveRouterPreferences] | None = None,
        seed: int = 0,
    ) -> None:
        if not models:
            raise ValueError("a decision maker needs at least one model")
        self._models: Final[tuple[str, ...]] = tuple(models)
        costs: Final = {m: float(model_costs.get(m, 0.0)) for m in self._models}
        self._costs: Final[dict[str, float]] = costs  # mutable-ok: pick_best takes a dict
        self._weights: Final[AdaptiveRouterWeights] = weights or AdaptiveRouterWeights()
        self._rng: Final = random.Random(seed)
        prefs: Final = model_prefs or {}
        grid: Final = tuple(itertools.product(RequestType, self._models))
        cells: Final = {(rt, model): initial_cell(prefs.get(model, _UNIFORM_PRIOR), rt) for rt, model in grid}
        self._cells: Final[dict[tuple[RequestType, str], BanditCell]] = cells  # mutable-ok: replaced on update
        self.pulls: Final[dict[str, int]] = {m: 0 for m in self._models}  # mutable-ok: counters

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    async def select(self, context: ProgramContext) -> str:
        cells: Final = {model: self._cells[(context.request_type, model)] for model in self._models}
        chosen: Final = pick_best(cells, self._costs, self._weights.quality, self._weights.cost, rng=self._rng)
        self.pulls[chosen] += 1
        return chosen

    def update(self, context: ProgramContext, model: str, score: float) -> None:
        _check_model(self._models, model)
        key: Final = (context.request_type, model)
        self._cells[key] = apply_delta(self._cells[key], score, 1.0 - score)

    def estimate(self, context: ProgramContext, model: str) -> float:
        _check_model(self._models, model)
        return self._cells[(context.request_type, model)].mean

    def snapshot(self) -> Mapping[str, object]:
        cells: Final = {
            f"{request_type.value}/{model}": {"mean": round(cell.mean, 4), "samples": cell.total_samples}
            for (request_type, model), cell in self._cells.items()
            if cell.total_samples > 0
        }
        return {
            "type": "thompson",
            "weights": {"quality": self._weights.quality, "cost": self._weights.cost},
            "pulls": dict(self.pulls),
            "cells": cells,
        }


class FixedDecisionMaker:
    """Always the same model; the single-model baseline that still gets verified and reported."""

    def __init__(self, models: Sequence[str], model: str) -> None:
        _check_model(models, model)
        self._models: Final[tuple[str, ...]] = tuple(models)
        self._model: Final[str] = model
        self._updates: int = 0

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    async def select(self, context: ProgramContext) -> str:
        return self._model

    def update(self, context: ProgramContext, model: str, score: float) -> None:
        _check_model(self._models, model)
        self._updates += 1

    def snapshot(self) -> Mapping[str, object]:
        return {"type": "fixed", "model": self._model, "updates": self._updates}


class PreRoutingDecisionMaker:
    """Let one of LiteLLM's own strategy routers propose the model, once per program.

    The wrapped router (complexity, quality, adaptive or semantic) sees the program's initial request as a
    single user message. Its per-request decision becomes a per-program binding, and the program's verified
    outcome is recorded against that model, so the state endpoint shows what the strategy actually achieved.
    """

    def __init__(self, models: Sequence[str], router_name: str, strategy: PreRoutingStrategy) -> None:
        if not models:
            raise ValueError("a decision maker needs at least one model")
        self._models: Final[tuple[str, ...]] = tuple(models)
        self._router_name: Final[str] = router_name
        self._strategy: Final[PreRoutingStrategy] = strategy
        self._proposals: Final[dict[str, int]] = {}  # mutable-ok: proposal counts per model, incremented per program
        self._fallbacks: int = 0
        self._updates: int = 0

    @property
    def models(self) -> tuple[str, ...]:
        return self._models

    async def select(self, context: ProgramContext) -> str:
        response: Final = await self._strategy.async_pre_routing_hook(
            model=self._router_name,
            request_kwargs={"metadata": {}},
            messages=[{"role": "user", "content": context.prompt}],
        )
        proposed: Final = response.model if response is not None and response.model in self._models else None
        if proposed is None:
            self._fallbacks += 1
        chosen: Final = proposed if proposed is not None else self._models[0]
        self._proposals[chosen] = self._proposals.get(chosen, 0) + 1
        return chosen

    def update(self, context: ProgramContext, model: str, score: float) -> None:
        _check_model(self._models, model)
        self._updates += 1

    def snapshot(self) -> Mapping[str, object]:
        return {
            "type": "pre_routing",
            "router": self._router_name,
            "proposals": dict(self._proposals),
            "fallbacks": self._fallbacks,
            "updates": self._updates,
        }


def load_object(path: str) -> object:
    """Import ``package.module:attribute``."""
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"expected 'package.module:Attribute', got {path!r}")
    loaded: Final[object] = getattr(importlib.import_module(module_name), attribute)  # pyright: ignore[reportAny]  # module attributes are untyped
    return loaded


def build_decision_maker(
    config: OracleDecisionMakerConfig,
    models: Sequence[str],
    model_costs: Mapping[str, float],
    resolve_strategy: Callable[[str], PreRoutingStrategy | None],
    model_prefs: Mapping[str, AdaptiveRouterPreferences] | None = None,
) -> DecisionMaker:
    """Instantiate the configured decision maker over ``models`` (strongest first)."""
    match config.type:
        case "thompson":
            return ThompsonDecisionMaker(
                models, model_costs, weights=config.weights, model_prefs=model_prefs, seed=config.seed
            )
        case "fixed":
            return FixedDecisionMaker(models, model=config.model or models[0])
        case "pre_routing":
            router_name: Final = config.router or ""
            strategy: Final = resolve_strategy(router_name)
            if strategy is None:
                raise ValueError(
                    f"oracle_router decision_maker.router {router_name!r} is not a configured strategy router"
                )
            return PreRoutingDecisionMaker(models, router_name=router_name, strategy=strategy)
        case "custom":
            factory: Final = load_object(config.path or "")
            if not callable(factory):
                raise TypeError(f"oracle_router decision_maker.path {config.path!r} is not callable")
            instance: Final = factory(tuple(models))
            if not isinstance(instance, DecisionMaker):
                raise TypeError(f"{config.path!r} must build an object with select, update, snapshot and models")
            return instance
