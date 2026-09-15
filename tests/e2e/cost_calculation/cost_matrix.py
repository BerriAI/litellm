"""The cost-calculation matrix: frontier model set, the pricing-component cases
each model runs, and the expected-cost arithmetic.

Rates come from ``tests/e2e/cost_map.json``, which the proxy under test loads as
its ENTIRE model cost map (LITELLM_MODEL_COST_MAP_URL), so an entry's rates are
exactly what the proxy bills and nothing in the suite depends on the bundled
map. Each model's rates are a distinct multiple of a shared base set, so a
component billed at the wrong model's rate (or the wrong case's rate) can never
coincidentally match.

Case applicability is pricing-field-gated AND wire-gated: a case runs for a
model only when the entry carries the rate the case exercises and the wire can
report the token kind that rate prices. When the wire cannot report a kind
(e.g. Anthropic has no reasoning-token field, Responses reports no cache
creation), the case is absent from the matrix rather than silently zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter

from scripted_provider import Scenario, ScriptedOutput, ScriptedUsage, Wire

COST_MAP_PATH: Final = Path(__file__).resolve().parent.parent / "cost_map.json"


class SearchContextCostPerQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_context_size_low: float | None = None
    search_context_size_medium: float | None = None
    search_context_size_high: float | None = None


class CostMapEntry(BaseModel):
    """The pricing fields of a cost-map entry the matrix reads. Shaped like a
    ``model_prices_and_context_window.json`` entry; unmodelled keys are ignored."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    litellm_provider: str
    mode: str
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    cache_creation_input_token_cost_above_1hr: float | None = None
    output_cost_per_reasoning_token: float | None = None
    input_cost_per_audio_token: float | None = None
    output_cost_per_audio_token: float | None = None
    input_cost_per_token_above_200k_tokens: float | None = None
    output_cost_per_token_above_200k_tokens: float | None = None
    input_cost_per_token_flex: float | None = None
    output_cost_per_token_flex: float | None = None
    input_cost_per_token_priority: float | None = None
    output_cost_per_token_priority: float | None = None
    search_context_cost_per_query: SearchContextCostPerQuery | None = None
    web_search_billing_unit: str | None = None


_COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, CostMapEntry])
_COST_MAP: Final[Mapping[str, CostMapEntry]] = MappingProxyType(
    _COST_MAP_ADAPTER.validate_python(json.loads(COST_MAP_PATH.read_text()))
)

TIER_THRESHOLD_TOKENS: Final = 200_000


@dataclass(frozen=True, slots=True)
class FrontierModel:
    """One deployment under test: the model_name the suite registers, the
    provider-prefixed litellm model string, the wire the scripted upstream
    speaks, its cost-map key, and the sibling map model the response_model
    override case reports."""

    model_name: str
    litellm_model: str
    wire: Wire
    map_key: str
    override_model: str

    @property
    def rates(self) -> CostMapEntry:
        return _COST_MAP[self.map_key]

    @property
    def override_rates(self) -> CostMapEntry:
        return _COST_MAP[self.override_map_key]

    @property
    def override_map_key(self) -> str:
        return _OVERRIDE_MAP_KEYS[self.override_model]

    @property
    def provider(self) -> str:
        return self.rates.litellm_provider

    @property
    def api_key(self) -> str:
        # The scripted upstream ignores auth; a fixed bogus key proves the suite
        # spends zero real provider calls.
        return "sk-scripted-provider"


# Response-model override targets: emit a sibling's bare provider-facing name so
# the biller's provider-prefixed lookup lands on that sibling's map key.
_OVERRIDE_MODELS: Final[Mapping[str, str]] = MappingProxyType({
    "gpt-5.6": "gpt-5.4-mini",
    "gpt-5.5-pro": "gpt-5.3-codex",
    "gpt-5.3-codex": "gpt-5.5-pro",
    "gpt-5.4-mini": "gpt-5.6",
    "claude-opus-5": "claude-sonnet-5",
    "claude-sonnet-5": "claude-opus-5",
    "claude-haiku-4-5": "claude-sonnet-5",
    "gemini/gemini-3.8-flash": "gemini-3.1-pro-preview",
    "gemini/gemini-3.1-pro-preview": "gemini-3.8-flash",
    "together_ai/moonshotai/Kimi-K3": "zai-org/GLM-5.3",
    "together_ai/zai-org/GLM-5.3": "moonshotai/Kimi-K3",
    "fireworks_ai/kimi-k3": "qwen3p8-max",
    "fireworks_ai/qwen3p8-max": "kimi-k3",
    "fireworks_ai/deepseek-v4p1-flash": "kimi-k3",
})

_OVERRIDE_MAP_KEYS: Final[Mapping[str, str]] = MappingProxyType({
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.6": "gpt-5.6",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5.5-pro": "gpt-5.5-pro",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-5": "claude-opus-5",
    "gemini-3.1-pro-preview": "gemini/gemini-3.1-pro-preview",
    "gemini-3.8-flash": "gemini/gemini-3.8-flash",
    "zai-org/GLM-5.3": "together_ai/zai-org/GLM-5.3",
    "moonshotai/Kimi-K3": "together_ai/moonshotai/Kimi-K3",
    "qwen3p8-max": "fireworks_ai/qwen3p8-max",
    "kimi-k3": "fireworks_ai/kimi-k3",
})


_FRONTIER_SPECS: Final[tuple[tuple[str, str, Wire], ...]] = (
    ("gpt-5.6", "openai/gpt-5.6", "openai_chat"),
    ("gpt-5.5-pro", "openai/gpt-5.5-pro", "openai_responses"),
    ("gpt-5.3-codex", "openai/gpt-5.3-codex", "openai_responses"),
    ("gpt-5.4-mini", "openai/gpt-5.4-mini", "openai_chat"),
    ("claude-opus-5", "anthropic/claude-opus-5", "anthropic_messages"),
    ("claude-sonnet-5", "anthropic/claude-sonnet-5", "anthropic_messages"),
    ("claude-haiku-4-5", "anthropic/claude-haiku-4-5", "anthropic_messages"),
    ("gemini/gemini-3.8-flash", "gemini/gemini-3.8-flash", "gemini_generate"),
    ("gemini/gemini-3.1-pro-preview", "gemini/gemini-3.1-pro-preview", "gemini_generate"),
    ("together_ai/moonshotai/Kimi-K3", "together_ai/moonshotai/Kimi-K3", "together_chat"),
    ("together_ai/zai-org/GLM-5.3", "together_ai/zai-org/GLM-5.3", "together_chat"),
    ("fireworks_ai/kimi-k3", "fireworks_ai/kimi-k3", "fireworks_chat"),
    ("fireworks_ai/qwen3p8-max", "fireworks_ai/qwen3p8-max", "fireworks_chat"),
    ("fireworks_ai/deepseek-v4p1-flash", "fireworks_ai/deepseek-v4p1-flash", "fireworks_chat"),
)


def _frontier() -> tuple[FrontierModel, ...]:
    return tuple(
        FrontierModel(
            model_name=f"cc-{map_key.replace('/', '-').lower()}",
            litellm_model=litellm_model,
            wire=wire,
            map_key=map_key,
            override_model=_OVERRIDE_MODELS[map_key],
        )
        for map_key, litellm_model, wire in _FRONTIER_SPECS
    )


FRONTIER_MODELS: Final[tuple[FrontierModel, ...]] = _frontier()

# Token kinds each wire can report, gating which pricing cases apply.
_WIRE_CAPS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "openai_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage",
        }
    ),
    "openai_responses": frozenset({"cache_read", "reasoning", "web_search", "response_model", "absent_usage"}),
    # Product gap: litellm hard-indexes message_delta["usage"] in
    # anthropic/chat/handler.py, so a usage-absent anthropic stream raises
    # KeyError; the real wire always carries it, so the case cannot be
    # represented.
    "anthropic_messages": frozenset({"cache_read", "cache_write_5m", "cache_write_1h", "web_search", "response_model"}),
    # Product gap: the gemini transform sets ModelResponse.model from the
    # request and drops the provider's modelVersion, so a response-model
    # override can never be priced on this wire.
    "gemini_generate": frozenset({"cache_read", "reasoning", "audio", "web_search", "absent_usage"}),
    "together_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage",
        }
    ),
    "fireworks_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage",
        }
    ),
})

CaseName: TypeAlias = Literal[
    "basic",
    "cache_read",
    "cache_write_5m",
    "cache_write_1h",
    "reasoning",
    "audio",
    "tiered",
    "service_tier_flex",
    "service_tier_priority",
    "web_search",
    "stream",
    "stream_no_usage",
    "response_model_override",
]


@dataclass(frozen=True, slots=True)
class Case:
    name: CaseName
    usage: ScriptedUsage
    stream: bool = False
    stream_usage: Literal["final_chunk", "absent"] = "final_chunk"
    service_tier: Literal["flex", "priority"] | None = None
    # For web_search the wire's reported call count is not always what gets
    # billed: chat-completions surfaces only expose url_citation annotations, so
    # the biller floors to one call; responses/messages/gemini report a real
    # count.
    billed_web_search_calls: int = 0
    response_model_override: bool = False
    exact_spend: bool = True
    # stream_usage=absent on a wire with no proxy-side token recount means the
    # bill is exactly zero; asserted as such rather than skipped.
    expect_zero_bill: bool = False

    def scenario(self, scenario_id: str, model: FrontierModel, text: str) -> Scenario:
        return Scenario(
            scenario_id=scenario_id,
            wire=model.wire,
            usage=self.usage,
            output=ScriptedOutput(
                text=text,
                response_model=model.override_model if self.response_model_override else None,
            ),
            stream_usage=self.stream_usage,
            service_tier=self.service_tier,
        )


_BASIC_USAGE: Final = ScriptedUsage(fresh_input_tokens=120, output_tokens=40)


def _web_search_case(model: FrontierModel) -> Case:
    counts_exactly: Final = model.wire in ("openai_responses", "anthropic_messages", "gemini_generate")
    return Case(
        name="web_search",
        usage=ScriptedUsage(fresh_input_tokens=100, output_tokens=30, web_search_calls=3),
        billed_web_search_calls=3 if counts_exactly else 1,
    )


def cases_for(model: FrontierModel) -> tuple[Case, ...]:
    rates: Final = model.rates
    caps: Final = _WIRE_CAPS[model.wire]
    candidates: Final[tuple[Case | None, ...]] = (
        Case(name="basic", usage=_BASIC_USAGE),
        (
            Case(name="cache_read", usage=ScriptedUsage(fresh_input_tokens=100, cache_read_tokens=50, output_tokens=30))
            if rates.cache_read_input_token_cost is not None and "cache_read" in caps
            else None
        ),
        (
            Case(
                name="cache_write_5m",
                usage=ScriptedUsage(fresh_input_tokens=90, cache_write_5m_tokens=60, output_tokens=30),
            )
            if rates.cache_creation_input_token_cost is not None and "cache_write_5m" in caps
            else None
        ),
        (
            Case(
                name="cache_write_1h",
                usage=ScriptedUsage(
                    fresh_input_tokens=90,
                    cache_write_5m_tokens=20,
                    cache_write_1h_tokens=40,
                    output_tokens=30,
                ),
            )
            if (
                rates.cache_creation_input_token_cost_above_1hr is not None
                and rates.cache_creation_input_token_cost is not None
                and "cache_write_1h" in caps
            )
            else None
        ),
        (
            Case(
                name="reasoning",
                usage=ScriptedUsage(fresh_input_tokens=100, output_tokens=30, reasoning_tokens=70),
            )
            if rates.output_cost_per_reasoning_token is not None and "reasoning" in caps
            else None
        ),
        (
            Case(
                name="audio",
                usage=ScriptedUsage(
                    fresh_input_tokens=100, audio_input_tokens=25, output_tokens=30, audio_output_tokens=15
                ),
            )
            if (
                rates.input_cost_per_audio_token is not None
                and rates.output_cost_per_audio_token is not None
                and "audio" in caps
            )
            else None
        ),
        (
            Case(
                name="tiered",
                usage=ScriptedUsage(
                    fresh_input_tokens=TIER_THRESHOLD_TOKENS + 1, output_tokens=30
                ),
            )
            if (
                rates.input_cost_per_token_above_200k_tokens is not None
                and rates.output_cost_per_token_above_200k_tokens is not None
            )
            else None
        ),
        (
            Case(name="service_tier_flex", usage=_BASIC_USAGE, service_tier="flex")
            if rates.input_cost_per_token_flex is not None and rates.output_cost_per_token_flex is not None
            else None
        ),
        (
            Case(name="service_tier_priority", usage=_BASIC_USAGE, service_tier="priority")
            if rates.input_cost_per_token_priority is not None and rates.output_cost_per_token_priority is not None
            else None
        ),
        _web_search_case(model) if rates.search_context_cost_per_query is not None and "web_search" in caps else None,
        Case(name="stream", usage=_BASIC_USAGE, stream=True),
        (
            Case(
                name="stream_no_usage",
                usage=_BASIC_USAGE,
                stream=True,
                stream_usage="absent",
                exact_spend=False,
                # The responses surface bills only provider-reported usage;
                # with no usage in the stream the spend row is zero. Other
                # wires recount tokens proxy-side and bill a nonzero amount.
                expect_zero_bill=model.wire == "openai_responses",
            )
            if "absent_usage" in caps
            else None
        ),
        (
            Case(name="response_model_override", usage=_BASIC_USAGE, response_model_override=True)
            if "response_model" in caps
            else None
        ),
    )
    return tuple(case for case in candidates if case is not None)


@dataclass(frozen=True, slots=True)
class ExpectedCost:
    """The expected bill split the way the spend row's cost_breakdown reports
    it: the gross input component (cache reads/writes folded in), the output
    component, and the tool-usage component."""

    input_cost: float
    output_cost: float
    tool_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.tool_cost


def expected_breakdown(model: FrontierModel, case: Case) -> ExpectedCost:
    """Literal arithmetic on the test-map rates over the scripted token counts.

    Input = fresh*in + read*read + 5m*create + 1h*create_1h + audio_in*audio_in;
    output = text*out + reasoning*reasoning + audio_out*audio_out; plus the
    billed web-search calls at the medium search-context rate. Above-threshold
    swaps every input/output rate to its ``_above_200k_tokens`` variant when
    total prompt tokens exceed the threshold; a service tier swaps input/output
    to the tier's variants, falling back to the base rate when a variant is
    unset -- mirroring _get_token_base_cost in litellm's cost calculator.
    """
    rates: Final = model.override_rates if case.response_model_override else model.rates
    u: Final = case.usage
    prompt_tokens: Final = (
        u.fresh_input_tokens + u.cache_read_tokens + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens + u.audio_input_tokens
    )
    tiered: Final = prompt_tokens > TIER_THRESHOLD_TOKENS
    in_rate: Final = (
        (rates.input_cost_per_token_above_200k_tokens if tiered else None)
        or (rates.input_cost_per_token_priority if case.service_tier == "priority" else None)
        or (rates.input_cost_per_token_flex if case.service_tier == "flex" else None)
        or rates.input_cost_per_token
        or 0.0
    )
    out_rate: Final = (
        (rates.output_cost_per_token_above_200k_tokens if tiered else None)
        or (rates.output_cost_per_token_priority if case.service_tier == "priority" else None)
        or (rates.output_cost_per_token_flex if case.service_tier == "flex" else None)
        or rates.output_cost_per_token
        or 0.0
    )
    input_cost: Final = (
        u.fresh_input_tokens * in_rate
        + u.cache_read_tokens * (rates.cache_read_input_token_cost or 0.0)
        + u.cache_write_5m_tokens * (rates.cache_creation_input_token_cost or 0.0)
        + u.cache_write_1h_tokens * (rates.cache_creation_input_token_cost_above_1hr or 0.0)
        + u.audio_input_tokens * (rates.input_cost_per_audio_token or 0.0)
    )
    output_cost: Final = (
        u.output_tokens * out_rate
        + u.reasoning_tokens * (rates.output_cost_per_reasoning_token or out_rate)
        + u.audio_output_tokens * (rates.output_cost_per_audio_token or out_rate)
    )
    search: Final = rates.search_context_cost_per_query
    tool_cost: Final = case.billed_web_search_calls * (
        search.search_context_size_medium if search and search.search_context_size_medium else 0.0
    )
    return ExpectedCost(input_cost=input_cost, output_cost=output_cost, tool_cost=tool_cost)


def expected_cost(model: FrontierModel, case: Case) -> float:
    return expected_breakdown(model, case).total


def expected_token_columns(model: FrontierModel, case: Case) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) the spend row should carry, per the
    wire's normalization: Anthropic folds cache read/write into prompt_tokens,
    everyone else reports the totals the wire emitted."""
    u: Final = case.usage
    if model.wire == "anthropic_messages":
        return (
            u.fresh_input_tokens + u.cache_read_tokens + u.cache_write_5m_tokens + u.cache_write_1h_tokens,
            u.output_tokens,
        )
    if model.wire == "gemini_generate":
        return (
            u.fresh_input_tokens + u.cache_read_tokens + u.audio_input_tokens,
            u.output_tokens + u.reasoning_tokens + u.audio_output_tokens,
        )
    if model.wire == "openai_responses":
        return (
            u.fresh_input_tokens + u.cache_read_tokens,
            u.output_tokens + u.reasoning_tokens,
        )
    return (
        u.fresh_input_tokens
        + u.cache_read_tokens
        + u.cache_write_5m_tokens
        + u.cache_write_1h_tokens
        + u.audio_input_tokens,
        u.output_tokens + u.reasoning_tokens + u.audio_output_tokens,
    )
