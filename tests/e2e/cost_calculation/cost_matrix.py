"""The cost-calculation matrix: the model set derived from the test cost map,
the request/response cases from ``cases.json``, and the loaders both use.

Three data files drive the suite; nothing in Python lists models or cases:
- ``tests/e2e/cost_map.json`` is the proxy's ENTIRE model cost map
  (LITELLM_MODEL_COST_MAP_URL); every entry becomes a deployment under test.
- ``tests/e2e/cost_calculation/cases.json`` is the case list; each case runs
  for a model when the entry carries the rates it exercises (``requires_rates``)
  and the wire can report the token kinds involved (``requires_caps`` /
  ``wires``).
- ``tests/e2e/cost_calculation/expected.json`` holds the reviewed goldens; the
  tests assert them verbatim and never compute a price themselves. The rate
  arithmetic that proposes goldens lives in ``generate_expected.py``, not here.
"""

from __future__ import annotations

import base64
import json
import random
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter
from scripted_provider import Scenario, ScriptedOutput, ScriptedToolCall, ScriptedUsage, Wire

COST_MAP_PATH: Final = Path(__file__).resolve().parent.parent / "cost_map.json"
CASES_PATH: Final = Path(__file__).resolve().parent / "cases.json"
EXPECTED_PATH: Final = Path(__file__).resolve().parent / "expected.json"


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


COST_MAP_ADAPTER: Final = TypeAdapter(dict[str, CostMapEntry])
COST_MAP: Final[Mapping[str, CostMapEntry]] = MappingProxyType(
    COST_MAP_ADAPTER.validate_python(json.loads(COST_MAP_PATH.read_text()))
)

TIER_THRESHOLD_TOKENS: Final = 200_000


class DeploymentSpec(BaseModel):
    """A deployment-level fact from cases.json: when a map key needs a
    registered deployment name that is not its provider model (or a
    model_info.base_model pin), the matrix uses these instead of the defaults."""

    model_config = ConfigDict(frozen=True)

    map_key: str
    litellm_model: str | None = None
    base_model: str | None = None


class Case(BaseModel):
    """One request/response shape from cases.json; gated onto a model by
    ``requires_rates`` (entry must carry each rate field), ``requires_caps``
    (the wire must report the token kind) and ``wires`` (shape is wire-specific)."""

    model_config = ConfigDict(frozen=True)

    name: str
    usage: ScriptedUsage
    stream: bool = False
    stream_usage: Literal["final_chunk", "absent"] = "final_chunk"
    service_tier: Literal["flex", "priority"] | None = None
    response_model_override: bool = False
    exact_spend: bool = True
    tool_call: bool = False
    image_input: bool = False
    terminal: Literal["completed", "incomplete", "unvalidated", "prompt_blocked"] = "completed"
    requires_rates: tuple[str, ...] = ()
    requires_caps: tuple[str, ...] = ()
    wires: tuple[Wire, ...] | None = None

    def applies_to(self, model: FrontierModel) -> bool:
        if self.wires is not None and model.wire not in self.wires:
            return False
        caps: Final = _WIRE_CAPS[model.wire]
        if not frozenset(self.requires_caps) <= caps:
            return False
        return all(
            getattr(model.rates, field, None) is not None for field in self.requires_rates
        )

    def scenario(self, scenario_id: str, model: FrontierModel, text: str) -> Scenario:
        return Scenario(
            scenario_id=scenario_id,
            wire=model.wire,
            usage=self.usage,
            model=model.provider_model,
            output=ScriptedOutput(
                text=text,
                response_model=model.override_model if self.response_model_override else None,
                tool_call=ScriptedToolCall(name="get_weather", arguments=TOOL_CALL_ARGUMENTS)
                if self.tool_call
                else None,
                terminal=self.terminal,
            ),
            stream_usage=self.stream_usage,
            service_tier=self.service_tier,
        )


class _CasesFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    deployments: tuple[DeploymentSpec, ...] = ()
    cases: tuple[Case, ...] = ()


CASES_FILE: Final = _CasesFile.model_validate(json.loads(CASES_PATH.read_text()))
CASES: Final[tuple[Case, ...]] = CASES_FILE.cases
_DEPLOYMENTS: Final[Mapping[str, DeploymentSpec]] = MappingProxyType(
    {spec.map_key: spec for spec in CASES_FILE.deployments}
)


@dataclass(frozen=True, slots=True)
class _ProviderWiring:
    """How a (litellm_provider, mode) pair maps to a sidecar wire, the provider
    prefix on the registered litellm model string, and extra litellm_params."""

    wire: Wire
    model_prefix: str | None
    litellm_params: Mapping[str, str]


_AZURE_PARAMS: Final[Mapping[str, str]] = MappingProxyType({"api_version": "2025-04-01-preview"})
_BEDROCK_PARAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
        "aws_secret_access_key": "scripted-secret",
        "aws_region_name": "us-east-1",
    }
)
_VERTEX_PARAMS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "vertex_project": "cc-scripted-project",
        "vertex_location": "us-central1",
    }
)

_PROVIDER_WIRING: Final[Mapping[tuple[str, str], _ProviderWiring]] = MappingProxyType(
    {
        ("openai", "chat"): _ProviderWiring("openai_chat", "openai", MappingProxyType({})),
        ("openai", "responses"): _ProviderWiring(
            "openai_responses", "openai", MappingProxyType({})
        ),
        ("anthropic", "chat"): _ProviderWiring(
            "anthropic_messages", "anthropic", MappingProxyType({})
        ),
        ("gemini", "chat"): _ProviderWiring("gemini_generate", None, MappingProxyType({})),
        ("together_ai", "chat"): _ProviderWiring("together_chat", None, MappingProxyType({})),
        ("fireworks_ai", "chat"): _ProviderWiring("fireworks_chat", None, MappingProxyType({})),
        ("azure", "chat"): _ProviderWiring("azure_chat", None, _AZURE_PARAMS),
        ("bedrock_converse", "chat"): _ProviderWiring(
            "bedrock_converse", "bedrock/converse", _BEDROCK_PARAMS
        ),
        ("vertex_ai-language-models", "chat"): _ProviderWiring(
            "vertex_generate", "vertex_ai", _VERTEX_PARAMS
        ),
    }
)


@dataclass(frozen=True, slots=True)
class FrontierModel:
    """One deployment under test, derived from a cost-map entry: the model_name
    the suite registers, the provider-prefixed litellm model string, the wire
    the scripted upstream speaks, and the sibling map model the response_model
    override case reports."""

    model_name: str
    litellm_model: str
    wire: Wire
    map_key: str
    override_model: str | None = None
    override_map_key: str | None = None
    # Registered as model_info.base_model; when set, the provider-reported
    # model loses to it and every case bills at this deployment's own rates.
    base_model: str | None = None
    litellm_params: Mapping[str, str] = MappingProxyType({})

    @property
    def rates(self) -> CostMapEntry:
        return COST_MAP[self.map_key]

    @property
    def override_rates(self) -> CostMapEntry:
        if self.base_model is not None or self.override_map_key is None:
            return self.rates
        return COST_MAP[self.override_map_key]

    @property
    def provider_model(self) -> str:
        """The bare provider-facing model name: litellm_model minus the provider
        prefix and any routing segment (converse/, responses/)."""
        return _provider_model(self.litellm_model)

    @property
    def provider(self) -> str:
        return self.rates.litellm_provider

    @property
    def api_key(self) -> str:
        # The scripted upstream ignores auth; a fixed bogus key proves the suite
        # spends zero real provider calls.
        return "sk-scripted-provider"


def _provider_model(litellm_model: str) -> str:
    tail: Final = litellm_model.split("/")[1:]
    return "/".join(tail[1:] if tail and tail[0] in ("converse", "responses") else tail)


def _litellm_model_for(map_key: str, wiring: _ProviderWiring) -> str:
    if wiring.model_prefix is None:
        return map_key
    if map_key.startswith(f"{wiring.model_prefix}/"):
        return map_key
    return f"{wiring.model_prefix}/{map_key}"


def _frontier() -> tuple[FrontierModel, ...]:
    groups: Final[Mapping[tuple[str, str], tuple[str, ...]]] = MappingProxyType(
        {
            pair: tuple(sorted(k for k, e in COST_MAP.items() if (e.litellm_provider, e.mode) == pair))
            for pair in {(e.litellm_provider, e.mode) for e in COST_MAP.values()}
        }
    )
    models: list[FrontierModel] = []  # mutable-ok: accumulated once at import into a tuple
    for map_key in sorted(COST_MAP):
        entry = COST_MAP[map_key]
        pair = (entry.litellm_provider, entry.mode)
        wiring = _PROVIDER_WIRING.get(pair)
        if wiring is None:
            raise ValueError(
                f"cost_map entry {map_key} has no wiring for "
                f"(litellm_provider={pair[0]}, mode={pair[1]}); add a "
                f"_ProviderWiring row in cost_matrix.py"
            )
        siblings = groups[pair]
        override_key = (
            siblings[(siblings.index(map_key) + 1) % len(siblings)] if len(siblings) > 1 else None
        )
        override_litellm = (
            _litellm_model_for(override_key, wiring) if override_key is not None else None
        )
        deployment = _DEPLOYMENTS.get(map_key)
        models.append(
            FrontierModel(
                model_name=f"cc-{map_key.replace('/', '-').replace(':', '-').replace('.', '-').lower()}",
                litellm_model=(
                    deployment.litellm_model
                    if deployment is not None and deployment.litellm_model is not None
                    else _litellm_model_for(map_key, wiring)
                ),
                wire=wiring.wire,
                map_key=map_key,
                override_model=(
                    _provider_model(override_litellm)
                    if override_litellm is not None
                    else None
                ),
                override_map_key=override_key,
                base_model=deployment.base_model if deployment is not None else None,
                litellm_params=wiring.litellm_params,
            )
        )
    return tuple(models)


FRONTIER_MODELS: Final[tuple[FrontierModel, ...]] = _frontier()

# Token kinds each wire can report, gating which pricing cases apply.
_WIRE_CAPS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "openai_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage", "tool_call", "image_input",
        }
    ),
    "openai_responses": frozenset(
        {
            "cache_read", "reasoning", "web_search", "response_model", "absent_usage",
            "tool_call", "image_input", "responses_terminal",
        }
    ),
    "anthropic_messages": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "web_search",
            "response_model", "absent_usage", "tool_call", "image_input",
        }
    ),
    "gemini_generate": frozenset(
        {
            "cache_read", "reasoning", "audio", "web_search", "response_model",
            "absent_usage", "tool_call", "image_input", "prompt_blocked",
        }
    ),
    "together_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage", "tool_call", "image_input",
        }
    ),
    "fireworks_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage", "tool_call", "image_input",
        }
    ),
    "azure_chat": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "audio",
            "web_search", "response_model", "absent_usage", "tool_call", "image_input",
        }
    ),
    "bedrock_converse": frozenset(
        {
            "cache_read", "cache_write_5m", "cache_write_1h", "absent_usage",
            "tool_call", "image_input",
        }
    ),
    "vertex_generate": frozenset(
        {
            "cache_read", "reasoning", "audio", "web_search", "response_model",
            "absent_usage", "tool_call", "image_input", "prompt_blocked",
        }
    ),
})

TOOL_CALL_ARGUMENTS: Final = json.dumps({
    "city": "Berlin",
    "days": 7,
    "units": "metric",
    "notes": "filler " * 30,
})


def cases_for(model: FrontierModel) -> tuple[Case, ...]:
    return tuple(case for case in CASES if case.applies_to(model))


def recount_cost(
    model: FrontierModel, case: Case, prompt_tokens: int, completion_tokens: int
) -> float:
    """What the proxy's own token recount should cost at the case's rates,
    without pinning the tokenizer's exact counts."""
    rates: Final = model.override_rates if case.response_model_override else model.rates
    return prompt_tokens * (rates.input_cost_per_token or 0.0) + completion_tokens * (
        rates.output_cost_per_token or 0.0
    )


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))


def image_input_data_url() -> str:
    """A deterministic 256x256 RGB noise PNG as a data URL; noise compresses
    poorly on purpose so the base64 payload stays well above 100 KB and would
    blow up the prompt recount if the URL were ever tokenized as text."""
    rng: Final = random.Random(0)
    side: Final = 256
    raw: Final = b"".join(
        b"\x00" + rng.randbytes(side * 3) for _ in range(side)
    )
    png: Final = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


IMAGE_INPUT_DATA_URL: Final = image_input_data_url()


class ExpectedCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float
    input_cost: float
    output_cost: float
    prompt_tokens: int
    completion_tokens: int


_EXPECTED_ADAPTER: Final = TypeAdapter(dict[str, ExpectedCell])
EXPECTED: Final[Mapping[str, ExpectedCell]] = MappingProxyType(
    _EXPECTED_ADAPTER.validate_python(json.loads(EXPECTED_PATH.read_text()))
    if EXPECTED_PATH.exists()
    else {}
)


def expected_key(model: FrontierModel, case: Case) -> str:
    return f"{model.map_key}|{case.name}"
