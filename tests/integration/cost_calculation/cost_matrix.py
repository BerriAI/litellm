"""The cost-calculation matrix: the model set derived from the test cost map,
the request/response cases from ``cases.json``, and the loaders both use.

Two data files drive the suite; nothing in Python lists models or cases:
- ``tests/integration/cost_calculation/cost_map.json`` is the proxy's ENTIRE model cost map
  (LITELLM_MODEL_COST_MAP_URL); every entry becomes a deployment under test.
- ``tests/integration/cost_calculation/cases.json`` is the case list plus the reviewed
  goldens: each exact-spend case carries an ``expected`` cell per map key it
  runs against, each recount case carries its ``models`` list, so matrix
  membership and expected values are literal data read side by side.
"""

from __future__ import annotations

import base64
import io
import json
import math
import random
import struct
import wave
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from integration._support.scripted_wires import Scenario, ScriptedOutput, ScriptedToolCall, ScriptedUsage, Wire

COST_MAP_PATH: Final = Path(__file__).resolve().parent / "cost_map.json"
CASES_PATH: Final = Path(__file__).resolve().parent / "cases.json"

class SearchContextCostPerQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_context_size_low: float | None = None
    search_context_size_medium: float | None = None
    search_context_size_high: float | None = None


class ProviderSpecificEntry(BaseModel):
    """Provider-specific key rates, keyed by the named suffix litellm looks up
    (``fast`` for Anthropic fast mode, ``us`` for US inference geography)."""

    model_config = ConfigDict(frozen=True)

    fast: float | None = None
    us: float | None = None


class CostMapEntry(BaseModel):
    """The pricing fields of a cost-map entry the matrix reads. Shaped like a
    ``model_prices_and_context_window.json`` entry; the file is test-owned so
    undeclared keys are forbidden rather than ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    litellm_provider: str
    mode: str
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_function_calling: bool | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    cache_creation_input_token_cost_above_1hr: float | None = None
    cache_read_input_token_cost_above_200k_tokens: float | None = None
    cache_creation_input_token_cost_above_200k_tokens: float | None = None
    output_cost_per_reasoning_token: float | None = None
    input_cost_per_audio_token: float | None = None
    output_cost_per_audio_token: float | None = None
    input_cost_per_image_token: float | None = None
    input_cost_per_video_token: float | None = None
    input_cost_per_token_above_200k_tokens: float | None = None
    output_cost_per_token_above_200k_tokens: float | None = None
    input_cost_per_token_flex: float | None = None
    output_cost_per_token_flex: float | None = None
    input_cost_per_token_priority: float | None = None
    output_cost_per_token_priority: float | None = None
    search_context_cost_per_query: SearchContextCostPerQuery | None = None
    web_search_billing_unit: str | None = None
    google_maps_grounding_cost_per_query: float | None = None
    file_search_cost_per_1k_calls: float | None = None
    provider_specific_entry: ProviderSpecificEntry | None = None


_METADATA_FIELDS: Final = frozenset(
    {
        "litellm_provider",
        "mode",
        "max_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "supports_function_calling",
    }
)
_CONTAINER_FIELDS: Final = frozenset({"search_context_cost_per_query", "provider_specific_entry"})


def _submodel_rate_keys(
    field: str, sub: SearchContextCostPerQuery | ProviderSpecificEntry | None
) -> tuple[str, ...]:
    if sub is None:
        return ()
    return tuple(
        f"{field}.{name}"
        for name in type(sub).model_fields
        if getattr(sub, name) is not None
    )


def _entry_rate_keys(entry: CostMapEntry) -> frozenset[str]:
    """Every cost key an entry carries, with container subfields expanded to
    dotted names (``search_context_cost_per_query.search_context_size_low``).
    ``web_search_billing_unit`` counts as a rate key whenever present,
    for both ``per_query`` and ``per_prompt`` values."""
    plain: Final = frozenset(
        name
        for name in CostMapEntry.model_fields
        if name not in _METADATA_FIELDS
        and name not in _CONTAINER_FIELDS
        and getattr(entry, name) is not None
    )
    return (
        plain
        | frozenset(
            _submodel_rate_keys("search_context_cost_per_query", entry.search_context_cost_per_query)
        )
        | frozenset(_submodel_rate_keys("provider_specific_entry", entry.provider_specific_entry))
    )


def _entry_has_rate_key(entry: CostMapEntry, rate_key: str) -> bool:
    outer, _, inner = rate_key.partition(".")
    if outer == "search_context_cost_per_query":
        return f"{outer}.{inner}" in _submodel_rate_keys(outer, entry.search_context_cost_per_query)
    if outer == "provider_specific_entry":
        return f"{outer}.{inner}" in _submodel_rate_keys(outer, entry.provider_specific_entry)
    value: Final[object] = getattr(entry, outer, None)
    return value is not None


SERVICE_TIER_REQUEST_WIRES: Final = frozenset(
    {"openai_chat", "azure_chat", "openai_responses", "bedrock_converse"}
)


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


class ExpectedCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float
    input_cost: float
    output_cost: float
    prompt_tokens: int
    completion_tokens: int


class Case(BaseModel):
    """One request/response shape from cases.json.

    ``family`` splits the matrix: ``pricing`` cases own cost keys (``owns``,
    dotted subfield names allowed) or declare which keys they deliberately
    leave absent (``fallback_for``) so every cost key in the map has exactly
    one owning case; ``transport`` cases exercise counting/transport only and
    run wherever they list membership. An exact-spend case names its models
    implicitly by carrying one ``expected`` golden per map key; a recount
    case (``exact_spend=False``) names them in ``models`` instead. The
    feature flags drive request realism in ``_chat_body``."""

    model_config = ConfigDict(frozen=True)

    name: str
    family: Literal["pricing", "transport"]
    usage: ScriptedUsage
    usage_by_model: Mapping[str, ScriptedUsage] = Field(default_factory=lambda: MappingProxyType({}))
    stream: bool = False
    stream_usage: Literal["final_chunk", "absent"] = "final_chunk"
    service_tier: Literal["flex", "priority"] | None = None
    speed: Literal["fast"] | None = None
    inference_geo: Literal["us"] | None = None
    response_model_override: bool = False
    exact_spend: bool = True
    tool_call: bool = False
    image_input: bool = False
    audio_input: bool = False
    audio_output: bool = False
    video_input: bool = False
    reasoning: bool = False
    web_search: Literal["low", "medium", "high"] | None = None
    google_maps: bool = False
    file_search: bool = False
    terminal: Literal["completed", "incomplete", "unvalidated", "prompt_blocked"] = "completed"
    owns: tuple[str, ...] = ()
    fallback_for: tuple[str, ...] = ()
    expected: Mapping[str, ExpectedCell] = Field(default_factory=lambda: MappingProxyType({}))
    models: tuple[str, ...] = ()

    def applies_to(self, model: FrontierModel) -> bool:
        if self.exact_spend:
            return model.map_key in self.expected
        return model.map_key in self.models

    def expected_for(self, model: FrontierModel) -> ExpectedCell:
        return self.expected[model.map_key]

    def usage_for(self, map_key: str) -> ScriptedUsage:
        return self.usage_by_model.get(map_key, self.usage)

    def scenario(self, scenario_id: str, model: FrontierModel, text: str) -> Scenario:
        return Scenario(
            scenario_id=scenario_id,
            wire=model.wire,
            usage=self.usage_for(model.map_key),
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
            speed=self.speed,
            inference_geo=self.inference_geo,
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
            "openai_responses", "openai/responses", MappingProxyType({})
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
        # bedrock_converse responses carry no model field, so a reported-model
        # override can never repoint pricing there, same as a base_model pin.
        if (
            self.base_model is not None
            or self.wire == "bedrock_converse"
            or self.override_map_key is None
        ):
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


def audio_input_data_url() -> str:
    """A deterministic 0.5 s 16-bit PCM WAV (8 kHz, 220 Hz sine) as a data
    URL, small enough to stay a fixture but real audio to the provider."""
    frames: Final = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 220 * i / 8000)))
        for i in range(4000)
    )
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(frames)
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def video_input_data_url() -> str:
    """A deterministic mp4-looking blob (ftyp box plus a fixed mdat payload)
    as a data URL; only the media type and bytes matter to the wire."""
    ftyp: Final = struct.pack(">I4s4sI4s4s", 24, b"ftyp", b"isom", 0x200, b"isom", b"iso6")
    mdat_payload: Final = bytes((i * 7 + 13) % 256 for i in range(4096))
    mdat: Final = struct.pack(">I4s", 8 + len(mdat_payload), b"mdat") + mdat_payload
    return "data:video/mp4;base64," + base64.b64encode(ftyp + mdat).decode()


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
AUDIO_INPUT_DATA_URL: Final = audio_input_data_url()
VIDEO_INPUT_DATA_URL: Final = video_input_data_url()


def matrix_data_errors() -> tuple[str, ...]:
    """Consistency findings for the data files, as human-readable strings.

    Called at collection time by the integration suite, so a map key named by a case
    but absent from cost_map.json fails the suite's collection loudly.
    """
    unknown_deployments: Final = sorted(
        spec.map_key for spec in CASES_FILE.deployments if spec.map_key not in COST_MAP
    )
    unknown_case_models: Final = sorted(
        {
            map_key
            for case in CASES
            for map_key in (*case.expected, *case.models)
            if map_key not in COST_MAP
        }
    )
    misshapen_cases: Final = sorted(
        case.name
        for case in CASES
        if case.exact_spend == bool(case.models) or case.exact_spend != bool(case.expected)
    )
    all_pairs: Final = frozenset(
        (map_key, key)
        for map_key, entry in COST_MAP.items()
        for key in _entry_rate_keys(entry)
    )
    owned_pairs: Final = tuple(
        (map_key, key)
        for case in CASES
        if case.family == "pricing"
        for map_key in case.expected
        for key in case.owns
        if map_key in COST_MAP and _entry_has_rate_key(COST_MAP[map_key], key)
    )
    unowned_pairs: Final = sorted(
        f"{map_key}:{key}" for map_key, key in all_pairs - frozenset(owned_pairs)
    )
    duplicate_pairs: Final = sorted(
        f"{map_key}:{key}"
        for map_key, key in set(owned_pairs)
        if owned_pairs.count((map_key, key)) > 1
    )
    owns_without_holder: Final = sorted(
        f"{case.name}:{key}"
        for case in CASES
        for key in case.owns
        if not any(
            map_key in COST_MAP and _entry_has_rate_key(COST_MAP[map_key], key)
            for map_key in case.expected
        )
    )
    fallback_violations: Final = sorted(
        f"{case.name}:{map_key}:{key}"
        for case in CASES
        for key in case.fallback_for
        for map_key in (*case.expected, *case.models)
        if map_key in COST_MAP and _entry_has_rate_key(COST_MAP[map_key], key)
    )
    family_violations: Final = sorted(
        case.name
        for case in CASES
        if (case.family == "transport") != (not case.owns and not case.fallback_for)
    )
    input_rates: Final = tuple(entry.input_cost_per_token for entry in COST_MAP.values())
    findings: Final = (
        (
            f"deployments entries name map keys absent from cost_map.json: {unknown_deployments}"
            if unknown_deployments
            else None
        ),
        (
            f"case expected/models name map keys absent from cost_map.json: {unknown_case_models}"
            if unknown_case_models
            else None
        ),
        (
            f"cases must carry expected xor models (exact_spend matches the field): {misshapen_cases}"
            if misshapen_cases
            else None
        ),
        (
            "two cost_map entries share input_cost_per_token; the suite relies on "
            "distinct rates so a wrong-model bill can never coincidentally match"
            if len(input_rates) != len(set(input_rates))
            else None
        ),
        (
            f"(model, rate key) pairs with no owning case: {unowned_pairs}"
            if unowned_pairs
            else None
        ),
        (
            f"(model, rate key) pairs owned by more than one case: {duplicate_pairs}"
            if duplicate_pairs
            else None
        ),
        (
            f"owns keys absent on all of the case's expected models: {owns_without_holder}"
            if owns_without_holder
            else None
        ),
        (
            f"fallback_for keys a case's models actually carry: {fallback_violations}"
            if fallback_violations
            else None
        ),
        (
            f"cases with owns/fallback_for inconsistent with family: {family_violations}"
            if family_violations
            else None
        ),
    )
    return tuple(finding for finding in findings if finding is not None)
