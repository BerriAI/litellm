from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

CASES_PATH: Final = Path(__file__).resolve().parent / "cost_tracking_cases.json"


class SearchContextCostPerQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_context_size_low: float | None = None
    search_context_size_medium: float | None = None
    search_context_size_high: float | None = None


class ProviderSpecificEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast: float | None = None
    us: float | None = None


class CostMapEntry(BaseModel):
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


class Deployment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None
    base_model: str | None = None


class JsonResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/json"]
    body: dict[str, JsonValue]
    status: int = 200


class SseResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["text/event-stream"]
    frames: tuple[str, ...]


class EventStreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str
    payload: dict[str, JsonValue]


class EventStreamResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/vnd.amazon.eventstream"]
    events: tuple[EventStreamEvent, ...]


StoredResponse: TypeAlias = Annotated[
    JsonResponse | SseResponse | EventStreamResponse,
    Field(discriminator="content_type"),
]


class ExactExpected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spend: float
    input_cost: float
    output_cost: float
    prompt_tokens: int
    completion_tokens: int
    cache_read_cost: float | None = None
    cache_creation_cost: float | None = None
    reasoning_cost: float | None = None
    tool_usage_cost: float | None = None


class RecountRates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_cost_per_token: float
    output_cost_per_token: float


class RecountExpected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recount: RecountRates


class FailureDetails(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: int


class FailureExpected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    failure: FailureDetails


Expected: TypeAlias = ExactExpected | RecountExpected | FailureExpected


class CostTrackingTestCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    covers: str
    model: str
    endpoint: Literal[
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/completions",
        "/v1/moderations",
    ] = "/v1/chat/completions"
    deployment: Deployment | None = None
    request: dict[str, JsonValue]
    response: StoredResponse
    expected: Expected

    @property
    def rates(self) -> CostMapEntry:
        return COST_MAP[self.model]

    @property
    def litellm_model(self) -> str:
        provider: Final = self.rates.litellm_provider
        prefix: Final = (
            "openai"
            if provider == "openai" and self.rates.mode == "chat"
            else "openai/responses"
            if provider == "openai"
            else _PROVIDER_PREFIXES.get(provider)
        )
        if prefix is None:
            raise ValueError(f"unsupported cost-map provider {provider} for {self.model}")
        return self.deployment.model if self.deployment and self.deployment.model is not None else (
            self.model if prefix == "" else f"{prefix}/{self.model}"
        )

    @property
    def litellm_params(self) -> Mapping[str, str]:
        return _LITELLM_PARAMS[self.rates.litellm_provider]

    @property
    def api_key(self) -> str:
        return "sk-scripted-provider"

    @property
    def base_model(self) -> str | None:
        return self.deployment.base_model if self.deployment else None


class _CasesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_map: dict[str, CostMapEntry]
    cases: tuple[CostTrackingTestCase, ...]


_PROVIDER_PREFIXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic": "anthropic",
        "bedrock_converse": "bedrock/converse",
        "vertex_ai-language-models": "vertex_ai",
        "gemini": "",
        "together_ai": "",
        "fireworks_ai": "",
        "azure": "",
    }
)
_LITELLM_PARAMS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "anthropic": MappingProxyType({}),
        "bedrock_converse": MappingProxyType(
            {
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            }
        ),
        "vertex_ai-language-models": MappingProxyType(
            {"vertex_project": "cc-scripted-project", "vertex_location": "us-central1"}
        ),
        "gemini": MappingProxyType({}),
        "together_ai": MappingProxyType({}),
        "fireworks_ai": MappingProxyType({}),
        "azure": MappingProxyType({"api_version": "2025-04-01-preview"}),
        "openai": MappingProxyType({}),
    }
)

_LOADED: Final = _CasesFile.model_validate_json(CASES_PATH.read_bytes())
COST_MAP: Final[Mapping[str, CostMapEntry]] = MappingProxyType(dict(_LOADED.cost_map))
CASES: Final[tuple[CostTrackingTestCase, ...]] = _LOADED.cases
_LITELLM_MODELS: Final = tuple(case.litellm_model for case in CASES)


def data_errors() -> tuple[str, ...]:
    case_models: Final = frozenset(case.model for case in CASES)
    unknown_models: Final = sorted(case.model for case in CASES if case.model not in COST_MAP)
    missing_cases: Final = sorted(model for model in COST_MAP if model not in case_models)
    duplicate_names: Final = sorted(
        name for name in {case.name for case in CASES} if sum(case.name == name for case in CASES) > 1
    )
    input_rates: Final = tuple(
        (entry.input_cost_per_token, model) for model, entry in COST_MAP.items()
    )
    shared_input_rates: Final = sorted(
        f"{rate}: {tuple(model for value, model in input_rates if value == rate)}"
        for rate in {value for value, _ in input_rates if value is not None}
        if sum(value == rate for value, _ in input_rates) > 1
    )
    recount_mismatches: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, RecountExpected)
        and case.model in COST_MAP
        and (
            case.expected.recount.input_cost_per_token != (COST_MAP[case.model].input_cost_per_token or 0.0)
            or case.expected.recount.output_cost_per_token != (COST_MAP[case.model].output_cost_per_token or 0.0)
        )
    )
    component_mismatches: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and any(
            component is not None
            for component in (
                case.expected.cache_read_cost,
                case.expected.cache_creation_cost,
                case.expected.reasoning_cost,
                case.expected.tool_usage_cost,
            )
        )
        and (
            (case.expected.cache_read_cost or 0.0) + (case.expected.cache_creation_cost or 0.0)
            > case.expected.input_cost
            or (case.expected.reasoning_cost or 0.0) > case.expected.output_cost
            or not _approx_equal(
                case.expected.input_cost
                + case.expected.output_cost
                + (case.expected.tool_usage_cost or 0.0),
                case.expected.spend,
            )
        )
    )
    failure_response_mismatches: Final = sorted(
        case.name
        for case in CASES
        if (
            isinstance(case.expected, FailureExpected)
            and (not isinstance(case.response, JsonResponse) or case.response.status < 400)
        )
        or (
            not isinstance(case.expected, FailureExpected)
            and isinstance(case.response, JsonResponse)
            and case.response.status != 200
        )
    )
    return tuple(
        message
        for message in (
            f"case models absent from cost_map: {unknown_models}" if unknown_models else None,
            f"cost-map entries without cases: {missing_cases}" if missing_cases else None,
            f"duplicate case names: {duplicate_names}" if duplicate_names else None,
            f"cost-map entries share input_cost_per_token: {shared_input_rates}" if shared_input_rates else None,
            f"recount rates differ from cost-map rates: {recount_mismatches}" if recount_mismatches else None,
            f"breakdown components are inconsistent: {component_mismatches}" if component_mismatches else None,
            f"failure response statuses are inconsistent: {failure_response_mismatches}"
            if failure_response_mismatches
            else None,
        )
        if message is not None
    )


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)
