from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

CASES_PATH: Final = Path(__file__).resolve().parent / "cost_tracking_cases.json"
PRIOR_RESPONSE_ID_MARKER: Final = "$PRIOR_RESPONSE_ID"


class SearchContextCostPerQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    search_context_size_low: float | None = None
    search_context_size_medium: float | None = None
    search_context_size_high: float | None = None


class ProviderSpecificEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    fast: float | None = None
    us: float | None = None


class TieredPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    range: tuple[float, float]
    input_cost_per_token: float
    output_cost_per_token: float


class CostMapEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    litellm_provider: str
    mode: str
    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    supports_function_calling: bool | None = None
    input_cost_per_token: float | None = None
    input_cost_per_query: float | None = None
    output_cost_per_token: float | None = None
    input_cost_per_token_batches: float | None = None
    output_cost_per_token_batches: float | None = None
    input_cost_per_token_above_128k_tokens: float | None = None
    output_cost_per_token_above_128k_tokens: float | None = None
    output_vector_size: int | None = None
    input_cost_per_token_batches: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None
    cache_creation_input_token_cost_above_1hr: float | None = None
    cache_creation_input_token_cost_above_1hr_above_200k_tokens: float | None = None
    cache_read_input_token_cost_above_200k_tokens: float | None = None
    cache_creation_input_token_cost_above_200k_tokens: float | None = None
    input_cost_per_token_above_200k_tokens: float | None = None
    output_cost_per_token_above_200k_tokens: float | None = None
    cache_read_input_audio_token_cost: float | None = None
    tiered_pricing: tuple[TieredPrice, ...] | None = None
    output_cost_per_reasoning_token: float | None = None
    input_cost_per_audio_token: float | None = None
    input_cost_per_second: float | None = None
    output_cost_per_second: float | None = None
    input_cost_per_character: float | None = None
    output_cost_per_character: float | None = None
    input_cost_per_image: float | None = None
    output_cost_per_image: float | None = None
    output_cost_per_audio_token: float | None = None
    input_cost_per_image_token: float | None = None
    output_cost_per_image_token: float | None = None
    input_cost_per_video_token: float | None = None
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
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


class WavUpload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["wav"]
    seconds: float


class PngUpload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["png"]


Upload: TypeAlias = Annotated[WavUpload | PngUpload, Field(discriminator="kind")]


class JsonResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/json"]
    body: dict[str, JsonValue]
    status: int = 200


class SseResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["text/event-stream"]
    frames: tuple[str, ...]
    frame_delay_ms: int = Field(default=0, ge=0)


class EventStreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str
    payload: dict[str, JsonValue]


class EventStreamResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/vnd.amazon.eventstream"]
    events: tuple[EventStreamEvent, ...]
    framing: Literal["converse", "invoke"] = "converse"


class BinaryResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["audio/mpeg"]
    length: int


class TextResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/jsonl"]
    body: str
    status: int = 200


class RoutedResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/x-routed"]
    routes: dict[str, JsonResponse | TextResponse]


class RealtimeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    content_type: Literal["application/x-realtime"]
    events: tuple[dict[str, JsonValue], ...]
    session_model: str | None = None


StoredResponse: TypeAlias = Annotated[
    JsonResponse | SseResponse | EventStreamResponse | BinaryResponse | RoutedResponse | RealtimeResponse,
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
    breakdown_persisted: bool = True
    cost_header: bool = True
    rollups: bool = False


class RecountRates(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_cost_per_token: float
    output_cost_per_token: float


class RecountExpected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    recount: RecountRates
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    min_completion_tokens: int | None = None
    max_completion_tokens: int | None = None


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
    endpoint: (
        Literal[
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/completions",
            "/v1/moderations",
            "/v1/audio/transcriptions",
            "/v1/audio/speech",
            "/v1/images/generations",
            "/v1/images/edits",
        ]
        | Annotated[str, Field(pattern=r"^/(gemini|anthropic|bedrock)/")]
    ) = "/v1/chat/completions"
    deployment: Deployment | None = None
    upload: Upload | None = None
    request: dict[str, JsonValue]
    response: StoredResponse
    expected: Expected
    fallback_from: StoredResponse | None = None
    disconnect_after_frames: int | None = Field(default=None, ge=1)

    @property
    def rates(self) -> CostMapEntry:
        return COST_MAP[self.model]

    @property
    def litellm_model(self) -> str:
        provider: Final = self.rates.litellm_provider
        prefix: Final = (
            "openai"
            if provider == "openai"
            and (
                self.endpoint == "/v1/responses"
                or self.rates.mode
                in {"chat", "embedding", "moderation", "audio_transcription", "audio_speech", "image_generation"}
            )
            else "openai/responses"
            if provider == "openai"
            else _PROVIDER_PREFIXES.get(provider)
        )
        if prefix is None:
            raise ValueError(f"unsupported cost-map provider {provider} for {self.model}")
        if self.deployment and self.deployment.model is not None:
            return self.deployment.model
        if prefix == "" or self.model.startswith(f"{prefix}/"):
            return self.model
        return f"{prefix}/{self.model}"

    @property
    def litellm_params(self) -> Mapping[str, str]:
        return _LITELLM_PARAMS[self.rates.litellm_provider]

    @property
    def api_key(self) -> str:
        return "sk-scripted-provider"

    @property
    def base_model(self) -> str | None:
        return self.deployment.base_model if self.deployment else None

    @property
    def passthrough_provider(self) -> Literal["gemini", "anthropic", "bedrock"] | None:
        provider: Final = self.endpoint.removeprefix("/").split("/", 1)[0]
        if provider == "gemini":
            return "gemini"
        if provider == "anthropic":
            return "anthropic"
        if provider == "bedrock":
            return "bedrock"
        return None

    @property
    def reports_provider_cost(self) -> bool:
        if not isinstance(self.response, JsonResponse):
            return False
        usage: Final = self.response.body.get("usage")
        return isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float))

    @property
    def chains_prior_response(self) -> bool:
        return self.request.get("previous_response_id") == PRIOR_RESPONSE_ID_MARKER

    @property
    def can_chain_prior_response(self) -> bool:
        return (
            self.chains_prior_response
            and self.endpoint == "/v1/responses"
            and isinstance(self.response, JsonResponse)
            and isinstance(self.response.body.get("id"), str)
            and not isinstance(self.expected, FailureExpected)
            and not (isinstance(self.expected, ExactExpected) and self.expected.rollups)
        )


class BatchOutputLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status_code: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None

    @field_validator("status_code")
    @classmethod
    def validate_status_code(cls, value: int) -> int:
        if value != 200 and not 400 <= value <= 499:
            raise ValueError("status_code must be 200 or a 4xx status")
        return value

    @model_validator(mode="after")
    def validate_success_tokens(self) -> BatchOutputLine:
        if self.status_code == 200 and (self.prompt_tokens is None or self.completion_tokens is None):
            raise ValueError("successful batch output lines require prompt and completion tokens")
        return self

    def render(self, index: int, model: str, request_id: str) -> dict[str, JsonValue]:
        if self.status_code != 200:
            return {
                "id": f"batch_req_{index}",
                "custom_id": f"r{index}",
                "response": None,
                "error": {"code": "bad_request", "message": "failed"},
            }
        if self.prompt_tokens is None or self.completion_tokens is None:
            raise ValueError("successful batch output lines require prompt and completion tokens")
        usage: Final = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            **(
                {"prompt_tokens_details": {"cached_tokens": self.cached_tokens}}
                if self.cached_tokens is not None
                else {}
            ),
        }
        return {
            "id": f"batch_req_{index}",
            "custom_id": f"r{index}",
            "response": {
                "status_code": 200,
                "request_id": f"{request_id}-{index}",
                "body": {
                    "id": f"chatcmpl-{request_id}-{index}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            },
            "error": None,
        }


class BatchCostCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    covers: str
    model: str
    litellm_model: str
    output_lines: tuple[BatchOutputLine, ...]
    expected: ExactExpected

    @property
    def request_count(self) -> int:
        return len(self.output_lines) or 2

    @property
    def completed_count(self) -> int:
        return sum(line.status_code == 200 for line in self.output_lines)

    @property
    def failed_count(self) -> int:
        return self.request_count - self.completed_count


class RealtimeTurn(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int
    output_tokens: int
    input_text_tokens: int
    input_audio_tokens: int
    input_cached_tokens: int
    output_text_tokens: int
    output_audio_tokens: int

    @model_validator(mode="after")
    def validate_token_totals(self) -> RealtimeTurn:
        if self.input_text_tokens + self.input_audio_tokens != self.input_tokens:
            raise ValueError("input text and audio tokens must equal input_tokens")
        if self.output_text_tokens + self.output_audio_tokens != self.output_tokens:
            raise ValueError("output text and audio tokens must equal output_tokens")
        if self.input_cached_tokens > self.input_text_tokens:
            raise ValueError("input_cached_tokens must not exceed input_text_tokens")
        return self

    def render(self, index: int, request_id: str) -> dict[str, JsonValue]:
        return {
            "type": "response.done",
            "event_id": f"evt_{request_id}_{index}",
            "response": {
                "id": f"resp_{request_id}_{index}",
                "object": "realtime.response",
                "status": "completed",
                "output": [],
                "usage": {
                    "total_tokens": self.input_tokens + self.output_tokens,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "input_token_details": {
                        "text_tokens": self.input_text_tokens,
                        "audio_tokens": self.input_audio_tokens,
                        "cached_tokens": self.input_cached_tokens,
                        "cached_tokens_details": {
                            "text_tokens": self.input_cached_tokens,
                            "audio_tokens": 0,
                        },
                    },
                    "output_token_details": {
                        "text_tokens": self.output_text_tokens,
                        "audio_tokens": self.output_audio_tokens,
                    },
                },
            },
        }


class RealtimeCostCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    covers: str
    model: str
    litellm_model: str
    turns: tuple[RealtimeTurn, ...] = Field(min_length=0)
    session_model: str | None = None
    expected: ExactExpected


class _CasesFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cost_map: dict[str, CostMapEntry]
    cases: tuple[CostTrackingTestCase, ...]
    batch_cases: tuple[BatchCostCase, ...] = ()
    realtime_cases: tuple[RealtimeCostCase, ...] = ()


_PROVIDER_PREFIXES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic": "anthropic",
        "bedrock": "bedrock",
        "bedrock_converse": "bedrock/converse",
        "deepgram": "deepgram",
        "text-completion-openai": "text-completion-openai",
        "cohere": "cohere",
        "vertex_ai-language-models": "vertex_ai",
        "vertex_ai-image-models": "vertex_ai",
        "vertex_ai-embedding-models": "vertex_ai",
        "gemini": "",
        "together_ai": "",
        "fireworks_ai": "",
        "azure": "",
        "dashscope": "",
        "openrouter": "",
        "perplexity": "",
        "deepseek": "",
        "xai": "",
        "azure_ai": "azure_ai",
        "groq": "groq",
        "mistral": "mistral",
        "cohere_chat": "cohere_chat",
    }
)
_LITELLM_PARAMS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "anthropic": MappingProxyType({}),
        "bedrock": MappingProxyType(
            {
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            }
        ),
        "bedrock_converse": MappingProxyType(
            {
                "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
                "aws_secret_access_key": "scripted-secret",
                "aws_region_name": "us-east-1",
            }
        ),
        "deepgram": MappingProxyType({}),
        "text-completion-openai": MappingProxyType({}),
        "cohere": MappingProxyType({}),
        "vertex_ai-language-models": MappingProxyType(
            {"vertex_project": "cc-scripted-project", "vertex_location": "us-central1"}
        ),
        "vertex_ai-image-models": MappingProxyType(
            {"vertex_project": "cc-scripted-project", "vertex_location": "us-central1"}
        ),
        "vertex_ai-embedding-models": MappingProxyType(
            {"vertex_project": "cc-scripted-project", "vertex_location": "us-central1"}
        ),
        "gemini": MappingProxyType({}),
        "together_ai": MappingProxyType({}),
        "fireworks_ai": MappingProxyType({}),
        "azure": MappingProxyType({"api_version": "2025-04-01-preview"}),
        "openai": MappingProxyType({}),
        "dashscope": MappingProxyType({}),
        "openrouter": MappingProxyType({}),
        "perplexity": MappingProxyType({}),
        "deepseek": MappingProxyType({}),
        "xai": MappingProxyType({}),
        "azure_ai": MappingProxyType({}),
        "groq": MappingProxyType({}),
        "mistral": MappingProxyType({}),
        "cohere_chat": MappingProxyType({}),
    }
)

_LOADED: Final = _CasesFile.model_validate_json(CASES_PATH.read_bytes())
COST_MAP: Final[Mapping[str, CostMapEntry]] = MappingProxyType(dict(_LOADED.cost_map))
CASES: Final[tuple[CostTrackingTestCase, ...]] = _LOADED.cases
BATCH_CASES: Final[tuple[BatchCostCase, ...]] = _LOADED.batch_cases
REALTIME_CASES: Final[tuple[RealtimeCostCase, ...]] = _LOADED.realtime_cases
_ALL_CASES: Final = CASES + BATCH_CASES + REALTIME_CASES
_LITELLM_MODELS: Final = tuple(case.litellm_model for case in _ALL_CASES)


def data_errors() -> tuple[str, ...]:
    case_models: Final = frozenset(case.model for case in _ALL_CASES) | frozenset(
        case.session_model for case in REALTIME_CASES if case.session_model is not None
    )
    unknown_models: Final = sorted(model for model in case_models if model not in COST_MAP)
    missing_cases: Final = sorted(model for model in COST_MAP if model not in case_models)
    duplicate_names: Final = sorted(
        name for name in {case.name for case in _ALL_CASES} if sum(case.name == name for case in _ALL_CASES) > 1
    )
    input_rates: Final = tuple(
        (entry.input_cost_per_token, model)
        for model, entry in COST_MAP.items()
        if entry.mode != "realtime"
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
            and (
                not isinstance(case.response, JsonResponse)
                or not 400 <= case.response.status <= 599
                or not 400 <= case.expected.failure.status <= 599
            )
        )
        or (
            not isinstance(case.expected, FailureExpected)
            and isinstance(case.response, JsonResponse)
            and case.response.status != 200
        )
    )
    invalid_opt_outs: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and (
            (
                not case.expected.breakdown_persisted
                and case.passthrough_provider is None
                and case.rates.mode != "image_generation"
                and not case.reports_provider_cost
            )
            or (
                not case.expected.cost_header
                and case.passthrough_provider is None
                and not isinstance(case.response, SseResponse)
                and case.expected.spend != 0.0
            )
        )
    )
    invalid_fallbacks: Final = sorted(
        case.name
        for case in CASES
        if case.fallback_from is not None
        and (
            not isinstance(case.fallback_from, JsonResponse)
            or not 400 <= case.fallback_from.status <= 599
        )
    )
    invalid_disconnects: Final = sorted(
        case.name
        for case in CASES
        if case.disconnect_after_frames is not None
        and (
            not isinstance(case.response, SseResponse)
            or case.response.frame_delay_ms <= 0
            or not isinstance(case.expected, RecountExpected)
        )
    )
    invalid_rollup_ids: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and case.expected.rollups
        and "$UNIQUE_ID" not in case.response.model_dump_json()
    )
    invalid_pinned_tool_ids: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, RecountExpected)
        and (case.expected.prompt_tokens is not None or case.expected.completion_tokens is not None)
        and any(
            marker in case.response.model_dump_json()
            for marker in ('"id": "call_$REQUEST_ID"', '"id": "toolu_$REQUEST_ID"')
        )
    )
    invalid_prior_response_chains: Final = sorted(
        case.name
        for case in CASES
        if PRIOR_RESPONSE_ID_MARKER in json.dumps(case.request) and not case.can_chain_prior_response
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
            f"invalid passthrough opt-outs: {invalid_opt_outs}" if invalid_opt_outs else None,
            f"invalid fallback responses: {invalid_fallbacks}" if invalid_fallbacks else None,
            f"invalid disconnect cases: {invalid_disconnects}" if invalid_disconnects else None,
            f"rollup responses lack $UNIQUE_ID: {invalid_rollup_ids}" if invalid_rollup_ids else None,
            f"pinned tool IDs contain $REQUEST_ID: {invalid_pinned_tool_ids}"
            if invalid_pinned_tool_ids
            else None,
            f"{PRIOR_RESPONSE_ID_MARKER} needs a non-rollup, non-failure /v1/responses JSON response with a string id"
            f" as previous_response_id: {invalid_prior_response_chains}"
            if invalid_prior_response_chains
            else None,
        )
        if message is not None
    )


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)
