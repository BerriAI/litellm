"""Python's side of the Rust pricing core (``NativeModelCostMap.price``, ``price_units``, ``bound``, ``rates`` and
``adjust``).

Each helper reads the facts Python holds into the document the bridge reads, and hands back what the old Python
API returned. ``None`` from ``price`` means Rust left the call to Python, which then prices it the old way."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    BilledTokenRates,
    BilledUsage,
    CacheReadCounts,
    CacheWriteCounts,
    ModalityCounts,
    billed_usage,
    parse_completion_tokens_details,
    parse_prompt_tokens_details,
)
from litellm.rust_bridge._native import NativeRateCard, NativeReceipt, NativeUnpriced
from litellm.types.utils import CacheCreationTokenDetails, Usage

SearchContextSize = Literal["low", "medium", "high"]


class WebSearchUse(TypedDict):
    requests: ReadOnly[int]
    context: ReadOnly[SearchContextSize | None]


class PricedUsage(BilledUsage):
    web_search: ReadOnly[WebSearchUse | None]
    speed: ReadOnly[Literal["fast"] | None]
    inference_geo: ReadOnly[str | None]


class CallFacts(TypedDict):
    """What ``completion_cost`` knows about a finished call before it picks a row."""

    by: ReadOnly[Literal["call"]]
    provider: ReadOnly[str]
    request_model: ReadOnly[str | None]
    response_model: ReadOnly[str | None]
    base_model: ReadOnly[str | None]
    custom_pricing: ReadOnly[bool]
    deployment_id: ReadOnly[str | None]
    region: ReadOnly[str | None]
    invoked_model_id: ReadOnly[str | None]
    provider_response_model: ReadOnly[str | None]
    hidden_model: ReadOnly[str | None]


class ModelFacts(TypedDict):
    """One model a ``cost_per_token`` caller names. ``row_provider`` lowers the row the caller passed when it is not
    the call's provider, as Fireworks AI bills a row it found under another provider."""

    by: ReadOnly[Literal["model"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    region: ReadOnly[str | None]
    row_provider: ReadOnly[NotRequired[str]]


class AdjustFacts(TypedDict):
    provider: ReadOnly[str | None]
    model: ReadOnly[str | None]
    region: ReadOnly[str | None]


class RerankDocument(TypedDict):
    kind: ReadOnly[Literal["rerank"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    search_units: ReadOnly[int | None]
    total_tokens: ReadOnly[int | None]


class VectorStoreSearchDocument(TypedDict):
    kind: ReadOnly[Literal["vector_store_search"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]


class OcrDocument(TypedDict):
    kind: ReadOnly[Literal["ocr"]]
    model: ReadOnly[str]
    provider: ReadOnly[str | None]
    pages: ReadOnly[int | None]
    annotation_pages: ReadOnly[int]
    credits: ReadOnly[float | None]
    batch: ReadOnly[bool]
    catalog_deferred: ReadOnly[bool]


class SpeechDocument(TypedDict):
    kind: ReadOnly[Literal["speech"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    characters: ReadOnly[int | None]
    generations: ReadOnly[int | None]


class TranscriptionDocument(TypedDict):
    kind: ReadOnly[Literal["transcription"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    seconds: ReadOnly[float]
    tokens: ReadOnly[bool]


class SearchDocument(TypedDict):
    kind: ReadOnly[Literal["search"]]
    model: ReadOnly[str]
    provider: ReadOnly[str | None]
    queries: ReadOnly[int]
    additional_results: ReadOnly[int]
    adjust: ReadOnly[AdjustFacts | None]


class VideoDocument(TypedDict):
    kind: ReadOnly[Literal["video"]]
    model: ReadOnly[str]
    provider: ReadOnly[str | None]
    seconds: ReadOnly[float | None]
    resolution: ReadOnly[str | None]
    videos: ReadOnly[int]
    input_images: ReadOnly[int]
    deployment: ReadOnly[bool]
    requested_row: ReadOnly[bool]


class ImageDocument(TypedDict):
    kind: ReadOnly[Literal["image"]]
    model: ReadOnly[str]
    provider: ReadOnly[str | None]
    calculator: ReadOnly[Literal["default", "per_image", "per_image_and_input"]]
    images: ReadOnly[int]
    pixels: ReadOnly[int]
    input_images: ReadOnly[int]


class WritesByTtl(TypedDict):
    five_minutes: ReadOnly[int | None]
    thirty_minutes: ReadOnly[int | None]
    one_hour: ReadOnly[int | None]
    twelve_hours: ReadOnly[int | None]


class BatchUsage(TypedDict):
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    completion_image_tokens: ReadOnly[int]
    cache_hit_tokens: ReadOnly[int]
    cache_hit_audio_tokens: ReadOnly[int]
    cached_image_tokens: ReadOnly[int]
    cache_creation_tokens: ReadOnly[int]
    cache_creation_by_ttl: ReadOnly[WritesByTtl | None]
    audio_tokens: ReadOnly[int]
    image_tokens: ReadOnly[int]
    video_tokens: ReadOnly[int]


TokenRules = Literal["branch", "generic", "generic_tier", "override"]


class Elapsed(TypedDict):
    """How long the call ran and the audio file it read, for a row billed by the second."""

    response_ms: ReadOnly[float]
    audio_seconds: ReadOnly[float]


class Characters(TypedDict):
    """The prompt and completion characters Vertex AI bills, ``None`` for a side nobody counted."""

    prompt: ReadOnly[int | None]
    completion: ReadOnly[int | None]


class RawCounts(TypedDict):
    """Counts as the provider reported them, for the provider branches that bill those rather than the billed split."""

    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    total_tokens: ReadOnly[int]
    cached_tokens: ReadOnly[int]
    cache_creation_tokens: ReadOnly[int]
    cache_writes: ReadOnly[WritesByTtl | None]
    reasoning_tokens: ReadOnly[int]
    stated_reasoning_tokens: ReadOnly[int]
    citation_tokens: ReadOnly[int]
    search_queries: ReadOnly[int]


class ReportedCost(TypedDict):
    source: ReadOnly[str]
    usd: ReadOnly[float]


class CustomRates(TypedDict):
    input: ReadOnly[float]
    output: ReadOnly[float]
    cache_read: ReadOnly[float | None]
    cache_write: ReadOnly[float | None]


class CustomPricing(TypedDict):
    """``custom_cost_per_token`` or ``custom_cost_per_second``, whichever the caller set."""

    per_token: ReadOnly[CustomRates | None]
    per_second: ReadOnly[float | None]


BreakdownFrom = Literal["priced", "named", "row", "absent"]

FeeBilling = Literal["prompt", "additional"]


class RouterFeeUse(TypedDict):
    billed: ReadOnly[FeeBilling]


class _PriceDocument(TypedDict):
    named: ReadOnly[CallFacts | ModelFacts]
    usage: ReadOnly[PricedUsage]
    requested_model: ReadOnly[str | None]
    service_tier: ReadOnly[str | None]
    data_residency: ReadOnly[str | None]
    vertex_location: ReadOnly[str | None]
    at: ReadOnly[str]
    adjust: ReadOnly[AdjustFacts | None]
    rules: ReadOnly[TokenRules]
    elapsed: ReadOnly[Elapsed | None]
    characters: ReadOnly[Characters | None]
    embedding: ReadOnly[bool]
    raw: ReadOnly[RawCounts | None]
    reported: ReadOnly[ReportedCost | None]
    custom: ReadOnly[CustomPricing | None]
    decisions: ReadOnly[bool]
    breakdown: ReadOnly[BreakdownFrom]
    tool_fees: ReadOnly[float]
    router_fee: ReadOnly[RouterFeeUse | None]


class BatchLineDocument(TypedDict):
    kind: ReadOnly[Literal["batch_line"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    usage: ReadOnly[BatchUsage]
    data_residency: ReadOnly[str | None]
    vertex_location: ReadOnly[str | None]


class ManyDocument(BatchLineDocument):
    """One line of a ``price_many`` request and the index of its row among the rows passed beside it."""

    row: ReadOnly[int | None]


UnitsDocument = (
    BatchLineDocument
    | ImageDocument
    | VideoDocument
    | SearchDocument
    | RerankDocument
    | VectorStoreSearchDocument
    | OcrDocument
    | SpeechDocument
    | TranscriptionDocument
)


class AgentUsage(TypedDict):
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]


class A2aDocument(TypedDict):
    kind: ReadOnly[Literal["a2a"]]
    cost_per_query: ReadOnly[float | None]
    input_cost_per_token: ReadOnly[float | None]
    output_cost_per_token: ReadOnly[float | None]
    usage: ReadOnly[AgentUsage | None]


class McpDocument(TypedDict):
    kind: ReadOnly[Literal["mcp"]]
    tool: ReadOnly[str]
    tool_prices: ReadOnly[Mapping[str, float]]
    default_price: ReadOnly[float | None]


class GuardrailUse(TypedDict):
    counter: ReadOnly[str]
    units: ReadOnly[int]


class GuardrailDocument(TypedDict):
    kind: ReadOnly[Literal["guardrail"]]
    keys: ReadOnly[Sequence[str]]
    units: ReadOnly[Sequence[GuardrailUse]]


class PromptShieldDocument(TypedDict):
    kind: ReadOnly[Literal["prompt_shield"]]
    units: ReadOnly[Sequence[GuardrailUse]]
    free_tier: ReadOnly[bool]
    per_thousand: ReadOnly[float | None]


class DurationClip(TypedDict):
    type: ReadOnly[Literal["duration"]]
    seconds: ReadOnly[float]


class TokenClip(TypedDict):
    type: ReadOnly[Literal["tokens"]]
    audio: ReadOnly[float]
    text: ReadOnly[float]
    output: ReadOnly[float]


class OtherClip(TypedDict):
    type: ReadOnly[Literal["other"]]


ClipUsage: TypeAlias = DurationClip | TokenClip | OtherClip


class ClipUse(TypedDict):
    rows: ReadOnly[Sequence[int]]
    usage: ReadOnly[ClipUsage]


class TranslationUse(TypedDict):
    input_seconds: ReadOnly[float]
    output_seconds: ReadOnly[float]
    rows: ReadOnly[Sequence[int]]


class MeteredUse(TypedDict):
    quantity: ReadOnly[float]
    rows: ReadOnly[Sequence[int]]


class SessionCandidate(TypedDict):
    model: ReadOnly[str]
    provider: ReadOnly[str]


class TokenCandidateUse(TypedDict):
    candidate: ReadOnly[SessionCandidate]
    row: ReadOnly[int]
    declares: ReadOnly[bool]


class SessionTokensUse(TypedDict):
    usage: ReadOnly[PricedUsage]
    candidates: ReadOnly[Sequence[TokenCandidateUse]]


class SessionTurnUse(TypedDict):
    candidate: ReadOnly[SessionCandidate]
    usage: ReadOnly[PricedUsage]
    row: ReadOnly[int | None]


class SessionDocument(TypedDict):
    """What a realtime or Live session bills: its tokens, its Live turns and the parts beside them. Each names its rows
    by index among the rows passed beside the document, in the order main tries the session's models."""

    transcription: ReadOnly[Sequence[ClipUse]]
    translation: ReadOnly[TranslationUse | None]
    live_voice: ReadOnly[MeteredUse | None]
    audio_duration: ReadOnly[MeteredUse | None]
    text_messages: ReadOnly[MeteredUse | None]
    tokens: ReadOnly[NotRequired[SessionTokensUse | None]]
    turns: ReadOnly[NotRequired[Sequence[SessionTurnUse]]]
    data_residency: ReadOnly[NotRequired[str | None]]


FeeDocument: TypeAlias = A2aDocument | McpDocument | GuardrailDocument | PromptShieldDocument


@dataclass(frozen=True, slots=True)
class BranchFacts:
    """What a ``cost_per_token`` provider branch reads beside the billed usage."""

    rules: TokenRules = "branch"
    elapsed: Elapsed | None = None
    characters: Characters | None = None
    embedding: bool = False
    raw: RawCounts | None = None
    reported: ReportedCost | None = None
    custom: CustomPricing | None = None
    decisions: bool = False


@dataclass(frozen=True, slots=True)
class ReceiptFacts:
    """What ``completion_cost`` prices beside a token call's own tokens: the row its stored breakdown reads, the built-in
    tool fees Python still prices and the Azure AI Model Router fee off ``fee_row``."""

    breakdown: BreakdownFrom = "priced"
    breakdown_row: Mapping[str, object] | None = None
    tool_fees: float = 0.0
    router_fee: FeeBilling | None = None
    fee_row: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AdjustedCost:
    cost: float
    discount_percent: float
    discount_amount: float
    margin_percent: float
    margin_fixed_amount: float
    margin_total_amount: float


def web_search_use(requests: int | None, search_context_size: str | None = None) -> WebSearchUse | None:
    """The web searches a call reports, at the context size it asked for, ``None`` when it reports no count."""
    if requests is None:
        return None
    return WebSearchUse(requests=max(requests, 0), context=_search_context(search_context_size))


def priced_usage(usage: Usage, web_search: WebSearchUse | None) -> PricedUsage:
    """``usage``'s billed quantities with ``web_search`` and the speed and geo the provider reported."""
    billed: Final = billed_usage(usage)
    speed: Final[object] = getattr(usage, "speed", None)
    geo: Final[object] = getattr(usage, "inference_geo", None)
    return PricedUsage(
        **billed,
        web_search=web_search,
        speed="fast" if speed == "fast" else None,
        inference_geo=geo if isinstance(geo, str) else None,
    )


def _count(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _writes_by_ttl(details: CacheCreationTokenDetails | None) -> WritesByTtl | None:
    if details is None:
        return None
    return WritesByTtl(
        five_minutes=details.ephemeral_5m_input_tokens,
        thirty_minutes=details.ephemeral_30m_input_tokens,
        one_hour=details.ephemeral_1h_input_tokens,
        twelve_hours=details.ephemeral_12h_input_tokens,
    )


def raw_counts(usage: Usage) -> RawCounts:
    """``usage``'s counts as the provider reported them, before the billed split removes their overlap."""
    prompt: Final = parse_prompt_tokens_details(usage)
    completion: Final = parse_completion_tokens_details(usage)
    details: Final = usage.prompt_tokens_details
    stated_reasoning: Final = _count(getattr(usage, "reasoning_tokens", 0))
    return RawCounts(
        prompt_tokens=_count(usage.prompt_tokens),
        completion_tokens=_count(usage.completion_tokens),
        total_tokens=_count(usage.total_tokens),
        cached_tokens=_count(prompt["cache_hit_tokens"]),
        cache_creation_tokens=_count(prompt["cache_creation_tokens"]),
        cache_writes=_writes_by_ttl(prompt["cache_creation_token_details"]),
        reasoning_tokens=_count(completion["reasoning_tokens"]),
        stated_reasoning_tokens=stated_reasoning
        or _count(getattr(usage.completion_tokens_details, "reasoning_tokens", 0)),
        citation_tokens=_count(getattr(usage, "citation_tokens", 0)),
        search_queries=_count(getattr(details, "web_search_requests", 0)),
    )


def _search_context(size: str | None) -> SearchContextSize | None:
    if size == "low":
        return "low"
    if size == "medium":
        return "medium"
    return "high" if size == "high" else None


def _billed_at(current_time: datetime | None) -> str:
    reference: Final = current_time if current_time is not None else current_billing_time()
    aware: Final = (
        reference.astimezone(timezone.utc) if reference.tzinfo is not None else reference.replace(tzinfo=timezone.utc)
    )
    return aware.isoformat()


def price(
    named: CallFacts | ModelFacts,
    usage: PricedUsage,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    requested_model: str | None = None,
    adjust: AdjustFacts | None = None,
    row: Mapping[str, object] | None = None,
    current_time: datetime | None = None,
    branch: BranchFacts | None = None,
    receipt: ReceiptFacts | None = None,
) -> NativeReceipt | None:
    facts: Final = branch or BranchFacts()
    beside: Final = receipt or ReceiptFacts()
    document: Final = _PriceDocument(
        named=named,
        usage=usage,
        requested_model=requested_model,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        at=_billed_at(current_time),
        adjust=adjust,
        rules=facts.rules,
        elapsed=facts.elapsed,
        characters=facts.characters,
        embedding=facts.embedding,
        raw=facts.raw,
        reported=facts.reported,
        custom=facts.custom,
        decisions=facts.decisions,
        breakdown=beside.breakdown,
        tool_fees=beside.tool_fees,
        router_fee=None if beside.router_fee is None else RouterFeeUse(billed=beside.router_fee),
    )
    return model_cost_inputs.price(document, row, beside.breakdown_row, beside.fee_row)


def branch_receipt(
    model: str,
    provider: str,
    usage: Usage,
    row: Mapping[str, object] | None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    branch: BranchFacts | None = None,
    row_provider: str | None = None,
    current_time: datetime | None = None,
    receipt: ReceiptFacts | None = None,
    adjust: AdjustFacts | None = None,
) -> NativeReceipt:
    """The receipt of ``usage`` on ``row``, the row ``provider``'s branch of ``cost_per_token`` looked up for ``model``,
    priced by that branch's rules in Rust. Raises where main's branch raised."""
    facts: Final = branch or BranchFacts()
    named: Final = (
        ModelFacts(by="model", model=model, provider=provider, region=None)
        if row_provider is None
        else ModelFacts(by="model", model=model, provider=provider, region=None, row_provider=row_provider)
    )
    priced: Final = price(
        named,
        priced_usage(usage, web_search=None),
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        adjust=adjust,
        row=row,
        current_time=current_time,
        branch=BranchFacts(
            rules=facts.rules,
            elapsed=facts.elapsed,
            characters=facts.characters,
            embedding=facts.embedding,
            raw=raw_counts(usage),
            reported=facts.reported,
            custom=facts.custom,
            decisions=facts.decisions,
        ),
        receipt=receipt,
    )
    if priced is None:
        raise ValueError(f"No price for model={model}, custom_llm_provider={provider}")
    return priced


def branch_cost(
    model: str,
    provider: str,
    usage: Usage,
    row: Mapping[str, object] | None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    branch: BranchFacts | None = None,
    row_provider: str | None = None,
    current_time: datetime | None = None,
) -> tuple[float, float]:
    """``cost_per_token``'s prompt and completion cost of ``usage`` on ``row``, the row ``provider``'s branch looked up
    for ``model``, priced by that branch's rules in Rust."""
    receipt: Final = branch_receipt(
        model,
        provider,
        usage,
        row,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        branch=branch,
        row_provider=row_provider,
        current_time=current_time,
    )
    return receipt.prompt_cost, receipt.completion_cost


def decisions_request_cost(provider: str, row: Mapping[str, object]) -> float:
    """What one Decisions request bills before its tokens on ``row``: the query price of a provider that bills per
    request, 0.0 for every other row."""
    prompt_cost, _ = branch_cost(
        "decisions",
        provider,
        Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        row,
        branch=BranchFacts(rules="generic", decisions=True),
    )
    return prompt_cost


def custom_pricing_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
    reasoning_tokens: int = 0,
) -> PricedUsage:
    """The counts ``custom_cost_per_token`` bills: the prompt with every cache read and write in it, and the
    completion, its reasoning tokens counted apart for the stored breakdown."""
    no_tokens: Final = ModalityCounts(text=0, audio=0, image=0, video=0)
    reasoning: Final = min(reasoning_tokens, completion_tokens)
    return PricedUsage(
        prompt_tokens=prompt_tokens,
        input=no_tokens,
        cache_read=CacheReadCounts(text=cached_tokens, audio=0, image=0, explicit=False),
        cache_write=CacheWriteCounts(default=cache_creation_tokens, one_hour=0, twelve_hours=0),
        output=ModalityCounts(text=completion_tokens - reasoning, audio=0, image=0, video=0),
        reasoning=reasoning,
        input_characters=0,
        input_images=0,
        input_audio_seconds=0.0,
        input_video_seconds=0.0,
        queries=0,
        web_search=None,
        speed=None,
        inference_geo=None,
    )


def custom_receipt(
    model: str,
    provider: str,
    usage: PricedUsage,
    custom: CustomPricing,
    response_time_ms: float | None,
    service_tier: str | None = None,
    data_residency: str | None = None,
    vertex_location: str | None = None,
    receipt: ReceiptFacts | None = None,
    adjust: AdjustFacts | None = None,
) -> NativeReceipt:
    """``custom_cost_per_token`` or ``custom_cost_per_second`` applied to ``usage``."""
    priced: Final = price(
        ModelFacts(by="model", model=model, provider=provider, region=None),
        usage,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        adjust=adjust,
        branch=BranchFacts(
            elapsed=Elapsed(response_ms=response_time_ms or 0.0, audio_seconds=0.0),
            custom=custom,
        ),
        receipt=receipt,
    )
    if priced is None:
        raise ValueError(f"Custom pricing left model={model} unpriced")
    return priced


def price_units(
    document: UnitsDocument,
    tables: Sequence[Mapping[str, object]] = (),
    catalog: Mapping[str, object] | None = None,
) -> NativeReceipt | NativeUnpriced:
    """A per-unit call priced from ``document``'s facts, or why no row prices it. ``tables`` are the deployment's own
    price tables, read in order ahead of ``catalog``, the row ``get_model_info`` found for the call."""
    return model_cost_inputs.price_units(document, tables, catalog)


def batch_usage(usage: Usage) -> BatchUsage:
    """The token counts a batch line bills, read off its usage the way main's batch calculator read them."""
    details: Final = parse_prompt_tokens_details(usage)
    ttl: Final = details["cache_creation_token_details"]
    completion_details: Final = usage.completion_tokens_details
    return BatchUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens,
        completion_image_tokens=(completion_details.image_tokens or 0) if completion_details is not None else 0,
        cache_hit_tokens=details["cache_hit_tokens"],
        cache_hit_audio_tokens=details["cache_hit_audio_tokens"],
        cached_image_tokens=details["cached_image_tokens"],
        cache_creation_tokens=details["cache_creation_tokens"],
        cache_creation_by_ttl=_writes_by_ttl(ttl),
        audio_tokens=details["audio_tokens"],
        image_tokens=details["image_tokens"],
        video_tokens=details["video_tokens"],
    )


def price_many(
    documents: Sequence[ManyDocument], rows: Sequence[Mapping[str, object]]
) -> Sequence[NativeReceipt | NativeUnpriced]:
    """Each call priced from its document as ``price_units`` would, in one crossing. A document's ``row`` is the index
    of the row ``get_model_info`` found for it, each row lowered once however many documents name it."""
    return model_cost_inputs.price_many(documents, rows)


def price_fee(document: FeeDocument) -> NativeReceipt | NativeUnpriced:
    """An agent call, MCP tool call or guardrail run priced from ``document``'s facts, or why nothing prices it."""
    return model_cost_inputs.price_fee(document)


def price_session(document: SessionDocument, rows: Sequence[Mapping[str, object]]) -> NativeReceipt | NativeUnpriced:
    """A session's tokens and Live turns on the receipt's prompt and completion, and the parts beside them, one
    receipt part each: ``transcription_cost``, ``translation_cost``, ``live_voice_cost``, ``audio_duration_cost`` and
    ``text_message_cost``."""
    return model_cost_inputs.price_session(document, rows)


def part_costs(outcome: NativeReceipt | NativeUnpriced) -> Mapping[str, float]:
    """Each priced part's cost by label, a part no price covers left out, the 0.0 main billed it."""
    if not isinstance(outcome, NativeReceipt):
        return MappingProxyType({})
    return MappingProxyType({label: cost for label, cost in outcome.parts if cost is not None})


def total_or_zero(outcome: NativeReceipt | NativeUnpriced) -> float:
    """The receipt's total, 0.0 for a call nothing prices, the legacy return of the flat-priced kinds."""
    return outcome.total if isinstance(outcome, NativeReceipt) else 0.0


def prompt_and_completion(outcome: NativeReceipt | NativeUnpriced) -> tuple[float, float]:
    """The receipt's prompt and completion cost, ``(0.0, 0.0)`` for a call no row prices."""
    return (outcome.prompt_cost, outcome.completion_cost) if isinstance(outcome, NativeReceipt) else (0.0, 0.0)


def billed_token_rates(receipt: NativeReceipt) -> BilledTokenRates | None:
    rates: Final = receipt.rates
    if rates is None:
        return None
    return BilledTokenRates(
        input_cost_per_token=rates.input,
        output_cost_per_token=rates.output,
        cache_read_input_token_cost=rates.cache_read,
        cache_read_input_audio_token_cost=rates.cache_read_audio,
        cache_creation_input_token_cost=rates.cache_write,
        cache_creation_input_token_cost_by_ttl=MappingProxyType(
            {"1hr": rates.cache_write_1h, "12hr": rates.cache_write_12h}
        ),
        output_cost_per_reasoning_token=rates.reasoning,
        cache_read_input_image_token_cost=rates.cache_read_image,
    )


def adjusted_cost(
    cost: float, provider: str | None, model: str | None = None, region: str | None = None
) -> AdjustedCost:
    """``cost`` after the provider-keyed ``cost_discount_config`` and then ``cost_margin_config``."""
    (
        final_cost,
        discount_percent,
        discount_amount,
        margin_percent,
        margin_fixed_amount,
        margin_total_amount,
    ) = model_cost_inputs.adjust(cost, provider, model, region)
    return AdjustedCost(
        cost=final_cost,
        discount_percent=discount_percent,
        discount_amount=discount_amount,
        margin_percent=margin_percent,
        margin_fixed_amount=margin_fixed_amount,
        margin_total_amount=margin_total_amount,
    )


def rate_card(key: str) -> NativeRateCard | None:
    """The rates and raw key facts of the ``litellm.model_cost`` row under exactly ``key``."""
    return model_cost_inputs.rates(key)
