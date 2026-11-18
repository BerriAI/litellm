"""Python's side of the Rust pricing core (``NativeModelCostMap.price``, ``price_units``, ``bound``, ``rates`` and
``adjust``).

Each helper reads the facts Python holds into the document the bridge reads, and hands back what the old Python
API returned. ``None`` from ``price`` means Rust left the call to Python, which then prices it the old way."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal

from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import current_billing_time
from litellm.litellm_core_utils.get_model_cost_map import model_cost_inputs
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    BilledTokenRates,
    BilledUsage,
    billed_usage,
)
from litellm.rust_bridge._native import NativeRateCard, NativeReceipt, NativeUnpriced
from litellm.types.utils import Usage

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
    """One model a ``cost_per_token`` caller names."""

    by: ReadOnly[Literal["model"]]
    model: ReadOnly[str]
    provider: ReadOnly[str]
    region: ReadOnly[str | None]


class AdjustFacts(TypedDict):
    provider: ReadOnly[str | None]
    model: ReadOnly[str | None]
    region: ReadOnly[str | None]


class _PriceDocument(TypedDict):
    named: ReadOnly[CallFacts | ModelFacts]
    usage: ReadOnly[PricedUsage]
    requested_model: ReadOnly[str | None]
    service_tier: ReadOnly[str | None]
    data_residency: ReadOnly[str | None]
    vertex_location: ReadOnly[str | None]
    at: ReadOnly[str]
    adjust: ReadOnly[AdjustFacts | None]


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


UnitsDocument = (
    SearchDocument | RerankDocument | VectorStoreSearchDocument | OcrDocument | SpeechDocument | TranscriptionDocument
)


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
) -> NativeReceipt | None:
    document: Final = _PriceDocument(
        named=named,
        usage=usage,
        requested_model=requested_model,
        service_tier=service_tier,
        data_residency=data_residency,
        vertex_location=vertex_location,
        at=_billed_at(current_time),
        adjust=adjust,
    )
    return model_cost_inputs.price(document, row)


def price_model(
    model: str,
    provider: str,
    usage: Usage,
    service_tier: str | None = None,
    row: Mapping[str, object] | None = None,
) -> tuple[float, float] | None:
    """``cost_per_token``'s prompt and completion cost of ``usage`` on ``model``, ``None`` when Rust leaves it to
    Python. ``row`` is a price row the caller resolved for this call. ``cost_per_token`` never bills the web search
    fee, ``completion_cost`` adds it, so the search count stays out of the price."""
    receipt: Final = price(
        ModelFacts(by="model", model=model, provider=provider, region=None),
        priced_usage(usage, web_search=None),
        service_tier=service_tier,
        row=row,
    )
    return None if receipt is None else (receipt.prompt_cost, receipt.completion_cost)


def price_units(
    document: UnitsDocument,
    tables: Sequence[Mapping[str, object]] = (),
    catalog: Mapping[str, object] | None = None,
) -> NativeReceipt | NativeUnpriced:
    """A per-unit call priced from ``document``'s facts, or why no row prices it. ``tables`` are the deployment's own
    price tables, read in order ahead of ``catalog``, the row ``get_model_info`` found for the call."""
    return model_cost_inputs.price_units(document, tables, catalog)


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
