"""Coverage cell vocabulary, id grammar, and the human-overlay row shape.

A cell is one customer-noticeable behavior a single e2e test can assert pass/fail
on, identified by a dotted id whose first segment is the module. The structural
facets an LLM id encodes (endpoint, route, capability, streaming) are parsed back
out of the id rather than stored a second time, so an id and its fields can never
drift. The only per-cell data a human curates lives in `OverlayRow`; the set of
cells itself (the denominator) is generated in `product_surface.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

Module = Literal[
    "llm",
    "mgmt",
    "mcp",
    "reliability",
    "quota_management",
    "logging",
    "guardrail",
    "other",
]


class Tier(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"


class FailBeforeFix(str, Enum):
    proven = "proven"
    unproven = "unproven"


LlmEndpoint = Literal[
    "chat_completions",
    "messages",
    "responses",
    "embeddings",
    "batches",
    "files",
    "rerank",
    "images_generations",
    "audio_speech",
    "audio_transcriptions",
    "moderations",
    "realtime",
]

LlmRoute = Literal[
    "anthropic",
    "azure_foundry",
    "azure_openai",
    "bedrock_converse",
    "bedrock_invoke",
    "cohere",
    "openai",
    "together_ai",
    "vertex",
]

LlmCapability = Literal[
    "basic",
    "count_tokens",
    "long_context_1m",
    "mid_conversation_system",
    "pdf_input",
    "prompt_cache_1h",
    "prompt_cache_5m",
    "service_tier",
    "structured_output",
    "thinking",
    "thinking_with_tool_use",
    "tool_search",
    "tool_use",
    "vision",
    "web_search",
]

LlmStreaming = Literal["stream", "nonstream", "na"]


@dataclass(frozen=True, slots=True)
class LlmCellId:
    endpoint: LlmEndpoint
    route: LlmRoute
    capability: LlmCapability
    streaming: LlmStreaming
    assertion: str


def format_llm_id(
    endpoint: LlmEndpoint,
    route: LlmRoute,
    capability: LlmCapability,
    streaming: LlmStreaming,
    assertion: str,
) -> str:
    return f"llm.{endpoint}.{route}.{capability}.{streaming}.{assertion}"


_LLM_CELL_ID_ADAPTER: TypeAdapter[LlmCellId] = TypeAdapter(LlmCellId)


def parse_llm_id(cell_id: str) -> LlmCellId | None:
    """The structural facets of an LLM id, or None when the id is not an LLM cell
    whose segments all match the typed vocabulary. Non-core LLM endpoints (batches,
    files, rerank, ...) use an operation grammar that is not part of this vocabulary
    and return None here by design; they are carried by the overlay, not generated."""
    parts = tuple(cell_id.split("."))
    if len(parts) != 6 or parts[0] != "llm":
        return None
    _, endpoint, route, capability, streaming, assertion = parts
    try:
        return _LLM_CELL_ID_ADAPTER.validate_python(
            {
                "endpoint": endpoint,
                "route": route,
                "capability": capability,
                "streaming": streaming,
                "assertion": assertion,
            }
        )
    except ValidationError:
        return None


class OverlayRow(BaseModel):
    """The human-curated fields for one cell, keyed by id in overlay.yaml. Holds no
    structural facet; those are parsed from the id or generated. A generated id with
    no overlay row defaults to P2 (see registry.py), so a newly grown surface shows
    up as an uncovered gap rather than vanishing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: Tier
    source: str = ""
    rationale: str = ""
    fail_before_fix: FailBeforeFix = FailBeforeFix.unproven
    supported: bool = True


@dataclass(frozen=True, slots=True)
class Cell:
    """A denominator cell: its id, the module parsed from that id, and the curated
    overlay fields (defaulted when the id has no overlay row)."""

    id: str
    module: Module
    tier: Tier
    source: str = ""
    rationale: str = ""
    fail_before_fix: FailBeforeFix = FailBeforeFix.unproven
    supported: bool = True


_MODULE_ADAPTER: TypeAdapter[Module] = TypeAdapter(Module)


def parse_module(cell_id: str) -> Module:
    """The module (id's first segment), validated against the vocabulary. Raises on
    an unknown prefix, since that would corrupt the per-module rollups."""
    return _MODULE_ADAPTER.validate_python(cell_id.split(".", 1)[0])


CORE_LLM_ENDPOINTS: frozenset[str] = frozenset(
    {
        "chat_completions",
        "messages",
        "responses",
    }
)

PREFIX_ROLLUP: dict[str, str] = {
    "mcp": "MCPs",
    "mgmt": "Management/UI",
    "reliability": "Reliability & Performance",
    "quota_management": "Quota Management",
    "logging": "Logging & Guardrails",
    "guardrail": "Logging & Guardrails",
    "other": "Other",
}

MODULE_ORDER: tuple[str, ...] = (
    "Core LLMs",
    "Non-Core LLMs",
    "MCPs",
    "Management/UI",
    "Reliability & Performance",
    "Quota Management",
    "Logging & Guardrails",
    "Other",
)

LOKI_MODULE_LABELS: dict[str, str] = {
    "Core LLMs": "core_llms",
    "Non-Core LLMs": "non_core_llms",
    "MCPs": "mcp",
    "Management/UI": "management_ui",
    "Reliability & Performance": "reliability_performance",
    "Quota Management": "quota_management",
    "Logging & Guardrails": "logging_guardrails",
    "Other": "other",
}


def dashboard_module(cell: Cell) -> str:
    """The Grafana/reporting module for a cell, decided from its id. LLM cells split
    into Core vs Non-Core on the endpoint segment; every other module maps by prefix."""
    if cell.module == "llm":
        endpoint = cell.id.split(".")[1]
        return "Core LLMs" if endpoint in CORE_LLM_ENDPOINTS else "Non-Core LLMs"
    return PREFIX_ROLLUP[cell.module]


def loki_module_label(module: str) -> str:
    """Return the log-safe Loki label for a dashboard module."""
    return LOKI_MODULE_LABELS[module]
