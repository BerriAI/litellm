"""Registry row schema: the contract every denominator cell validates against.

A cell is one customer-noticeable behavior a single e2e test can assert pass/fail
on. `module` is the id's segment-1 prefix (eight of them); dashboard rollups can
split or merge those prefixes. The union is discriminated on `module`, so an LLM
row cannot carry a guardrail field and vice versa.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


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
    "gemini",
    "hosted_vllm",
    "openai",
    "together_ai",
    "vertex",
]

LlmCapability = Literal[
    "assume_role",
    "basic",
    "count_tokens",
    "drop_params",
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


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    tier: Tier
    assertions: tuple[str, ...]
    source: str
    rationale: str = ""
    fail_before_fix: FailBeforeFix = FailBeforeFix.unproven
    supported: bool = True


class LlmCell(_Base):
    module: Literal["llm"]
    subject_endpoint: LlmEndpoint
    route: LlmRoute
    capability: LlmCapability
    streaming: Literal["stream", "nonstream", "na"]


class MgmtCell(_Base):
    module: Literal["mgmt"]
    surface: Literal["api", "ui"]


class McpCell(_Base):
    module: Literal["mcp"]
    operation: str
    auth_family: Literal["none", "api_key", "bearer", "oauth"]


class ReliabilityCell(_Base):
    module: Literal["reliability"]
    behavior: str
    variant: str
    exercised_on: tuple[str, ...]


class QuotaCell(_Base):
    module: Literal["quota_management"]
    behavior: Literal["ratelimit", "budget", "spend_tracking"]
    variant: str
    exercised_on: tuple[str, ...]


class LoggingCell(_Base):
    module: Literal["logging"]
    event: str
    exercised_on: tuple[str, ...]


class GuardrailCell(_Base):
    module: Literal["guardrail"]
    hook_point: str
    exercised_on: tuple[str, ...]


class OtherCell(_Base):
    module: Literal["other"]
    area: str


Cell = Annotated[
    LlmCell
    | MgmtCell
    | McpCell
    | ReliabilityCell
    | QuotaCell
    | LoggingCell
    | GuardrailCell
    | OtherCell,
    Field(discriminator="module"),
]

CELL_ADAPTER: TypeAdapter[Cell] = TypeAdapter(Cell)

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
    """Return the Grafana/reporting module for a registry cell."""
    if isinstance(cell, LlmCell):
        if cell.subject_endpoint in CORE_LLM_ENDPOINTS:
            return "Core LLMs"
        return "Non-Core LLMs"
    return PREFIX_ROLLUP[cell.module]


def loki_module_label(module: str) -> str:
    """Return the log-safe Loki label for a dashboard module."""
    return LOKI_MODULE_LABELS[module]
