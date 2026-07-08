"""Registry row schema: the contract every denominator cell validates against.

A cell is one customer-noticeable behavior a single e2e test can assert pass/fail
on. `module` is the id's segment-1 prefix (seven of them); the six-way dashboard
rollup merges logging + guardrail via ROLLUP. The union is discriminated on
`module`, so an LLM row cannot carry a guardrail field and vice versa.
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
    subject_endpoint: str
    route: str
    capability: str
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
    LlmCell | MgmtCell | McpCell | ReliabilityCell | LoggingCell | GuardrailCell | OtherCell,
    Field(discriminator="module"),
]

CELL_ADAPTER: TypeAdapter[Cell] = TypeAdapter(Cell)

ROLLUP: dict[str, str] = {
    "llm": "LLMs",
    "mcp": "MCPs",
    "mgmt": "Management/UI",
    "reliability": "Reliability & Performance",
    "logging": "Logging & Guardrails",
    "guardrail": "Logging & Guardrails",
    "other": "Other",
}

MODULE_ORDER: tuple[str, ...] = (
    "LLMs",
    "MCPs",
    "Management/UI",
    "Reliability & Performance",
    "Logging & Guardrails",
    "Other",
)
