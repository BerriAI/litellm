from enum import Enum
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .base import GuardrailConfigModel


class PolicyMode(str, Enum):
    ENFORCE = "ENFORCE"
    MONITOR = "MONITOR"


class PiiPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["pii"] = "pii"
    action: Literal["REDACT", "BLOCK"] = Field(default="REDACT", description="Redact matches, or block the request.")
    patterns: Literal["all"] | tuple[str, ...] = Field(
        default="all",
        description=(
            "'all' scans the high-precision default set. A list of pattern names and/or category names "
            "scans exactly those, including loose patterns such as ssn_no_dash or swift_bic."
        ),
    )


class CostPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["cost"] = "cost"
    daily_limit_usd: float | None = Field(default=None, gt=0, description="Per-user daily spend limit.")
    session_limit_usd: float | None = Field(default=None, gt=0, description="Per-session spend limit.")
    request_limit_usd: float | None = Field(default=None, gt=0, description="Estimated cost limit for one request.")
    cost_per_1k_tokens: float = Field(default=0.002, ge=0, description="Rate used to turn estimated tokens into USD.")


class ToolAuthPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["tool_auth"] = "tool_auth"
    allowlist: tuple[str, ...] | None = Field(default=None, description="Only these tools may be used.")
    blocklist: tuple[str, ...] = Field(default=(), description="These tools are always blocked.")


Policy: TypeAlias = Annotated[PiiPolicy | CostPolicy | ToolAuthPolicy, Field(discriminator="type")]

DEFAULT_POLICIES: Final[tuple[Policy, ...]] = (PiiPolicy(),)


class TealTigerGuardrailConfigModel(GuardrailConfigModel[BaseModel]):
    model_config = ConfigDict(frozen=True)

    policies: tuple[Policy, ...] | None = Field(
        default=None,
        description="Policies to enforce: pii, cost and tool_auth. Defaults to PII redaction only.",
    )
    policy_mode: PolicyMode | None = Field(
        default=None,
        description="ENFORCE blocks or redacts. MONITOR only logs decision receipts. Defaults to ENFORCE.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "TealTiger"
