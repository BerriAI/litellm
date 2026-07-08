"""Schema for e2e coverage declared directly on pytest tests.

Each collected e2e pytest must provide:

    @pytest.mark.e2e_coverage(
        module="core_llms",
        endpoint="/chat/completions",
        provider="openai",
        params=["tools"],
    )

The collector turns those markers into endpoint x provider x parameter coverage
units and Grafana-ready module summaries.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CoverageModule = Literal[
    "core_llms",
    "non_core_llms",
    "access_control",
    "budgets",
    "spend_tracking",
    "management",
    "mcp",
    "rate_limits",
    "reliability",
    "logging",
    "guardrails",
    "other",
]

MODULE_ORDER: tuple[CoverageModule, ...] = (
    "core_llms",
    "non_core_llms",
    "access_control",
    "budgets",
    "spend_tracking",
    "management",
    "mcp",
    "rate_limits",
    "reliability",
    "logging",
    "guardrails",
    "other",
)

MODULE_DISPLAY_NAMES: dict[str, str] = {
    "core_llms": "Core LLMs",
    "non_core_llms": "Non-Core LLMs",
    "access_control": "Access Control",
    "budgets": "Budgets",
    "spend_tracking": "Spend Tracking",
    "management": "Management",
    "mcp": "MCP",
    "rate_limits": "Rate Limits",
    "reliability": "Reliability",
    "logging": "Logging",
    "guardrails": "Guardrails",
    "other": "Other",
}

KNOWN_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/batches",
        "/v1/realtime",
        "/v1/audio/speech",
        "/v1/embeddings",
        "/v1/images/generations",
        "/rerank",
        "/anthropic/*",
        "/vertex_ai/*",
        "/model/*",
        "/key/*",
        "/team/*",
        "/user/*",
        "/organization/*",
        "/budget/*",
        "/spend/*",
        "/global/spend/*",
        "/mcp/*",
        "/guardrails/*",
        "/health/*",
        "coverage_registry",
        "e2e_harness",
        "logging",
        "reliability",
    }
)

KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "azure",
        "azure_ai",
        "azure_document_intelligence",
        "bedrock",
        "cohere",
        "deepseek",
        "gemini",
        "litellm",
        "mistral",
        "multiple",
        "openai",
        "prometheus",
        "proxy",
        "vertex_ai",
        "xai",
    }
)

_PARAM_RE = re.compile(r"^[a-z0-9][a-z0-9_.:/-]*$")


class CoveragePoint(BaseModel):
    """Validated data carried by @pytest.mark.e2e_coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    module: CoverageModule
    endpoint: str
    provider: str
    params: tuple[str, ...] = Field(min_length=1)

    @field_validator("endpoint")
    @classmethod
    def endpoint_must_be_known(cls, value: str) -> str:
        if value not in KNOWN_ENDPOINTS:
            raise ValueError(f"unknown endpoint {value!r}")
        return value

    @field_validator("provider")
    @classmethod
    def provider_must_be_known(cls, value: str) -> str:
        if value not in KNOWN_PROVIDERS:
            raise ValueError(f"unknown provider {value!r}")
        return value

    @field_validator("params")
    @classmethod
    def params_must_be_normalized(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        invalid = [param for param in value if not _PARAM_RE.fullmatch(param)]
        if invalid:
            raise ValueError(f"invalid params: {invalid}")
        return value


class CoverageUnit(BaseModel):
    """One endpoint x provider x parameter combination covered by a test."""

    model_config = ConfigDict(frozen=True)

    module: CoverageModule
    endpoint: str
    provider: str
    param: str

    @property
    def key(self) -> str:
        return f"{self.module}|{self.endpoint}|{self.provider}|{self.param}"


def units_for_point(point: CoveragePoint) -> tuple[CoverageUnit, ...]:
    return tuple(
        CoverageUnit(
            module=point.module,
            endpoint=point.endpoint,
            provider=point.provider,
            param=param,
        )
        for param in point.params
    )
