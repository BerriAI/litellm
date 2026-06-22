"""The declarative provider x routing-scenario matrix the lifecycle test runs.

One Capability per supported (provider, scenario) pair, so the parametrized test
has no dead/skipped cells. `provider` is litellm's custom_llm_provider, used to
route provider-fallback calls to /{provider}/v1/... and to assert the raw batch id
shape (the only scenario whose id is not re-encoded by the proxy). Operations that
a provider does not support (Bedrock: no cancel, no list) are gated per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Scenario = Literal["encoded", "unified", "model_param", "provider_fallback"]

SCENARIOS: tuple[Scenario, ...] = (
    "encoded",
    "unified",
    "model_param",
    "provider_fallback",
)


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    model: str
    raw_model: str
    can_cancel: bool
    can_list: bool


@dataclass(frozen=True, slots=True)
class Capability:
    provider: str
    model: str
    raw_model: str
    scenario: Scenario
    can_cancel: bool
    can_list: bool

    @property
    def id(self) -> str:
        return f"{self.provider}-{self.scenario}"

    @property
    def jsonl_model(self) -> str:
        """The model name to embed in the input JSONL. Model-routed scenarios let
        litellm rewrite it to the deployment's real model, so the litellm name is
        fine; provider_fallback has no mapping, so the real provider model is needed."""
        return self.raw_model if self.scenario == "provider_fallback" else self.model


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "openai-batch", "gpt-4o-mini", can_cancel=True, can_list=True),
    Provider("azure", "azure-batch", "gpt-4o-mini", can_cancel=True, can_list=True),
    Provider(
        "vertex_ai", "vertex-batch", "gemini-2.5-flash", can_cancel=True, can_list=True
    ),
    Provider(
        "bedrock",
        "bedrock-batch",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        can_cancel=False,
        can_list=False,
    ),
)

CAPABILITIES: tuple[Capability, ...] = tuple(
    Capability(p.name, p.model, p.raw_model, scenario, p.can_cancel, p.can_list)
    for p in PROVIDERS
    for scenario in SCENARIOS
)


def raw_id_matches_provider(provider: str, batch_id: str) -> bool:
    """The provider-fallback path returns the provider's native batch id (unencoded),
    so its shape discriminates which provider actually handled the batch."""
    if provider in ("openai", "azure"):
        return batch_id.startswith("batch")
    if provider == "vertex_ai":
        return batch_id.startswith("projects/") or "batchPredictionJobs" in batch_id
    if provider == "bedrock":
        return batch_id.startswith("arn:aws")
    return True
