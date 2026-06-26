"""The declarative provider x routing-scenario matrix the lifecycle test runs.

One Capability per supported (provider, scenario) pair, so the parametrized test
has no dead/skipped cells. `provider` is litellm's custom_llm_provider, used to
route provider-fallback calls to /{provider}/v1/... and to assert the raw batch id
shape (the only scenario whose id is not re-encoded by the proxy). Operations that
a provider does not support (Bedrock: no cancel, no list) are gated per row.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Literal

Scenario = Literal["encoded", "unified", "model_param", "provider_fallback"]

IdShape = Literal["managed", "model_encoded", "raw"]

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
        """Model name embedded in the uploaded JSONL ``body.model``.

        Only the unified upload path rewrites JSONL on upload
        (``target_model_names`` → ``llm_router.acreate_file`` →
        ``replace_model_in_jsonl``), so that scenario can use the LiteLLM alias
        and rely on the proxy to swap it to the deployment model. Every other
        scenario uploads raw JSONL with no rewrite, so the provider's real
        deployment name is required or create fails upstream validation."""
        return self.model if self.scenario == "unified" else self.raw_model


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "openai-batch", "gpt-4o-mini", can_cancel=True, can_list=True),
    Provider("azure", "azure-batch", "gpt-4.1-mini-batch", can_cancel=True, can_list=True),
    Provider(
        "vertex_ai", "vertex-batch", "gemini-2.5-flash", can_cancel=True, can_list=True
    ),
    # Provider(
    #     "bedrock",
    #     "bedrock-batch",
    #     "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    #     can_cancel=False,
    #     can_list=False,
    # ),
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
        # Vertex returns the batch prediction job id, which depending on the
        # routing path arrives either as the full resource name
        # (projects/.../batchPredictionJobs/<id>) or as just the trailing
        # numeric id, so accept either form.
        return (
            batch_id.startswith("projects/")
            or "batchPredictionJobs" in batch_id
            or batch_id.isdigit()
        )
    if provider == "bedrock":
        return batch_id.startswith("arn:aws")
    return True


FILE_ID_SHAPE: dict[Scenario, IdShape] = {
    "encoded": "model_encoded",
    "unified": "managed",
    "model_param": "raw",
    "provider_fallback": "raw",
}

BATCH_ID_SHAPE: dict[Scenario, IdShape] = {
    "encoded": "model_encoded",
    "unified": "managed",
    "model_param": "model_encoded",
    "provider_fallback": "raw",
}


def _b64_decode(value: str) -> str:
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode()
    except Exception:
        return ""


def is_managed_id(id_str: str) -> bool:
    """A litellm managed unified file/batch id base64-decodes to a litellm_proxy marker."""
    return _b64_decode(id_str).startswith("litellm_proxy")


def is_model_encoded_id(id_str: str) -> bool:
    """A model-encoded id keeps the provider prefix and base64-encodes litellm:<id>;model,<m>."""
    for prefix in ("file-", "batch_"):
        if id_str.startswith(prefix):
            decoded = _b64_decode(id_str[len(prefix) :])
            return decoded.startswith("litellm:") and ";model," in decoded
    return False


def matches_id_shape(shape: IdShape, id_str: str) -> bool:
    if shape == "managed":
        return is_managed_id(id_str)
    if shape == "model_encoded":
        return is_model_encoded_id(id_str)
    return not is_managed_id(id_str) and not is_model_encoded_id(id_str)
