"""Provider x routing-scenario matrix for the batches lifecycle e2e."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Literal

from models import LiteLLMParamsBody


def _env_ref(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return f"os.environ/{name}"
    return f"os.environ/{names[0]}"

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

    def litellm_params(self) -> LiteLLMParamsBody:
        match self.name:
            case "openai":
                return LiteLLMParamsBody(
                    model="openai/gpt-4o-mini",
                    api_key="os.environ/OPENAI_API_KEY",
                )
            case "azure":
                return LiteLLMParamsBody(
                    model="azure/gpt-5.4-mini-batch",
                    api_base="os.environ/AZURE_API_BASE",
                    api_key="os.environ/AZURE_API_KEY",
                    api_version="2025-04-01-preview",
                )
            case "vertex_ai":
                return LiteLLMParamsBody(
                    model="vertex_ai/gemini-2.5-flash",
                    vertex_project="os.environ/VERTEXAI_PROJECT",
                    vertex_location="us-central1",
                    vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
                    gcs_bucket_name="os.environ/GCS_BUCKET_NAME",
                    bucket_name="os.environ/GCS_BUCKET_NAME",
                )
            case "bedrock":
                return LiteLLMParamsBody(
                    model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                    aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                    aws_region_name="os.environ/AWS_REGION",
                    s3_region_name="os.environ/AWS_REGION",
                    s3_bucket_name=_env_ref("AWS_BATCH_S3_BUCKET", "AWS_S3_BUCKET_NAME"),
                    s3_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                    s3_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                    aws_batch_role_arn="os.environ/AWS_BATCH_ROLE_ARN",
                )
            case _:
                raise ValueError(f"unknown batch provider: {self.name!r}")


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
        return self.model if self.scenario == "unified" else self.raw_model


PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "openai-batch", "gpt-4o-mini", can_cancel=True, can_list=True),
    Provider("azure", "azure-batch", "gpt-5.4-mini-batch", can_cancel=True, can_list=True),
    Provider(
        "vertex_ai", "vertex-batch", "gemini-2.5-flash", can_cancel=True, can_list=True
    ),
    Provider(
        "bedrock",
        "bedrock-batch",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        can_cancel=False,
        can_list=False,
    ),
)

BEDROCK_SCENARIOS: tuple[Scenario, ...] = ("unified",)


def scenarios_for_provider(provider: Provider) -> tuple[Scenario, ...]:
    if provider.name == "bedrock":
        return BEDROCK_SCENARIOS
    return SCENARIOS


CAPABILITIES: tuple[Capability, ...] = tuple(
    Capability(p.name, p.model, p.raw_model, scenario, p.can_cancel, p.can_list)
    for p in PROVIDERS
    for scenario in scenarios_for_provider(p)
)


def raw_id_matches_provider(provider: str, batch_id: str) -> bool:
    if provider in ("openai", "azure"):
        return batch_id.startswith("batch")
    if provider == "vertex_ai":
        return (
            batch_id.startswith("projects/")
            or "batchPredictionJobs" in batch_id
            or batch_id.isdigit()
        )
    if provider == "bedrock":
        return batch_id.startswith("arn:aws:bedrock:")
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
    return _b64_decode(id_str).startswith("litellm_proxy")


def is_model_encoded_id(id_str: str) -> bool:
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
