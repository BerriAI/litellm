# Derived from tm-v1-ai-guard-litellm-plugin revision 6cafc143f62962a98d4eb5abe9f608c61ff194d4.
# This file has been modified for integration into LiteLLM.
# Licensed under the Apache License, Version 2.0. See LICENSE.txt in this directory.

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class TrendAISettings(BaseModel):
    app_name: str | None = None
    fallback_on_error: Literal["block", "allow"] = "block"
    timeout: float = 5.0
    stream_batch_size: int = 2048
    stream_overlap_size: int = 256
    response_content_chunk_size_bytes: int = 49_500
    logging_only_scan: Literal["request", "response", "both"] = "both"


class TrendAIChatMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class TrendAIChatChoice(BaseModel):
    index: int = 0
    message: TrendAIChatMessage
    finish_reason: Literal["stop"] = "stop"


class TrendAIChatCompletionPayload(BaseModel):
    """The OpenAI chat-completion shape Trend AI Guard scans model output as."""

    id: str = "chatcmpl-stream"
    object: Literal["chat.completion"] = "chat.completion"
    created: int = 0
    model: str
    choices: tuple[TrendAIChatChoice, ...]

    @classmethod
    def for_content(cls, content: str, model: str) -> "TrendAIChatCompletionPayload":
        return cls(model=model, choices=(TrendAIChatChoice(message=TrendAIChatMessage(content=content)),))


class TrendAIRedactedMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None


class TrendAIRedactedChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: TrendAIRedactedMessage | None = None


class TrendAIRedactedResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choices: tuple[TrendAIRedactedChoice, ...] = ()


class TrendAIRedactedPrompt(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str | None = None


class TrendAISensitiveRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = ""


class TrendAISensitiveInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    has_policy_violation: bool = Field(default=False, alias="hasPolicyViolation")
    rules: tuple[TrendAISensitiveRule, ...] = ()


class TrendAIResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action: str
    reasons: tuple[str, ...] = ()
    reason: str = ""
    redacted_request: dict[str, object] | None = Field(default=None, alias="redactedRequest")
    sensitive_information: TrendAISensitiveInformation | None = Field(default=None, alias="sensitiveInformation")


@dataclass(frozen=True, slots=True)
class TrendAIAllow:
    kind: Literal["allow"] = "allow"
    redacted_content: str | None = None
    masked_entity_count: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class TrendAIBlock:
    reason: str
    status_code: int = 400
    kind: Literal["block"] = "block"


@dataclass(frozen=True, slots=True)
class TrendAIProviderFailure:
    reason: str
    status_code: int | None = None
    kind: Literal["provider_failure"] = "provider_failure"


TrendAIScanResult: TypeAlias = TrendAIAllow | TrendAIBlock | TrendAIProviderFailure


@dataclass(frozen=True, slots=True)
class TrendAITextWindow:
    text: str
    start: int


@dataclass(frozen=True, slots=True)
class TrendAIWindowScan:
    window: TrendAITextWindow
    result: TrendAIScanResult


@dataclass(frozen=True, slots=True)
class TrendAIRequestPrompt:
    prompt: str
    start: int
    end: int
