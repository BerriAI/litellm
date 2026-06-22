"""Typed request bodies for the Bedrock Invoke sub-providers.

Each sub-provider declares the exact wire keys it accepts in its inference
params. Parsing `optional_params` into one of these models splits the payload
into known fields (kept) and unknown ones captured as `extra_body` passthrough.
The passthrough is forwarded by default and stripped when `drop_params` is set,
so a strict caller never ships keys the provider would reject.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict

from litellm.types.llms.bedrock import LITELLM_CONTROL_PARAM_KEYS

BEDROCK_INVOKE_PROVIDER = str


class BedrockInvokeInferenceParams(BaseModel):
    """Base for a sub-provider's inference-param body.

    Declared fields are the provider's wire keys; anything else is captured as
    passthrough (`model_extra`) and dropped only when `drop_params` is set.
    """

    model_config = ConfigDict(extra="allow")

    def to_body(self, *, drop_params: bool) -> Dict[str, object]:
        known = self.model_dump(
            exclude_none=True, exclude=set((self.model_extra or {}).keys())
        )
        if drop_params:
            return known
        return {**known, **(self.model_extra or {})}


class _CohereCommandRBody(BedrockInvokeInferenceParams):
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    p: Optional[float] = None
    k: Optional[float] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    preamble: Optional[str] = None
    prompt_truncation: Optional[str] = None
    return_prompt: Optional[bool] = None
    raw_prompting: Optional[bool] = None
    search_queries_only: Optional[bool] = None


class _CohereLegacyBody(BedrockInvokeInferenceParams):
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    p: Optional[float] = None
    seed: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    num_generations: Optional[int] = None
    return_likelihood: Optional[str] = None


class _AI21Body(BedrockInvokeInferenceParams):
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stream: Optional[bool] = None
    stopSequences: Optional[List[str]] = None


class _MistralBody(BedrockInvokeInferenceParams):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None


class _TitanTextGenerationConfig(BedrockInvokeInferenceParams):
    maxTokenCount: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[List[str]] = None


class _LlamaBody(BedrockInvokeInferenceParams):
    max_gen_len: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    topP: Optional[float] = None
    stream: Optional[bool] = None


_COHERE_COMMAND_R_PREFIX = "cohere.command-r"

_INVOKE_BODY_MODELS: Dict[str, Type[BedrockInvokeInferenceParams]] = {
    "cohere_command_r": _CohereCommandRBody,
    "cohere": _CohereLegacyBody,
    "ai21": _AI21Body,
    "mistral": _MistralBody,
    "amazon": _TitanTextGenerationConfig,
    "meta": _LlamaBody,
    "llama": _LlamaBody,
    "deepseek_r1": _LlamaBody,
}


def _resolve_invoke_body_model(
    provider: Optional[str], model: str
) -> Optional[Type[BedrockInvokeInferenceParams]]:
    if provider == "cohere" and model.startswith(_COHERE_COMMAND_R_PREFIX):
        return _CohereCommandRBody
    if provider is None:
        return None
    return _INVOKE_BODY_MODELS.get(provider)


def parse_invoke_inference_params(
    provider: Optional[str],
    model: str,
    params: Dict[str, object],
    drop_params: bool,
) -> Dict[str, object]:
    """Validate inference params against the sub-provider's typed body.

    Providers without a typed body (e.g. delegating ones) pass through
    unchanged, preserving today's behavior.
    """
    body_model = _resolve_invoke_body_model(provider, model)
    if body_model is None:
        return params
    return body_model.model_validate(params).to_body(drop_params=drop_params)


def assert_no_control_params(body: Dict[str, object]) -> None:
    """Guard run right before dispatch: control keys must never reach the wire."""
    leaked = LITELLM_CONTROL_PARAM_KEYS & body.keys()
    if leaked:
        raise ValueError(
            "litellm control params leaked into the Bedrock request body: "
            f"{sorted(leaked)}"
        )


def assert_no_control_params_in_payload(data: str) -> None:
    """Dispatch-time guard over the serialized request payload."""
    try:
        parsed = json.loads(data)
    except (TypeError, ValueError):
        return
    if isinstance(parsed, dict):
        assert_no_control_params(parsed)
