"""
Support for OrcaRouter's `/v1/chat/completions` endpoint.

OrcaRouter is OpenAI-compatible and proxies 150+ upstream models. Routing
preferences are exposed via OrcaRouter-specific extra_body fields (`models`,
`route`).

Per-vendor reasoning protocol (see orcarouter-shared-notes.md §15):
- anthropic/* → top-level `thinking: {type: "enabled", budget_tokens: N}` block
- openai/*, qwen/*, grok/*, gemini/*, kimi/* → top-level `reasoning_effort`
- deepseek/*-reasoner → no reasoning fields (model auto-reasons)

Model quirks (see §16) require client-side parameter dropping because upstream
providers reject unsupported fields with 400.

Docs: https://docs.orcarouter.ai
"""

from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx

import litellm
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.llms.orcarouter import OrcaRouterErrorMessage
from litellm.types.utils import ModelResponse, ModelResponseStream

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import OrcaRouterException


class CacheControlSupportedModels(str, Enum):
    """Upstream vendors that accept cache_control hints in content blocks."""

    CLAUDE = "claude"
    GEMINI = "gemini"
    MINIMAX = "minimax"
    GLM = "glm"
    ZAI = "z-ai"


# Fields that upstream providers reject with 400 for specific models.
# Source: orcarouter-shared-notes.md §16 A.
_MODEL_FIELD_BLACKLIST: Dict[str, Tuple[str, ...]] = {
    "anthropic/claude-opus-4.7": ("temperature", "top_k"),
    "openai/gpt-4o": ("top_k",),
    "openai/gpt-4o-mini": ("top_k",),
    "openai/gpt-4.1": ("top_k",),
    "openai/gpt-4.1-mini": ("top_k",),
    "openai/gpt-4.1-nano": ("top_k",),
    "openai/gpt-4-turbo": ("top_k",),
    "grok/grok-4.3": ("presence_penalty", "frequency_penalty"),
}

# Prefix patterns: any model whose normalized id starts with the key drops the
# listed fields. Covers entire model families (gpt-5*, deepseek-reasoner).
_MODEL_PREFIX_BLACKLIST: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("openai/gpt-5", ("temperature",)),  # gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.X
    ("deepseek/deepseek-reasoner", ("temperature",)),
)

# Forced parameter values: upstream accepts only one value (§16 B).
_MODEL_FIELD_FORCE: Dict[str, Dict[str, Any]] = {
    "kimi/kimi-k2.6": {"temperature": 1, "top_p": 0.95},
}

# Models that auto-reason without accepting any reasoning control fields.
_REASONING_AUTO_PREFIXES: Tuple[str, ...] = ("deepseek/deepseek-reasoner",)

# Map standard reasoning_effort strings to Anthropic thinking budget_tokens.
# budget_tokens must be >= 1024 and < max_tokens per Anthropic API.
_REASONING_EFFORT_TO_BUDGET: Dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
}


def _normalize_model(model: str) -> str:
    """Strip the LiteLLM-style 'orcarouter/' routing prefix if present."""
    if model.startswith("orcarouter/"):
        return model[len("orcarouter/") :]
    return model


def _is_anthropic_model(model: str) -> bool:
    return _normalize_model(model).startswith("anthropic/")


def _is_reasoning_auto_model(model: str) -> bool:
    normalized = _normalize_model(model)
    return any(normalized.startswith(prefix) for prefix in _REASONING_AUTO_PREFIXES)


class OrcaRouterConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Allow reasoning parameters for models flagged as reasoning-capable.
        """
        supported_params = super().get_supported_openai_params(model=model)
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider="orcarouter"
            ) or litellm.supports_reasoning(model=model):
                supported_params.append("reasoning_effort")
                supported_params.append("thinking")
        except Exception:
            pass
        return list(dict.fromkeys(supported_params))

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Pop OrcaRouter-only routing fields before delegating to the OpenAI
        # mapper so they don't surface as unsupported-param warnings.
        models = non_default_params.pop("models", None)
        route = non_default_params.pop("route", None)

        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )

        extra_body: Dict[str, Any] = {}
        if models is not None:
            extra_body["models"] = models
        if route is not None:
            extra_body["route"] = route
        if extra_body:
            mapped_openai_params["extra_body"] = extra_body

        # Per-vendor reasoning protocol translation.
        self._normalize_reasoning_params(model, mapped_openai_params)

        # Drop / force fields per the upstream quirks table (§16).
        self._apply_model_quirks(model, mapped_openai_params)

        return mapped_openai_params

    def _normalize_reasoning_params(self, model: str, params: dict) -> None:
        """
        Convert LiteLLM's standard `reasoning_effort` / `thinking` inputs into
        the vendor-specific form OrcaRouter expects:

        - Anthropic: drop `reasoning_effort`, build a `thinking` block.
        - DeepSeek reasoner: drop both, model auto-reasons.
        - All other vendors: pass `reasoning_effort` through, drop `thinking`.

        Mutates `params` in place.
        """
        if _is_reasoning_auto_model(model):
            params.pop("reasoning_effort", None)
            params.pop("thinking", None)
            return

        if _is_anthropic_model(model):
            existing_thinking = params.pop("thinking", None)
            effort = params.pop("reasoning_effort", None)
            if existing_thinking is not None:
                params["thinking"] = existing_thinking
            elif effort is not None:
                budget = _REASONING_EFFORT_TO_BUDGET.get(
                    effort, _REASONING_EFFORT_TO_BUDGET["medium"]
                )
                params["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return

        # OpenAI / Qwen / Grok / Gemini / Kimi reasoning families all use the
        # top-level `reasoning_effort` field; thinking blocks are unsupported.
        params.pop("thinking", None)

    def _apply_model_quirks(self, model: str, params: dict) -> None:
        """Drop or force fields per the upstream quirks table (§16). Mutates."""
        normalized = _normalize_model(model)

        blacklist: List[str] = list(_MODEL_FIELD_BLACKLIST.get(normalized, ()))
        for prefix, fields in _MODEL_PREFIX_BLACKLIST:
            if normalized.startswith(prefix):
                blacklist.extend(fields)
        for field in blacklist:
            params.pop(field, None)

        forced = _MODEL_FIELD_FORCE.get(normalized)
        if forced:
            for key, value in forced.items():
                params[key] = value

    def _supports_cache_control_in_content(self, model: str) -> bool:
        model_lower = model.lower()
        return any(
            supported_model.value in model_lower
            for supported_model in CacheControlSupportedModels
        )

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List["ChatCompletionToolParam"]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List["ChatCompletionToolParam"]]]:
        if self._supports_cache_control_in_content(model):
            return messages, tools
        return super().remove_cache_control_flag_from_messages_and_tools(
            model, messages, tools
        )

    def _move_cache_control_to_content(
        self, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """
        Move cache_control from message level to content blocks.
        OrcaRouter forwards Anthropic cache_control hints to the upstream Claude
        API, which requires the hint inside content blocks (not at message
        level). Only the LAST block per message gets the hint to respect
        Anthropic's 4-breakpoint cap.
        """
        transformed_messages: List[AllMessageValues] = []
        for message in messages:
            message_dict = dict(message)
            cache_control = message_dict.pop("cache_control", None)

            if cache_control is not None:
                content = message_dict.get("content")

                if isinstance(content, list):
                    if len(content) > 0:
                        content_copy = []
                        for i, block in enumerate(content):
                            block_dict = dict(block)
                            if i == len(content) - 1:
                                block_dict["cache_control"] = cache_control
                            content_copy.append(block_dict)
                        message_dict["content"] = content_copy
                else:
                    message_dict["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
                    ]

            transformed_messages.append(cast(AllMessageValues, message_dict))

        return transformed_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # LiteLLM's provider resolver strips the outer "orcarouter/" prefix
        # before dispatch, leaving a bare name like "auto" for native routers.
        # OrcaRouter's canonical model id is "orcarouter/auto" (per /api/pricing),
        # so re-prefix bare router names here. Vendor-namespaced ids like
        # "openai/gpt-5" are not affected (they still contain a "/").
        if "/" not in model:
            model = f"orcarouter/{model}"

        if self._supports_cache_control_in_content(model):
            messages = self._move_cache_control_to_content(messages)

        extra_body = optional_params.pop("extra_body", {})
        response = super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )
        response.update(extra_body)
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OrcaRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return OrcaRouterChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class OrcaRouterChatCompletionStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            if "error" in chunk:
                error_chunk = chunk["error"]
                error_message = OrcaRouterErrorMessage(
                    message="Message: {}, Metadata: {}, User ID: {}".format(
                        error_chunk["message"],
                        error_chunk.get("metadata", {}),
                        error_chunk.get("user_id", ""),
                    ),
                    code=error_chunk["code"],
                    metadata=error_chunk.get("metadata", {}),
                )
                raise OrcaRouterException(
                    message=error_message["message"],
                    status_code=error_message["code"],
                    headers=error_message["metadata"].get("headers", {}),
                )

            new_choices = []
            for choice in chunk["choices"]:
                choice["delta"]["reasoning_content"] = choice["delta"].get("reasoning")
                new_choices.append(choice)
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                usage=chunk.get("usage"),
                model=chunk["model"],
                choices=new_choices,
            )
        except KeyError as e:
            raise OrcaRouterException(
                message=f"KeyError: {e}, Got unexpected response from OrcaRouter: {chunk}",
                status_code=400,
                headers={"Content-Type": "application/json"},
            )
        except Exception as e:
            raise e
