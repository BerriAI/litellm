from collections.abc import Mapping
from typing import Final

from litellm._logging import verbose_logger
from litellm.types.llms.openai import AllMessageValues, OpenAITextCompletionUserMessage
from litellm.utils import supports_reasoning

from ...base_llm.completion.transformation import BaseTextCompletionConfig
from ...openai.completion.utils import _transform_prompt
from ..chat.transformation import (
    EFFORT_KWARG_KEYS,
    NIM_VLLM_STRIP_PARAMS,
    FireworksAIConfig,
    effort_from_chat_template_kwargs,
)
from ..common_utils import FireworksAIMixin, resolve_fireworks_resource_name

_TEXT_COMPLETION_STRIP_PARAMS: Final = (
    frozenset({"truncate_prompt_tokens", "prompt_truncate_len"}) | NIM_VLLM_STRIP_PARAMS
)


class FireworksAITextCompletionConfig(FireworksAIMixin, BaseTextCompletionConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        See how LiteLLM supports Provider-specific parameters - https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage
        """
        return [
            "max_tokens",
            "logprobs",
            "echo",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "response_format",
            "stream",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def map_extra_body_params(
        self, optional_params: Mapping[str, object], model: str
    ) -> dict:  # mutable-ok: returned dict is spread into the OpenAI SDK call as kwargs
        raw_extra_body: Final = optional_params.get("extra_body")
        initial_body: Final = (
            dict(raw_extra_body) if isinstance(raw_extra_body, dict) else {}  # mutable-ok: JSON request body
        )
        stripped_body: Final = self._strip_unsupported_params(initial_body, model)
        moved_body: Final = self._move_native_params_into_extra_body(stripped_body, optional_params)
        effort_body: Final = self._translate_chat_template_kwargs(moved_body, optional_params, model)
        final_body: Final = self._translate_guided_into_extra_body(effort_body, optional_params)
        base: Final = {  # mutable-ok: JSON request body
            k: v
            for k, v in optional_params.items()
            if k not in ("extra_body", "response_format", "reasoning_effort", "thinking")
        }
        if final_body:
            base["extra_body"] = final_body
        return base

    @staticmethod
    def _strip_unsupported_params(
        extra_body: Mapping[str, object], model: str
    ) -> dict:  # mutable-ok: JSON request body
        stripped: Final = tuple(sorted(k for k in extra_body if k in _TEXT_COMPLETION_STRIP_PARAMS))
        if stripped:
            verbose_logger.debug(
                "fireworks_ai does not support NIM/vLLM params %s for model=%s; dropping them from the request.",
                stripped,
                model,
            )
        return {  # mutable-ok: JSON request body
            k: v for k, v in extra_body.items() if k not in _TEXT_COMPLETION_STRIP_PARAMS
        }

    @staticmethod
    def _move_native_params_into_extra_body(
        extra_body: Mapping[str, object], optional_params: Mapping[str, object]
    ) -> dict:  # mutable-ok: JSON request body
        moved: Final = dict(extra_body)  # mutable-ok: JSON request body
        for key in ("response_format", "reasoning_effort", "thinking"):
            value = optional_params.get(key)
            if value is None:
                continue
            if key in moved:
                verbose_logger.debug("fireworks_ai overriding extra_body.%s with the top-level %s.", key, key)
            moved[key] = value
        return moved

    def _translate_chat_template_kwargs(
        self, extra_body: Mapping[str, object], optional_params: Mapping[str, object], model: str
    ) -> dict:  # mutable-ok: JSON request body
        chat_template_kwargs: Final = extra_body.get("chat_template_kwargs")
        if chat_template_kwargs is None:
            return dict(extra_body)  # mutable-ok: JSON request body
        result: Final = {  # mutable-ok: JSON request body
            k: v for k, v in extra_body.items() if k != "chat_template_kwargs"
        }
        if not isinstance(chat_template_kwargs, dict):
            verbose_logger.debug(
                "fireworks_ai dropping chat_template_kwargs for model=%s; expected an object, got %s.",
                model,
                type(chat_template_kwargs).__name__,
            )
            return result
        other_keys: Final = tuple(sorted(k for k in chat_template_kwargs if k not in EFFORT_KWARG_KEYS))
        if other_keys:
            verbose_logger.debug(
                "fireworks_ai does not support chat_template_kwargs keys %s for model=%s; dropping them.",
                other_keys,
                model,
            )
        effort: Final = effort_from_chat_template_kwargs(chat_template_kwargs)
        if effort is None:
            return result
        if any(key in result or key in optional_params for key in ("reasoning_effort", "thinking")):
            verbose_logger.debug(
                "fireworks_ai ignoring chat_template_kwargs; explicit reasoning_effort/thinking takes precedence."
            )
            return result
        if not supports_reasoning(model=model, custom_llm_provider="fireworks_ai"):
            verbose_logger.debug(
                "fireworks_ai model %r does not support reasoning; dropping chat_template_kwargs effort keys.",
                model,
            )
            return result
        return {**result, "reasoning_effort": effort}  # mutable-ok: JSON request body

    @staticmethod
    def _translate_guided_into_extra_body(
        extra_body: Mapping[str, object], optional_params: Mapping[str, object]
    ) -> dict:  # mutable-ok: JSON request body
        guided_response_format: Final = FireworksAIConfig.translate_guided_params(extra_body, optional_params)
        remaining: Final = {  # mutable-ok: JSON request body
            k: v for k, v in extra_body.items() if k not in ("guided_json", "guided_grammar", "guided_choice")
        }
        if guided_response_format:
            return {  # mutable-ok: JSON request body
                **remaining,
                guided_response_format[0][0]: guided_response_format[0][1],
            }
        return remaining

    def transform_text_completion_request(
        self,
        model: str,
        messages: list[AllMessageValues] | list[OpenAITextCompletionUserMessage],
        optional_params: dict,
        headers: dict,
    ) -> dict:
        translated_params: Final = self.map_extra_body_params(optional_params=optional_params, model=model)
        prompt: Final = _transform_prompt(messages=messages)

        data: Final = {
            "model": resolve_fireworks_resource_name(model),
            "prompt": prompt,
            **translated_params,
        }
        return data
