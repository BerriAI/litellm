"""
Support for OpenAI's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/chat-completions
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ProviderSpecificModelInfo
from litellm.utils import _get_model_cost_key

from ..openai.chat.gpt_transformation import OpenAIGPTConfig

TOGETHER_AI_API_BASE: Final = "https://api.together.xyz/v1"

# Together-native chat-completions params: accepted by the API, absent from OpenAI's schema.
_NATIVE_PARAMS: Final = (
    "top_k",
    "min_p",
    "repetition_penalty",
    "echo",
    "context_length_exceeded_behavior",
    "safety_model",
    "chat_template_kwargs",
    "reasoning",
)
_REASONING_PARAMS: Final = ("reasoning_effort", "thinking")
_FUNCTION_CALLING_PARAMS: Final = frozenset({"tools", "tool_choice", "function_call"})
_TEXT_RESPONSE_FORMAT: Final = MappingProxyType({"type": "text"})
_EMPTY_ENTRY: Final[Mapping[str, object]] = MappingProxyType({})

# Every serverless chat model takes tools, tool_choice and response_format, so an id that
# predates the cost map keeps them instead of silently losing them. An explicit map entry
# still wins, and capabilities that vary per model (reasoning, vision) stay unknown here.
_CAPABILITY_DEFAULTS: Final = MappingProxyType(
    {
        "supports_function_calling": True,
        "supports_tool_choice": True,
        "supports_response_schema": True,
    }
)


class TogetherAIReasoningParam(TypedDict):
    """Together's reasoning toggle: https://docs.together.ai/docs/inference/chat/reasoning"""

    enabled: ReadOnly[bool]


@dataclass(frozen=True, slots=True)
class _ParamFragment:
    """One rewrite step: keys to drop, keys to set, and keys to nest under `extra_body`.

    Anything the OpenAI SDK does not accept as a keyword argument has to travel in
    `extra_body`, which the SDK spreads back into the request body.
    """

    drop: frozenset[str] = frozenset()
    overrides: tuple[tuple[str, object], ...] = ()
    extra_body: tuple[tuple[str, object], ...] = ()


def _reasoning_toggle(enabled: bool) -> TogetherAIReasoningParam:
    toggle: Final[TogetherAIReasoningParam] = {"enabled": enabled}
    return toggle


def _max_tokens_fragment(params: Mapping[str, object]) -> _ParamFragment:
    """Together only reads `max_tokens`; it accepts and ignores `max_completion_tokens`."""
    requested: Final = params.get("max_completion_tokens")
    if requested is None:
        return _ParamFragment()
    if params.get("max_tokens") is not None:
        return _ParamFragment(drop=frozenset({"max_completion_tokens"}))
    return _ParamFragment(drop=frozenset({"max_completion_tokens"}), overrides=(("max_tokens", requested),))


def _logprobs_count(logprobs: object, top_logprobs: object) -> int | None:
    """Together's `logprobs` is how many top tokens to return, not OpenAI's boolean."""
    requested_count: Final = (
        top_logprobs if isinstance(top_logprobs, int) and not isinstance(top_logprobs, bool) else None
    )
    if isinstance(logprobs, bool):
        return (requested_count or 1) if logprobs else None
    if isinstance(logprobs, int):
        return logprobs
    return requested_count


def _logprobs_fragment(params: Mapping[str, object]) -> _ParamFragment:
    if "logprobs" not in params and "top_logprobs" not in params:
        return _ParamFragment()
    count: Final = _logprobs_count(params.get("logprobs"), params.get("top_logprobs"))
    dropped: Final = frozenset({"logprobs", "top_logprobs"})
    if count is None:
        return _ParamFragment(drop=dropped)
    return _ParamFragment(drop=dropped, overrides=(("logprobs", count),))


def _reasoning_fragment(params: Mapping[str, object]) -> _ParamFragment:
    """Fold litellm's `thinking` and `reasoning_effort="none"` onto Together's `reasoning` toggle."""
    dropped: Final = frozenset({"thinking", "reasoning"})
    native_toggle: Final = params.get("reasoning")
    if native_toggle is not None:
        return _ParamFragment(drop=dropped, extra_body=(("reasoning", native_toggle),))
    effort: Final = params.get("reasoning_effort")
    if effort == "none":
        return _ParamFragment(
            drop=dropped | frozenset({"reasoning_effort"}),
            extra_body=(("reasoning", _reasoning_toggle(False)),),
        )
    if effort == "minimal":
        return _ParamFragment(drop=dropped, overrides=(("reasoning_effort", "low"),))
    thinking: Final = params.get("thinking")
    if isinstance(thinking, Mapping):
        if thinking.get("budget_tokens") is not None:
            verbose_logger.debug("together_ai has no reasoning token budget; dropping thinking.budget_tokens.")
        return _ParamFragment(
            drop=dropped,
            extra_body=(("reasoning", _reasoning_toggle(thinking.get("type") != "disabled")),),
        )
    return _ParamFragment(drop=dropped)


def _response_format_fragment(params: Mapping[str, object]) -> _ParamFragment:
    if params.get("response_format") == _TEXT_RESPONSE_FORMAT:
        return _ParamFragment(drop=frozenset({"response_format"}))
    return _ParamFragment()


def _translation_plan(params: Mapping[str, object]) -> _ParamFragment:
    fragments: Final = (
        _max_tokens_fragment(params),
        _logprobs_fragment(params),
        _reasoning_fragment(params),
        _response_format_fragment(params),
    )
    return _ParamFragment(
        drop=frozenset(key for fragment in fragments for key in fragment.drop),
        overrides=tuple(override for fragment in fragments for override in fragment.overrides),
        extra_body=tuple(nested for fragment in fragments for nested in fragment.extra_body),
    )


def _merged_extra_body(
    existing: object,
    additions: tuple[tuple[str, object], ...],
) -> dict:  # mutable-ok: litellm only merges further params into extra_body when it is a real dict
    current: Final = existing.items() if isinstance(existing, Mapping) else ()
    return dict((*current, *additions))  # mutable-ok: same


class TogetherAIConfig(OpenAIGPTConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return (
            api_key
            or get_secret_str("TOGETHER_API_KEY")
            or get_secret_str("TOGETHER_AI_API_KEY")
            or get_secret_str("TOGETHERAI_API_KEY")
            or get_secret_str("TOGETHER_AI_TOKEN")
            or litellm.togetherai_api_key
        )

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("TOGETHER_AI_API_BASE") or TOGETHER_AI_API_BASE

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

    @staticmethod
    def _model_cost_entry(model: str) -> Mapping[str, object]:
        """The model's cost-map entry, or an empty mapping when nothing maps it.

        The provider-prefixed key is tried first: Together model ids are bare Hugging Face
        names, and several are also served by other providers under that same bare key.
        """
        bare: Final = model.removeprefix("together_ai/")
        for candidate in (f"together_ai/{bare}", model):
            resolved = _get_model_cost_key(candidate)
            if resolved is not None:
                return litellm.model_cost[resolved]
        return _EMPTY_ENTRY

    @staticmethod
    def _capability(entry: Mapping[str, object], capability: str) -> bool | None:
        declared: Final = entry.get(capability)
        return declared if isinstance(declared, bool) else _CAPABILITY_DEFAULTS.get(capability)

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        """Capability baseline for every Together model, overridden by any explicit map entry.

        Read by `litellm.supports_*` and merged over the cost-map entry in `get_model_info`, so
        an unmapped or newly launched model resolves to Together's real feature set instead of to
        "unknown", which the param gate below reads as "unsupported".
        """
        entry: Final = self._model_cost_entry(model)
        resolved: Final[ProviderSpecificModelInfo] = {
            "supports_function_calling": self._capability(entry, "supports_function_calling"),
            "supports_tool_choice": self._capability(entry, "supports_tool_choice"),
            "supports_response_schema": self._capability(entry, "supports_response_schema"),
            "supports_parallel_function_calling": self._capability(entry, "supports_parallel_function_calling"),
            "supports_reasoning": self._capability(entry, "supports_reasoning"),
            "supports_vision": self._capability(entry, "supports_vision"),
            # Together caches prompt prefixes with no opt-in, and bills the hit at the
            # cached rate only for the models that publish one.
            "supports_prompt_caching": True if entry.get("cache_read_input_token_cost") is not None else None,
        }
        return resolved

    def get_supported_openai_params(self, model: str) -> list:
        """
        Only some together models support tool calling / structured outputs / reasoning.

        Docs: https://docs.together.ai/docs/inference/function-calling/overview
        """
        capabilities: Final = self.get_provider_info(model)
        supports_tools: Final = capabilities["supports_function_calling"] is True
        supports_schema: Final = capabilities["supports_response_schema"] is True
        if not supports_tools:
            verbose_logger.debug(
                "together_ai model %r is marked as not supporting function calling in "
                "model_prices_and_context_window.json; `tools`, `tool_choice` and `function_call` "
                "will be dropped from the request.",
                model,
            )
        allowed: Final = tuple(
            param
            for param in super().get_supported_openai_params(model)
            if (supports_tools or param not in _FUNCTION_CALLING_PARAMS)
            and (supports_schema or param != "response_format")
        )
        reasoning_params: Final = _REASONING_PARAMS if capabilities["supports_reasoning"] is True else ()
        return [  # mutable-ok: BaseConfig contract returns a list
            *allowed,
            *_NATIVE_PARAMS,
            *reasoning_params,
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        plan: Final = _translation_plan(mapped)
        retained: Final = tuple(
            (key, value) for key, value in mapped.items() if key not in plan.drop and key != "extra_body"
        )
        extra_body: Final = _merged_extra_body(mapped.get("extra_body"), plan.extra_body)
        return dict(  # mutable-ok: request body handed to the OpenAI SDK
            (
                *retained,
                *plan.overrides,
                *((("extra_body", extra_body),) if extra_body else ()),
            )
        )

    def get_models(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo contract returns a list
        """
        Calls Together AI's `/v1/models` endpoint and returns the list of models.

        Docs: https://docs.together.ai/reference/models
        """
        resolved_base: Final = self.get_api_base(api_base)
        resolved_key: Final = self.get_api_key(api_key)
        if resolved_base is None or resolved_key is None:
            raise ValueError(
                "TOGETHER_AI_API_BASE or TOGETHER_API_KEY is not set. Please set the environment "
                "variable, to query Together AI's `/models` endpoint."
            )

        response: Final = litellm.module_level_client.get(
            url=f"{resolved_base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {resolved_key}"},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Failed to fetch models from Together AI. "
                f"Status code: {response.status_code}, Response: {response.text}"
            )

        payload: Final = response.json()
        listed: Final = (payload.get("data") if isinstance(payload, Mapping) else payload) or ()
        return [  # mutable-ok: BaseLLMModelInfo contract returns a list
            f"together_ai/{model['id']}" for model in listed if isinstance(model, Mapping) and model.get("id")
        ]
