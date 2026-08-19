"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from collections.abc import Coroutine, Mapping
from typing import Any, Final, Literal, cast, overload

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

# Model families the DeepSeek API runs in thinking mode unless the caller turns
# it off, as opposed to the families where thinking is opt-in. Compared against
# the model name with any provider prefix stripped.
#
# This cannot be read off model_prices_and_context_window.json: every
# reasoning-capable DeepSeek entry there carries `supports_reasoning: true`,
# including deepseek-v3.1 / v3.2 / reasoner, where thinking is opt-in and
# tool_choice="required" is accepted while thinking is off. No capability flag
# distinguishes "reasoning is on unless disabled" from "reasoning is available",
# so the family is listed here.
THINKING_ON_BY_DEFAULT_MODELS: Final = ("deepseek-v4",)

# tool_choice values the DeepSeek API accepts outside thinking mode but rejects
# while it is active. `"any"` is deliberately absent: DeepSeek rejects it in
# every mode, thinking or not, with a deserialization error naming the values it
# accepts - that is a separate bug from this one, and `"any"` is not an
# OpenAI-spec tool_choice value to begin with.
TOOL_CHOICES_REJECTED_IN_THINKING_MODE: Final = ("required",)


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params: Final = super().get_supported_openai_params(model)
        params.extend(["thinking", "reasoning_effort"])
        return params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to DeepSeek params.

        Handles `thinking` and `reasoning_effort` parameters for DeepSeek reasoner models.
        DeepSeek supports `{"type": "enabled"}` and `{"type": "disabled"}` - no budget_tokens
        like Anthropic. `reasoning_effort="none"` is the OpenAI-style way to ask for thinking
        off, so it maps to `{"type": "disabled"}`; any other effort keeps thinking on.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value: Final = optional_params.pop("thinking", None)
        reasoning_effort: Final = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - accept both enabled and disabled, ignore budget_tokens
        if isinstance(thinking_value, dict) and thinking_value.get("type") in ("enabled", "disabled"):
            optional_params["thinking"] = {"type": thinking_value["type"]}

        # Otherwise fall back to reasoning_effort: "none" disables, anything else enables
        elif reasoning_effort is not None:
            optional_params["thinking"] = {"type": "disabled" if reasoning_effort == "none" else "enabled"}

        if self._tool_choice_is_rejected_in_thinking_mode(model=model, optional_params=optional_params):
            litellm.verbose_logger.warning(
                "DeepSeek thinking mode does not accept tool_choice=%r; sending "
                "tool_choice='auto' for model %r instead. The model now decides "
                "whether to call a tool, so a tool call is no longer forced. Turn "
                "thinking off (thinking={'type': 'disabled'} or "
                "reasoning_effort='none') to keep the requested tool_choice.",
                optional_params["tool_choice"],
                model,
            )
            optional_params["tool_choice"] = "auto"
            return optional_params

        return optional_params

    @staticmethod
    def _thinking_on_by_default(model: str) -> bool:
        """
        Whether the API enables thinking mode for this model unless the caller disables it.
        """
        model_name: Final = model.rsplit("/", 1)[-1].lower()
        return model_name.startswith(THINKING_ON_BY_DEFAULT_MODELS)

    def _thinking_mode_will_be_active(self, model: str, optional_params: Mapping[str, Any]) -> bool:
        """
        Whether the request being built will run in thinking mode - the caller
        enabled it, or the model runs it by default and the caller did not
        disable it.

        Reads the `thinking` value already resolved into `optional_params`, i.e.
        what is about to be sent, rather than the `supports_reasoning`
        capability flag: that flag is also true for models where thinking is
        merely available, and a deployment missing from the cost map still runs
        in thinking mode.
        """
        thinking: Final = optional_params.get("thinking")
        thinking_type: Final = thinking.get("type") if isinstance(thinking, dict) else None
        if thinking_type in ("enabled", "disabled"):
            return thinking_type == "enabled"
        return self._thinking_on_by_default(model)

    def _tool_choice_is_rejected_in_thinking_mode(self, model: str, optional_params: Mapping[str, Any]) -> bool:
        """
        DeepSeek rejects `tool_choice="required"` and the
        `{"type": "function", ...}` form while thinking mode is active:

            400 - Thinking mode does not support this tool_choice

        Only "auto" and "none" get through. deepseek-v4-pro / deepseek-v4-flash
        run in thinking mode by default, so callers hit this without ever
        passing `thinking` - every SDK that hardcodes `tool_choice="required"`
        to force a tool call (the OpenAI Agents SDK delegation path, LangChain's
        `bind_tools(tool_choice=...)`) fails on the first turn.

        The caller asked for a stronger guarantee than DeepSeek gives here, so
        downgrading to "auto" is lossy - but the alternative is a 400 that
        honours the requested tool_choice even less. A caller that needs the
        guarantee can turn thinking off with `thinking={"type": "disabled"}` or
        `reasoning_effort="none"`, which leaves tool_choice untouched.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        tool_choice: Final = optional_params.get("tool_choice")
        is_rejected: Final = tool_choice in TOOL_CHOICES_REJECTED_IN_THINKING_MODE or isinstance(tool_choice, dict)
        return is_rejected and self._thinking_mode_will_be_active(model=model, optional_params=optional_params)

    def _fill_reasoning_content(self, messages: list[AllMessageValues]) -> list[AllMessageValues]:
        """
        DeepSeek thinking mode requires `reasoning_content` to be passed back on
        every assistant message in multi-turn conversations. If it is missing,
        the API returns:
          "The reasoning_content in the thinking mode must be passed back to the API."

        For each assistant message that is missing `reasoning_content`:
          1. Promote it from `provider_specific_fields["reasoning_content"]` if present
             (LiteLLM stores provider-specific response fields there).
          2. Otherwise inject a single space — the minimum value the API accepts.
        """
        result: Final[list[AllMessageValues]] = []
        for msg in messages:
            if msg.get("role") == "assistant" and not msg.get("reasoning_content"):
                patched = dict(cast(dict, msg))
                provider_fields = patched.get("provider_specific_fields") or {}
                stored = provider_fields.get("reasoning_content")
                if stored:
                    patched["reasoning_content"] = stored
                    cleaned = dict(provider_fields)
                    cleaned.pop("reasoning_content", None)
                    patched["provider_specific_fields"] = cleaned
                else:
                    litellm.verbose_logger.warning(
                        "DeepSeek thinking mode: assistant message is missing "
                        "`reasoning_content` and none was saved in "
                        "`provider_specific_fields`. A single-space placeholder "
                        "is being injected to satisfy API validation, but the "
                        "model will receive a blank reasoning chain for this turn, "
                        "which may silently degrade multi-turn response quality. "
                        "Preserve `reasoning_content` from the original assistant "
                        "response when building multi-turn conversation history."
                    )
                    patched["reasoning_content"] = " "
                result.append(cast(AllMessageValues, patched))
            else:
                result.append(msg)
        return result

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, list[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: ...

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[Any, Any, list[AllMessageValues]]:
        """
        DeepSeek does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(messages=messages, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=messages, model=model, is_async=False)

    def _thinking_mode_active(self, model: str, optional_params: dict) -> bool:
        """
        Returns True only when thinking mode is actually active for this request:
          - model supports reasoning (capability check)
          - user explicitly passed thinking={"type": "enabled"} (opt-in check)
        """
        thinking: Final = optional_params.get("thinking")
        return (
            supports_reasoning(model=model, custom_llm_provider="deepseek")
            and isinstance(thinking, dict)
            and thinking.get("type") == "enabled"
        )

    @staticmethod
    def _drop_unsupported_tools(optional_params: dict) -> dict:
        """
        DeepSeek's /chat/completions only accepts tools of type "function".

        Requests bridged from /v1/responses can carry responses-API-native tool
        types (e.g. a Codex CLI tool typed "namespace"); DeepSeek rejects the
        whole request with `unknown variant '<type>', expected 'function'` (issue
        #30722). Drop the unsupported entries so the function tools still go
        through, and drop the now-dangling tool_choice/parallel_tool_calls when
        nothing callable survives.

        When a specific `tool_choice` points at a dropped tool, clear it so the
        sanitized request does not reference a tool DeepSeek will never receive.
        """
        tools: Final = optional_params.get("tools")
        if not isinstance(tools, list) or not tools:
            return optional_params

        def _is_function_tool(tool: object) -> bool:
            return isinstance(tool, dict) and tool.get("type") == "function"

        def _get_function_tool_name(tool: object) -> str | None:
            if not isinstance(tool, dict):
                return None
            function: Final = tool.get("function")
            if not isinstance(function, dict):
                return None
            name: Final = function.get("name")
            return name if isinstance(name, str) else None

        def _tool_choice_matches_function_tool(tool_choice: object, function_tool_names: set[str]) -> bool:
            if not isinstance(tool_choice, dict):
                return True
            if tool_choice.get("type") != "function":
                return False
            function: Final = tool_choice.get("function")
            if not isinstance(function, dict):
                return False
            name: Final = function.get("name")
            return isinstance(name, str) and name in function_tool_names

        function_tools: Final = [tool for tool in tools if _is_function_tool(tool)]
        if len(function_tools) == len(tools):
            return optional_params

        dropped_types: Final = sorted(
            {
                str(tool.get("type")) if isinstance(tool, dict) else type(tool).__name__
                for tool in tools
                if not _is_function_tool(tool)
            }
        )
        litellm.verbose_logger.warning(
            "DeepSeek chat completions only supports function tools; dropping "
            "unsupported tool type(s) %s before sending the request",
            dropped_types,
        )

        cleaned = {k: v for k, v in optional_params.items() if k != "tools"}
        if function_tools:
            function_tool_names: Final = {
                name for tool in function_tools for name in (_get_function_tool_name(tool),) if name is not None
            }
            if not _tool_choice_matches_function_tool(cleaned.get("tool_choice"), function_tool_names):
                cleaned = {k: v for k, v in cleaned.items() if k != "tool_choice"}
            return {**cleaned, "tools": function_tools}
        return {k: v for k, v in cleaned.items() if k not in ("tool_choice", "parallel_tool_calls")}

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Ensures `reasoning_content` is forwarded on assistant messages for
        multi-turn thinking-mode conversations (issue #28045).

        Only runs when thinking mode is actually active - guarded by both
        supports_reasoning() (model capability) and optional_params["thinking"]
        (user explicitly enabled it), preventing spurious injection on models
        like deepseek-v3.2 that support thinking as opt-in but not always-on.
        """
        optional_params = self._drop_unsupported_tools(optional_params)
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Async equivalent of transform_request — applies the same reasoning_content
        fix for multi-turn thinking-mode conversations.
        """
        optional_params = self._drop_unsupported_tools(optional_params)
        if self._thinking_mode_active(model=model, optional_params=optional_params):
            messages = self._fill_reasoning_content(messages)
        return await super().async_transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("DEEPSEEK_API_BASE") or "https://api.deepseek.com/beta"
        dynamic_api_key: Final = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
