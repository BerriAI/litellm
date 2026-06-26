"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.litellm_core_utils.prompt_templates.factory import response_schema_prompt
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class DeepSeekChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        DeepSeek reasoner models support thinking parameter.
        """
        params = super().get_supported_openai_params(model)
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
        DeepSeek only supports `{"type": "enabled"}` - no budget_tokens like Anthropic.

        Reference: https://api-docs.deepseek.com/guides/thinking_mode
        """
        # Let parent handle standard params first
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        # Pop thinking/reasoning_effort from optional_params first (parent may have added them)
        # Then re-add only if valid for DeepSeek
        thinking_value = optional_params.pop("thinking", None)
        reasoning_effort = optional_params.pop("reasoning_effort", None)

        # Handle thinking parameter - only accept {"type": "enabled"}
        if thinking_value is not None:
            if isinstance(thinking_value, dict) and thinking_value.get("type") == "enabled":
                # DeepSeek only accepts {"type": "enabled"}, ignore budget_tokens
                optional_params["thinking"] = {"type": "enabled"}

        # Handle reasoning_effort - map to thinking enabled
        elif reasoning_effort is not None and reasoning_effort != "none":
            optional_params["thinking"] = {"type": "enabled"}

        return optional_params

    def _fill_reasoning_content(self, messages: List[AllMessageValues]) -> List[AllMessageValues]:
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
        result: List[AllMessageValues] = []
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
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]: ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]: ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
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
        return (
            supports_reasoning(model=model, custom_llm_provider="deepseek")
            and (optional_params.get("thinking") or {}).get("type") == "enabled"
        )

    def _downgrade_json_schema_to_json_object(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
    ) -> tuple[list[AllMessageValues], dict]:
        """
        DeepSeek's /chat/completions rejects `response_format={"type": "json_schema"}`
        with "This response_format type is unavailable now" — no DeepSeek model
        supports native Structured Outputs, only JSON mode (`{"type": "json_object"}`).
        See recurring reports #7580 and #7646.

        DeepSeek models are marked `supports_response_schema=True` in the model-cost
        map (#20885) and downstream consumers filter their model lists on that flag, so
        flipping it is not an option. Instead, honour the flag: downgrade json_schema to
        json_object and move the schema into the prompt via `response_schema_prompt`
        (the same helper the Gemini path uses when it can't enforce a schema natively).
        The injected message also satisfies DeepSeek's requirement that the word "json"
        appear in the input when json_object mode is used.

        The downgrade is unconditional (not gated on `supports_response_schema`) because
        no DeepSeek model supports native json_schema.
        """
        response_format = optional_params.get("response_format")
        if (
            not isinstance(response_format, dict)
            or response_format.get("type") != "json_schema"
        ):
            return messages, optional_params

        json_schema: Optional[dict] = None
        nested_schema = response_format.get("json_schema")
        if isinstance(nested_schema, dict):
            json_schema = nested_schema.get("schema")
        elif "response_schema" in response_format:
            json_schema = response_format["response_schema"]

        # DeepSeek rejects the json_schema *type* itself, so the format must always be
        # downgraded to json_object — even when the schema body is absent/empty, where
        # there is simply nothing to inject. Only convey the schema (and satisfy the
        # "json" keyword requirement) via the prompt when one is actually present.
        if json_schema:
            messages = messages + [
                {
                    "role": "user",
                    "content": response_schema_prompt(
                        model=model, response_schema=json_schema
                    ),
                }
            ]
        optional_params = {
            **optional_params,
            "response_format": {"type": "json_object"},
        }
        return messages, optional_params

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
        tools = optional_params.get("tools")
        if not isinstance(tools, list) or not tools:
            return optional_params

        def _is_function_tool(tool: object) -> bool:
            return isinstance(tool, dict) and tool.get("type") == "function"

        def _get_function_tool_name(tool: object) -> str | None:
            if not isinstance(tool, dict):
                return None
            function = tool.get("function")
            if not isinstance(function, dict):
                return None
            name = function.get("name")
            return name if isinstance(name, str) else None

        def _tool_choice_matches_function_tool(tool_choice: object, function_tool_names: set[str]) -> bool:
            if not isinstance(tool_choice, dict):
                return True
            if tool_choice.get("type") != "function":
                return False
            function = tool_choice.get("function")
            if not isinstance(function, dict):
                return False
            name = function.get("name")
            return isinstance(name, str) and name in function_tool_names

        function_tools = [tool for tool in tools if _is_function_tool(tool)]
        if len(function_tools) == len(tools):
            return optional_params

        dropped_types = sorted(
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
            function_tool_names = {
                name for tool in function_tools for name in (_get_function_tool_name(tool),) if name is not None
            }
            if not _tool_choice_matches_function_tool(cleaned.get("tool_choice"), function_tool_names):
                cleaned = {k: v for k, v in cleaned.items() if k != "tool_choice"}
            return {**cleaned, "tools": function_tools}
        return {k: v for k, v in cleaned.items() if k not in ("tool_choice", "parallel_tool_calls")}

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
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
        messages, optional_params = self._downgrade_json_schema_to_json_object(
            model=model, messages=messages, optional_params=optional_params
        )
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
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Async equivalent of transform_request — applies the same reasoning_content
        fix for multi-turn thinking-mode conversations.
        """
        optional_params = self._drop_unsupported_tools(optional_params)
        messages, optional_params = self._downgrade_json_schema_to_json_object(
            model=model, messages=messages, optional_params=optional_params
        )
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
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("DEEPSEEK_API_BASE") or "https://api.deepseek.com/beta"  # type: ignore
        dynamic_api_key = api_key or get_secret_str("DEEPSEEK_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default DeepSeek /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.deepseek.com/beta"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
