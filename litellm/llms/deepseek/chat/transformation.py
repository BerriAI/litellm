"""
Translates from OpenAI's `/v1/chat/completions` to DeepSeek's `/v1/chat/completions`
"""

from collections.abc import Coroutine, Mapping, Sequence
from typing import Any, Final, Literal, cast, overload

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
    extract_search_results_text,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import supports_reasoning, supports_vision

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


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

        return optional_params

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
        DeepSeek vision models accept image_url content blocks in user
        messages (https://api-docs.deepseek.com/guides/vision), so those
        content lists are forwarded as-is, with any search_results text
        appended as a trailing text block. Every other message keeps the
        historical string collapse (which also folds search_results text
        into string content); a list with no extractable text stays
        unchanged, matching what DeepSeek historically received.
        """
        forward_images: Final = any(
            isinstance(message.get("content"), list) for message in messages
        ) and supports_vision(model=model, custom_llm_provider="deepseek")
        transformed: Final = [  # mutable-ok: provider messages must stay JSON-array lists the base transform mutates
            self._forward_or_collapse_content(message=message, forward_images=forward_images) for message in messages
        ]

        if is_async:
            return super()._transform_messages(messages=transformed, model=model, is_async=True)
        else:
            return super()._transform_messages(messages=transformed, model=model, is_async=False)

    def _forward_or_collapse_content(self, message: AllMessageValues, forward_images: bool) -> AllMessageValues:
        """
        Returns the vision-forwardable message with any search_results text
        appended as a text block; every other message keeps the historical
        string collapse, which extracts the text from a content list and
        folds search_results text into string content.
        """
        content: Final = message.get("content")
        if (
            forward_images
            and isinstance(content, list)
            and self._is_vision_forwardable_content(message=message, content=content)
        ):
            return self._with_search_results_text_block(message=message, content=content)
        collapsed: Final = convert_content_list_to_str(message=message)
        if not collapsed or collapsed == content:
            return message
        collapsed_message: Final = {**message, "content": collapsed}  # mutable-ok: wire messages are plain JSON dicts
        return cast(AllMessageValues, collapsed_message)  # cast-ok: TypedDict spread narrows to dict

    def _is_vision_forwardable_content(self, message: AllMessageValues, content: Sequence[object]) -> bool:
        """
        True only for a user message whose content list holds well-formed
        text and image_url blocks with at least one image; a block missing
        its payload falls back to the string collapse instead of crashing
        or reaching the wire malformed. The model capability gate lives in
        the caller.
        """
        if message.get("role") != "user":
            return False
        if not all(self._is_forwardable_block(block) for block in content):
            return False
        return any(isinstance(block, dict) and block.get("type") == "image_url" for block in content)

    @staticmethod
    def _is_forwardable_block(block: object) -> bool:
        """A dict block typed text or image_url that carries its payload."""
        if not isinstance(block, dict):
            return False
        block_type: Final = block.get("type")
        if block_type == "image_url":
            return DeepSeekChatConfig._is_image_url_payload(block.get("image_url"))
        if block_type == "text":
            return isinstance(block.get("text"), str)
        return False

    @staticmethod
    def _is_image_url_payload(payload: object) -> bool:
        """A url string or an object carrying one, per the OpenAI image_url shape."""
        if isinstance(payload, str):
            return bool(payload)
        if not isinstance(payload, Mapping):
            return False
        url: Final = payload.get("url")
        return isinstance(url, str) and bool(url)

    def _with_search_results_text_block(self, message: AllMessageValues, content: Sequence[object]) -> AllMessageValues:
        """
        Appends the message's search_results text as a trailing text block,
        keeping the context that the string collapse used to fold in, and
        drops the non-OpenAI search_results key from the wire message.
        """
        message_fields: Final = cast(Mapping[str, object], message)  # cast-ok: search_results is not on the TypedDicts
        search_text: Final = extract_search_results_text(message_fields.get("search_results"))
        if not search_text:
            return message
        forwarded_content: Final = [*content, {"type": "text", "text": search_text}]  # mutable-ok: JSON-array content
        forwarded: Final = {  # mutable-ok: wire messages are plain JSON dicts
            **{key: value for key, value in message_fields.items() if key != "search_results"},
            "content": forwarded_content,
        }
        return cast(AllMessageValues, forwarded)  # cast-ok: TypedDict spread narrows to dict

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
