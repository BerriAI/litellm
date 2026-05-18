"""
This hook is used to inject cache control directives into the messages of a chat completion.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

"""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.types.integrations.anthropic_cache_control_hook import (
    CacheControlInjectionPoint,
    CacheControlMessageInjectionPoint,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionCachedContent
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AnthropicCacheControlHook(CustomPromptManagement):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: Optional[PromptSpec] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Apply cache control directives based on specified injection points.

        Returns:
        - model: str - the model to use
        - messages: List[AllMessageValues] - messages with applied cache controls
        - non_default_params: dict - params with any global cache controls
        """
        # Extract cache control injection points
        injection_points: List[CacheControlInjectionPoint] = non_default_params.pop(
            "cache_control_injection_points", []
        )
        if not injection_points:
            return model, messages, non_default_params

        # Create a deep copy of messages to avoid modifying the original list
        processed_messages = copy.deepcopy(messages)

        # Separate message-level and non-message-level injection points
        remaining_points = []
        for point in injection_points:
            if point.get("location") == "message":
                point = cast(CacheControlMessageInjectionPoint, point)
                processed_messages = self._process_message_injection(
                    point=point, messages=processed_messages
                )
            else:
                remaining_points.append(point)

        # Pass through non-message injection points for provider-specific handling
        if remaining_points:
            non_default_params["cache_control_injection_points"] = remaining_points

        return model, processed_messages, non_default_params

    @staticmethod
    def _process_message_injection(
        point: CacheControlMessageInjectionPoint, messages: List[AllMessageValues]
    ) -> List[AllMessageValues]:
        """Process message-level cache control injection."""
        control: ChatCompletionCachedContent = point.get(
            "control", None
        ) or ChatCompletionCachedContent(type="ephemeral")

        _targetted_index: Optional[Union[int, str]] = point.get("index", None)
        targetted_index: Optional[int] = None
        if isinstance(_targetted_index, str):
            try:
                targetted_index = int(_targetted_index)
            except ValueError:
                pass
        else:
            targetted_index = _targetted_index

        targetted_role = point.get("role", None)

        # Case 1: Target by specific index
        if targetted_index is not None:
            original_index = targetted_index
            # Handle negative indices (convert to positive)
            if targetted_index < 0:
                targetted_index += len(messages)

            if 0 <= targetted_index < len(messages):
                messages[targetted_index] = (
                    AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                        messages[targetted_index], control
                    )
                )
            else:
                verbose_logger.warning(
                    f"AnthropicCacheControlHook: Provided index {original_index} is out of bounds for message list of length {len(messages)}. "
                    f"Targeted index was {targetted_index}. Skipping cache control injection for this point."
                )
        # Case 2: Target by role
        elif targetted_role is not None:
            for msg in messages:
                if msg.get("role") == targetted_role:
                    msg = (
                        AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                            message=msg, control=control
                        )
                    )
        return messages

    @staticmethod
    def _safe_insert_cache_control_in_message(
        message: AllMessageValues, control: ChatCompletionCachedContent
    ) -> AllMessageValues:
        """
        Safe way to insert cache control in a message

        OpenAI Message content can be either:
            - string
            - list of objects

        This method handles inserting cache control in both cases.
        Per Anthropic's API specification, when using multiple content blocks,
        only the last content block can have cache_control.
        """
        message_content = message.get("content", None)

        # 1. if string, insert cache control in the message
        if isinstance(message_content, str):
            message["cache_control"] = control  # type: ignore
        # 2. list of objects - only apply to last item per Anthropic spec
        elif isinstance(message_content, list):
            if len(message_content) > 0 and isinstance(message_content[-1], dict):
                message_content[-1]["cache_control"] = control  # type: ignore
        return message

    @property
    def integration_name(self) -> str:
        """Return the integration name for this hook."""
        return "anthropic_cache_control_hook"

    def should_run_prompt_management(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """Always return False since this is not a true prompt management system."""
        return False

    def _compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_spec: Optional[PromptSpec],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        """Not used - this hook only modifies messages, doesn't fetch prompts."""
        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=[],
            prompt_template_model=None,
            prompt_template_optional_params=None,
            completed_messages=None,
        )

    async def async_compile_prompt_helper(
        self,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: Optional[PromptSpec] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
    ) -> PromptManagementClient:
        """Not used - this hook only modifies messages, doesn't fetch prompts."""
        return self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_spec=prompt_spec,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: Optional[PromptSpec] = None,
        tools: Optional[List[Dict]] = None,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
        ignore_prompt_manager_model: Optional[bool] = False,
        ignore_prompt_manager_optional_params: Optional[bool] = False,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """Async version - delegates to sync since no async operations needed."""
        return self.get_chat_completion_prompt(
            model=model,
            messages=messages,
            non_default_params=non_default_params,
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )

    @staticmethod
    def should_use_anthropic_cache_control_hook(non_default_params: Dict) -> bool:
        if non_default_params.get("cache_control_injection_points", None):
            return True
        return False

    # ------------------------------------------------------------------
    # Anthropic /v1/messages (native Messages API) processing
    #
    # The OpenAI-shaped path (``get_chat_completion_prompt`` above) writes
    # ``cache_control`` at the message level for string content so it can be
    # detected later by both OpenAI- and Anthropic-style downstream
    # transformers. Anthropic's /v1/messages does NOT accept message-level
    # ``cache_control`` — it must live on a content block. Additionally,
    # /v1/messages separates ``system`` and ``tools`` from ``messages``, so
    # those need their own injection paths.
    # ------------------------------------------------------------------

    @staticmethod
    def apply_to_anthropic_messages_request(
        messages: List[Dict[str, Any]],
        system: Optional[Union[str, List[Dict[str, Any]]]],
        tools: Optional[List[Dict[str, Any]]],
        non_default_params: Dict[str, Any],
    ) -> Tuple[
        List[Dict[str, Any]],
        Optional[Union[str, List[Dict[str, Any]]]],
        Optional[List[Dict[str, Any]]],
    ]:
        """Apply ``cache_control_injection_points`` to an Anthropic /v1/messages
        request.

        Unlike :meth:`get_chat_completion_prompt`, this entrypoint:
            - inserts ``cache_control`` at the *block* level (Anthropic's API
              rejects message-level cache_control),
            - handles the top-level ``system`` parameter (which is not a
              message in /v1/messages), and
            - handles ``tools`` (cache_control on the final tool covers the
              whole list).

        The provided ``non_default_params`` is mutated to remove the
        ``cache_control_injection_points`` key so it is not forwarded
        downstream as an unknown field.
        """
        injection_points: List[CacheControlInjectionPoint] = non_default_params.pop(
            "cache_control_injection_points", []
        )
        if not injection_points:
            return messages, system, tools

        processed_messages = copy.deepcopy(messages)
        processed_system: Optional[Union[str, List[Dict[str, Any]]]] = (
            copy.deepcopy(system) if system is not None else None
        )
        processed_tools: Optional[List[Dict[str, Any]]] = (
            copy.deepcopy(tools) if tools is not None else None
        )

        for point in injection_points:
            location = point.get("location")
            control: ChatCompletionCachedContent = point.get(
                "control", None
            ) or ChatCompletionCachedContent(type="ephemeral")
            if location == "message":
                # Backward-compat sugar for the very common Claude Code config
                # ``location: message, role: system``. On /v1/chat/completions
                # that targets the system message in the ``messages`` array;
                # on /v1/messages there IS no system message (system is a
                # top-level param) so the original config silently no-op'd
                # for customers (this PR's motivating bug report). When we
                # detect that exact mismatch — role=system requested, no
                # system-role message present, but a top-level system param
                # IS — treat it as ``location: system``.
                if (
                    point.get("role") == "system"
                    and point.get("index") is None
                    and processed_system is not None
                    and not any(m.get("role") == "system" for m in processed_messages)
                ):
                    processed_system = AnthropicCacheControlHook._insert_cache_control_in_anthropic_system(
                        system=processed_system, control=control
                    )
                    continue
                processed_messages = (
                    AnthropicCacheControlHook._process_anthropic_message_injection(
                        point=cast(CacheControlMessageInjectionPoint, point),
                        messages=processed_messages,
                    )
                )
            elif location == "system":
                processed_system = (
                    AnthropicCacheControlHook._insert_cache_control_in_anthropic_system(
                        system=processed_system, control=control
                    )
                )
            elif location == "tools":
                processed_tools = (
                    AnthropicCacheControlHook._insert_cache_control_in_anthropic_tools(
                        tools=processed_tools, control=control
                    )
                )
            else:
                # Unknown location (e.g. "tool_config" for Bedrock Converse) –
                # not applicable on the Anthropic Messages API path. Leave
                # everything untouched and log so the user can debug silent
                # misconfiguration.
                verbose_logger.debug(
                    "AnthropicCacheControlHook: ignoring injection point "
                    f"with location={location!r} on /v1/messages path."
                )

        return processed_messages, processed_system, processed_tools

    @staticmethod
    def _process_anthropic_message_injection(
        point: CacheControlMessageInjectionPoint,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """``location: "message"`` for Anthropic /v1/messages format.

        Differs from :meth:`_process_message_injection` in that cache_control
        is *always* written at the block level, never on the message dict
        itself.  String-content messages are upgraded to a single-block list
        so the cache_control field has somewhere valid to live.

        Note: on /v1/messages the system prompt is a separate top-level
        parameter, NOT a message with ``role: "system"``. The customer report
        in https://github.com/BerriAI/litellm/issues (Claude Code) hit this
        exact pitfall.  We still honor ``role: "system"`` here for
        compatibility, but it will be a no-op when the request has no
        system-role message (which is the case for /v1/messages).
        """
        control: ChatCompletionCachedContent = point.get(
            "control", None
        ) or ChatCompletionCachedContent(type="ephemeral")

        _targetted_index: Optional[Union[int, str]] = point.get("index", None)
        targetted_index: Optional[int] = None
        if isinstance(_targetted_index, str):
            try:
                targetted_index = int(_targetted_index)
            except ValueError:
                pass
        else:
            targetted_index = _targetted_index

        targetted_role = point.get("role", None)

        if targetted_index is not None:
            original_index = targetted_index
            if targetted_index < 0:
                targetted_index += len(messages)
            if 0 <= targetted_index < len(messages):
                messages[targetted_index] = (
                    AnthropicCacheControlHook._safe_insert_cache_control_in_anthropic_message(
                        message=messages[targetted_index], control=control
                    )
                )
            else:
                verbose_logger.warning(
                    f"AnthropicCacheControlHook: Provided index {original_index} is out of bounds for message list of length {len(messages)}. "
                    f"Targeted index was {targetted_index}. Skipping cache control injection for this point."
                )
        elif targetted_role is not None:
            for i, msg in enumerate(messages):
                if msg.get("role") == targetted_role:
                    messages[i] = (
                        AnthropicCacheControlHook._safe_insert_cache_control_in_anthropic_message(
                            message=msg, control=control
                        )
                    )
        return messages

    @staticmethod
    def _safe_insert_cache_control_in_anthropic_message(
        message: Dict[str, Any], control: ChatCompletionCachedContent
    ) -> Dict[str, Any]:
        """Block-level cache_control insertion for an Anthropic message.

        - ``content: str``  → wrap into ``[{"type": "text", "text": str, "cache_control": control}]``.
          Anthropic accepts this shape and treats it identically to the
          original string body for token-counting purposes, while making the
          cache marker valid.
        - ``content: list`` → set ``cache_control`` on the last block. Per
          Anthropic's spec only the final block in a sequence carries the
          marker; everything preceding it is covered.
        """
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": control,
                }
            ]
        elif isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict):
                last["cache_control"] = control
        return message

    @staticmethod
    def _insert_cache_control_in_anthropic_system(
        system: Optional[Union[str, List[Dict[str, Any]]]],
        control: ChatCompletionCachedContent,
    ) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Apply ``cache_control`` to the Anthropic /v1/messages ``system``.

        Returns a list-of-blocks shape even when the input was a string,
        because cache_control on the string form is not valid Anthropic
        request syntax.  Returns ``None`` unchanged if there is no system
        prompt (caller should not have requested ``location: "system"`` in
        that case, but we no-op gracefully).
        """
        if system is None:
            verbose_logger.warning(
                "AnthropicCacheControlHook: cache_control_injection_points "
                "requested location='system' but request has no system prompt. "
                "Skipping."
            )
            return None
        if isinstance(system, str):
            return [{"type": "text", "text": system, "cache_control": control}]
        if isinstance(system, list) and system:
            last = system[-1]
            if isinstance(last, dict):
                last["cache_control"] = control
            return system
        return system

    @staticmethod
    def _insert_cache_control_in_anthropic_tools(
        tools: Optional[List[Dict[str, Any]]],
        control: ChatCompletionCachedContent,
    ) -> Optional[List[Dict[str, Any]]]:
        """Apply ``cache_control`` to the final tool definition.

        Anthropic only honors ``cache_control`` on the last entry of the
        ``tools`` array; doing so caches all preceding tool definitions as a
        single chunk.  No-ops gracefully when there are no tools (logged at
        debug level — this is a common shape for non-tool requests).
        """
        if not tools:
            verbose_logger.debug(
                "AnthropicCacheControlHook: cache_control_injection_points "
                "requested location='tools' but request has no tools. Skipping."
            )
            return tools
        last = tools[-1]
        if isinstance(last, dict):
            last["cache_control"] = control
        return tools

    @staticmethod
    def get_custom_logger_for_anthropic_cache_control_hook(
        non_default_params: Dict,
    ) -> Optional[CustomLogger]:
        from litellm.litellm_core_utils.litellm_logging import (
            _init_custom_logger_compatible_class,
        )

        if AnthropicCacheControlHook.should_use_anthropic_cache_control_hook(
            non_default_params
        ):
            return _init_custom_logger_compatible_class(
                logging_integration="anthropic_cache_control_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
        return None
