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


# Anthropic (and Bedrock Claude) reject requests with more than 4 cache_control
# breakpoints: "A maximum of 4 blocks with cache_control may be provided."
MAX_CACHE_CONTROL_BLOCKS = 4


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
        message_points: List[CacheControlMessageInjectionPoint] = []
        remaining_points: List[CacheControlInjectionPoint] = []
        for point in injection_points:
            if point.get("location") == "message":
                message_points.append(cast(CacheControlMessageInjectionPoint, point))
            else:
                remaining_points.append(point)

        # Non-message points (currently Bedrock tool_config) are handled in the
        # provider transform, where each tool_config point appends at most one
        # cachePoint to the tools. That block also counts toward Anthropic's
        # limit, so reserve a slot for it here to leave room.
        reserved_blocks = (
            1
            if any(p.get("location") == "tool_config" for p in remaining_points)
            else 0
        )

        processed_messages = self._apply_message_injections(
            points=message_points,
            messages=processed_messages,
            max_blocks=MAX_CACHE_CONTROL_BLOCKS - reserved_blocks,
        )

        # Pass through non-message injection points for provider-specific handling
        if remaining_points:
            non_default_params["cache_control_injection_points"] = remaining_points

        return model, processed_messages, non_default_params

    @staticmethod
    def _apply_message_injections(
        points: List[CacheControlMessageInjectionPoint],
        messages: List[AllMessageValues],
        max_blocks: int,
    ) -> List[AllMessageValues]:
        """Apply message-level cache control injection points in order.

        Anthropic allows at most ``MAX_CACHE_CONTROL_BLOCKS`` cache_control
        breakpoints per request. Client-supplied breakpoints count toward that
        limit, so we never inject onto a message that already carries
        cache_control (preserving the client's TTL) and we stop injecting once
        ``max_blocks`` is reached. Injection points are honored in config order,
        so earlier points win when slots are scarce.
        """
        used_blocks = sum(
            AnthropicCacheControlHook._count_cache_control_blocks(msg)
            for msg in messages
        )

        limit_reached = False
        for point in points:
            if used_blocks >= max_blocks:
                limit_reached = True
                break

            control: ChatCompletionCachedContent = point.get(
                "control", None
            ) or ChatCompletionCachedContent(type="ephemeral")

            for target_index in AnthropicCacheControlHook._resolve_target_indices(
                point=point, messages=messages
            ):
                if used_blocks >= max_blocks:
                    limit_reached = True
                    break

                if AnthropicCacheControlHook._message_has_cache_control(
                    messages[target_index]
                ):
                    # Client already marked this message; don't overwrite it.
                    continue

                messages[target_index] = (
                    AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                        messages[target_index], control
                    )
                )
                used_blocks += 1

            if limit_reached:
                break

        if limit_reached:
            verbose_logger.warning(
                f"AnthropicCacheControlHook: Reached the Anthropic limit of "
                f"{MAX_CACHE_CONTROL_BLOCKS} cache_control blocks. Skipping further injection."
            )

        return messages

    @staticmethod
    def _resolve_target_indices(
        point: CacheControlMessageInjectionPoint, messages: List[AllMessageValues]
    ) -> List[int]:
        """Resolve which message indices an injection point targets."""
        _targetted_index: Optional[Union[int, str]] = point.get("index", None)
        targetted_index: Optional[int] = None
        if isinstance(_targetted_index, str):
            try:
                targetted_index = int(_targetted_index)
            except ValueError:
                pass
        else:
            targetted_index = _targetted_index

        # Case 1: Target by specific index
        if targetted_index is not None:
            original_index = targetted_index
            if targetted_index < 0:
                targetted_index += len(messages)

            if 0 <= targetted_index < len(messages):
                return [targetted_index]

            verbose_logger.warning(
                f"AnthropicCacheControlHook: Provided index {original_index} is out of bounds for message list of length {len(messages)}. "
                f"Targeted index was {targetted_index}. Skipping cache control injection for this point."
            )
            return []

        # Case 2: Target by role
        targetted_role = point.get("role", None)
        if targetted_role is not None:
            return [
                idx
                for idx, msg in enumerate(messages)
                if msg.get("role") == targetted_role
            ]

        return []

    @staticmethod
    def _count_cache_control_blocks(message: AllMessageValues) -> int:
        """Count cache_control breakpoints on a message (message + content level)."""
        count = 0
        if message.get("cache_control") is not None:
            count += 1
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control") is not None:
                    count += 1
        return count

    @staticmethod
    def _message_has_cache_control(message: AllMessageValues) -> bool:
        """Return True if the message already carries any cache_control."""
        return AnthropicCacheControlHook._count_cache_control_blocks(message) > 0

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
