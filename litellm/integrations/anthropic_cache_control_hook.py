"""
This hook is used to inject cache control directives into the messages of a chat completion.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

"""

import copy
from typing import Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.integrations.anthropic_cache_control_hook import (
    CacheControlInjectionPoint,
    CacheControlMessageInjectionPoint,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionCachedContent
from litellm.types.utils import StandardCallbackDynamicParams


class AnthropicCacheControlHook(CustomPromptManagement):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: Optional[str] = None,
        prompt_version: Optional[int] = None,
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

        # Process message-level cache controls
        for point in injection_points:
            if point.get("location") == "message":
                point = cast(CacheControlMessageInjectionPoint, point)
                processed_messages = self._process_message_injection(
                    point=point, messages=processed_messages
                )

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
            if _targetted_index.isdigit():
                targetted_index = int(_targetted_index)
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
