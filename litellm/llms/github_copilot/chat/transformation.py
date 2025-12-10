from typing import Any, Dict, Optional, Tuple, cast, List, Union

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import GetAPIKeyError, GITHUB_COPILOT_API_BASE


class GithubCopilotConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = (
            self.authenticator.get_api_base() or GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm

        disable_copilot_system_to_assistant = (
            litellm.disable_copilot_system_to_assistant
        )
        if not disable_copilot_system_to_assistant:
            for message in messages:
                if "role" in message and message["role"] == "system":
                    cast(Any, message)["role"] = "assistant"

        # Consolidate duplicate tool results for Claude models on GitHub Copilot
        # GitHub Copilot's Claude API expects each tool_use to have exactly one tool_result
        messages = self._consolidate_tool_results(messages)

        return messages

    def _consolidate_tool_results(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """
        Consolidate multiple tool results with the same tool_call_id/tool_use_id into a single result.

        GitHub Copilot's Claude API expects each tool_use to have exactly one tool_result.
        When multiple tool_result blocks have the same ID, their content is merged into a single block.

        Handles both:
        - OpenAI format: multiple 'tool' role messages with the same tool_call_id
        - Anthropic format: content arrays with multiple tool_result blocks with the same tool_use_id
        """
        # First, consolidate OpenAI-style tool messages
        messages = self._consolidate_openai_tool_messages(messages)

        # Then, consolidate Anthropic-style tool_result blocks within message content
        messages = self._consolidate_anthropic_tool_results(messages)

        return messages

    def _consolidate_openai_tool_messages(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """
        Consolidate multiple OpenAI-style 'tool' role messages with the same tool_call_id.

        When multiple tool messages have the same tool_call_id, their content is merged
        into a single tool message.
        """
        # Find all tool messages and group by tool_call_id
        tool_messages_by_id: Dict[str, List[Dict[str, Any]]] = {}
        non_tool_messages: List[Tuple[int, AllMessageValues]] = []
        tool_message_first_indices: Dict[str, int] = {}

        for idx, message in enumerate(messages):
            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id:
                    if tool_call_id not in tool_messages_by_id:
                        tool_messages_by_id[tool_call_id] = []
                        tool_message_first_indices[tool_call_id] = idx
                    tool_messages_by_id[tool_call_id].append(message)  # type: ignore
                else:
                    non_tool_messages.append((idx, message))
            else:
                non_tool_messages.append((idx, message))

        # If no tool messages or no duplicates found, return original messages
        if not tool_messages_by_id or all(len(msgs) == 1 for msgs in tool_messages_by_id.values()):
            return messages

        # Build new message list with consolidated tool messages
        consolidated_messages: List[AllMessageValues] = []
        processed_tool_ids: set = set()

        for idx, message in enumerate(messages):
            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id and tool_call_id not in processed_tool_ids:
                    # Consolidate all messages with this tool_call_id
                    tool_msgs = tool_messages_by_id[tool_call_id]
                    if len(tool_msgs) == 1:
                        consolidated_messages.append(tool_msgs[0])  # type: ignore
                    else:
                        consolidated_msg = self._merge_tool_messages(tool_msgs)
                        consolidated_messages.append(consolidated_msg)  # type: ignore
                    processed_tool_ids.add(tool_call_id)
                elif not tool_call_id:
                    consolidated_messages.append(message)
            else:
                consolidated_messages.append(message)

        return consolidated_messages

    def _merge_tool_messages(
        self,
        tool_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge multiple tool messages with the same tool_call_id into a single message.

        The content from all messages is concatenated with newlines.
        """
        if not tool_messages:
            raise ValueError("Cannot merge empty list of tool messages")

        first_msg = tool_messages[0]
        merged_content_parts: List[str] = []

        for msg in tool_messages:
            content = msg.get("content")
            if isinstance(content, str):
                merged_content_parts.append(content)
            elif isinstance(content, list):
                # Handle list content (extract text parts)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        merged_content_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        merged_content_parts.append(item)

        merged_msg: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": first_msg.get("tool_call_id"),
            "content": "\n".join(merged_content_parts),
        }

        # Preserve name if present
        if "name" in first_msg:
            merged_msg["name"] = first_msg["name"]

        return merged_msg

    def _consolidate_anthropic_tool_results(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """
        Consolidate multiple Anthropic-style tool_result blocks with the same tool_use_id
        within message content arrays.

        When multiple tool_result blocks have the same tool_use_id, their content is merged
        into a single tool_result block.
        """
        result_messages: List[AllMessageValues] = []

        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Check if this message has tool_result blocks that need consolidation
                consolidated_content = self._consolidate_content_tool_results(content)
                if consolidated_content is not content:
                    # Content was modified, create new message
                    new_message = dict(message)
                    new_message["content"] = consolidated_content
                    result_messages.append(new_message)  # type: ignore
                else:
                    result_messages.append(message)
            else:
                result_messages.append(message)

        return result_messages

    def _consolidate_content_tool_results(
        self,
        content: List[Any],
    ) -> List[Any]:
        """
        Consolidate tool_result blocks in a content array by tool_use_id.

        Returns the original content list if no consolidation is needed,
        or a new list with consolidated tool_results.
        """
        # Group tool_results by tool_use_id
        tool_results_by_id: Dict[str, List[Dict[str, Any]]] = {}
        tool_result_first_indices: Dict[str, int] = {}
        other_content: List[Tuple[int, Any]] = []

        for idx, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_use_id = item.get("tool_use_id")
                if tool_use_id:
                    if tool_use_id not in tool_results_by_id:
                        tool_results_by_id[tool_use_id] = []
                        tool_result_first_indices[tool_use_id] = idx
                    tool_results_by_id[tool_use_id].append(item)
                else:
                    other_content.append((idx, item))
            else:
                other_content.append((idx, item))

        # If no tool_results or no duplicates, return original content
        if not tool_results_by_id or all(len(results) == 1 for results in tool_results_by_id.values()):
            return content

        # Build new content list with consolidated tool_results
        consolidated_content: List[Any] = []
        processed_tool_ids: set = set()

        for idx, item in enumerate(content):
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_use_id = item.get("tool_use_id")
                if tool_use_id and tool_use_id not in processed_tool_ids:
                    # Consolidate all tool_results with this tool_use_id
                    tool_results = tool_results_by_id[tool_use_id]
                    if len(tool_results) == 1:
                        consolidated_content.append(tool_results[0])
                    else:
                        consolidated_result = self._merge_tool_results(tool_results)
                        consolidated_content.append(consolidated_result)
                    processed_tool_ids.add(tool_use_id)
                elif not tool_use_id:
                    consolidated_content.append(item)
            else:
                consolidated_content.append(item)

        return consolidated_content

    def _merge_tool_results(
        self,
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Merge multiple tool_result blocks with the same tool_use_id into a single block.

        The content from all blocks is merged:
        - String content is concatenated with newlines
        - List content is flattened into a single list
        """
        if not tool_results:
            raise ValueError("Cannot merge empty list of tool results")

        first_result = tool_results[0]
        merged_content: Union[str, List[Any]]

        # Collect all content
        all_string_content: List[str] = []
        all_list_content: List[Any] = []
        has_string_content = False
        has_list_content = False

        for result in tool_results:
            result_content = result.get("content")
            if isinstance(result_content, str):
                all_string_content.append(result_content)
                has_string_content = True
            elif isinstance(result_content, list):
                all_list_content.extend(result_content)
                has_list_content = True

        # Determine merged content format
        if has_list_content and not has_string_content:
            # All list content
            merged_content = all_list_content
        elif has_string_content and not has_list_content:
            # All string content
            merged_content = "\n".join(all_string_content)
        else:
            # Mixed content - convert strings to text blocks and combine
            merged_list: List[Any] = []
            for s in all_string_content:
                merged_list.append({"type": "text", "text": s})
            merged_list.extend(all_list_content)
            merged_content = merged_list

        merged_result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": first_result.get("tool_use_id"),
            "content": merged_content,
        }

        # Preserve is_error if any result has it set to True
        if any(r.get("is_error") for r in tool_results):
            merged_result["is_error"] = True

        # Preserve cache_control from first result if present
        if "cache_control" in first_result:
            merged_result["cache_control"] = first_result["cache_control"]

        return merged_result

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Get base headers from parent
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        validated_headers["X-Initiator"] = initiator

        # Add Copilot-Vision-Request header if request contains images
        if self._has_vision_content(messages):
            validated_headers["Copilot-Vision-Request"] = "true"

        return validated_headers

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for GitHub Copilot.

        For Claude models that support extended thinking (Claude 4 family and Claude 3-7), includes thinking and reasoning_effort parameters.
        For other models, returns standard OpenAI parameters (which may include reasoning_effort for o-series models).
        """
        from litellm.utils import supports_reasoning
        
        # Get base OpenAI parameters
        base_params = super().get_supported_openai_params(model)

        # Add Claude-specific parameters for models that support extended thinking
        if "claude" in model.lower() and supports_reasoning(
            model=model.lower(),
        ):
            if "thinking" not in base_params:
                base_params.append("thinking")
            # reasoning_effort is not included by parent for Claude models, so add it
            if "reasoning_effort" not in base_params:
                base_params.append("reasoning_effort")

        return base_params

    def _determine_initiator(self, messages: List[AllMessageValues]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            role = message.get("role")
            if role in ["tool", "assistant"]:
                return "agent"
        return "user"

    def _has_vision_content(self, messages: List[AllMessageValues]) -> bool:
        """
        Check if any message contains vision content (images).
        Returns True if any message has content with vision-related types, otherwise False.
        
        Checks for:
        - image_url content type (OpenAI format)
        - Content items with type 'image_url'
        """
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                # Check if any content item indicates vision content
                for content_item in content:
                    if isinstance(content_item, dict):
                        # Check for image_url field (direct image URL)
                        if "image_url" in content_item:
                            return True
                        # Check for type field indicating image content
                        content_type = content_item.get("type")
                        if content_type == "image_url":
                            return True
        return False
