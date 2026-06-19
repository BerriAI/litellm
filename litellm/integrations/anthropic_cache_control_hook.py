"""
This hook is used to inject cache control directives into the messages of a chat completion.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

"""

import copy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_NON_CACHEABLE_TOOL_TYPES
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

# Re-export for any existing internal callers while the canonical definition
# lives in litellm.constants (shared with transformation.py).
NON_CACHEABLE_TOOL_TYPES = ANTHROPIC_NON_CACHEABLE_TOOL_TYPES


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

    @staticmethod
    def apply_to_anthropic_messages_request(
        messages: list[dict],
        system: str | list[dict] | None,
        tools: list[dict] | None,
        non_default_params: dict,
    ) -> tuple[list[dict], str | list[dict] | None, list[dict] | None]:
        """Apply ``cache_control_injection_points`` to a native Anthropic
        ``/v1/messages`` request.

        The OpenAI chat/completions path injects cache control via
        ``get_chat_completion_prompt`` (message-level ``cache_control``). The
        native Anthropic Messages endpoint differs in two ways that the OpenAI
        helper does not handle, so this method exists alongside it:

        1. The system prompt is a top-level ``system`` parameter, not a
           ``role: system`` entry in ``messages``.
        2. Anthropic rejects message-level ``cache_control``; the marker must
           live on a content *block* (or directly on a ``system`` / tool block).

        Injection is capped at ``MAX_CACHE_CONTROL_BLOCKS`` markers shared across
        system, tools, and messages (client-supplied markers count toward the cap
        and are never overwritten), so the outbound payload never exceeds the
        Anthropic/Bedrock limit.

        Args:
            messages: Anthropic-format messages (each ``content`` is a string or
                a list of content blocks).
            system: Top-level system prompt (string or list of system blocks).
            tools: Tool definitions sent to the model.
            non_default_params: Request params; ``cache_control_injection_points``
                is read and popped from here so it is not forwarded upstream as
                an unknown field. Any injection points this method does not
                handle (e.g. Bedrock ``tool_config``) are written back so
                downstream provider transforms still receive them.

        Returns:
            A tuple of (messages, system, tools). When there are no injection
            points, the original references are returned as-is (no copy is
            made). When injection occurs, the inputs are deep-copied before
            modification so the caller's originals are not mutated.
        """
        injection_points: list[CacheControlInjectionPoint] = non_default_params.pop(
            "cache_control_injection_points", []
        )
        if not injection_points:
            return messages, system, tools

        processed_messages = copy.deepcopy(messages)
        processed_system = copy.deepcopy(system)
        processed_tools = copy.deepcopy(tools)

        # Injection points this native path cannot represent in the payload
        # (e.g. Bedrock `tool_config`). Forwarded rather than dropped so the
        # param is never silently swallowed. Note these are inert on the native
        # `/v1/messages` path: it uses the Anthropic Invoke transform, while only
        # the Converse transform on the chat/completions path consumes
        # `tool_config`. No block budget is reserved for them here (unlike
        # get_chat_completion_prompt, which reserves a slot because Converse does
        # append a tool_config cachePoint downstream).
        remaining_points: list[CacheControlInjectionPoint] = []

        # Anthropic/Bedrock reject requests with more than MAX_CACHE_CONTROL_BLOCKS
        # cache_control breakpoints. Client-supplied markers count toward that
        # limit, so inject in config order, never overwrite a client's existing
        # marker, and stop once the budget is exhausted.
        max_blocks = MAX_CACHE_CONTROL_BLOCKS
        used_blocks = AnthropicCacheControlHook._count_request_cache_control_blocks(
            system=processed_system, tools=processed_tools, messages=processed_messages
        )
        limit_reached = False

        for point in injection_points:
            location = point.get("location")

            if location not in ("system", "tools", "message"):
                # Unhandled location (tool_config / future types): forward it.
                remaining_points.append(point)
                continue

            if used_blocks >= max_blocks:
                # Out of payload-level budget; keep scanning so any remaining
                # tool_config points are still forwarded downstream.
                limit_reached = True
                continue

            control = AnthropicCacheControlHook._control_from_point(point)

            if location == "system":
                processed_system, added = (
                    AnthropicCacheControlHook._insert_cache_control_in_system(
                        system=processed_system, control=control
                    )
                )
                used_blocks += int(added)
            elif location == "tools":
                processed_tools, added = (
                    AnthropicCacheControlHook._insert_cache_control_in_tools(
                        tools=processed_tools, control=control
                    )
                )
                used_blocks += int(added)
            else:  # message
                message_point = cast(CacheControlMessageInjectionPoint, point)
                # `role: system` has no message-level analogue on /v1/messages
                # (the system prompt is the top-level `system` param), so
                # redirect it to the system prompt when one exists. This keeps
                # the same YAML config working across both endpoints.
                if (
                    message_point.get("role") == "system"
                    and message_point.get("index") is None
                    and not any(m.get("role") == "system" for m in processed_messages)
                    and processed_system is not None
                ):
                    processed_system, added = (
                        AnthropicCacheControlHook._insert_cache_control_in_system(
                            system=processed_system, control=control
                        )
                    )
                    used_blocks += int(added)
                else:
                    processed_messages, added_count, hit_limit = (
                        AnthropicCacheControlHook._process_anthropic_message_injection(
                            point=message_point,
                            messages=processed_messages,
                            used_blocks=used_blocks,
                            max_blocks=max_blocks,
                        )
                    )
                    used_blocks += added_count
                    limit_reached = limit_reached or hit_limit

        if limit_reached:
            verbose_logger.warning(
                f"AnthropicCacheControlHook: Reached the Anthropic limit of "
                f"{MAX_CACHE_CONTROL_BLOCKS} cache_control blocks on the "
                f"/v1/messages path. Skipping further injection."
            )

        if remaining_points:
            non_default_params["cache_control_injection_points"] = remaining_points

        return processed_messages, processed_system, processed_tools

    @staticmethod
    def _control_from_point(
        point: CacheControlInjectionPoint,
    ) -> ChatCompletionCachedContent:
        """Resolve the ``cache_control`` value for an injection point.

        Defaults to ``{"type": "ephemeral"}`` when no explicit control is given.
        ``control`` is not declared on every member of the
        ``CacheControlInjectionPoint`` union (e.g. ``tool_config`` has none), so
        it is read via a structural dict view rather than a misleading
        ``Optional`` cast on a value that is immediately defaulted to non-None.
        """
        control = cast(dict[str, Any], point).get("control")
        return control or ChatCompletionCachedContent(type="ephemeral")

    @staticmethod
    def _count_request_cache_control_blocks(
        system: str | list[dict] | None,
        tools: list[dict] | None,
        messages: list[dict],
    ) -> int:
        """Count cache_control markers already present across a /v1/messages
        payload (system blocks, tool definitions, and message content blocks).

        Client-supplied markers count toward Anthropic's limit, so they must be
        tallied before injecting more. A string ``system`` carries no marker.
        """
        count = 0
        if isinstance(system, list):
            count += sum(
                1
                for block in system
                if isinstance(block, dict) and block.get("cache_control") is not None
            )
        if isinstance(tools, list):
            count += sum(
                1
                for tool in tools
                if isinstance(tool, dict) and tool.get("cache_control") is not None
            )
        for message in messages:
            count += AnthropicCacheControlHook._count_cache_control_blocks(
                cast(AllMessageValues, message)
            )
            # `_count_cache_control_blocks` only sees message-level and top-level
            # content-block markers. Anthropic `tool_result` blocks nest their
            # own content list, whose items may carry their own cache_control;
            # count those too so the cap reflects the real marker total and we
            # never push a near-full request over the limit.
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    nested = block.get("content")
                    if isinstance(nested, list):
                        count += sum(
                            1
                            for item in nested
                            if isinstance(item, dict)
                            and item.get("cache_control") is not None
                        )
        return count

    @staticmethod
    def _insert_cache_control_in_system(
        system: str | list[dict] | None,
        control: ChatCompletionCachedContent,
    ) -> tuple[str | list[dict] | None, bool]:
        """Insert cache control on the last block of an Anthropic system prompt.

        A string system prompt is promoted to a single-block list so the marker
        can live at block level (Anthropic rejects a bare string + cache marker).
        No-op when there is no system prompt or the last block already carries a
        marker (a client's marker is preserved).

        Returns ``(system, added)`` where ``added`` is True iff a new marker was
        written, so the caller can track the cache_control block budget.
        """
        if system is None:
            return system, False
        if isinstance(system, str):
            if system == "":
                return system, False
            return [{"type": "text", "text": system, "cache_control": control}], True
        if isinstance(system, list) and len(system) > 0:
            last_block = system[-1]
            if isinstance(last_block, dict) and last_block.get("cache_control") is None:
                last_block["cache_control"] = control
                return system, True
        return system, False

    @staticmethod
    def _insert_cache_control_in_tools(
        tools: list[dict] | None,
        control: ChatCompletionCachedContent,
    ) -> tuple[list[dict] | None, bool]:
        """Insert cache control on the last *cacheable* tool definition.

        Marking a tool caches the tool-list prefix up to and including it
        (Anthropic caches the prefix). Tool-search tools (see
        ``NON_CACHEABLE_TOOL_TYPES``) reject ``cache_control`` and are skipped,
        so the marker lands on the last tool that supports it — otherwise a valid
        request would 400. No-op when there are no cacheable tools or that tool
        already carries a marker.

        Returns ``(tools, added)`` where ``added`` is True iff a new marker was
        written, so the caller can track the cache_control block budget.
        """
        if not tools:
            return tools, False
        for tool in reversed(tools):
            if not isinstance(tool, dict):
                continue
            if tool.get("type") in NON_CACHEABLE_TOOL_TYPES:
                continue
            if tool.get("cache_control") is None:
                tool["cache_control"] = control
                return tools, True
            return tools, False  # last cacheable tool already marked
        return tools, False

    @staticmethod
    def _process_anthropic_message_injection(
        point: CacheControlMessageInjectionPoint,
        messages: list[dict],
        used_blocks: int,
        max_blocks: int,
    ) -> tuple[list[dict], int, bool]:
        """Apply block-level cache control to targeted messages within budget.

        Reuses :meth:`_resolve_target_indices` for index / role targeting
        (negative index support, out-of-bounds warning) but writes the marker on
        a content *block* rather than the message object, because Anthropic's
        ``/v1/messages`` endpoint rejects message-level ``cache_control``.
        Messages already carrying a marker are left untouched (client TTL
        preserved) and injection stops once the shared block budget is exhausted.

        Returns ``(messages, added, limit_reached)``.
        """
        control = AnthropicCacheControlHook._control_from_point(point)
        target_indices = AnthropicCacheControlHook._resolve_target_indices(
            point=point, messages=cast(list[AllMessageValues], messages)
        )

        added = 0
        limit_reached = False
        for target_index in target_indices:
            if used_blocks + added >= max_blocks:
                limit_reached = True
                break
            if AnthropicCacheControlHook._message_has_cache_control(
                cast(AllMessageValues, messages[target_index])
            ):
                # Client already marked this message; don't overwrite it.
                continue
            AnthropicCacheControlHook._insert_cache_control_in_message_block(
                messages[target_index], control
            )
            added += 1
        return messages, added, limit_reached

    @staticmethod
    def _insert_cache_control_in_message_block(
        message: dict, control: ChatCompletionCachedContent
    ) -> dict:
        """Insert cache control on the last content block of one message.

        A string ``content`` is promoted to a single text block so the marker
        can live at block level, as required by ``/v1/messages``.
        """
        message_content = message.get("content", None)
        if isinstance(message_content, str):
            message["content"] = [
                {"type": "text", "text": message_content, "cache_control": control}
            ]
        elif isinstance(message_content, list) and len(message_content) > 0:
            last_block = message_content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = control
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
