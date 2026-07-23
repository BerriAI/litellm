"""
This hook is used to inject cache control directives into messages.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

Supported for both `v1/chat/completions` (via the prompt-management hook) and
`v1/messages` (via `apply_to_anthropic_messages_request`).

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
        reserved_blocks = 1 if any(p.get("location") == "tool_config" for p in remaining_points) else 0

        processed_messages = self._apply_message_injections(
            points=message_points,
            messages=processed_messages,
            max_blocks=MAX_CACHE_CONTROL_BLOCKS - reserved_blocks,
        )

        # Pass through non-message injection points for provider-specific handling
        if remaining_points:
            non_default_params["cache_control_injection_points"] = AnthropicCacheControlHook._stamped_as_judged(
                remaining_points
            )

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
        used_blocks = sum(AnthropicCacheControlHook._count_cache_control_blocks(msg) for msg in messages)

        limit_reached = False
        for point in points:
            if used_blocks >= max_blocks:
                limit_reached = True
                break

            control: ChatCompletionCachedContent = point.get("control", None) or ChatCompletionCachedContent(
                type="ephemeral"
            )

            for target_index in AnthropicCacheControlHook._resolve_target_indices(point=point, messages=messages):
                if used_blocks >= max_blocks:
                    limit_reached = True
                    break

                if AnthropicCacheControlHook._message_has_cache_control(messages[target_index]):
                    # Client already marked this message; don't overwrite it.
                    continue

                messages[target_index] = AnthropicCacheControlHook._safe_insert_cache_control_in_message(
                    messages[target_index], control
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
            return [idx for idx, msg in enumerate(messages) if msg.get("role") == targetted_role]

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
        messages: List[Dict],
        system: str | list | None,
        injection_points: List[CacheControlInjectionPoint],
    ) -> Tuple[List[Dict], str | list | None, List[CacheControlInjectionPoint]]:
        """Apply cache control injection for the Anthropic-native v1/messages endpoint.

        Returns (messages, system, remaining_non_message_points).
        """
        if not injection_points:
            return messages, system, []

        processed_messages: List[Dict] = copy.deepcopy(messages)
        processed_system = copy.deepcopy(system) if system is not None else None

        message_points: List[CacheControlMessageInjectionPoint] = []
        system_points: List[CacheControlMessageInjectionPoint] = []
        remaining_points: List[CacheControlInjectionPoint] = []

        for point in injection_points:
            if point.get("location") == "message":
                msg_point = cast(CacheControlMessageInjectionPoint, point)
                if msg_point.get("role") == "system":
                    system_points.append(msg_point)
                else:
                    message_points.append(msg_point)
            else:
                remaining_points.append(point)

        reserved_blocks = 1 if any(p.get("location") == "tool_config" for p in remaining_points) else 0
        max_blocks = MAX_CACHE_CONTROL_BLOCKS - reserved_blocks

        used_blocks = sum(
            AnthropicCacheControlHook._count_cache_control_blocks(cast(AllMessageValues, msg))
            for msg in processed_messages
        )
        if isinstance(processed_system, list):
            used_blocks += sum(
                1 for b in processed_system if isinstance(b, dict) and b.get("cache_control") is not None
            )

        if system_points and processed_system is not None and used_blocks < max_blocks:
            system_already_has_cc = isinstance(processed_system, list) and any(
                isinstance(b, dict) and b.get("cache_control") is not None for b in processed_system
            )
            if not system_already_has_cc:
                control = system_points[0].get("control") or ChatCompletionCachedContent(type="ephemeral")
                if isinstance(processed_system, str):
                    processed_system = [{"type": "text", "text": processed_system, "cache_control": control}]
                    used_blocks += 1
                elif len(processed_system) > 0 and isinstance(processed_system[-1], dict):
                    processed_system[-1] = {**processed_system[-1], "cache_control": control}
                    used_blocks += 1

        for i, msg in enumerate(processed_messages):
            content = msg.get("content")
            if isinstance(content, str):
                processed_messages[i] = {**msg, "content": [{"type": "text", "text": content}]}

        processed_messages = AnthropicCacheControlHook._apply_message_injections(
            points=message_points,
            messages=cast(List[AllMessageValues], processed_messages),
            max_blocks=max_blocks - used_blocks,
        )

        return processed_messages, processed_system, remaining_points

    @staticmethod
    def _default_control() -> ChatCompletionCachedContent:
        """Build the cache_control block for auto-injected breakpoints.

        Defaults to Anthropic's 5-minute ephemeral cache; honors the optional
        ``litellm.anthropic_prompt_caching_ttl`` override ("5m" or "1h").
        """
        import litellm

        ttl = litellm.anthropic_prompt_caching_ttl
        if ttl == "5m" or ttl == "1h":
            return ChatCompletionCachedContent(type="ephemeral", ttl=ttl)
        return ChatCompletionCachedContent(type="ephemeral")

    @staticmethod
    def _stamped_as_judged(points: list[CacheControlInjectionPoint]) -> list[dict[str, object]]:
        """Mark written-back points as having passed the client cache_control judgment.

        Builds copies because config-owned point dicts are shared across
        requests; mutating them would leak the stamp into future requests.
        """
        return [{**point, "_litellm_judged": True} for point in points]

    @staticmethod
    def _should_stand_down(
        points: list[CacheControlInjectionPoint],
        messages: list[AllMessageValues],
        system: str | list | None,
        tools: list | None,
    ) -> bool:
        """Whether configured injection points must yield to client-set cache_control.

        Points that a prior pass over this request already judged and wrote
        back carry the internal judged stamp; any re-entry (acompletion
        re-entering completion, the async-to-sync /v1/messages dispatch,
        interceptor sub-calls reusing the request kwargs) must not re-judge
        them, because by then the messages carry litellm's own injected marks
        and the judgment would misread those as client breakpoints.
        """
        if all(point.get("_litellm_judged") for point in points):
            return False
        return AnthropicCacheControlHook._request_has_cache_control(messages, system, tools)

    @staticmethod
    def _request_has_cache_control(
        messages: list[AllMessageValues],
        system: str | list | None,
        tools: list | None = None,
    ) -> bool:
        """Return True if the request already carries any client-supplied cache_control.

        When the client (e.g. Claude Code) already marks its own breakpoints we
        stand down entirely rather than add more, per the auto-caching contract.
        Tools count: they are a breakpoint the client can mark, they count toward
        the provider's four-block limit, and caching only the tool definitions is
        a common pattern, so injecting alongside them can exceed the cap. Tools
        carry the mark either at the top level (Anthropic shape) or nested under
        ``function`` (OpenAI shape); the Anthropic chat transform accepts both.
        """
        if any(AnthropicCacheControlHook._count_cache_control_blocks(msg) for msg in messages):
            return True
        if isinstance(system, list):
            if any(isinstance(block, dict) and block.get("cache_control") is not None for block in system):
                return True
        if tools is not None:
            return any(
                isinstance(tool, dict)
                and (
                    tool.get("cache_control") is not None
                    or (isinstance(tool.get("function"), dict) and tool["function"].get("cache_control") is not None)
                )
                for tool in tools
            )
        return False

    @staticmethod
    def get_default_injection_points(
        messages: list[AllMessageValues],
        system: str | list | None,
        model: str,
        custom_llm_provider: str | None,
        tools: list | None = None,
    ) -> list[CacheControlInjectionPoint]:
        """Default breakpoints when ``litellm.enable_anthropic_prompt_caching`` is on.

        Caches the system prompt and the trailing turn, so the stable prefix
        (system + tools + history) is reused while the breakpoint advances with
        the conversation. Returns [] (stand down) when the flag is off, the
        provider does not consume cache_control breakpoints (only anthropic /
        bedrock do), the model lacks prompt-caching support, or the request
        already carries client-supplied cache_control.
        """
        import litellm

        if litellm.enable_anthropic_prompt_caching is not True:
            return []

        provider = custom_llm_provider
        if provider is None:
            from litellm.litellm_core_utils.get_llm_provider_logic import (
                get_llm_provider,
            )

            try:
                _, provider, _, _ = get_llm_provider(model=model)
            except Exception:  # noqa: BLE001  # unroutable model must never block the call, just skip auto-caching
                return []

        if provider not in ("anthropic", "bedrock"):
            return []

        from litellm.utils import supports_prompt_caching

        if not supports_prompt_caching(model=model, custom_llm_provider=provider):
            return []

        if AnthropicCacheControlHook._request_has_cache_control(messages, system, tools):
            return []

        control = AnthropicCacheControlHook._default_control()
        points: list[CacheControlInjectionPoint] = [
            CacheControlMessageInjectionPoint(location="message", role="system", index=None, control=control),
            CacheControlMessageInjectionPoint(location="message", role=None, index=-1, control=control),
        ]
        return points

    @staticmethod
    def maybe_seed_default_injection_points(
        non_default_params: dict[str, Any],
        messages: list[AllMessageValues],
        model: str,
        custom_llm_provider: str | None,
        tools: list | None = None,
    ) -> None:
        """For /chat/completions: resolve the injection points the request should carry.

        Configured injection points win over the automatic defaults, but stand
        down entirely when the client already marked its own cache_control
        breakpoints (messages or tools): injecting alongside them clashes with
        the client's caching strategy and can exceed the provider's four-block
        limit. The judgment happens once per request; points a prior pass
        wrote back carry the judged stamp and are never re-judged (see
        ``_should_stand_down``). Seeding the param lets the existing
        prompt-management gate and the AnthropicCacheControlHook run
        unchanged.
        """
        if non_default_params.get("cache_control_injection_points"):
            if AnthropicCacheControlHook._should_stand_down(
                non_default_params["cache_control_injection_points"], messages, None, tools
            ):
                non_default_params.pop("cache_control_injection_points")
            return
        points = AnthropicCacheControlHook.get_default_injection_points(
            messages=messages,
            system=None,
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
        )
        if points:
            non_default_params["cache_control_injection_points"] = points

    @staticmethod
    def maybe_inject_cache_control(
        messages: List[Dict],
        system: str | list | None,
        kwargs: Dict[str, Any],
        model: str | None = None,
        custom_llm_provider: str | None = None,
        tools: list[dict] | None = None,
    ) -> Tuple[List[Dict], str | list | None]:
        """Extract cache_control_injection_points from kwargs and apply if present.

        Configured points stand down entirely when the client already marked
        its own cache_control breakpoints anywhere in the request. The
        judgment happens once per request; points a prior pass wrote back
        carry the judged stamp and are never re-judged (see
        ``_should_stand_down``). When none are configured but
        ``litellm.enable_anthropic_prompt_caching`` is on, synthesize default
        breakpoints for the native /v1/messages path. Pops the key from kwargs;
        if remaining (non-message) points exist they are written back so
        downstream transforms can handle them.
        """
        typed_messages = cast(list[AllMessageValues], messages)  # cast-ok: Anthropic-shaped dicts from v1/messages
        configured = cast(  # cast-ok: kwargs is untyped; this key only holds the documented injection-point list
            list[CacheControlInjectionPoint] | None, kwargs.pop("cache_control_injection_points", None)
        )
        if configured and AnthropicCacheControlHook._should_stand_down(configured, typed_messages, system, tools):
            return messages, system
        injection_points: list[CacheControlInjectionPoint] = configured or []
        if not injection_points and model is not None:
            injection_points = AnthropicCacheControlHook.get_default_injection_points(
                messages=typed_messages,
                system=system,
                tools=tools,
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        if not injection_points:
            return messages, system

        messages, system, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
        )
        if remaining:
            kwargs["cache_control_injection_points"] = AnthropicCacheControlHook._stamped_as_judged(remaining)
        return messages, system

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

        if AnthropicCacheControlHook.should_use_anthropic_cache_control_hook(non_default_params):
            return _init_custom_logger_compatible_class(
                logging_integration="anthropic_cache_control_hook",
                internal_usage_cache=None,
                llm_router=None,
            )
        return None
