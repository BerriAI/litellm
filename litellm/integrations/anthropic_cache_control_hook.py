"""
This hook is used to inject cache control directives into messages.

Users can define
- `cache_control_injection_points` in the completion params and litellm will inject the cache control directives into the messages at the specified injection points.

Supported for both `v1/chat/completions` (via the prompt-management hook) and
`v1/messages` (via `apply_to_anthropic_messages_request`).

"""

import copy
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    with_prompt_cache_breakpoint,
)
from litellm.types.integrations.anthropic_cache_control_hook import (
    CacheControlInjectionPoint,
    CacheControlMessageInjectionPoint,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionTextObject,
    ChatCompletionToolParam,
    PromptCacheBreakpoint,
    PromptCacheOptions,
)
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


# Anthropic (and Bedrock Claude) reject requests with more than 4 cache_control
# breakpoints: "A maximum of 4 blocks with cache_control may be provided."
MAX_CACHE_CONTROL_BLOCKS: Final = 4

CACHE_BREAKPOINT_KEYS: Final = ("cache_control", "prompt_cache_breakpoint")
OPENAI_PROMPT_CACHE_BREAKPOINT_MIN_GPT_VERSION: Final = (5, 6)
_GPT_VERSION_PATTERN: Final = re.compile(r"^gpt-(\d+)(?:\.(\d+))?")
OPENAI_PROMPT_CACHE_BREAKPOINT_BLOCK_TYPES: Final = frozenset(
    {"text", "image", "image_url", "file", "input_audio", "input_text", "input_image", "input_file"}
)
OPENAI_API_HOST: Final = "api.openai.com"
OPENAI_API_BASE_ENV_VARS: Final = ("OPENAI_BASE_URL", "OPENAI_API_BASE")

AllToolParamValues = ChatCompletionToolParam | AllAnthropicToolsValues


def supports_openai_prompt_cache_breakpoint(model: str) -> bool:
    model_map_flag: Final = _model_map_prompt_cache_breakpoint_flag(model)
    if model_map_flag is not None:
        return model_map_flag
    version_match: Final = _GPT_VERSION_PATTERN.match(model.rsplit("/", 1)[-1].lower())
    if version_match is None:
        return False
    version: Final = (int(version_match.group(1)), int(version_match.group(2) or 0))
    return version >= OPENAI_PROMPT_CACHE_BREAKPOINT_MIN_GPT_VERSION


def _model_map_prompt_cache_breakpoint_flag(model: str) -> bool | None:
    import litellm

    entries: Final = (litellm.model_cost.get(key) for key in (model, model.rsplit("/", 1)[-1]))
    flags: Final = (entry.get("supports_prompt_cache_breakpoint") for entry in entries if isinstance(entry, dict))
    return next((bool(flag) for flag in flags if flag is not None), None)


def targets_openai_api(api_base: object) -> bool:
    import litellm

    resolved: Final = next(
        (value for value in (api_base, litellm.api_base, *map(os.getenv, OPENAI_API_BASE_ENV_VARS)) if value),
        None,
    )
    if not isinstance(resolved, str):
        return True
    host: Final = urlparse(resolved).hostname
    return host is not None and (host == OPENAI_API_HOST or host.endswith(f".{OPENAI_API_HOST}"))


def _carries_cache_breakpoint(block: object) -> bool:
    return isinstance(block, dict) and any(block.get(key) is not None for key in CACHE_BREAKPOINT_KEYS)


def _accepts_prompt_cache_breakpoint(block: object) -> bool:
    return isinstance(block, dict) and block.get("type") in OPENAI_PROMPT_CACHE_BREAKPOINT_BLOCK_TYPES


# Set by a caller whose message list is not the one that goes upstream -- today the
# Responses API layer, whose `instructions` only becomes a system message further down.
# Tells this hook to hand role-targeted points to the pass holding the final messages
# rather than spending them on a list that is still missing some of their targets.
CARRY_UNMATCHED_MESSAGE_POINTS: Final = "_litellm_carry_unmatched_cache_control_points"


class AnthropicCacheControlHook(CustomPromptManagement):
    def get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        """
        Apply cache control directives based on specified injection points.

        Returns:
        - model: str - the model to use
        - messages: List[AllMessageValues] - messages with applied cache controls
        - non_default_params: dict - params with any global cache controls
        """
        # Extract cache control injection points
        carry_unmatched: Final = bool(non_default_params.pop(CARRY_UNMATCHED_MESSAGE_POINTS, False))
        injection_points: Final[list[CacheControlInjectionPoint]] = non_default_params.pop(
            "cache_control_injection_points", []
        )
        if not injection_points:
            return model, messages, non_default_params

        # Create a deep copy of messages to avoid modifying the original list
        processed_messages = copy.deepcopy(messages)

        # Separate message-level and non-message-level injection points
        message_points: Final[list[CacheControlMessageInjectionPoint]] = []
        remaining_points: Final[list[CacheControlInjectionPoint]] = []
        for point in injection_points:
            if point.get("location") == "message":
                message_points.append(cast(CacheControlMessageInjectionPoint, point))
            else:
                remaining_points.append(point)

        # Non-message points (currently Bedrock tool_config) are handled in the
        # provider transform, where each tool_config point appends at most one
        # cachePoint to the tools. That block also counts toward Anthropic's
        # limit, so reserve a slot for it here to leave room.
        stamped_dialect: Final = injection_points[0].get("_litellm_openai_dialect")
        openai_dialect: Final = (
            stamped_dialect
            if isinstance(stamped_dialect, bool)
            else AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
                model,
                non_default_params.get("custom_llm_provider"),
                non_default_params.get("api_base") or non_default_params.get("base_url"),
                non_default_params.get("prompt_cache_options"),
            )
        )
        # A provisional message list defers every role-targeted point to the pass holding
        # the final one: a role with no message here may have one there, and settling all
        # of them in one pass is what lets config order decide the shared breakpoint
        # budget. An ordinal names a different message once a later layer builds its own
        # list, so it is placed here or not at all.
        carried_message_points: Final[Sequence[CacheControlMessageInjectionPoint]] = (
            tuple(point for point in message_points if point.get("index") is None) if carry_unmatched else ()
        )
        applied_message_points: Final[Sequence[CacheControlMessageInjectionPoint]] = (
            tuple(point for point in message_points if point.get("index") is not None)
            if carry_unmatched
            else tuple(message_points)
        )
        reserved_blocks: Final = (
            1 if not openai_dialect and any(p.get("location") == "tool_config" for p in remaining_points) else 0
        )
        breakpoints_before: Final = AnthropicCacheControlHook._count_request_cache_breakpoints(processed_messages)
        processed_messages = self._apply_message_injections(
            points=applied_message_points,
            messages=processed_messages,
            max_blocks=MAX_CACHE_CONTROL_BLOCKS - reserved_blocks,
            openai_dialect=openai_dialect,
        )
        if (
            openai_dialect
            and AnthropicCacheControlHook._count_request_cache_breakpoints(processed_messages) > breakpoints_before
        ):
            non_default_params.setdefault("prompt_cache_options", PromptCacheOptions(mode="explicit"))

        # Points this pass did not place: non-message ones for the provider transform, and
        # the deferred role-targeted ones. Deferring is what reaches the Responses API's
        # `instructions`, which is only a system message once the bridge builds one. The
        # judged stamp is what makes it safe: the next pass must not re-judge points
        # against messages this pass already marked (see `_should_stand_down`).
        carried_points: Final[Sequence[CacheControlInjectionPoint]] = (*remaining_points, *carried_message_points)
        if carried_points:
            non_default_params["cache_control_injection_points"] = AnthropicCacheControlHook._stamped_as_judged(
                carried_points
            )

        return model, processed_messages, non_default_params

    @staticmethod
    def _targets_openai_prompt_cache_breakpoint(
        model: str | None,
        custom_llm_provider: str | None,
        api_base: object = None,
        prompt_cache_options: object = None,
    ) -> bool:
        if model is None or not supports_openai_prompt_cache_breakpoint(model):
            return False
        if (custom_llm_provider or AnthropicCacheControlHook._resolve_provider(model)) != "openai":
            return False
        return prompt_cache_options is not None or targets_openai_api(api_base)

    @staticmethod
    def _resolve_provider(model: str) -> str | None:
        from litellm.exceptions import BadRequestError
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        try:
            _, provider, _, _ = get_llm_provider(model=model)
        except BadRequestError:
            return None
        return provider

    @staticmethod
    def _count_request_cache_breakpoints(messages: Iterable[object], system: object = None) -> int:
        system_blocks: Final = (
            sum(1 for block in system if _carries_cache_breakpoint(block)) if isinstance(system, list) else 0
        )
        return system_blocks + sum(AnthropicCacheControlHook._count_cache_control_blocks(msg) for msg in messages)

    @staticmethod
    def _apply_message_injections(
        points: Sequence[CacheControlMessageInjectionPoint],
        messages: list[AllMessageValues],
        max_blocks: int,
        openai_dialect: bool = False,
    ) -> list[AllMessageValues]:
        """Apply message-level cache control injection points in order.

        Anthropic allows at most ``MAX_CACHE_CONTROL_BLOCKS`` cache_control
        breakpoints per request. Client-supplied breakpoints count toward that
        limit, so we never inject onto a message that already carries
        cache_control (preserving the client's TTL) and we stop injecting once
        ``max_blocks`` is reached. Injection points are honored in config order,
        so earlier points win when slots are scarce.
        """
        used_blocks = AnthropicCacheControlHook._count_request_cache_breakpoints(messages)

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
                    messages[target_index], control, openai_dialect
                )
                if AnthropicCacheControlHook._message_has_cache_control(messages[target_index]):
                    used_blocks += 1

            if limit_reached:
                break

        if limit_reached:
            verbose_logger.warning(
                "AnthropicCacheControlHook: Reached the provider limit of %s cache breakpoints. Skipping further injection.",
                MAX_CACHE_CONTROL_BLOCKS,
            )

        return messages

    @staticmethod
    def _resolve_target_indices(
        point: CacheControlMessageInjectionPoint, messages: list[AllMessageValues]
    ) -> list[int]:
        """Resolve which message indices an injection point targets."""
        _targetted_index: Final[int | str | None] = point.get("index", None)
        targetted_index: int | None = None
        if isinstance(_targetted_index, str):
            try:
                targetted_index = int(_targetted_index)
            except ValueError:
                pass
        else:
            targetted_index = _targetted_index

        # Case 1: Target by specific index
        if targetted_index is not None:
            original_index: Final = targetted_index
            if targetted_index < 0:
                targetted_index += len(messages)

            if 0 <= targetted_index < len(messages):
                return [targetted_index]

            verbose_logger.warning(
                "AnthropicCacheControlHook: Provided index %s is out of bounds for message list of length %s. Targeted index was %s. Skipping cache control injection for this point.",
                original_index,
                len(messages),
                targetted_index,
            )
            return []

        # Case 2: Target by role
        targetted_role: Final = point.get("role", None)
        if targetted_role is not None:
            return [idx for idx, msg in enumerate(messages) if msg.get("role") == targetted_role]

        return []

    @staticmethod
    def _count_cache_control_blocks(message: object) -> int:
        if not isinstance(message, dict):
            return 0
        count = 1 if _carries_cache_breakpoint(message) else 0
        content: Final = message.get("content")
        if isinstance(content, list):
            count += sum(1 for block in content if _carries_cache_breakpoint(block))
        return count

    @staticmethod
    def _message_has_cache_control(message: AllMessageValues) -> bool:
        """Return True if the message already carries any cache_control."""
        return AnthropicCacheControlHook._count_cache_control_blocks(message) > 0

    @staticmethod
    def _safe_insert_cache_control_in_message(
        message: AllMessageValues, control: ChatCompletionCachedContent, openai_dialect: bool = False
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
        if openai_dialect:
            return AnthropicCacheControlHook._insert_prompt_cache_breakpoint_in_message(message)

        message_content: Final = message.get("content", None)

        # 1. if string, insert cache control in the message
        if isinstance(message_content, str):
            message["cache_control"] = control
        # 2. list of objects - only apply to last item per Anthropic spec
        elif isinstance(message_content, list):
            if len(message_content) > 0 and isinstance(message_content[-1], dict):
                message_content[-1]["cache_control"] = control
        return message

    @staticmethod
    def _insert_prompt_cache_breakpoint_in_message(message: AllMessageValues) -> AllMessageValues:
        if message.get("role") == "assistant":
            return message
        message_content: Final = message.get("content", None)
        if isinstance(message_content, str):
            marked: Final = copy.copy(message)
            marked["content"] = [
                with_prompt_cache_breakpoint(
                    ChatCompletionTextObject(type="text", text=message_content), PromptCacheBreakpoint(mode="explicit")
                )
            ]
            return marked
        if isinstance(message_content, list):
            target_index: Final = next(
                (
                    index
                    for index in range(len(message_content) - 1, -1, -1)
                    if _accepts_prompt_cache_breakpoint(message_content[index])
                ),
                None,
            )
            if target_index is not None:
                message_content[target_index] = with_prompt_cache_breakpoint(
                    message_content[target_index], PromptCacheBreakpoint(mode="explicit")
                )
        return message

    @staticmethod
    def _system_block_with_breakpoint(
        block: Mapping[str, object], control: ChatCompletionCachedContent, openai_dialect: bool
    ) -> Mapping[str, object]:
        marker: Final = (
            ("prompt_cache_breakpoint", PromptCacheBreakpoint(mode="explicit"))
            if openai_dialect
            else ("cache_control", control)
        )
        return {**block, marker[0]: marker[1]}

    @staticmethod
    def apply_to_anthropic_messages_request(
        messages: list[dict],
        system: str | list | None,
        injection_points: list[CacheControlInjectionPoint],
        openai_dialect: bool = False,
    ) -> tuple[list[dict], str | list | None, list[CacheControlInjectionPoint]]:
        """Apply cache control injection for the Anthropic-native v1/messages endpoint.

        Returns (messages, system, remaining_non_message_points).
        """
        if not injection_points:
            return messages, system, []

        processed_messages: list[dict] = copy.deepcopy(messages)
        processed_system = copy.deepcopy(system) if system is not None else None

        message_points: Final[list[CacheControlMessageInjectionPoint]] = []
        system_points: Final[list[CacheControlMessageInjectionPoint]] = []
        remaining_points: Final[list[CacheControlInjectionPoint]] = []

        for point in injection_points:
            if point.get("location") == "message":
                msg_point = cast(CacheControlMessageInjectionPoint, point)
                if msg_point.get("role") == "system":
                    system_points.append(msg_point)
                else:
                    message_points.append(msg_point)
            else:
                remaining_points.append(point)

        reserved_blocks: Final = (
            1 if not openai_dialect and any(p.get("location") == "tool_config" for p in remaining_points) else 0
        )
        max_blocks: Final = MAX_CACHE_CONTROL_BLOCKS - reserved_blocks

        message_blocks: Final = AnthropicCacheControlHook._count_request_cache_breakpoints(processed_messages)
        system_blocks = AnthropicCacheControlHook._count_request_cache_breakpoints((), processed_system)

        if system_points and processed_system is not None and message_blocks + system_blocks < max_blocks:
            system_already_has_cc: Final = isinstance(processed_system, list) and any(
                _carries_cache_breakpoint(b) for b in processed_system
            )
            if not system_already_has_cc:
                control: Final = system_points[0].get("control") or ChatCompletionCachedContent(type="ephemeral")
                if isinstance(processed_system, str):
                    processed_system = [
                        AnthropicCacheControlHook._system_block_with_breakpoint(
                            AnthropicSystemMessageContent(type="text", text=processed_system), control, openai_dialect
                        )
                    ]
                    system_blocks += 1
                elif len(processed_system) > 0 and isinstance(processed_system[-1], dict):
                    processed_system[-1] = AnthropicCacheControlHook._system_block_with_breakpoint(
                        processed_system[-1], control, openai_dialect
                    )
                    system_blocks += 1

        for i, msg in enumerate(processed_messages):
            content = msg.get("content")
            if isinstance(content, str):
                processed_messages[i] = {**msg, "content": [{"type": "text", "text": content}]}

        processed_messages = AnthropicCacheControlHook._apply_message_injections(
            points=message_points,
            messages=cast(list[AllMessageValues], processed_messages),
            max_blocks=max_blocks - system_blocks,
            openai_dialect=openai_dialect,
        )

        return processed_messages, processed_system, remaining_points

    @staticmethod
    def _default_control() -> ChatCompletionCachedContent:
        """Build the cache_control block for auto-injected breakpoints.

        Defaults to Anthropic's 5-minute ephemeral cache; honors the optional
        ``litellm.anthropic_prompt_caching_ttl`` override ("5m" or "1h").
        """
        import litellm

        ttl: Final = litellm.anthropic_prompt_caching_ttl
        if ttl == "5m" or ttl == "1h":
            return ChatCompletionCachedContent(type="ephemeral", ttl=ttl)
        return ChatCompletionCachedContent(type="ephemeral")

    @staticmethod
    def _stamped_as_judged(points: Sequence[CacheControlInjectionPoint]) -> Sequence[Mapping[str, object]]:
        """Mark written-back points as having passed the client cache_control judgment.

        Builds copies because config-owned point dicts are shared across
        requests; mutating them would leak the stamp into future requests.
        """
        return AnthropicCacheControlHook._stamped(points, "_litellm_judged", True)

    @staticmethod
    def _judged_configured_points(
        points: Sequence[CacheControlInjectionPoint],
        messages: list[AllMessageValues],
        tools: list[object] | None,
        model: str,
        custom_llm_provider: str | None,
        api_base: object,
        prompt_cache_options: object,
    ) -> Sequence[Mapping[str, object]] | None:
        if AnthropicCacheControlHook._should_stand_down(points, messages, None, tools):
            return None
        return AnthropicCacheControlHook._stamped_with_dialect(
            points, model, custom_llm_provider, api_base, prompt_cache_options
        )

    @staticmethod
    def _stamped_with_dialect(
        points: Sequence[CacheControlInjectionPoint],
        model: str,
        custom_llm_provider: str | None,
        api_base: object,
        prompt_cache_options: object,
    ) -> Sequence[Mapping[str, object]]:
        if not supports_openai_prompt_cache_breakpoint(model):
            return points
        return AnthropicCacheControlHook._stamped(
            points,
            "_litellm_openai_dialect",
            AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
                model, custom_llm_provider, api_base, prompt_cache_options
            ),
        )

    @staticmethod
    def _stamped(
        points: Sequence[CacheControlInjectionPoint], key: str, value: object
    ) -> Sequence[Mapping[str, object]]:
        return [{**point, key: value} for point in points]

    @staticmethod
    def _should_stand_down(
        points: Sequence[CacheControlInjectionPoint],
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
        if AnthropicCacheControlHook._count_request_cache_breakpoints(messages, system) > 0:
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
        enable_prompt_caching: bool | None = None,
    ) -> list[CacheControlInjectionPoint]:
        """Default breakpoints when ``litellm.enable_anthropic_prompt_caching`` is on.

        ``enable_prompt_caching`` is the per-request override (stamped from key
        metadata by the proxy); True turns auto-injection on for this request
        even when the global flag is off. Caches the system prompt and the
        trailing turn, so the stable prefix (system + tools + history) is
        reused while the breakpoint advances with the conversation. Returns []
        (stand down) when neither flag is on, the provider does not consume
        cache_control breakpoints (only anthropic / bedrock do), the model
        lacks prompt-caching support, or the request already carries
        client-supplied cache_control.
        """
        import litellm

        if litellm.enable_anthropic_prompt_caching is not True and enable_prompt_caching is not True:
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

        control: Final = AnthropicCacheControlHook._default_control()
        points: Final[list[CacheControlInjectionPoint]] = [
            CacheControlMessageInjectionPoint(location="message", role="system", index=None, control=control),
            CacheControlMessageInjectionPoint(location="message", role=None, index=-1, control=control),
        ]
        return points

    @staticmethod
    def messages_with_default_injections(
        messages: list[AllMessageValues],
        models: Iterable[str],
        tools: list[AllToolParamValues] | None = None,
        enable_prompt_caching: bool | None = None,
    ) -> list[AllMessageValues]:
        """Return the messages auto prompt caching will send, default breakpoints included.

        Router cache affinity depends on this. Deployment selection runs before the injection in
        `litellm.acompletion`, so it has to reproduce the markers to derive the same cache key the
        success event later writes from the sent messages. `models` is every candidate model of the
        group: the first that would auto-inject decides, since the default breakpoints (system
        prompt and trailing turn) do not depend on which deployment serves the call. Returns the
        input list itself when auto-injection would not apply
        """
        points: Final = next(
            (
                candidate
                for candidate in (
                    AnthropicCacheControlHook.get_default_injection_points(
                        messages=messages,
                        system=None,
                        model=model,
                        custom_llm_provider=None,
                        tools=tools,
                        enable_prompt_caching=enable_prompt_caching,
                    )
                    for model in models
                )
                if candidate
            ),
            None,
        )
        if not points:
            return messages
        return AnthropicCacheControlHook._apply_message_injections(
            points=cast(  # cast-ok: the default points are all message-location points
                list[CacheControlMessageInjectionPoint], points
            ),
            messages=copy.deepcopy(messages),
            max_blocks=MAX_CACHE_CONTROL_BLOCKS,
        )

    @staticmethod
    def maybe_seed_default_injection_points(
        non_default_params: dict[str, Any],
        messages: list[AllMessageValues],
        model: str,
        custom_llm_provider: str | None,
        tools: list | None = None,
        enable_prompt_caching: bool | None = None,
        api_base: object = None,
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
            judged: Final = AnthropicCacheControlHook._judged_configured_points(
                non_default_params["cache_control_injection_points"],
                messages,
                tools,
                model,
                custom_llm_provider,
                api_base,
                non_default_params.get("prompt_cache_options"),
            )
            if judged is None:
                non_default_params.pop("cache_control_injection_points")
            else:
                non_default_params["cache_control_injection_points"] = judged
            return
        points: Final = AnthropicCacheControlHook.get_default_injection_points(
            messages=messages,
            system=None,
            model=model,
            custom_llm_provider=custom_llm_provider,
            tools=tools,
            enable_prompt_caching=enable_prompt_caching,
        )
        if points:
            non_default_params["cache_control_injection_points"] = points

    @staticmethod
    def maybe_inject_cache_control(
        messages: list[dict],
        system: str | list | None,
        kwargs: dict[str, Any],
        model: str | None = None,
        custom_llm_provider: str | None = None,
        tools: list[dict] | None = None,
        api_base: str | None = None,
    ) -> tuple[list[dict], str | list | None]:
        """Extract cache_control_injection_points from kwargs and apply if present.

        Configured points stand down entirely when the client already marked
        its own cache_control breakpoints anywhere in the request. The
        judgment happens once per request; points a prior pass wrote back
        carry the judged stamp and are never re-judged (see
        ``_should_stand_down``). When none are configured but
        ``litellm.enable_anthropic_prompt_caching`` or the per-request
        ``enable_prompt_caching`` kwarg (stamped from key metadata) is on,
        synthesize default breakpoints for the native /v1/messages path. Pops
        both keys from kwargs;
        if remaining (non-message) points exist they are written back so
        downstream transforms can handle them.
        """
        typed_messages = cast(list[AllMessageValues], messages)  # cast-ok: Anthropic-shaped dicts from v1/messages
        enable_prompt_caching: Final = cast(  # cast-ok: kwargs is untyped; key stamped as bool by the proxy
            bool | None, kwargs.pop("enable_prompt_caching", None)
        )
        configured: Final = cast(  # cast-ok: kwargs is untyped; this key only holds the documented injection-point list
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
                enable_prompt_caching=enable_prompt_caching,
            )
        if not injection_points:
            return messages, system

        openai_dialect: Final = AnthropicCacheControlHook._targets_openai_prompt_cache_breakpoint(
            model, custom_llm_provider, api_base, kwargs.get("prompt_cache_options")
        )
        breakpoints_before: Final = AnthropicCacheControlHook._count_request_cache_breakpoints(messages, system)
        messages, system, remaining = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            injection_points=injection_points,
            openai_dialect=openai_dialect,
        )
        if (
            openai_dialect
            and AnthropicCacheControlHook._count_request_cache_breakpoints(messages, system) > breakpoints_before
        ):
            kwargs.setdefault("prompt_cache_options", PromptCacheOptions(mode="explicit"))
        if remaining:
            kwargs["cache_control_injection_points"] = AnthropicCacheControlHook._stamped_as_judged(remaining)
        return messages, system

    @property
    def integration_name(self) -> str:
        """Return the integration name for this hook."""
        return "anthropic_cache_control_hook"

    def should_run_prompt_management(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        """Always return False since this is not a true prompt management system."""
        return False

    def _compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
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
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
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
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
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
    def should_use_anthropic_cache_control_hook(non_default_params: dict) -> bool:
        if non_default_params.get("cache_control_injection_points", None):
            return True
        return False

    @staticmethod
    def get_custom_logger_for_anthropic_cache_control_hook(
        non_default_params: dict,
    ) -> CustomLogger | None:
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
