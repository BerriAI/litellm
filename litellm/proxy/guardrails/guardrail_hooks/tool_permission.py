import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import Any, Final, Literal, TypeAlias, TypedDict

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, Required

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    replayed_tool_search_tools,
    without_tool_search_items_loading,
)
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.base_llm.guardrail_translation.utils import anthropic_tool_names
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    assemble_anthropic_sse_stream,
    is_raw_sse_stream,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
    ToolPermissionRule,
    ToolResult,
)
from litellm.types.utils import (
    CallTypesLiteral,
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME: Final = "tool_permission"
_RESPONSES_CALL_TYPES: Final = frozenset({"responses", "aresponses", "_aresponses_websocket"})
_ANTHROPIC_CALL_TYPES: Final = frozenset({"anthropic_messages", "aanthropic_messages"})
_HOSTED_SERVER_NAME_FIELDS: Final = MappingProxyType({"mcp": "server_label", "mcp_toolset": "mcp_server_name"})
_DECLARATION_KEYS: Final = ("functionDeclarations", "function_declarations")
_NAMED_CHOICE_TYPES: Final = frozenset({"tool", "function", "custom", "allowed_tools", "any", "auto", "none"})
_DECODED_ARGUMENTS: Final = TypeAdapter(object)
_REQUEST_DATA: Final = TypeAdapter(object)
_OPTIONAL_REQUEST_LIST: Final[TypeAdapter[list[object] | None]] = TypeAdapter(list[object] | None)
_ToolTarget: TypeAlias = tuple[str | None, str | None]


def _object_mapping(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as an opaque mapping when it is a dict."""
    return value if isinstance(value, dict) else None


def _object_list(value: object) -> Sequence[object] | None:
    """Return ``value`` as an opaque sequence when it is a list."""
    return value if isinstance(value, list) else None


def _function_tool_name(tool: object) -> str | None:
    fields: Final = _object_mapping(tool)
    name: Final = fields.get("name") if fields is not None and fields.get("type") == "function" else None
    return name if isinstance(name, str) else None


def _non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _request_tool_names(tool: object) -> tuple[str, ...]:
    fields: Final = _object_mapping(tool)
    custom: Final = (
        _object_mapping(fields.get("custom")) if fields is not None and fields.get("type") == "custom" else None
    )
    custom_name: Final = _non_empty_string(custom.get("name")) if custom is not None else None
    server_field: Final = _HOSTED_SERVER_NAME_FIELDS.get(str(fields.get("type"))) if fields is not None else None
    server_name: Final = (
        _non_empty_string(fields.get(server_field)) if fields is not None and server_field is not None else None
    )
    return anthropic_tool_names(tool) + tuple(name for name in (custom_name, server_name) if name is not None)


def _namespace_members(tool: object) -> Sequence[object]:
    fields: Final = _object_mapping(tool)
    members: Final = (
        _object_list(fields.get("tools")) if fields is not None and fields.get("type") == "namespace" else None
    )
    return members or ()


def _namespace_member_target(namespace: str | None, member: object) -> _ToolTarget | None:
    fields: Final = _object_mapping(member)
    if fields is None:
        return None
    member_type: Final = _non_empty_string(fields.get("type"))
    name: Final = _non_empty_string(fields.get("name"))
    qualified: Final = name is not None and namespace is not None
    return (f"{namespace}__{name}" if qualified else name, member_type)


def _declaration_target(declaration: object) -> _ToolTarget | None:
    fields: Final = _object_mapping(declaration)
    name: Final = _non_empty_string(fields.get("name")) if fields is not None else None
    return (name, "function") if name is not None else None


@dataclass(frozen=True, slots=True)
class _MemberGroup:
    key: str
    members: Sequence[object]
    target_of: Callable[[object], _ToolTarget | None]


def _member_group(tool: object) -> _MemberGroup | None:
    fields: Final = _object_mapping(tool)
    if fields is None:
        return None
    members: Final = _namespace_members(tool)
    if members:
        return _MemberGroup("tools", members, partial(_namespace_member_target, _non_empty_string(fields.get("name"))))
    key: Final = next((key for key in _DECLARATION_KEYS if _object_list(fields.get(key))), None)
    declarations: Final = _object_list(fields.get(key)) if key is not None else None
    return _MemberGroup(key, declarations, _declaration_target) if key is not None and declarations else None


def _sole_key(tool: object) -> str | None:
    fields: Final = _object_mapping(tool)
    keys: Final = tuple(fields) if fields is not None else ()
    return keys[0] if len(keys) == 1 and keys[0] not in _DECLARATION_KEYS else None


def _tool_search_output_tools(item: object) -> Sequence[object] | None:
    fields: Final = _object_mapping(item)
    return (
        _object_list(fields.get("tools")) if fields is not None and fields.get("type") == "tool_search_output" else None
    )


def _input_items(data: object) -> Sequence[object]:
    fields: Final = _object_mapping(data)
    return (_object_list(fields.get("input")) if fields is not None else None) or ()


def _loaded_input_tools(data: object) -> tuple[object, ...]:
    return tuple(chain.from_iterable(_tool_search_output_tools(item) or () for item in _input_items(data)))


def _reference_name(part: object) -> str | None:
    fields: Final = _object_mapping(part)
    is_reference: Final = fields is not None and fields.get("type") == "tool_reference"
    return _non_empty_string(fields.get("tool_name")) if fields is not None and is_reference else None


def _references_one_of(part: object, names: frozenset[str]) -> bool:
    name: Final = _reference_name(part)
    return name is not None and name in names


def _without_references_to(parts: Sequence[object], names: frozenset[str]) -> Sequence[object]:
    removed: Final = [name for name in map(_reference_name, parts) if name is not None and name in names]
    kept: Final = [part for part in parts if not _references_one_of(part, names)]
    if kept or not removed:
        return kept
    return [{"type": "text", "text": f"Permission denied: tool {', '.join(removed)} is not available"}]


def _tool_result_without_references(fields: Mapping[str, object], names: frozenset[str]) -> object:
    parts: Final = _object_list(fields.get("content"))
    return {**fields, "content": _without_references_to(parts, names)} if parts else fields


def _search_result_without_references(fields: Mapping[str, object], names: frozenset[str]) -> object:
    result: Final = _object_mapping(fields.get("content"))
    references: Final = _object_list(result.get("tool_references")) if result is not None else None
    if result is None or references is None:
        return fields
    kept: Final = [reference for reference in references if not _references_one_of(reference, names)]
    return {**fields, "content": {**result, "tool_references": kept}}


def _without_references_in_block(block: object, names: frozenset[str]) -> object:
    fields: Final = _object_mapping(block)
    if fields is None:
        return block
    match fields.get("type"):
        case "tool_result":
            return _tool_result_without_references(fields, names)
        case "tool_search_tool_result":
            return _search_result_without_references(fields, names)
        case _:
            return block


def _without_references_in_message(message: object, names: frozenset[str]) -> object:
    fields: Final = _object_mapping(message)
    content: Final = _object_list(fields.get("content")) if fields is not None else None
    if fields is None or content is None:
        return message
    if fields.get("role") == "tool":
        return {**fields, "content": _without_references_to(content, names)}
    return {**fields, "content": [_without_references_in_block(block, names) for block in content]}


def _allowed_tool_entries(tool_choice: Mapping[str, object]) -> Sequence[object] | None:
    if tool_choice.get("type") != "allowed_tools":
        return None
    nested: Final = _object_mapping(tool_choice.get("allowed_tools"))
    return _object_list((nested if nested is not None else tool_choice).get("tools"))


def _with_allowed_tool_entries(tool_choice: Mapping[str, object], entries: list[object]) -> dict[str, object]:
    nested: Final = _object_mapping(tool_choice.get("allowed_tools"))
    if nested is None:
        return {**tool_choice, "tools": entries}
    return {**tool_choice, "allowed_tools": {**nested, "tools": entries}}


def _is_denied_target(
    target: _ToolTarget, denied_targets: frozenset[_ToolTarget], denied_names: frozenset[str]
) -> bool:
    return target in denied_targets or target[0] in denied_names


def _none_tool_choice(tool_choice: object) -> object:
    choice: Final = _object_mapping(tool_choice)
    return {"type": "none"} if choice is not None and choice.get("type") in ("tool", "any") else "none"


def _requires_a_tool(tool_choice: object) -> bool:
    choice: Final = _object_mapping(tool_choice)
    return tool_choice == "required" or (choice is not None and choice.get("type") == "any")


class _ToolPermissionRuleFields(TypedDict, total=False):
    """The config-file shape a :class:`ToolPermissionRule` is built from."""

    id: ReadOnly[Required[str]]
    tool_name: ReadOnly[str | None]
    tool_type: ReadOnly[str | None]
    decision: ReadOnly[Required[Literal["allow", "deny"]]]
    allowed_param_patterns: ReadOnly[dict[str, str] | None]


def _rule_from_fields(fields: _ToolPermissionRuleFields) -> ToolPermissionRule:
    """Validate one config-file rule entry into a :class:`ToolPermissionRule`."""
    return ToolPermissionRule(**fields)


def _is_tool_use_block(block: object) -> bool:
    """Whether ``block`` is an Anthropic ``tool_use`` content block."""
    fields: Final = _object_mapping(block)
    return fields is not None and fields.get("type") == "tool_use"


class ToolPermissionGuardrail(CustomGuardrail):
    def __init__(
        self,
        rules: list[dict] | None = None,
        default_action: Literal["deny", "allow"] = "deny",
        on_disallowed_action: Literal["block", "rewrite"] = "block",
        **kwargs,
    ):
        """
        Initialize the Tool Permission Guardrail

        Args:
            rules: List of permission rules
            default_action: Default action when no rule matches ("allow" or "deny")
            on_disallowed_action:
            **kwargs: Additional arguments passed to CustomGuardrail
        """
        # Set supported event hooks - this guardrail only works on post_call
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))

        super().__init__(**kwargs)

        self._load_rules(rules)

        # Normalize to lowercase for case-insensitive handling
        self.default_action = default_action.lower() if isinstance(default_action, str) else default_action
        self.on_disallowed_action = (
            on_disallowed_action.lower() if isinstance(on_disallowed_action, str) else on_disallowed_action
        )

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail initialized with %d rules, default_action: %s",
            len(self.rules),
            self.default_action,
        )

    def _load_rules(self, rules: list[Any] | None) -> None:
        """Parse ``rules`` and (re)build the compiled target/pattern lookups.

        ``self.rules`` plus ``_compiled_rule_targets`` / ``_compiled_rule_patterns``
        are the state every matching path reads. Centralizing the build here lets
        both ``__init__`` and ``update_in_memory_litellm_params`` recompile from a
        single source of truth, so an in-place update (PUT /guardrails, immediate
        sync) reflects rule changes instead of keeping the construction-time maps.
        """
        parsed_rules: Final[list[ToolPermissionRule]] = []
        compiled_targets: Final[dict[str, dict[str, re.Pattern | None]]] = {}
        compiled_patterns: Final[dict[str, dict[str, re.Pattern]]] = {}

        for rule_item in rules or []:
            rule = rule_item if isinstance(rule_item, ToolPermissionRule) else _rule_from_fields(rule_item)

            target_patterns: dict[str, re.Pattern | None] = {
                "tool_name": None,
                "tool_type": None,
            }
            if rule.tool_name is not None:
                try:
                    target_patterns["tool_name"] = re.compile(rule.tool_name)
                except re.error as exc:
                    raise ValueError(f"Invalid regex for tool_name in rule '{rule.id}': {exc}") from exc
            if rule.tool_type is not None:
                try:
                    target_patterns["tool_type"] = re.compile(rule.tool_type)
                except re.error as exc:
                    raise ValueError(f"Invalid regex for tool_type in rule '{rule.id}': {exc}") from exc

            rule_patterns: dict[str, re.Pattern] = {}
            for path, pattern in (rule.allowed_param_patterns or {}).items():
                try:
                    rule_patterns[path] = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Invalid regex in allowed_param_patterns for rule '{rule.id}': {exc}") from exc

            parsed_rules.append(rule)
            compiled_targets[rule.id] = target_patterns
            if rule_patterns:
                compiled_patterns[rule.id] = rule_patterns

        # Swap in the fully-built maps only after every rule compiles, so an
        # invalid regex raises without leaving a partially-built ruleset (a
        # missing compiled target is read as a match-all wildcard).
        self.rules = parsed_rules
        self._compiled_rule_targets = compiled_targets
        self._compiled_rule_patterns = compiled_patterns

    def update_in_memory_litellm_params(self, litellm_params: LitellmParams | dict) -> None:
        """Apply updated params in place, rebuilding the compiled rule state.

        The base implementation only ``setattr``s raw fields, which would leave
        ``_compiled_rule_targets`` / ``_compiled_rule_patterns`` (built in
        ``__init__``) stale, so a guardrail updated without reinitialization would
        keep enforcing the old ruleset. Recompile here so PUT /guardrails and the
        immediate in-memory sync take effect, mirroring the PresidioGuardrail
        override of this method.
        """
        # ``litellm_params`` may arrive as the raw DB dict (the proxy ``cast()``s
        # it to ``LitellmParams`` without converting), so handle both shapes. The
        # base ``setattr`` loop is model-only, so apply the dict case here.
        previous_rules: Final = self.rules
        if isinstance(litellm_params, dict):
            params = litellm_params
            for key, value in params.items():
                setattr(self, key, value)
        else:
            super().update_in_memory_litellm_params(litellm_params)
            params = vars(litellm_params)

        # The generic update above sets ``self.rules`` from the incoming value
        # (None on a partial update that omits rules), but never rebuilds the
        # compiled maps. Rebuild them when rules are provided; otherwise restore
        # the previous ruleset so a partial update doesn't silently wipe it. An
        # explicit empty list still clears the rules.
        rules: Final = params.get("rules")
        if rules is not None:
            try:
                self._load_rules(rules)
            except Exception:
                # The generic update above may have overwritten self.rules with
                # the raw payload; restore the prior consistent ruleset so a
                # rejected update can't leave the live guardrail enforcing a
                # broken policy.
                self.rules = previous_rules
                raise
        else:
            self.rules = previous_rules
        default_action: Final = params.get("default_action")
        if isinstance(default_action, str):
            self.default_action = default_action.lower()
        on_disallowed_action: Final = params.get("on_disallowed_action")
        if isinstance(on_disallowed_action, str):
            self.on_disallowed_action = on_disallowed_action.lower()

    @staticmethod
    def get_config_model():
        from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
            ToolPermissionGuardrailConfigModel,
        )

        return ToolPermissionGuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def _matches_regex(self, pattern: re.Pattern | None, value: str | None) -> bool:
        if pattern is None:
            return True
        if value is None:
            return False
        return bool(pattern.fullmatch(value))

    def _rule_matches_tool(
        self,
        rule: ToolPermissionRule,
        *,
        tool_name: str | None,
        tool_type: str | None = None,
    ) -> tuple[bool, bool]:
        target_patterns: Final = self._compiled_rule_targets.get(rule.id, {})
        name_pattern: Final = target_patterns.get("tool_name")
        type_pattern: Final = target_patterns.get("tool_type")

        name_required: Final = rule.tool_name is not None
        type_required: Final = rule.tool_type is not None

        name_matched: Final = self._matches_regex(name_pattern, tool_name) if name_required else True
        type_matched: Final = self._matches_regex(type_pattern, tool_type) if type_required else True

        overall_match: Final = name_matched and type_matched
        should_check_params: Final = name_required and name_matched

        return overall_match, should_check_params

    def _check_tool_permission(
        self,
        tool_name: str | None,
        tool_type: str | None = None,
    ) -> tuple[bool, str | None, str | None]:
        """
        Check if a tool is allowed based on the configured rules

        Args:
            tool_name: Name of the tool to check
            tool_type: Type of the tool to check

        Returns:
            Tuple of (is_allowed, rule_id, message)
        """
        verbose_proxy_logger.debug("Checking permission for tool: %s", tool_name or tool_type)

        # Check each rule in order
        for rule in self.rules:
            matches, _ = self._rule_matches_tool(
                rule,
                tool_name=tool_name,
                tool_type=tool_type,
            )
            if matches:
                is_allowed = rule.decision == "allow"
                tool_identifier = tool_name or tool_type or "unknown_tool"
                default_message = (
                    f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by rule '{rule.id}'"
                )
                message = self.render_violation_message(
                    default=default_message,
                    context={
                        "tool_name": tool_name or tool_identifier,
                        "rule_id": rule.id,
                    },
                )
                verbose_proxy_logger.debug(message)
                return is_allowed, rule.id, message

        # No rule matched, use default action
        is_allowed = self.default_action == "allow"
        tool_identifier = tool_name or tool_type or "unknown_tool"
        default_message = f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by default action"
        message = self.render_violation_message(
            default=default_message,
            context={
                "tool_name": tool_name or tool_identifier,
                "rule_id": None,
            },
        )
        verbose_proxy_logger.debug(message)
        return is_allowed, None, message

    def _parse_tool_call_arguments(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> tuple[Mapping[str, object] | None, str | None]:
        arguments: Final = getattr(tool_call.function, "arguments", None)
        if not arguments:
            return None, "missing arguments"

        parsed_arguments: object = {}
        try:
            if isinstance(arguments, str):
                parsed_arguments = _DECODED_ARGUMENTS.validate_python(json.loads(arguments))
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                return None, "arguments must be a JSON object"
        except (json.JSONDecodeError, TypeError) as exc:
            verbose_proxy_logger.warning(
                "Tool Permission Guardrail: Failed to decode arguments for tool %s: %s",
                tool_call.function.name,
                exc,
            )
            return None, "arguments could not be parsed"

        if isinstance(parsed_arguments, dict):
            return parsed_arguments, None

        verbose_proxy_logger.debug(
            "Tool Permission Guardrail: Rejecting non-dict arguments for tool %s",
            tool_call.function.name,
        )
        return None, "arguments must be a JSON object"

    def _collect_argument_paths(
        self,
        value: object,
        current_path: str,
        collected: dict[str, list[object]],
        depth: int = 0,
    ) -> None:
        from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

        if depth > DEFAULT_MAX_RECURSE_DEPTH:
            return

        mapping_value: Final = _object_mapping(value)
        list_value: Final = _object_list(value)
        if mapping_value is not None:
            for key, sub_value in mapping_value.items():
                next_path = f"{current_path}.{key}" if current_path else key
                self._collect_argument_paths(sub_value, next_path, collected, depth + 1)
        elif list_value is not None:
            list_path: Final = f"{current_path}[]" if current_path else "[]"
            for item in list_value:
                self._collect_argument_paths(item, list_path, collected, depth + 1)
        else:
            if not current_path:
                return
            collected.setdefault(current_path, []).append(value)

    def _patterns_match_for_rule(
        self,
        *,
        arguments: Mapping[str, object],
        rule: ToolPermissionRule,
        tool_name: str | None,
    ) -> tuple[bool, str | None]:
        compiled_patterns: Final = self._compiled_rule_patterns.get(rule.id)
        if not compiled_patterns:
            return True, None

        path_value_map: Final[dict[str, list[object]]] = {}
        self._collect_argument_paths(arguments, "", path_value_map)

        for path, compiled_pattern in compiled_patterns.items():
            values = path_value_map.get(path)
            if not values:
                return (
                    False,
                    f"Missing value for path '{path}' required by rule '{rule.id}'",
                )
            for raw_value in values:
                if not compiled_pattern.fullmatch(str(raw_value)):
                    return (
                        False,
                        f"Value '{raw_value}' for path '{path}' does not match allowed pattern"
                        f" '{compiled_pattern.pattern}' for tool '{tool_name or 'unknown_tool'}'",
                    )

        return True, None

    def _get_permission_for_tool_call(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> tuple[bool, str | None, str | None]:
        tool_name: Final = tool_call.function.name if tool_call.function else None
        tool_type: Final = getattr(tool_call, "type", None)
        if not tool_name and not tool_type:
            return self.default_action == "allow", None, None

        tool_identifier: Final = tool_name or tool_type or "unknown_tool"

        last_pattern_failure_msg: str | None = None

        for rule in self.rules:
            matches, should_check_params = self._rule_matches_tool(
                rule,
                tool_name=tool_name,
                tool_type=tool_type,
            )
            if not matches:
                continue

            if rule.allowed_param_patterns and should_check_params:
                arguments, parse_error = self._parse_tool_call_arguments(tool_call)
                if parse_error:
                    default_message = f"Tool '{tool_identifier}' {parse_error} required by rule '{rule.id}'"
                    message = self.render_violation_message(
                        default=default_message,
                        context={"tool_name": tool_identifier, "rule_id": rule.id},
                    )
                    return False, rule.id, message
                if not arguments:
                    default_message = f"Tool '{tool_identifier}' is missing arguments required by rule '{rule.id}'"
                    message = self.render_violation_message(
                        default=default_message,
                        context={"tool_name": tool_identifier, "rule_id": rule.id},
                    )
                    return False, rule.id, message

                patterns_match, failure_message = self._patterns_match_for_rule(
                    arguments=arguments,
                    rule=rule,
                    tool_name=tool_name,
                )
                if not patterns_match:
                    last_pattern_failure_msg = failure_message
                    continue

            is_allowed = rule.decision == "allow"
            default_message = f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by rule '{rule.id}'"
            message = self.render_violation_message(
                default=default_message,
                context={"tool_name": tool_identifier, "rule_id": rule.id},
            )
            return is_allowed, rule.id, message

        is_allowed = self.default_action == "allow"
        default_message = (
            last_pattern_failure_msg
            if (last_pattern_failure_msg and not is_allowed)
            else f"Tool '{tool_identifier}' {'allowed' if is_allowed else 'denied'} by default action"
        )
        message = self.render_violation_message(
            default=default_message,
            context={"tool_name": tool_identifier, "rule_id": None},
        )
        return is_allowed, None, message

    @staticmethod
    def _get_mapping_value(item: object, key: str) -> Any:
        if isinstance(item, dict):
            return item.get(key)
        return getattr(item, key, None)

    @staticmethod
    def _legacy_function_call_id(choice_index: int) -> str:
        return f"legacy_function_call_{choice_index}"

    def _legacy_function_call_to_tool_call(
        self, function_call: object, choice_index: int
    ) -> ChatCompletionMessageToolCall | None:
        if function_call is None:
            return None

        function_name: Final = self._get_mapping_value(function_call, "name")
        arguments: Final = self._get_mapping_value(function_call, "arguments") or ""
        if not function_name:
            return None

        return ChatCompletionMessageToolCall(
            id=self._legacy_function_call_id(choice_index),
            type="function",
            function={"name": function_name, "arguments": arguments},
        )

    def _extract_tool_calls_from_response(self, response: ModelResponse) -> list[ChatCompletionMessageToolCall]:
        """
        Extract tool_calls from all choices in a model response.

        Args:
            response: The model response to analyze

        Returns:
            List of tool_calls blocks found in the response
        """
        tool_calls: Final = []

        for choice_index, choice in enumerate(response.choices):
            if isinstance(choice, Choices):
                for tool in choice.message.tool_calls or []:
                    tool_calls.append(tool)
                legacy_tool_call = self._legacy_function_call_to_tool_call(
                    getattr(choice.message, "function_call", None), choice_index
                )
                if legacy_tool_call is not None:
                    tool_calls.append(legacy_tool_call)

        return tool_calls

    @staticmethod
    def _anthropic_tool_use_to_tool_call(block: object) -> ChatCompletionMessageToolCall | None:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            return None
        name: Final = block.get("name")
        if not isinstance(name, str) or not name:
            return None
        tool_input: Final[object] = block.get("input")
        return ChatCompletionMessageToolCall(
            id=str(block.get("id") or ""),
            function=Function(name=name, arguments=json.dumps(tool_input) if isinstance(tool_input, dict) else "{}"),
            type="function",
        )

    @staticmethod
    def _get_anthropic_content_blocks(response: object) -> tuple[object, ...] | None:
        if not isinstance(response, dict):
            return None
        content: Final[object] = response.get("content")
        return tuple(content) if isinstance(content, list) else None

    def _extract_tool_calls_from_anthropic_content(
        self, content: tuple[object, ...]
    ) -> tuple[ChatCompletionMessageToolCall, ...]:
        return tuple(
            tool_call for block in content if (tool_call := self._anthropic_tool_use_to_tool_call(block)) is not None
        )

    def _evaluate_tool_calls(
        self, tool_calls: Sequence[ChatCompletionMessageToolCall]
    ) -> tuple[tuple[ChatCompletionMessageToolCall, PermissionError], ...]:
        checked: Final = tuple((tool_call, *self._get_permission_for_tool_call(tool_call)) for tool_call in tool_calls)

        for _tool_call, is_allowed, _rule_id, message in checked:
            if not is_allowed and message is not None:
                verbose_proxy_logger.info("Tool Permission Guardrail: %s", message)
                if self.on_disallowed_action == "block":
                    raise GuardrailRaisedException(
                        guardrail_name=self.guardrail_name, message=message, blocked_content=True
                    )

        return tuple(
            (
                tool_call,
                PermissionError(
                    tool_name=(
                        tool_call.function.name if tool_call.function and tool_call.function.name else "unknown_tool"
                    ),
                    rule_id=rule_id,
                    message=message,
                ),
            )
            for tool_call, is_allowed, rule_id, message in checked
            if not is_allowed and message is not None
        )

    def _modify_anthropic_content_with_permission_errors(
        self,
        response: object,
        content: tuple[object, ...],
        denied_tools: tuple[tuple[ChatCompletionMessageToolCall, PermissionError], ...],
    ) -> None:
        if not denied_tools or not isinstance(response, dict):
            return

        verbose_proxy_logger.info("Blocking %s unauthorized tool uses", len(denied_tools))

        error_by_tool_use_id: Final[Mapping[object, str]] = {
            tool_call.id: self._create_permission_error_result(tool_call, error).content
            for tool_call, error in denied_tools
        }

        def _denied_message(block: object) -> str | None:
            fields: Final = _object_mapping(block)
            if fields is None or fields.get("type") != "tool_use":
                return None
            return error_by_tool_use_id.get(fields.get("id"))

        error_messages: Final = tuple(
            message for message in (_denied_message(block) for block in content) if message is not None
        )
        kept_blocks: Final = tuple(block for block in content if _denied_message(block) is None)
        new_content: Final = [
            *kept_blocks,
            {"type": "text", "text": "\n".join(error_messages)},
        ]

        response["content"] = new_content  # rebind-ok: the guardrail rewrites the provider response in place
        if not any(_is_tool_use_block(block) for block in kept_blocks):
            response["stop_reason"] = "end_turn"  # rebind-ok: dropping every tool_use ends the turn

    def _get_request_tool_targets(self, tool: object, call_type: CallTypesLiteral) -> tuple[_ToolTarget, ...]:
        group: Final = _member_group(tool)
        if group is not None:
            return tuple(target for target in map(group.target_of, group.members) if target is not None)
        tool_type: Final = self._get_mapping_value(tool, "type")
        reads_as_function: Final = tool_type is None or (tool_type == "custom" and call_type in _ANTHROPIC_CALL_TYPES)
        normalized_type: Final = "function" if reads_as_function else tool_type
        names: Final = _request_tool_names(tool)
        if names:
            return tuple((name, normalized_type) for name in names)
        if isinstance(tool_type, str):
            return ((None, tool_type),)
        sole_key: Final = _sole_key(tool)
        return ((None, sole_key),) if sole_key is not None else ()

    def _get_legacy_function_name(self, function: object) -> str | None:
        return self._get_mapping_value(function, "name")

    def _get_named_tool_choice(self, tool_choice: object) -> str | None:
        if not tool_choice or tool_choice in ("auto", "none", "required"):
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        choice: Final = _object_mapping(tool_choice)
        if choice is None:
            return None
        choice_type: Final = choice.get("type")
        server_field: Final = _HOSTED_SERVER_NAME_FIELDS.get(str(choice_type))
        if server_field is not None:
            return _non_empty_string(choice.get(server_field))
        if choice_type not in ("tool", "function", "custom"):
            return None
        nested: Final = _object_mapping(choice.get(str(choice_type)))
        nested_name: Final = _non_empty_string(nested.get("name")) if nested is not None else None
        return nested_name or _non_empty_string(choice.get("name"))

    def _get_named_function_call(self, data: dict) -> str | None:
        function_call: Final = data.get("function_call")
        if not function_call or function_call in ("auto", "none"):
            return None
        if isinstance(function_call, str):
            return function_call
        return self._get_mapping_value(function_call, "name")

    def _collect_request_tools(self, data: dict, call_type: CallTypesLiteral) -> list[_ToolTarget]:
        request_tools: Final[list[_ToolTarget]] = []

        for tool in data.get("tools") or []:
            request_tools.extend(self._get_request_tool_targets(tool, call_type))

        for function in data.get("functions") or []:
            function_name = self._get_legacy_function_name(function)
            if function_name is not None:
                request_tools.append((function_name, "function"))

        listed_names: Final = frozenset(name for name, _ in request_tools)
        for forced_tool_name in (
            self._get_named_tool_choice(_REQUEST_DATA.validate_python(data.get("tool_choice"))),
            self._get_named_function_call(data),
        ):
            if forced_tool_name is not None and forced_tool_name not in listed_names:
                request_tools.append((forced_tool_name, "function"))

        replayed: Final = map(_function_tool_name, replayed_tool_search_tools(_REQUEST_DATA.validate_python(data)))
        request_tools.extend((name, "function") for name in replayed if name is not None)
        for loaded_tool in _loaded_input_tools(data):
            request_tools.extend(self._get_request_tool_targets(loaded_tool, call_type))

        return request_tools

    def _tool_names_in(self, tools: Sequence[object], call_type: CallTypesLiteral) -> frozenset[str]:
        targets: Final = chain.from_iterable(self._get_request_tool_targets(tool, call_type) for tool in tools)
        return frozenset(name for name, _ in targets if name is not None)

    def _kept_tools(
        self,
        tool: object,
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
        call_type: CallTypesLiteral,
    ) -> tuple[object, ...]:
        group: Final = _member_group(tool)
        fields: Final = _object_mapping(tool)
        if group is None or fields is None:
            targets: Final = self._get_request_tool_targets(tool, call_type)
            denied: Final = any(_is_denied_target(target, denied_targets, denied_names) for target in targets)
            return () if denied else (tool,)
        kept_members: Final = [
            member
            for member in group.members
            if not _is_denied_target(group.target_of(member) or (None, None), denied_targets, denied_names)
        ]
        if not kept_members:
            return ()
        return (tool,) if len(kept_members) == len(group.members) else ({**fields, group.key: kept_members},)

    def _with_kept_loaded_tools(
        self,
        item: object,
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
        call_type: CallTypesLiteral,
    ) -> object:
        fields: Final = _object_mapping(item)
        loaded: Final = _tool_search_output_tools(item)
        if fields is None or loaded is None:
            return item
        kept: Final = chain.from_iterable(
            self._kept_tools(tool, denied_targets, denied_names, call_type) for tool in loaded
        )
        return {**fields, "tools": list(kept)}

    def _tool_types_in(self, tools: Sequence[object], call_type: CallTypesLiteral) -> frozenset[str]:
        targets: Final = chain.from_iterable(self._get_request_tool_targets(tool, call_type) for tool in tools)
        return frozenset(tool_type for _, tool_type in targets if tool_type is not None)

    def _rewritten_tool_choice(
        self, tool_choice: object, removed_names: frozenset[str], removed_types: frozenset[str], tools_remain: bool
    ) -> object:
        named: Final = self._get_named_tool_choice(tool_choice)
        if named is not None and named in removed_names:
            return _none_tool_choice(tool_choice)
        choice: Final = _object_mapping(tool_choice)
        hosted_type: Final = choice.get("type") if choice is not None else None
        if hosted_type not in _NAMED_CHOICE_TYPES and hosted_type in removed_types:
            return _none_tool_choice(tool_choice)
        allowed: Final = _allowed_tool_entries(choice) if choice is not None else None
        if choice is not None and allowed is not None:
            kept: Final = [
                entry for entry in allowed if not any(name in removed_names for name in _request_tool_names(entry))
            ]
            return _with_allowed_tool_entries(choice, kept) if kept and tools_remain else _none_tool_choice(choice)
        if not tools_remain and (named is not None or _requires_a_tool(tool_choice)):
            return _none_tool_choice(tool_choice)
        return tool_choice

    def _modify_request_with_permission_errors(
        self,
        data: dict,
        denied_targets: frozenset[_ToolTarget],
        call_type: CallTypesLiteral,
    ):
        """
        Remove denied tools from the request, and relax any tool_choice that would still require one
        """
        if not denied_targets:
            return data

        verbose_proxy_logger.info("Blocking %s unauthorized tool uses", len(denied_targets))

        denied_names: Final = frozenset(name for name, _ in denied_targets if name is not None)
        tools: Final = _OPTIONAL_REQUEST_LIST.validate_python(data.get("tools"))
        kept_tools: Final = (
            None
            if tools is None
            else list(
                chain.from_iterable(self._kept_tools(tool, denied_targets, denied_names, call_type) for tool in tools)
            )
        )
        if kept_tools is not None:
            data["tools"] = kept_tools

        functions: Final = _OPTIONAL_REQUEST_LIST.validate_python(data.get("functions"))
        kept_functions: Final = (
            None
            if functions is None
            else [function for function in functions if self._get_legacy_function_name(function) not in denied_names]
        )
        if kept_functions is not None:
            data["functions"] = kept_functions

        removed_names: Final = denied_names | (
            self._tool_names_in(tools, call_type) - self._tool_names_in(kept_tools, call_type)
            if tools is not None and kept_tools is not None
            else frozenset()
        )
        removed_types: Final = (
            self._tool_types_in(tools, call_type) - self._tool_types_in(kept_tools, call_type)
            if tools is not None and kept_tools is not None
            else frozenset()
        )
        input_items: Final = _input_items(data)
        if any(_tool_search_output_tools(item) is not None for item in input_items):
            data["input"] = [
                self._with_kept_loaded_tools(item, denied_targets, denied_names, call_type) for item in input_items
            ]

        tools_remain: Final = bool(kept_tools) or bool(kept_functions) or bool(_loaded_input_tools(data))
        tool_choice: Final = _REQUEST_DATA.validate_python(data.get("tool_choice"))
        if tool_choice is not None:
            data["tool_choice"] = self._rewritten_tool_choice(tool_choice, removed_names, removed_types, tools_remain)

        named_function_call: Final = self._get_named_function_call(data)
        if named_function_call in denied_names:
            data["function_call"] = "none"

        messages: Final = without_tool_search_items_loading(_REQUEST_DATA.validate_python(data), denied_names)
        if messages is not None:
            data["messages"] = [  # rebind-ok: rewrite mode edits the request in place
                _without_references_in_message(message, removed_names) for message in messages
            ]

        return data

    def _create_permission_error_result(
        self, tool_call: ChatCompletionMessageToolCall, error: PermissionError
    ) -> ToolResult:
        """
        Create a tool_result block for a permission error

        Args:
            tool_use: The tool use that was denied
            error: The permission error details

        Returns:
            A tool_result block with the error message
        """
        error_message = f"Permission denied: {error.message}"
        if error.rule_id:
            error_message += f" (Rule: {error.rule_id})"

        return ToolResult(tool_use_id=tool_call.id, content=error_message, is_error=True)

    def _modify_response_with_permission_errors(
        self,
        response: ModelResponse,
        denied_tools: Sequence[tuple[ChatCompletionMessageToolCall, PermissionError]],
    ) -> None:
        """
        Modify the response to replace denied tool_calls blocks with error results

        Args:
            response: The model response to modify
            denied_tools: List of (tool_use, error) tuples for denied tools
        """
        if not denied_tools:
            return

        verbose_proxy_logger.info("Blocking %s unauthorized tool uses", len(denied_tools))

        # Create a mapping of tool_use_id to error result
        error_results: Final = {}
        for tool_use, error in denied_tools:
            error_result = self._create_permission_error_result(tool_use, error)
            error_results[tool_use.id] = error_result

        # Modify the response content
        for choice_index, choice in enumerate(response.choices):
            if isinstance(choice, Choices):
                filtered_tool_calls = []
                error_messages = []

                # Rewrite tool_calls
                for tool_call in choice.message.tool_calls or []:
                    tool_call_id = tool_call.id
                    if tool_call_id in error_results:
                        error_result = error_results[tool_call_id]
                        error_messages.append(error_result.content)
                    else:
                        filtered_tool_calls.append(tool_call)

                choice.message.tool_calls = filtered_tool_calls if filtered_tool_calls else None

                legacy_tool_call = self._legacy_function_call_to_tool_call(
                    getattr(choice.message, "function_call", None), choice_index
                )
                if legacy_tool_call is not None:
                    legacy_error_result = error_results.get(legacy_tool_call.id)
                    if legacy_error_result is not None:
                        choice.message.function_call = None
                        error_messages.append(legacy_error_result.content)

                # Add error messages to content
                if error_messages:
                    existing_content = choice.message.content
                    if existing_content:
                        choice.message.content = existing_content + "\n\n" + "\n".join(error_messages)
                    else:
                        choice.message.content = "\n".join(error_messages)

                if (
                    not choice.message.tool_calls
                    and getattr(choice.message, "function_call", None) is None
                    and choice.finish_reason in ("tool_calls", "function_call")
                ):
                    choice.finish_reason = "stop"

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict | None:
        """ """
        verbose_proxy_logger.debug("Tool Permission Guardrail Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: Final[GuardrailEventHooks] = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        new_tools: Final = self._collect_request_tools(data, call_type)
        if not new_tools:
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: not running guardrail. No tools or functions in data"
            )
            return data

        decisions: Final = tuple((target, self._check_tool_permission(*target)) for target in new_tools)
        denied: Final = tuple(
            (target, message)
            for target, (is_allowed, _, message) in decisions
            if not is_allowed and message is not None
        )
        for _, message in denied:
            verbose_proxy_logger.info("Tool Permission Guardrail: %s", message)
        if denied and self.on_disallowed_action == "block":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "detection_message": denied[0][1],
                },
            )

        if denied:
            self._modify_request_with_permission_errors(data, frozenset(target for target, _ in denied), call_type)

        verbose_proxy_logger.debug("Tool Permission Guardrail Pre-Call Hook: All tools allowed")

        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        """
        Check tool usage permissions after the LLM call

        Args:
            data: Request data
            user_api_key_dict: User API key information (unused but required by interface)
            response: The model response to check
        """
        anthropic_content: Final = (
            None if isinstance(response, ModelResponse) else self._get_anthropic_content_blocks(response)
        )
        if not isinstance(response, ModelResponse) and anthropic_content is None:
            return response

        verbose_proxy_logger.debug("Tool Permission Guardrail Post-Call Hook: Checking response")

        if not self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call):
            verbose_proxy_logger.debug("Tool Permission Guardrail: Skipping check (not enabled)")
            return response

        # Extract tool_calls from the response
        tool_calls: Final = (
            self._extract_tool_calls_from_response(response)
            if isinstance(response, ModelResponse)
            else self._extract_tool_calls_from_anthropic_content(anthropic_content or ())
        )

        if not tool_calls:
            verbose_proxy_logger.debug("Tool Permission Guardrail: No tool uses found")
            return response

        verbose_proxy_logger.debug("Tool Permission Guardrail: Found %s tool calls", len(tool_calls))

        denied_tools: Final = self._evaluate_tool_calls(tool_calls)

        if not denied_tools:
            verbose_proxy_logger.debug("Tool Permission Guardrail Post-Call Hook: All tools allowed")
        elif isinstance(response, ModelResponse):
            self._modify_response_with_permission_errors(response, denied_tools)
        else:
            self._modify_anthropic_content_with_permission_errors(response, anthropic_content or (), denied_tools)

        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterable[ModelResponseStream],
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Check tool usage permissions after the LLM stream call

        Args:
            user_api_key_dict: User API key information (unused but required by interface)
            response: The model response to check
            request_data: The model request (unused but required by interface)
        """

        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        # Collect all chunks to process them together
        all_chunks: Final[list[ModelResponseStream]] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled_model_response: Final[ModelResponse | TextCompletionResponse | None] = (
            stream_chunk_builder(chunks=all_chunks) if not is_raw_sse_stream(all_chunks) else None
        )
        if isinstance(assembled_model_response, ModelResponse):
            denied_tools = self._check_assembled_stream(assembled_model_response)
            if denied_tools:
                self._modify_response_with_permission_errors(assembled_model_response, denied_tools)

            mock_response: Final = MockResponseIterator(model_response=assembled_model_response)
            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk
            return

        anthropic_response: Final = assemble_anthropic_sse_stream(all_chunks)
        if anthropic_response is None:
            if is_raw_sse_stream(all_chunks):
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        "Streamed response could not be verified for tool permissions "
                        "(not a parseable Anthropic SSE stream), blocking it"
                    ),
                )
            for chunk in all_chunks:
                yield chunk
            return

        anthropic_denials: Final = self._check_assembled_stream(anthropic_response)
        if not anthropic_denials:
            for chunk in all_chunks:
                yield chunk
            return

        self._modify_response_with_permission_errors(anthropic_response, anthropic_denials)
        for sse_chunk in anthropic_sse_chunks_from_response(anthropic_response, all_chunks, keep_explanations=True):
            yield sse_chunk

    def _check_assembled_stream(
        self, assembled: ModelResponse
    ) -> tuple[tuple[ChatCompletionMessageToolCall, PermissionError], ...]:
        verbose_proxy_logger.debug("Tool Permission Guardrail: Checking response")
        tool_calls: Final = self._extract_tool_calls_from_response(assembled)
        if not tool_calls:
            verbose_proxy_logger.debug("Tool Permission Guardrail: No tool uses found")
            return ()
        verbose_proxy_logger.debug("Tool Permission Guardrail: Found %s tool calls", len(tool_calls))
        denied_tools: Final = self._evaluate_tool_calls(tool_calls)
        if not denied_tools:
            verbose_proxy_logger.debug("Tool Permission Guardrail Post-Call Hook: All tools allowed")
        return denied_tools
