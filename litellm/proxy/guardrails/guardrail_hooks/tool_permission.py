import json
import re
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, TypeVar

from fastapi import HTTPException
from pydantic import TypeAdapter
from typing_extensions import TypedDict

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
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none, keyword_arguments
from litellm.litellm_core_utils.tool_fields import declared_function_name, qualified_member_name
from litellm.llms.anthropic.common_utils import (
    anthropic_server_tool_call_params,
    anthropic_server_tool_calls,
    without_anthropic_server_tool_calls,
)
from litellm.llms.base_llm.guardrail_translation.utils import anthropic_tool_names
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    anthropic_sse_error_events,
    assemble_anthropic_sse_stream,
    is_raw_sse_stream,
    is_sse_error_stream,
)
from litellm.responses.additional_tools import ADDITIONAL_TOOLS_INPUT_ITEM_TYPE
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.llms.anthropic import ANTHROPIC_HOSTED_TOOLS
from litellm.types.llms.vertex_ai import VertexToolName
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
    ToolPermissionRule,
    ToolResult,
)
from litellm.types.utils import (
    CallTypesLiteral,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME: Final = "tool_permission"
_MCP_TOOLSET_TYPE: Final = "mcp_toolset"
_HOSTED_SERVER_NAME_FIELDS: Final = MappingProxyType({"mcp": "server_label", _MCP_TOOLSET_TYPE: "mcp_server_name"})
_TOOL_CARRYING_INPUT_TYPES: Final = frozenset({"tool_search_output", ADDITIONAL_TOOLS_INPUT_ITEM_TYPE})
_DECLARATION_KEYS: Final = ("functionDeclarations", "function_declarations")
_NON_TOOL_PART_KEYS: Final = frozenset({*_DECLARATION_KEYS, "cachePoint", "type"})
_TOOL_CONFIG_KEYS: Final = ("toolConfig", "tool_config")
_FUNCTION_CALLING_CONFIG_KEYS: Final = ("functionCallingConfig", "function_calling_config")
_ALLOWED_FUNCTION_NAMES_KEYS: Final = ("allowedFunctionNames", "allowed_function_names")
_FORCED_FUNCTION_CALLING_MODES: Final = frozenset({"ANY", "VALIDATED"})
_NAME_IDENTIFIED_TYPES: Final = frozenset({"function", "custom", "namespace", *_HOSTED_SERVER_NAME_FIELDS})
_TEXT_EDITOR_TOOL_NAMES: Final = frozenset({"str_replace_editor", "str_replace_based_edit_tool"})
_TYPE_VERSION_SUFFIX: Final = re.compile(r"_\d{8}$")
_GEMINI_TYPED_HOSTED_TOOLS: Final = ("computer_use", "web_search", "web_search_preview")
_NAMED_CHOICE_TYPES: Final = frozenset({"tool", "function", "custom", "allowed_tools", "any", "auto", "none"})
_DECODED_ARGUMENTS: Final = TypeAdapter(object)
_REQUEST_DATA: Final = TypeAdapter(object)
_ToolTarget: TypeAlias = tuple[str | None, str | None]
_JudgedCalls: TypeAlias = Mapping[str, tuple[tuple[str, str], ...]]
_GroupedT: Final = TypeVar("_GroupedT")


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _camel_case(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part.title() for part in rest)


_GEMINI_HOSTED_TOOL_KEYS: Final = frozenset(
    chain.from_iterable((_snake_case(tool.value), _camel_case(tool.value)) for tool in VertexToolName)
)


def _name_identifies_tool(name: str, tool_type: str) -> bool:
    stem: Final = _TYPE_VERSION_SUFFIX.sub("", tool_type)
    return (
        tool_type in _NAME_IDENTIFIED_TYPES
        or name == stem
        or (stem == ANTHROPIC_HOSTED_TOOLS.TEXT_EDITOR.value and name in _TEXT_EDITOR_TOOL_NAMES)
    )


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
    extra_names: Final = tuple(name for name in (_function_name(tool), custom_name, server_name) if name is not None)
    return tuple(dict.fromkeys(anthropic_tool_names(tool) + extra_names))


def _function_name(tool: object) -> str | None:
    fields: Final = _object_mapping(tool)
    function: Final = _object_mapping(fields.get("function")) if fields is not None else None
    return _non_empty_string(function.get("name")) if function is not None else None


def _namespace_function_targets(tool: object) -> tuple[_ToolTarget, ...]:
    function_name: Final = _function_name(tool)
    return ((function_name, "function"),) if function_name is not None else ()


def _gemini_hosted_key_targets(tool: object) -> tuple[_ToolTarget, ...]:
    fields: Final = _object_mapping(tool)
    if fields is None or fields.get("type") in _GEMINI_TYPED_HOSTED_TOOLS:
        return ()
    keys: Final = tuple(key for key in fields if key != "type")
    return (_part_target(fields, keys[0]),) if len(keys) == 1 and keys[0] in _GEMINI_HOSTED_TOOL_KEYS else ()


def _is_namespace(tool: object) -> bool:
    fields: Final = _object_mapping(tool)
    return fields is not None and fields.get("type") == "namespace"


def _namespace_members(tool: object) -> Sequence[object]:
    fields: Final = _object_mapping(tool)
    members: Final = _object_list(fields.get("tools")) if fields is not None and _is_namespace(tool) else None
    return members or ()


def _returned_namespace(tool_call: ChatCompletionMessageToolCall) -> str | None:
    tags: Final = as_str_mapping(attribute_or_none(tool_call, "provider_specific_fields"))
    return _non_empty_string(tags.get("namespace")) if tags is not None else None


def _judged_call_names(
    tool_call: ChatCompletionMessageToolCall, returned_name: str, judged_names: _JudgedCalls
) -> tuple[tuple[str, str], ...]:
    namespace: Final = _returned_namespace(tool_call)
    if namespace is not None:
        return ((qualified_member_name(namespace, returned_name), returned_name),)
    return judged_names.get(returned_name, ())


def _pair_namespace(qualified: str, bare: str) -> str | None:
    suffix: Final = qualified_member_name("", bare)
    if qualified == bare or not qualified.endswith(suffix):
        return None
    return _non_empty_string(qualified.removesuffix(suffix))


def _qualified_target(namespace: str | None, target: _ToolTarget) -> _ToolTarget:
    name, tool_type = target
    return (qualified_member_name(namespace, name), tool_type) if namespace is not None and name is not None else target


def _member_name(member: object) -> str | None:
    fields: Final = _object_mapping(member)
    return _non_empty_string(fields.get("name")) if fields is not None else None


def _declaration_targets(declaration: object) -> tuple[_ToolTarget, ...]:
    name: Final = _member_name(declaration)
    return ((name, "function"),) if name is not None else ()


@dataclass(frozen=True, slots=True)
class _MemberGroup:
    key: str
    members: Sequence[object]
    targets_of: Callable[[object], tuple[_ToolTarget, ...]]


def _member_group(
    tool: object, namespace_member_targets: Callable[[str | None, object], tuple[_ToolTarget, ...]]
) -> _MemberGroup | None:
    fields: Final = _object_mapping(tool)
    if fields is None:
        return None
    members: Final = _namespace_members(tool)
    if members:
        return _MemberGroup("tools", members, partial(namespace_member_targets, _member_name(tool)))
    key: Final = next((key for key in _DECLARATION_KEYS if _object_list(fields.get(key))), None)
    declarations: Final = _object_list(fields.get(key)) if key is not None else None
    return _MemberGroup(key, declarations, _declaration_targets) if key is not None and declarations else None


def _keyed_tool_fields(tool: object) -> Mapping[str, object] | None:
    fields: Final = _object_mapping(tool)
    if fields is None or isinstance(fields.get("type"), str) or _request_tool_names(tool):
        return None
    return fields


def _part_keys(fields: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(key for key in fields if key not in _NON_TOOL_PART_KEYS)


def _part_target(fields: Mapping[str, object], key: str) -> _ToolTarget:
    return (None if key in _GEMINI_HOSTED_TOOL_KEYS else _member_name(fields.get(key)), key)


def _input_item_tools(item: object) -> Sequence[object] | None:
    fields: Final = _object_mapping(item)
    carries_tools: Final = fields is not None and _non_empty_string(fields.get("type")) in _TOOL_CARRYING_INPUT_TYPES
    return _object_list(fields.get("tools")) if fields is not None and carries_tools else None


def _request_list(data: object, key: str) -> Sequence[object] | None:
    fields: Final = _object_mapping(data)
    return _object_list(fields.get(key)) if fields is not None else None


def _input_items(data: object) -> Sequence[object]:
    return _request_list(data, "input") or ()


def _tools_in_items(items: Sequence[object]) -> tuple[object, ...]:
    return tuple(chain.from_iterable(_input_item_tools(item) or () for item in items))


def _carries_tools(items: Sequence[object]) -> bool:
    return any(_input_item_tools(item) is not None for item in items)


def _pair_key(pair: tuple[str, object]) -> str:
    return pair[0]


def _grouped_values(pairs: Iterable[tuple[str, _GroupedT]]) -> Mapping[str, tuple[_GroupedT, ...]]:
    groups: Final = groupby(sorted(pairs, key=_pair_key), key=_pair_key)
    return MappingProxyType({key: tuple(dict.fromkeys(value for _, value in group)) for key, group in groups})


def _request_layers(data: object) -> tuple[Mapping[str, object], ...]:
    fields: Final = _object_mapping(data)
    extra_body: Final = _object_mapping(fields.get("extra_body")) if fields is not None else None
    return tuple(layer for layer in (fields, extra_body) if layer is not None)


def _layer_sent_tools(layer: Mapping[str, object]) -> tuple[object, ...]:
    return (
        *(_request_list(layer, "tools") or ()),
        *(_request_list(layer, "functions") or ()),
        *_tools_in_items(_input_items(layer)),
    )


def _sent_tools(data: object) -> tuple[object, ...]:
    return tuple(chain.from_iterable(map(_layer_sent_tools, _request_layers(data))))


def _sent_fields(data: object) -> Mapping[str, object]:
    return dict(chain.from_iterable(layer.items() for layer in _request_layers(data)))


@dataclass(frozen=True, slots=True)
class _KeptLayer:
    fields: Mapping[str, object]
    tools: tuple[object, ...] | None
    functions: tuple[object, ...] | None
    mcp_servers: tuple[object, ...] | None
    input: tuple[object, ...] | None

    def sent_tools(self) -> tuple[object, ...]:
        kept_input: Final = _input_items(self.fields) if self.input is None else self.input
        return (*(self.tools or ()), *(self.functions or ()), *_tools_in_items(kept_input))

    def kept_lists(self) -> dict[str, object]:
        lists: Final = (
            ("tools", self.tools),
            ("functions", self.functions),
            ("mcp_servers", self.mcp_servers),
            ("input", self.input),
        )
        return {key: list(kept) for key, kept in lists if kept is not None}

    def overlaid_by(self, over: "_KeptLayer") -> "_KeptLayer":
        return _KeptLayer(
            fields={**self.fields, **over.fields},
            tools=over.tools if "tools" in over.fields else self.tools,
            functions=over.functions if "functions" in over.fields else self.functions,
            mcp_servers=over.mcp_servers if "mcp_servers" in over.fields else self.mcp_servers,
            input=over.input if "input" in over.fields else self.input,
        )


def _mcp_server_target(server: object) -> _ToolTarget:
    fields: Final = _object_mapping(server)
    return (_non_empty_string(fields.get("name")) if fields is not None else None, _MCP_TOOLSET_TYPE)


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


def _is_removed_entry(entry: object, removed_names: frozenset[str], removed_types: frozenset[str]) -> bool:
    names: Final = _request_tool_names(entry)
    if names:
        return any(name in removed_names for name in names)
    fields: Final = _object_mapping(entry)
    entry_type: Final = _non_empty_string(fields.get("type")) if fields is not None else None
    return entry_type is not None and entry_type in removed_types


def _with_allowed_tool_entries(tool_choice: Mapping[str, object], entries: list[object]) -> dict[str, object]:
    nested: Final = _object_mapping(tool_choice.get("allowed_tools"))
    if nested is None:
        return {**tool_choice, "tools": entries}
    return {**tool_choice, "allowed_tools": {**nested, "tools": entries}}


def _is_denied_target(
    target: _ToolTarget, denied_targets: frozenset[_ToolTarget], denied_names: frozenset[str]
) -> bool:
    return target in denied_targets or target[0] in denied_names


def _kept_members(group: _MemberGroup | None, is_denied: Callable[[_ToolTarget], bool]) -> tuple[object, ...]:
    if group is None:
        return ()
    return tuple(member for member in group.members if not any(map(is_denied, group.targets_of(member))))


def _none_tool_choice(tool_choice: object) -> object:
    choice: Final = _object_mapping(tool_choice)
    return {"type": "none"} if choice is not None and choice.get("type") in ("tool", "any") else "none"


def _requires_a_tool(tool_choice: object) -> bool:
    choice: Final = _object_mapping(tool_choice)
    return tool_choice == "required" or (choice is not None and choice.get("type") == "any")


def _kept_allowed_function_names(config: Mapping[str, object], removed_names: frozenset[str]) -> Mapping[str, object]:
    listed: Final = tuple((key, _object_list(config.get(key))) for key in _ALLOWED_FUNCTION_NAMES_KEYS)
    return {
        key: [name for name in names if _non_empty_string(name) not in removed_names]
        for key, names in listed
        if names is not None
    }


def _rewritten_function_calling_config(
    config: Mapping[str, object], removed_names: frozenset[str], functions_remain: bool
) -> Mapping[str, object]:
    kept_names: Final = _kept_allowed_function_names(config, removed_names)
    had_names: Final = any(_object_list(config.get(key)) for key in _ALLOWED_FUNCTION_NAMES_KEYS)
    every_listed_name_removed: Final = had_names and not any(kept_names.values())
    forces_a_function: Final = str(config.get("mode", "")).upper() in _FORCED_FUNCTION_CALLING_MODES
    if not every_listed_name_removed and (functions_remain or not forces_a_function):
        return {**config, **kept_names}
    return {**{key: value for key, value in config.items() if key not in _ALLOWED_FUNCTION_NAMES_KEYS}, "mode": "NONE"}


def _rewritten_tool_config_field(
    key: str, value: object, removed_names: frozenset[str], functions_remain: bool
) -> object:
    config: Final = _object_mapping(value)
    if key not in _FUNCTION_CALLING_CONFIG_KEYS or config is None:
        return value
    return _rewritten_function_calling_config(config, removed_names, functions_remain)


def _rewritten_tool_config(tool_config: object, removed_names: frozenset[str], functions_remain: bool) -> object:
    fields: Final = _object_mapping(tool_config)
    if fields is None:
        return tool_config
    return {
        key: _rewritten_tool_config_field(key, value, removed_names, functions_remain) for key, value in fields.items()
    }


class _ToolPermissionRuleFields(TypedDict, total=False, extra_items=object):
    pass


def _rule_from_fields(fields: object) -> ToolPermissionRule:
    """Validate one config-file rule entry into a :class:`ToolPermissionRule`."""
    return ToolPermissionRule.model_validate(_ToolPermissionRuleFields(**keyword_arguments(fields, ToolPermissionRule)))


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

    def _load_rules(self, rules: Sequence[object] | None) -> None:
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

        name_matched: Final = self._matches_regex(name_pattern, tool_name or tool_type) if name_required else True
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

    def _is_name_rule_for(self, rule: ToolPermissionRule, tool_name: str, tool_type: str | None) -> bool:
        return rule.tool_name is not None and self._rule_matches_tool(rule, tool_name=tool_name, tool_type=tool_type)[0]

    def _decides_call(
        self, rule: ToolPermissionRule, tool_name: str, tool_call: ChatCompletionMessageToolCall | None
    ) -> bool:
        if tool_call is None or not rule.allowed_param_patterns:
            return True
        arguments, parse_error = self._parse_tool_call_arguments(tool_call)
        if parse_error or not arguments:
            return True
        return self._patterns_match_for_rule(arguments=arguments, rule=rule, tool_name=tool_name)[0]

    def _first_name_rule(
        self, tool_name: str, tool_type: str | None, tool_call: ChatCompletionMessageToolCall | None
    ) -> ToolPermissionRule | None:
        return next(
            (
                rule
                for rule in self.rules
                if self._is_name_rule_for(rule, tool_name, tool_type) and self._decides_call(rule, tool_name, tool_call)
            ),
            None,
        )

    def _narrower_name(
        self,
        name: str,
        other: str,
        tool_type: str | None,
        tool_call: ChatCompletionMessageToolCall | None,
    ) -> str:
        rule: Final = self._first_name_rule(name, tool_type, tool_call)
        other_rule: Final = self._first_name_rule(other, tool_type, tool_call)
        if other_rule is None or other_rule is rule:
            return name
        if rule is None or self._is_name_rule_for(rule, other, tool_type):
            return other
        if self._is_name_rule_for(other_rule, name, tool_type):
            return name
        return other if other_rule.decision == "deny" else name

    def _deciding_member_name(
        self,
        qualified: str,
        bare: str,
        tool_type: str | None,
        tool_call: ChatCompletionMessageToolCall | None = None,
    ) -> str:
        member: Final = self._narrower_name(qualified, bare, tool_type, tool_call)
        namespace: Final = _pair_namespace(qualified, bare)
        return member if namespace is None else self._narrower_name(member, namespace, tool_type, tool_call)

    def _check_target_permission(
        self, target: _ToolTarget, bare_names: Mapping[str, tuple[str, ...]]
    ) -> tuple[bool, str | None, str | None]:
        name, tool_type = target
        candidates: Final = tuple(sorted(bare_names.get(name, ()))) if name is not None else ()
        if name is None or not candidates:
            return self._check_tool_permission(name, tool_type)
        results: Final = tuple(
            self._check_tool_permission(self._deciding_member_name(name, bare_name, tool_type), tool_type)
            for bare_name in candidates
        )
        return next((result for result in results if not result[0]), results[0])

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
        except (json.JSONDecodeError, TypeError, RecursionError) as exc:
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
        self, tool_call: ChatCompletionMessageToolCall, judged_name: str | None = None
    ) -> tuple[bool, str | None, str | None]:
        tool_name: Final = judged_name or (tool_call.function.name if tool_call.function else None)
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
    def _get_mapping_value(item: object, key: str) -> object:
        if isinstance(item, dict):
            return item.get(key)
        return attribute_or_none(item, key)

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
                for tool in (*(choice.message.tool_calls or ()), *anthropic_server_tool_calls(choice.message)):
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

    def _returned_call_names(self, data: Mapping[str, object]) -> _JudgedCalls:
        sent_tools: Final = _sent_tools(_REQUEST_DATA.validate_python(data))
        member_names: Final = tuple(chain.from_iterable(map(self._bare_member_names, sent_tools)))
        if not member_names:
            return MappingProxyType({})
        top_level_tools: Final = (tool for tool in sent_tools if not _is_namespace(tool))
        top_level_names: Final = frozenset(name for name in map(declared_function_name, top_level_tools) if name)
        listed: Final = frozenset(name for name, _ in self._collect_request_tools(data) if name is not None)
        own_names: Final = frozenset(bare for _, bare in member_names if bare in listed) | top_level_names
        returned_members: Final = _grouped_values(
            chain(
                ((bare, (name, bare)) for name, bare in member_names if bare not in top_level_names),
                ((name, (name, bare)) for name, bare in member_names if name not in top_level_names),
            )
        )
        return MappingProxyType(
            {
                name: (*members, *(((name, name),) if name in own_names else ()))
                for name, members in returned_members.items()
            }
        )

    def _judged_name(self, tool_call: ChatCompletionMessageToolCall, qualified: str, bare: str) -> str:
        return qualified if qualified == bare else self._deciding_member_name(qualified, bare, "function", tool_call)

    def _judge_tool_call(
        self, tool_call: ChatCompletionMessageToolCall, judged_names: _JudgedCalls
    ) -> tuple[bool, str | None, str | None]:
        returned_name: Final = tool_call.function.name if tool_call.function else None
        names: Final = _judged_call_names(tool_call, returned_name, judged_names) if returned_name is not None else ()
        if returned_name is None or not names:
            return self._get_permission_for_tool_call(tool_call)
        judged: Final = dict.fromkeys(self._judged_name(tool_call, qualified, bare) for qualified, bare in names)
        results: Final = tuple(self._get_permission_for_tool_call(tool_call, name) for name in judged)
        return next((result for result in results if not result[0]), results[0])

    def _evaluate_tool_calls(
        self, tool_calls: Sequence[ChatCompletionMessageToolCall], judged_names: _JudgedCalls
    ) -> tuple[tuple[ChatCompletionMessageToolCall, PermissionError], ...]:
        checked: Final = tuple((tool_call, *self._judge_tool_call(tool_call, judged_names)) for tool_call in tool_calls)

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

    def _unqualified_member_targets(self, member: object) -> tuple[_ToolTarget, ...]:
        fields: Final = _object_mapping(member)
        if fields is None:
            return ()
        targets: Final = self._get_request_tool_targets(member)
        own_name: Final = _member_name(member)
        judged_names: Final = frozenset(name for name, _ in targets)
        own_target: Final = (
            ((own_name, _non_empty_string(fields.get("type"))),)
            if own_name is not None and own_name not in judged_names
            else ()
        )
        return targets + own_target

    def _namespace_member_targets(self, namespace: str | None, member: object) -> tuple[_ToolTarget, ...]:
        return tuple(_qualified_target(namespace, target) for target in self._unqualified_member_targets(member))

    def _bare_member_names(self, tool: object) -> tuple[tuple[str, str], ...]:
        namespace: Final = _member_name(tool)
        members: Final = _namespace_members(tool)
        if namespace is None or not members:
            return ()
        targets: Final = chain.from_iterable(map(self._unqualified_member_targets, members))
        return tuple((qualified_member_name(namespace, name), name) for name, _ in targets if name is not None)

    def _get_request_tool_targets(self, tool: object) -> tuple[_ToolTarget, ...]:
        group: Final = _member_group(tool, self._namespace_member_targets)
        member_targets: Final = (
            tuple(chain.from_iterable(map(group.targets_of, group.members))) if group is not None else ()
        )
        keyed: Final = _keyed_tool_fields(tool)
        part_targets: Final = tuple(_part_target(keyed, key) for key in _part_keys(keyed)) if keyed is not None else ()
        return member_targets + part_targets + self._own_tool_targets(tool)

    def _own_tool_targets(self, tool: object) -> tuple[_ToolTarget, ...]:
        if _keyed_tool_fields(tool) is not None:
            return ()
        named: Final = _namespace_function_targets(tool) if _namespace_members(tool) else self._named_tool_targets(tool)
        return named + _gemini_hosted_key_targets(tool)

    def _named_tool_targets(self, tool: object) -> tuple[_ToolTarget, ...]:
        raw_type: Final = self._get_mapping_value(tool, "type")
        tool_type: Final = raw_type if isinstance(raw_type, str) else None
        anthropic_custom: Final = (
            tool_type == "custom"
            and self._get_mapping_value(tool, "input_schema") is not None
            and _object_mapping(self._get_mapping_value(tool, "custom")) is None
        )
        normalized_type: Final = "function" if tool_type is None or anthropic_custom else tool_type
        names: Final = _request_tool_names(tool)
        named_targets: Final = tuple((name, normalized_type) for name in names)
        if names and all(_name_identifies_tool(name, normalized_type) for name in names):
            return named_targets
        if names:
            return (*named_targets, (None, normalized_type))
        return ((None, tool_type),) if tool_type is not None else ()

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

    def _get_named_function_call(self, data: Mapping[str, object]) -> str | None:
        function_call: Final = data.get("function_call")
        if not function_call or function_call in ("auto", "none"):
            return None
        if isinstance(function_call, str):
            return function_call
        return _non_empty_string(self._get_mapping_value(function_call, "name"))

    def _listed_layer_targets(self, layer: Mapping[str, object]) -> tuple[_ToolTarget, ...]:
        return (
            *chain.from_iterable(map(self._get_request_tool_targets, _request_list(layer, "tools") or ())),
            *map(_mcp_server_target, _request_list(layer, "mcp_servers") or ()),
            *chain.from_iterable(map(self._legacy_function_targets, _request_list(layer, "functions") or ())),
        )

    def _forced_layer_names(self, layer: Mapping[str, object]) -> tuple[str | None, ...]:
        return (self._get_named_tool_choice(layer.get("tool_choice")), self._get_named_function_call(layer))

    def _collect_request_tools(self, data: Mapping[str, object]) -> tuple[_ToolTarget, ...]:
        request: Final = _REQUEST_DATA.validate_python(data)
        layers: Final = _request_layers(request)
        listed: Final = tuple(chain.from_iterable(map(self._listed_layer_targets, layers)))
        listed_names: Final = frozenset(name for name, _ in listed)
        forced: Final = chain.from_iterable(map(self._forced_layer_names, layers))
        replayed: Final = map(_function_tool_name, replayed_tool_search_tools(request))
        input_tools: Final = _tools_in_items(tuple(chain.from_iterable(map(_input_items, layers))))
        return (
            *listed,
            *((name, "function") for name in forced if name is not None and name not in listed_names),
            *((name, "function") for name in replayed if name is not None),
            *chain.from_iterable(map(self._get_request_tool_targets, input_tools)),
        )

    def _legacy_function_targets(self, function: object) -> tuple[_ToolTarget, ...]:
        targets: Final = self._get_request_tool_targets(function)
        as_functions: Final = tuple((name, "function") for name, _ in targets if name is not None)
        return tuple(dict.fromkeys(targets + as_functions))

    def _kept_layer(
        self,
        layer: Mapping[str, object],
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
    ) -> _KeptLayer:
        is_denied: Final = partial(_is_denied_target, denied_targets=denied_targets, denied_names=denied_names)
        tools: Final = _request_list(layer, "tools")
        functions: Final = _request_list(layer, "functions")
        mcp_servers: Final = _request_list(layer, "mcp_servers")
        return _KeptLayer(
            fields=layer,
            tools=None if tools is None else self._kept_tool_list(tools, denied_targets, denied_names),
            functions=None if functions is None else self._kept_tool_list(functions, denied_targets, denied_names),
            mcp_servers=(
                None
                if mcp_servers is None
                else tuple(server for server in mcp_servers if not is_denied(_mcp_server_target(server)))
            ),
            input=self._kept_input_list(_input_items(layer), denied_targets, denied_names),
        )

    def _rewritten_layer_fields(
        self,
        kept: _KeptLayer,
        rewrite_choice: Callable[[object], object],
        removed_names: frozenset[str],
        functions_remain: bool,
    ) -> dict[str, object]:
        layer: Final = kept.fields
        choice: Final = layer.get("tool_choice")
        relaxes_function_call: Final = self._get_named_function_call(layer) in removed_names
        return {
            **kept.kept_lists(),
            **({"tool_choice": rewrite_choice(choice)} if choice is not None else {}),
            **({"function_call": "none"} if relaxes_function_call else {}),
            **{
                key: _rewritten_tool_config(layer[key], removed_names, functions_remain)
                for key in _TOOL_CONFIG_KEYS
                if key in layer
            },
        }

    def _tool_names_in(self, tools: Sequence[object]) -> frozenset[str]:
        targets: Final = chain.from_iterable(self._get_request_tool_targets(tool) for tool in tools)
        return frozenset(name for name, _ in targets if name is not None)

    def _referable_names_in(self, tools: Sequence[object]) -> frozenset[str]:
        member_names: Final = chain.from_iterable(map(self._bare_member_names, tools))
        namespace_names: Final = (_member_name(tool) for tool in tools if _namespace_members(tool))
        return (
            self._tool_names_in(tools)
            | frozenset(bare for _, bare in member_names)
            | frozenset(name for name in namespace_names if name is not None)
        )

    def _kept_tools(
        self,
        tool: object,
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
    ) -> tuple[object, ...]:
        is_denied: Final = partial(_is_denied_target, denied_targets=denied_targets, denied_names=denied_names)
        group: Final = _member_group(tool, self._namespace_member_targets)
        keyed: Final = _keyed_tool_fields(tool)
        fields: Final = _object_mapping(tool)
        if fields is None or (group is None and keyed is None):
            return () if any(map(is_denied, self._get_request_tool_targets(tool))) else (tool,)
        own_targets: Final = self._own_tool_targets(tool)
        if any(map(is_denied, own_targets)):
            return ()
        kept_members: Final = _kept_members(group, is_denied)
        denied_keys: Final = (
            frozenset(key for key in _part_keys(keyed) if is_denied(_part_target(keyed, key)))
            if keyed is not None
            else frozenset[str]()
        )
        if not denied_keys and len(kept_members) == len(group.members if group is not None else ()):
            return (tool,)
        emptied_group: Final = frozenset({group.key}) if group is not None and not kept_members else frozenset[str]()
        dropped_keys: Final = denied_keys | emptied_group
        kept_fields: Final = {
            key: list(kept_members) if group is not None and key == group.key else value
            for key, value in fields.items()
            if key not in dropped_keys
        }
        own_name_left: Final = any(name is not None for name, _ in own_targets)
        parts_left: Final = bool(kept_members) or own_name_left or (keyed is not None and bool(_part_keys(kept_fields)))
        return (kept_fields,) if parts_left else ()

    def _kept_tool_list(
        self,
        tools: Sequence[object],
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
    ) -> tuple[object, ...]:
        return tuple(chain.from_iterable(self._kept_tools(tool, denied_targets, denied_names) for tool in tools))

    def _kept_input_list(
        self,
        items: Sequence[object],
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
    ) -> tuple[object, ...] | None:
        if not _carries_tools(items):
            return None
        return tuple(chain.from_iterable(self._kept_input_items(item, denied_targets, denied_names) for item in items))

    def _kept_input_items(
        self,
        item: object,
        denied_targets: frozenset[_ToolTarget],
        denied_names: frozenset[str],
    ) -> tuple[object, ...]:
        fields: Final = _object_mapping(item)
        tools: Final = _input_item_tools(item)
        if fields is None or tools is None:
            return (item,)
        kept: Final = self._kept_tool_list(tools, denied_targets, denied_names)
        emptied: Final = bool(tools) and not kept and fields.get("type") == ADDITIONAL_TOOLS_INPUT_ITEM_TYPE
        return () if emptied else ({**fields, "tools": list(kept)},)

    def _tool_types_in(self, tools: Sequence[object]) -> frozenset[str]:
        targets: Final = chain.from_iterable(self._get_request_tool_targets(tool) for tool in tools)
        return frozenset(tool_type for _, tool_type in targets if tool_type is not None)

    def _rewritten_tool_choice(
        self, tool_choice: object, removed_names: frozenset[str], removed_types: frozenset[str], tools_remain: bool
    ) -> object:
        named: Final = self._get_named_tool_choice(tool_choice)
        if named is not None and named in removed_names:
            return _none_tool_choice(tool_choice)
        choice: Final = _object_mapping(tool_choice)
        hosted_type: Final = _non_empty_string(choice.get("type")) if choice is not None else None
        if hosted_type not in _NAMED_CHOICE_TYPES and hosted_type in removed_types:
            return _none_tool_choice(tool_choice)
        allowed: Final = _allowed_tool_entries(choice) if choice is not None else None
        if choice is not None and allowed is not None:
            kept: Final = [entry for entry in allowed if not _is_removed_entry(entry, removed_names, removed_types)]
            return _with_allowed_tool_entries(choice, kept) if kept and tools_remain else _none_tool_choice(choice)
        if not tools_remain and (named is not None or _requires_a_tool(tool_choice)):
            return _none_tool_choice(tool_choice)
        return tool_choice

    def _modify_request_with_permission_errors(
        self,
        data: dict,
        denied_targets: frozenset[_ToolTarget],
    ):
        """
        Remove denied tools from the request, and relax any tool_choice that would still require one
        """
        if not denied_targets:
            return data

        verbose_proxy_logger.info("Blocking %s unauthorized tool uses", len(denied_targets))

        denied_names: Final = frozenset(name for name, _ in denied_targets if name is not None)
        request: Final = _REQUEST_DATA.validate_python(data)
        kept_layers: Final = tuple(
            self._kept_layer(layer, denied_targets, denied_names) for layer in _request_layers(request)
        )

        sent_tools: Final = _layer_sent_tools(_sent_fields(request))
        kept_sent: Final = reduce(_KeptLayer.overlaid_by, kept_layers)
        kept_sent_tools: Final = kept_sent.sent_tools()
        removed_names: Final = denied_names | (
            self._referable_names_in(sent_tools) - self._referable_names_in(kept_sent_tools)
        )
        kept_types: Final = self._tool_types_in(kept_sent_tools)
        removed_types: Final = self._tool_types_in(sent_tools) - kept_types

        tools_remain: Final = bool(kept_sent_tools) or bool(kept_sent.mcp_servers)
        rewrite_choice: Final = partial(
            self._rewritten_tool_choice,
            removed_names=removed_names,
            removed_types=removed_types,
            tools_remain=tools_remain,
        )
        rewritten: Final = tuple(
            self._rewritten_layer_fields(kept, rewrite_choice, removed_names, "function" in kept_types)
            for kept in kept_layers
        )
        data.update(rewritten[0])
        if len(rewritten) > 1 and rewritten[1]:
            data["extra_body"] = {**kept_layers[1].fields, **rewritten[1]}  # rebind-ok: rewrite mode edits in place

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

    @staticmethod
    def _judged_tool_call(
        tool_call: ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall,
    ) -> tuple[str | None, ...]:
        if isinstance(tool_call, ChatCompletionMessageCustomToolCall):
            return (tool_call.id, tool_call.type, tool_call.custom.name, tool_call.custom.input)
        return (tool_call.id, tool_call.type, tool_call.function.name, tool_call.function.arguments)

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

        error_results: Final = {
            self._judged_tool_call(tool_use): self._create_permission_error_result(tool_use, error)
            for tool_use, error in denied_tools
        }

        # Modify the response content
        for choice_index, choice in enumerate(response.choices):
            if isinstance(choice, Choices):
                filtered_tool_calls = []
                error_messages = []

                # Rewrite tool_calls
                for tool_call in choice.message.tool_calls or []:
                    error_result = error_results.get(self._judged_tool_call(tool_call))
                    if error_result is not None:
                        error_messages.append(error_result.content)
                    else:
                        filtered_tool_calls.append(tool_call)

                choice.message.tool_calls = filtered_tool_calls if filtered_tool_calls else None

                server_error_results = tuple(
                    error_results.get(self._judged_tool_call(server_tool_call))
                    for server_tool_call in anthropic_server_tool_calls(choice.message)
                )
                if any(server_error_results):
                    error_messages.extend(result.content for result in server_error_results if result is not None)
                    without_anthropic_server_tool_calls(
                        choice.message,
                        frozenset(
                            server_tool_call["id"] or ""
                            for server_tool_call, result in zip(
                                anthropic_server_tool_call_params(choice.message), server_error_results
                            )
                            if result is not None
                        ),
                    )

                legacy_tool_call = self._legacy_function_call_to_tool_call(
                    getattr(choice.message, "function_call", None), choice_index
                )
                if legacy_tool_call is not None:
                    legacy_error_result = error_results.get(self._judged_tool_call(legacy_tool_call))
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

        new_tools: Final = self._collect_request_tools(data)
        if not new_tools:
            verbose_proxy_logger.debug(
                "Tool Permission Guardrail: not running guardrail. No tools or functions in data"
            )
            return data

        member_names: Final = tuple(
            chain.from_iterable(map(self._bare_member_names, _sent_tools(_REQUEST_DATA.validate_python(data))))
        )
        bare_names: Final = _grouped_values(member_names)
        decisions: Final = tuple((target, self._check_target_permission(target, bare_names)) for target in new_tools)
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
            self._modify_request_with_permission_errors(data, frozenset(target for target, _ in denied))

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

        denied_tools: Final = self._evaluate_tool_calls(tool_calls, self._returned_call_names(data))

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
            request_data: The model request
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
            denied_tools = self._check_assembled_stream(
                assembled_model_response, self._returned_call_names(request_data)
            )
            if denied_tools:
                self._modify_response_with_permission_errors(assembled_model_response, denied_tools)

            mock_response: Final = MockResponseIterator(model_response=assembled_model_response)
            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk
            return

        anthropic_response: Final = assemble_anthropic_sse_stream(
            all_chunks, restore_identity=bool(anthropic_sse_error_events(all_chunks)), through_upstream_error=True
        )
        if anthropic_response is None:
            if is_raw_sse_stream(all_chunks) and not is_sse_error_stream(all_chunks):
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

        for sse_chunk in self._anthropic_stream_to_deliver(
            anthropic_response, all_chunks, self._returned_call_names(request_data)
        ):
            yield sse_chunk

    def _anthropic_stream_to_deliver(
        self,
        anthropic_response: ModelResponse,
        source_chunks: Sequence[object],
        judged_names: _JudgedCalls,
    ) -> Sequence[object]:
        """The frames the client gets: on a failed upstream only what the check read, never the upstream's own.

        Assembly reads past an upstream error, and a client can read a frame of a broken stream that assembly
        does not, so forwarding those frames would pass a tool call nobody checked.
        """
        denials: Final = self._check_assembled_stream(anthropic_response, judged_names)
        if denials:
            self._modify_response_with_permission_errors(anthropic_response, denials)
        elif not anthropic_sse_error_events(source_chunks):
            return source_chunks
        return anthropic_sse_chunks_from_response(anthropic_response, source_chunks, keep_explanations=True)

    def _check_assembled_stream(
        self, assembled: ModelResponse, judged_names: _JudgedCalls
    ) -> tuple[tuple[ChatCompletionMessageToolCall, PermissionError], ...]:
        verbose_proxy_logger.debug("Tool Permission Guardrail: Checking response")
        tool_calls: Final = self._extract_tool_calls_from_response(assembled)
        if not tool_calls:
            verbose_proxy_logger.debug("Tool Permission Guardrail: No tool uses found")
            return ()
        verbose_proxy_logger.debug("Tool Permission Guardrail: Found %s tool calls", len(tool_calls))
        denied_tools: Final = self._evaluate_tool_calls(tool_calls, judged_names)
        if not denied_tools:
            verbose_proxy_logger.debug("Tool Permission Guardrail Post-Call Hook: All tools allowed")
        return denied_tools
