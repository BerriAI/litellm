from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from typing_extensions import Self

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.api_route_to_call_types import get_primary_call_type_for_route
from litellm.types.utils import CallTypes

from .config_parsing import config_values

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_KNOWN_CALL_TYPES: Final = frozenset(call_type.value for call_type in CallTypes)
_UNRESOLVABLE_CALL_TYPES: Final = frozenset({CallTypes.call_mcp_tool.value})


def _unknown_call_types_error(option_name: str, unknown: Sequence[str]) -> ValueError:
    renamed: Final = tuple(
        f"{value!r} -> {CallTypes[value].value!r}" for value in unknown if value in CallTypes.__members__
    )
    hint: Final = f" CallTypes member names map to these values: {', '.join(renamed)}." if renamed else ""
    return ValueError(
        f"{option_name} contains unknown call type(s) {list(unknown)}. Use CallTypes values, the strings "
        f"logged as call_type (e.g. acompletion, aembedding, anthropic_messages, pass_through_endpoint).{hint}"
    )


def _as_call_types(raw: Sequence[str] | None, *, option_name: str) -> frozenset[str]:
    values: Final = config_values(raw, option_name=option_name)
    unknown: Final = tuple(value for value in values if value not in _KNOWN_CALL_TYPES)
    if unknown:
        raise _unknown_call_types_error(option_name, unknown)
    unresolvable: Final = tuple(value for value in values if value in _UNRESOLVABLE_CALL_TYPES)
    if unresolvable:
        raise ValueError(
            f"{option_name} cannot filter {list(unresolvable)}: MCP tool calls reach the guardrail without a "
            "call type, so the filter could never match them"
        )
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class CallTypeFilter:
    run_only_on: frozenset[str] | None = None
    skip: frozenset[str] = frozenset()

    @classmethod
    def from_config(
        cls,
        *,
        run_only_on_call_types: Sequence[str] | None,
        skip_call_types: Sequence[str] | None,
        guardrail_name: str | None,
    ) -> Self:
        run_only_on: Final = (
            _as_call_types(run_only_on_call_types, option_name="run_only_on_call_types")
            if run_only_on_call_types
            else None
        )
        skip: Final = _as_call_types(skip_call_types, option_name="skip_call_types")
        if run_only_on is not None and skip:
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): both run_only_on_call_types and skip_call_types are set. "
                "The allowlist wins; skip_call_types=%s is ignored.",
                guardrail_name,
                sorted(skip),
            )
        return cls(run_only_on=run_only_on, skip=skip)

    def allows(self, call_type: str | None) -> bool:
        if call_type is None:
            return True
        if self.run_only_on is not None:
            return call_type in self.run_only_on
        return call_type not in self.skip

    def skip_reason(
        self,
        *,
        request_data: Mapping[str, object],
        logging_obj: "LiteLLMLoggingObj | None",
    ) -> str | None:
        if self.run_only_on is None and not self.skip:
            return None
        call_type: Final = resolve_call_type(request_data=request_data, logging_obj=logging_obj)
        if self.allows(call_type):
            return None
        rule: Final = "not in run_only_on_call_types" if self.run_only_on is not None else "in skip_call_types"
        return f"skipped: call type {call_type} {rule}"


def _non_empty_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _metadata_route(metadata: object) -> str | None:
    return _non_empty_str(metadata.get("user_api_key_request_route")) if isinstance(metadata, Mapping) else None


def _request_route(request_data: Mapping[str, object]) -> str | None:
    return _metadata_route(request_data.get("litellm_metadata")) or _metadata_route(request_data.get("metadata"))


def resolve_call_type(
    request_data: Mapping[str, object],
    logging_obj: "LiteLLMLoggingObj | None",
) -> str | None:
    route_call_type: Final = get_primary_call_type_for_route(_request_route(request_data))
    if route_call_type is not None:
        return route_call_type.value
    return _non_empty_str(logging_obj.call_type) if logging_obj is not None else None
