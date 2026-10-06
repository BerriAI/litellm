from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from typing_extensions import Self

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.api_route_to_call_types import (
    get_primary_call_type_for_route,
    primary_call_types,
)
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.config_parsing import config_strings
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_KNOWN_CALL_TYPES: Final = frozenset(call_type.value for call_type in CallTypes)
_ROUTE_CALL_TYPES: Final = frozenset(call_type.value for call_type in primary_call_types())
_RESOLVED_NAME_OF: Final = MappingProxyType(
    {
        **{value: f"a{value}" for value in _KNOWN_CALL_TYPES - _ROUTE_CALL_TYPES if f"a{value}" in _ROUTE_CALL_TYPES},
        CallTypes.video_generation.value: CallTypes.acreate_video.value,
        CallTypes.avideo_generation.value: CallTypes.acreate_video.value,
    }
)
_CALL_TYPE_HELP: Final = (
    "Use CallTypes values, the strings logged as call_type (e.g. acompletion, aembedding, anthropic_messages, "
    "pass_through_endpoint)."
)


def _suggestion(value: str) -> str | None:
    if value in _RESOLVED_NAME_OF:
        return _RESOLVED_NAME_OF[value]
    member: Final = CallTypes.__members__.get(value)
    return member.value if member is not None and member.value != value else None


def _rename_hint(unknown: Sequence[str]) -> str:
    renames: Final = tuple(
        f"{value!r} -> {suggestion!r}" for value in unknown if (suggestion := _suggestion(value)) is not None
    )
    return f" Use these call types instead: {', '.join(renames)}." if renames else ""


def _is_unknown(value: str) -> bool:
    return value not in _KNOWN_CALL_TYPES or value in _RESOLVED_NAME_OF


def _allowlist(raw: object) -> frozenset[str] | None:
    values: Final = config_strings(raw, option_name="run_only_on_call_types", fallback="Every call type is scanned")
    unknown: Final = tuple(value for value in values if _is_unknown(value))
    if unknown:
        verbose_proxy_logger.warning(
            "Ignoring run_only_on_call_types=%r: unknown call type(s) %s. %s%s Every call type is scanned",
            raw,
            list(unknown),
            _CALL_TYPE_HELP,
            _rename_hint(unknown),
        )
        return None
    return frozenset(values) or None


def _denylist(raw: object) -> frozenset[str]:
    values: Final = config_strings(raw, option_name="skip_call_types", fallback="Every call type is scanned")
    unknown: Final = tuple(value for value in values if _is_unknown(value))
    if unknown:
        verbose_proxy_logger.warning(
            "Ignoring skip_call_types entries %s: unknown call type(s). %s%s Those call types are scanned",
            list(unknown),
            _CALL_TYPE_HELP,
            _rename_hint(unknown),
        )
    return frozenset(value for value in values if not _is_unknown(value))


@dataclass(frozen=True, slots=True)
class CallTypeFilter:
    run_only_on: frozenset[str] | None = None
    skip: frozenset[str] = frozenset()

    @classmethod
    def from_config(
        cls,
        *,
        run_only_on_call_types: object,
        skip_call_types: object,
        guardrail_name: str | None,
    ) -> Self:
        allowlist_configured: Final = run_only_on_call_types not in (None, [], ())
        run_only_on: Final = _allowlist(run_only_on_call_types)
        skip: Final = _denylist(skip_call_types)
        if allowlist_configured and skip:
            verbose_proxy_logger.warning(
                "Generic Guardrail API (%s): both run_only_on_call_types and skip_call_types are set. "
                "The allowlist wins; skip_call_types=%s is ignored.",
                guardrail_name,
                sorted(skip),
            )
        return cls(run_only_on=run_only_on, skip=frozenset() if allowlist_configured else skip)

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
