import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal

from litellm._logging import verbose_proxy_logger
from litellm.proxy.guardrails._content_utils import iter_request_messages, message_text
from litellm.types.guardrails import GuardrailEventHooks, Mode

from .config_parsing import compile_patterns, config_values

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})

MAX_INSTRUCTION_SEARCH_CHARS: Final = 16 * 1024

SYSTEM_PROMPT_SKIP_REASON: Final = "skipped: skip_if_system_prompt_matches"
FIRST_ROLE_SKIP_REASON: Final = "skipped: skip_if_first_role_in"

_REQUEST_SIDE_HOOKS: Final = frozenset(
    hook.value
    for hook in (
        GuardrailEventHooks.pre_call,
        GuardrailEventHooks.during_call,
        GuardrailEventHooks.pre_mcp_call,
        GuardrailEventHooks.during_mcp_call,
    )
)


def _configured_hooks(event_hook: str | Sequence[str] | Mode | None) -> tuple[str, ...]:
    if event_hook is None:
        return ()
    if isinstance(event_hook, Mode):
        mode_values: Final = (event_hook.default, *event_hook.tags.values())
        return tuple(chain.from_iterable(_configured_hooks(value) for value in mode_values))
    if isinstance(event_hook, str):
        return (_hook_value(event_hook),)
    return tuple(_hook_value(hook) for hook in event_hook)


def _hook_value(hook: str) -> str:
    return hook.value if isinstance(hook, GuardrailEventHooks) else hook


def _has_request_side_hook(event_hook: str | Sequence[str] | Mode | None) -> bool:
    return event_hook is None or any(hook in _REQUEST_SIDE_HOOKS for hook in _configured_hooks(event_hook))


def _role(message: Mapping[str, object]) -> str | None:
    role: Final = message.get("role")
    return role if isinstance(role, str) else None


@dataclass(frozen=True, slots=True)
class MessageSkipPolicy:
    system_prompt_patterns: tuple[re.Pattern[str], ...] = ()
    first_role_in: frozenset[str] = frozenset()

    @property
    def enabled(self) -> bool:
        return bool(self.system_prompt_patterns or self.first_role_in)

    def skip_reason(self, request_data: Mapping[str, object]) -> str | None:
        messages: Final = tuple(iter_request_messages(request_data))
        if not messages:
            return None
        if _role(messages[0]) in self.first_role_in:
            return FIRST_ROLE_SKIP_REASON
        instructions: Final = (message for message in messages if _role(message) in INSTRUCTION_ROLES)
        return (
            SYSTEM_PROMPT_SKIP_REASON if any(self._instruction_matches(message) for message in instructions) else None
        )

    def _instruction_matches(self, message: Mapping[str, object]) -> bool:
        searched: Final = message_text(message)[:MAX_INSTRUCTION_SEARCH_CHARS]
        return any(pattern.search(searched) is not None for pattern in self.system_prompt_patterns)


@dataclass(frozen=True, slots=True)
class _SkipDecision:
    reason: str


@dataclass(frozen=True, slots=True)
class MessageSkipFilter:
    """Skips a matching request and replays that decision on its response, which has no system prompt to match.

    The decision rides on the call's logging object, which every request and response hook of the call shares.
    The key is unique per filter instance, since config allows two guardrails with the same name. The value is
    a ``_SkipDecision``: request body keys can reach ``model_call_details`` through ``optional_params``, but only
    as JSON values, so they cannot forge one.
    """

    policy: MessageSkipPolicy
    marker: str

    def skip_reason(
        self,
        *,
        input_type: Literal["request", "response"],
        request_data: Mapping[str, object],
        logging_obj: "LiteLLMLoggingObj | None",
    ) -> str | None:
        if not self.policy.enabled:
            return None
        if input_type == "response":
            return self._recorded_reason(logging_obj)
        reason: Final = self.policy.skip_reason(request_data)
        if reason is not None and logging_obj is not None:
            logging_obj.model_call_details[self.marker] = _SkipDecision(reason)
        return reason

    def _recorded_reason(self, logging_obj: "LiteLLMLoggingObj | None") -> str | None:
        recorded: Final = logging_obj.model_call_details.get(self.marker) if logging_obj is not None else None
        return recorded.reason if isinstance(recorded, _SkipDecision) else None


def build_message_skip_filter(
    *,
    skip_if_system_prompt_matches: Sequence[str] | None,
    skip_if_first_role_in: Sequence[str] | None,
    guardrail_name: str | None,
    event_hook: str | Sequence[str] | Mode | None,
) -> MessageSkipFilter:
    policy: Final = MessageSkipPolicy(
        system_prompt_patterns=compile_patterns(
            skip_if_system_prompt_matches, option_name="skip_if_system_prompt_matches"
        ),
        first_role_in=frozenset(config_values(skip_if_first_role_in, option_name="skip_if_first_role_in")),
    )
    if policy.enabled:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): skip_if_system_prompt_matches / skip_if_first_role_in match on the "
            "request body, which the caller controls, so a caller that knows the configured value can exempt "
            "itself from this guardrail. Use skip_if_key_alias_in / skip_if_team_id_in when the exemption must "
            "hold against the caller.",
            guardrail_name,
        )
    if policy.enabled and not _has_request_side_hook(event_hook):
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): skip_if_system_prompt_matches / skip_if_first_role_in need a "
            "request-side hook (pre_call, during_call, pre_mcp_call or during_mcp_call) to decide anything. "
            "mode=%s only sees responses, so nothing will be skipped.",
            guardrail_name,
            event_hook,
        )
    return MessageSkipFilter(
        policy=policy, marker=f"generic_guardrail_api_message_skip::{guardrail_name}::{uuid.uuid4().hex}"
    )
