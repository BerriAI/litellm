import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy.guardrails._content_utils import iter_request_messages, message_text
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.config_parsing import (
    config_patterns,
    config_strings,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

SkipReason: TypeAlias = Literal["skipped: skip_if_system_prompt_matches", "skipped: skip_if_first_role_in"]

INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})

MAX_INSTRUCTION_SEARCH_CHARS: Final = 16 * 1024

SYSTEM_PROMPT_SKIP_REASON: Final[SkipReason] = "skipped: skip_if_system_prompt_matches"
FIRST_ROLE_SKIP_REASON: Final[SkipReason] = "skipped: skip_if_first_role_in"


def _role(message: Mapping[str, object]) -> str | None:
    role: Final = message.get("role")
    return role if isinstance(role, str) else None


def _instruction_texts(messages: Iterable[Mapping[str, object]]) -> tuple[str, ...]:
    texts: Final = tuple(message_text(message) for message in messages if _role(message) in INSTRUCTION_ROLES)
    starts: Final = accumulate((len(text) for text in texts), initial=0)
    return tuple(text[: max(0, MAX_INSTRUCTION_SEARCH_CHARS - start)] for text, start in zip(texts, starts))


@dataclass(frozen=True, slots=True)
class MessageSkipPolicy:
    system_prompt_patterns: tuple[re.Pattern[str], ...] = ()
    first_role_in: frozenset[str] = frozenset()

    @property
    def enabled(self) -> bool:
        return bool(self.system_prompt_patterns or self.first_role_in)

    def skip_reason(self, request_data: Mapping[str, object]) -> SkipReason | None:
        messages: Final = tuple(iter_request_messages(request_data))
        if not messages:
            return None
        if _role(messages[0]) in self.first_role_in:
            return FIRST_ROLE_SKIP_REASON
        matched: Final = any(self._matches(text) for text in _instruction_texts(messages))
        return SYSTEM_PROMPT_SKIP_REASON if matched else None

    def _matches(self, text: str) -> bool:
        return any(pattern.search(text) is not None for pattern in self.system_prompt_patterns)


@dataclass(frozen=True, slots=True)
class _SkipDecision:
    reason: SkipReason | None


@dataclass(frozen=True, slots=True)
class MessageSkipFilter:
    """Records a request's skip decision on the call's logging object so its response follows it.

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
    ) -> SkipReason | None:
        if not self.policy.enabled:
            return None
        if input_type == "response":
            recorded: Final = self._recorded_decision(logging_obj)
            return recorded.reason if recorded is not None else self.policy.skip_reason(request_data)
        reason: Final = self.policy.skip_reason(request_data)
        if logging_obj is not None:
            logging_obj.model_call_details[self.marker] = _SkipDecision(reason)  # rebind-ok: the call's shared record
        return reason

    def _recorded_decision(self, logging_obj: "LiteLLMLoggingObj | None") -> _SkipDecision | None:
        recorded: Final = logging_obj.model_call_details.get(self.marker) if logging_obj is not None else None
        return recorded if isinstance(recorded, _SkipDecision) else None


def build_message_skip_filter(
    *,
    skip_if_system_prompt_matches: object,
    skip_if_first_role_in: object,
    guardrail_name: str | None,
) -> MessageSkipFilter:
    policy: Final = MessageSkipPolicy(
        system_prompt_patterns=config_patterns(
            skip_if_system_prompt_matches, option_name="skip_if_system_prompt_matches"
        ),
        first_role_in=frozenset(config_strings(skip_if_first_role_in, option_name="skip_if_first_role_in")),
    )
    if policy.enabled:
        verbose_proxy_logger.warning(
            "Generic Guardrail API (%s): skip_if_system_prompt_matches / skip_if_first_role_in match on the "
            "request body, which the caller controls, so a caller that knows the configured value can exempt "
            "itself from this guardrail. Only use them to scope traffic you trust, never as enforcement.",
            guardrail_name,
        )
    return MessageSkipFilter(
        policy=policy, marker=f"generic_guardrail_api_message_skip::{guardrail_name}::{uuid.uuid4().hex}"
    )
