"""
Built-in Sensitive Data Routing guardrail.

Detects sensitive data with prebuilt regex patterns, custom regex and keyword matching,
then reroutes the request to an on-premise model instead of blocking or redacting it.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from re import Pattern
from typing import TYPE_CHECKING, Final, Literal, Optional, TypeAlias

from litellm.exceptions import SensitiveDataRouteException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    get_session_id_from_request_data,
    log_guardrail_information,
)
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.patterns import (
    get_compiled_pattern,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.sensitive_data_routing import (
    DEFAULT_SESSION_TTL_SECONDS,
    SensitiveDataRoutingGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

SensitiveDataDetectorKind: TypeAlias = Literal["prebuilt_pattern", "regex_pattern", "keyword"]


@dataclass(frozen=True, slots=True)
class SensitiveDataDetector:
    kind: SensitiveDataDetectorKind
    rule: str
    matcher: Pattern[str]


def build_detectors(
    prebuilt_patterns: Sequence[str] | None,
    regex_patterns: Sequence[str] | None,
    keywords: Sequence[str] | None,
) -> tuple[SensitiveDataDetector, ...]:
    return (
        *(
            SensitiveDataDetector("prebuilt_pattern", name, get_compiled_pattern(name))
            for name in prebuilt_patterns or ()
        ),
        *(
            SensitiveDataDetector("regex_pattern", pattern, re.compile(pattern, re.IGNORECASE))
            for pattern in regex_patterns or ()
        ),
        *(
            SensitiveDataDetector("keyword", keyword, re.compile(re.escape(keyword), re.IGNORECASE))
            for keyword in keywords or ()
        ),
    )


def _string_leaves(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _string_leaves(key)
            yield from _string_leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_leaves(item)


def tool_call_argument_texts(tool_calls: Sequence[object] | None) -> Iterator[str]:
    """Yield tool-call argument strings, plus their decoded JSON string values so escapes cannot hide a match."""
    for tool_call in tool_calls or ():
        function = tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
        arguments = function.get("arguments") if isinstance(function, dict) else getattr(function, "arguments", None)
        if not isinstance(arguments, str) or not arguments:
            continue
        yield arguments
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        yield from _string_leaves(parsed)


class SensitiveDataRoutingGuardrail(CustomGuardrail):
    """
    Reroutes a request to an on-premise model when sensitive data is detected.

    Runs locally with no external API call, and never blocks or redacts: the prompt is
    forwarded unchanged to the on-premise model. When sticky_session is enabled and the
    request carries a session id, the whole session stays pinned to that model.
    """

    def __init__(
        self,
        on_premise_model: str,
        guardrail_name: str | None = None,
        prebuilt_patterns: Sequence[str] | None = None,
        regex_patterns: Sequence[str] | None = None,
        keywords: Sequence[str] | None = None,
        sticky_session: bool = True,
        session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | None = None,  # mutable-ok: base type
        default_on: bool = False,
    ) -> None:
        if not on_premise_model:
            raise ValueError("sensitive_data_routing guardrail requires 'on_premise_model'")

        detectors: Final = build_detectors(prebuilt_patterns, regex_patterns, keywords)
        if not detectors:
            raise ValueError(
                "sensitive_data_routing guardrail requires at least one of "
                "'prebuilt_patterns', 'regex_patterns' or 'keywords'"
            )

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=event_hook or GuardrailEventHooks.pre_call,
            default_on=default_on,
        )
        self.on_premise_model = on_premise_model
        self.detectors = detectors
        self.sticky_session = sticky_session
        self.session_ttl_seconds = session_ttl_seconds

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: base type
        return [GuardrailEventHooks.pre_call]

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel] | None:
        return SensitiveDataRoutingGuardrailConfigModel

    def detect(self, texts: Sequence[str]) -> SensitiveDataDetector | None:
        for detector in self.detectors:
            if any(text and detector.matcher.search(text) for text in texts):
                return detector
        return None

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,  # mutable-ok: proxy request body, owned by the caller
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        if input_type != "request":
            return inputs

        detected: Final = self.detect(
            [*(inputs.get("texts") or ()), *tool_call_argument_texts(inputs.get("tool_calls"))]
        )
        if detected is None:
            return inputs

        session_id: Final = get_session_id_from_request_data(request_data)
        raise SensitiveDataRouteException(
            route_to_model=self.on_premise_model,
            session_id=session_id or "",
            guardrail_name=self.guardrail_name,
            detection_info={"detection_type": detected.kind, "rule": detected.rule},
            sticky_session_routing=self.sticky_session and session_id is not None,
            session_ttl_seconds=self.session_ttl_seconds,
        )
