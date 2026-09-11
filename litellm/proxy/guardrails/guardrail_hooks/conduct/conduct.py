"""Conduct Guard as a LiteLLM guardrail, backed by the ``conduct-litellm-guard`` PyPI package.

Install: ``pip install "conduct-litellm-guard>=0.2.4"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol

from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.types.llms.openai import ChatCompletionUserMessage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs

MISSING_PACKAGE_MESSAGE: Final = (
    "conduct-litellm-guard is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.4"'
)

BLOCKING_VERDICTS: Final = frozenset({"block", "approval"})


class ConductDecision(Protocol):
    @property
    def verdict(self) -> str: ...


class ConductCheck(Protocol):
    def __call__(self, *, data: Mapping[str, object], call_type: str) -> Awaitable[ConductDecision]: ...


def request_payload(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
) -> Mapping[str, object] | None:
    if input_type != "request":
        return None
    messages: Final = inputs.get("structured_messages") or tuple(
        ChatCompletionUserMessage(role="user", content=text) for text in inputs.get("texts") or ()
    )
    return MappingProxyType({**request_data, "prompt": None, "messages": messages})


async def apply_conduct_guardrail(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
    check: ConductCheck,
    blocked: Callable[[ConductDecision], Exception],
) -> GenericGuardrailAPIInputs:
    payload: Final = request_payload(inputs, request_data, input_type)
    if payload is None:
        return inputs
    decision: Final = await check(data=payload, call_type=input_type)
    if decision.verdict in BLOCKING_VERDICTS:
        raise blocked(decision)
    return inputs


try:
    from conduct_litellm_guard.guardrail import ConductGuard, ConductGuardBlocked
except ImportError as import_error:
    _import_error: Final = import_error

    class ConductGuardrail(CustomGuardrail):
        def __init__(self, **kwargs: object) -> None:  # kwargs-ok: mirrors the plugin constructor, only raises
            raise ImportError(MISSING_PACKAGE_MESSAGE) from _import_error

else:

    class ConductGuardrail(ConductGuard):  # pyright: ignore[reportUntypedBaseClass]  # optional dep, absent at type-check
        @log_guardrail_information
        async def apply_guardrail(
            self,
            inputs: GenericGuardrailAPIInputs,
            request_data: Mapping[str, object],
            input_type: Literal["request", "response"],
            logging_obj: LiteLLMLoggingObj | None = None,
        ) -> GenericGuardrailAPIInputs:
            return await apply_conduct_guardrail(inputs, request_data, input_type, self.check, ConductGuardBlocked)


__all__ = (
    "BLOCKING_VERDICTS",
    "MISSING_PACKAGE_MESSAGE",
    "ConductCheck",
    "ConductDecision",
    "ConductGuardrail",
    "apply_conduct_guardrail",
    "request_payload",
)
