"""Conduct Guard as a LiteLLM guardrail, backed by the ``conduct-litellm-guard`` PyPI package.

Install: ``pip install "conduct-litellm-guard>=0.2.5"``
Source:  https://github.com/sseshachala/conductai/tree/main/packages/conduct-litellm-guard
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.types.llms.openai import ChatCompletionUserMessage
from litellm.types.proxy.guardrails.guardrail_hooks.conduct import ConductGuardrailConfigModel

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

MISSING_PACKAGE_MESSAGE: Final = (
    "conduct-litellm-guard>=0.2.5 is required for the Conduct guardrail. "
    'Install it with: pip install "conduct-litellm-guard>=0.2.5"'
)

BLOCKING_VERDICTS: Final = frozenset({"block", "approval"})
FLAGGED_VERDICTS: Final = frozenset({"warning", "advisory"})


class ConductDecision(Protocol):
    @property
    def verdict(self) -> str: ...

    @property
    def rule_id(self) -> str | None: ...


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


def decision_status(decision: ConductDecision) -> GuardrailStatus:
    return "guardrail_flagged" if decision.verdict in FLAGGED_VERDICTS else "success"


class ConductVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: str
    rule_id: str | None = None


def record_decision(
    guardrail: CustomGuardrail,
    request_data: dict[str, object],  # mutable-ok: the logging helper writes metadata into it
    decision: ConductDecision,
) -> None:
    guardrail.add_standard_logging_guardrail_information_to_request_data(
        guardrail_json_response=ConductVerdict(verdict=decision.verdict, rule_id=decision.rule_id).model_dump(),
        request_data=request_data,
        guardrail_status=decision_status(decision),
    )


async def apply_conduct_guardrail(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: Literal["request", "response"],
    check: ConductCheck,
    blocked: Callable[[ConductDecision], Exception],
    record: Callable[[ConductDecision], None],
) -> GenericGuardrailAPIInputs:
    payload: Final = request_payload(inputs, request_data, input_type)
    if payload is None:
        return inputs
    decision: Final = await check(data=payload, call_type=input_type)
    if decision.verdict in BLOCKING_VERDICTS:
        raise blocked(decision)
    record(decision)
    return inputs


def binds_unreachable_fallback(guardrail_cls: type[object]) -> bool:
    return "unreachable_fallback" in inspect.signature(guardrail_cls.__init__).parameters


try:
    from conduct_litellm_guard.guardrail import ConductGuard, ConductGuardBlocked

    if not binds_unreachable_fallback(ConductGuard):
        raise ImportError(MISSING_PACKAGE_MESSAGE)
except ImportError as import_error:
    _import_error: Final = import_error

    class ConductGuardrail(CustomGuardrail):
        def __init__(self, **kwargs: object) -> None:  # kwargs-ok: mirrors the plugin constructor, only raises
            raise ImportError(MISSING_PACKAGE_MESSAGE) from _import_error

        @staticmethod
        def get_config_model() -> type[ConductGuardrailConfigModel]:
            return ConductGuardrailConfigModel

else:

    class ConductGuardrail(ConductGuard):  # pyright: ignore[reportUntypedBaseClass]  # optional dep, absent at type-check
        @staticmethod
        def get_config_model() -> type[ConductGuardrailConfigModel]:
            return ConductGuardrailConfigModel

        @log_guardrail_information
        async def apply_guardrail(
            self,
            inputs: GenericGuardrailAPIInputs,
            request_data: dict[str, object],  # mutable-ok: CustomGuardrail.apply_guardrail contract
            input_type: Literal["request", "response"],
            logging_obj: LiteLLMLoggingObj | None = None,
        ) -> GenericGuardrailAPIInputs:
            return await apply_conduct_guardrail(
                inputs,
                request_data,
                input_type,
                self.check,
                ConductGuardBlocked,
                partial(record_decision, self, request_data),
            )


__all__ = (
    "BLOCKING_VERDICTS",
    "FLAGGED_VERDICTS",
    "MISSING_PACKAGE_MESSAGE",
    "ConductCheck",
    "ConductDecision",
    "ConductGuardrail",
    "ConductVerdict",
    "apply_conduct_guardrail",
    "binds_unreachable_fallback",
    "decision_status",
    "record_decision",
    "request_payload",
)
