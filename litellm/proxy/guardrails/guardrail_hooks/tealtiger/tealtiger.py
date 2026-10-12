"""
TealTigerGuardrail: self-contained PII / cost / tool-authorization guardrail.

Implements apply_guardrail(), the recommended interface for guardrails that run uniformly across chat completions,
/v1/messages, the responses API, embeddings and so on. No network calls, no LLM calls, no external API keys.

Every governance check returns a Decision value. LiteLLM's guardrail contract blocks a call by raising, so the
single raise lives at the edge of apply_guardrail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # upstream decorator is untyped
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import GenericGuardrailAPIInputs

from litellm.types.proxy.guardrails.guardrail_hooks.tealtiger import DEFAULT_POLICIES, Policy, PolicyMode

from .decision import NO_VIOLATIONS, Action, Decision
from .engine import TealEngine
from .request import identity_of, tool_name_of

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from pydantic import BaseModel


def _to_event_hook(
    mode: GuardrailEventHooks  # mutable-ok: matches LitellmParams.mode's real type
    | list[GuardrailEventHooks]
    | Mode
    | str
    | list[str]
    | None,
) -> (
    GuardrailEventHooks | list[GuardrailEventHooks] | Mode | None  # mutable-ok: matches CustomGuardrail's event_hook
):
    """Narrow LitellmParams.mode's wider type down to what CustomGuardrail's own __init__ accepts."""
    if mode is None or isinstance(mode, (GuardrailEventHooks, Mode)):
        return mode
    if isinstance(mode, str):
        return GuardrailEventHooks(mode)
    return [item if isinstance(item, GuardrailEventHooks) else GuardrailEventHooks(item) for item in mode]


def _log_receipt(decision: Decision) -> None:
    if decision.reason_code == NO_VIOLATIONS:
        verbose_proxy_logger.debug("tealtiger.decision_receipt=%s", decision.receipt_json())
        return
    verbose_proxy_logger.info("tealtiger.decision_receipt=%s", decision.receipt_json())


def _raise_if_blocked(decisions: Sequence[Decision]) -> None:
    blocked: Final = next((d for d in decisions if d.action is Action.BLOCK), None)
    if blocked is None:
        return
    raise HTTPException(
        status_code=400,
        detail=f"TealTiger: blocked: {blocked.reason_code} (correlation_id={blocked.correlation_id})",
    )


class TealTigerGuardrail(CustomGuardrail):
    def __init__(
        self,
        policies: Sequence[Policy] = DEFAULT_POLICIES,
        policy_mode: PolicyMode = PolicyMode.ENFORCE,
        guardrail_name: str | None = None,
        # LitellmParams.mode is str | list[str] | Mode, wider than CustomGuardrail's event_hook; narrowed below.
        event_hook: GuardrailEventHooks  # mutable-ok: matches LitellmParams.mode / CustomGuardrail.event_hook
        | list[GuardrailEventHooks]
        | Mode
        | str
        | list[str]
        | None = None,
        default_on: bool | None = False,
    ) -> None:
        self.engine: Final = TealEngine(policies=policies, mode=policy_mode)
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # upstream CustomGuardrail.__init__ takes untyped **kwargs
            guardrail_name=guardrail_name,
            event_hook=_to_event_hook(event_hook),
            default_on=default_on or False,
        )

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[BaseModel]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.tealtiger import TealTigerGuardrailConfigModel

        return TealTigerGuardrailConfigModel

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Runs on request and response content, since TealTiger scans for PII leakage in both directions.

        Tool authorization covers tool definitions sent with the request and tool calls in both directions (a model
        response can invoke a tool the request never declared). The budget check only runs on the request path.
        In MONITOR mode violations are logged as receipts and never raised.

        Spend accrues per call from an estimate (about 4 characters per token) of the text passing through, since
        apply_guardrail is not handed usage data.
        """
        identity: Final = identity_of(request_data)
        texts: Final = tuple(inputs.get("texts") or ())
        tool_decisions: Final = tuple(self.engine.check_tool(name) for name in self._tool_names(inputs, input_type))
        budget_decisions: Final = (self.engine.check_budget(identity, texts),) if input_type == "request" else ()
        text_decisions: Final = tuple(self.engine.check_text(text) for text in texts)

        decisions: Final = (*tool_decisions, *budget_decisions, *text_decisions)
        for decision in decisions:
            _log_receipt(decision)
        _raise_if_blocked(decisions)

        self.engine.record_usage(identity, texts)
        redacted: Final = tuple(
            text if decision.redacted_text is None else decision.redacted_text
            for text, decision in zip(texts, text_decisions, strict=True)
        )
        return {**inputs, "texts": list(redacted)}

    @staticmethod
    def _tool_names(inputs: GenericGuardrailAPIInputs, input_type: Literal["request", "response"]) -> tuple[str, ...]:
        definitions: Final[Sequence[object]] = (inputs.get("tools") or ()) if input_type == "request" else ()
        calls: Final[Sequence[object]] = inputs.get("tool_calls") or ()
        return tuple(name for name in map(tool_name_of, (*definitions, *calls)) if name is not None)
