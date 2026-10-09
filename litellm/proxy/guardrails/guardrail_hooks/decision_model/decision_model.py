"""Decision-model guardrail: a decisions-API model (e.g. Jev) scores the request or
response against predicate checks and blocks when any check crosses its threshold."""

from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop, Semaphore
from collections.abc import Callable, Mapping, Sequence
from itertools import chain
from time import time
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,  # pyright: ignore[reportUnknownVariableType]  # decorator is untyped upstream
)
from litellm.litellm_core_utils.llm_judge import default_router_provider, judge_target
from litellm.types.decisions import MAX_DECISION_QUESTIONS, OpenAIDecisionResponse, OpenAIPredicateAnswer
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.decision_model import (
    DECISION_MODEL_CHECK_PRESETS,
    DecisionModelCheck,
    DecisionModelGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm import Router
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class _CheckVerdict(TypedDict):
    name: ReadOnly[str]
    probability: ReadOnly[float | None]
    threshold: ReadOnly[float]
    action: ReadOnly[Literal["block", "log"]]
    flagged: ReadOnly[bool]
    reason: ReadOnly[str | None]


def _resolve_checks(checks: tuple[DecisionModelCheck, ...]) -> tuple[DecisionModelCheck, ...]:
    """Fill preset instructions and enforce the check invariants (non-empty, unique names,
    instructions present) that decide what a run sends, wherever the guardrail is built."""
    if not checks:
        raise ValueError("decision_model guardrail requires at least one check")
    if len(checks) > MAX_DECISION_QUESTIONS:
        raise ValueError(
            f"decision_model guardrail supports at most {MAX_DECISION_QUESTIONS} checks, got {len(checks)}"
        )
    names: Final = [check.name for check in checks]
    if len(set(names)) != len(names):
        raise ValueError("decision_model guardrail check names must be unique")
    resolved: Final = tuple(
        check
        if check.instructions is not None
        else (
            check.model_copy(update={"instructions": DECISION_MODEL_CHECK_PRESETS[check.name][1]})
            if check.name in DECISION_MODEL_CHECK_PRESETS
            else check
        )
        for check in checks
    )
    missing: Final = [check.name for check in resolved if check.instructions is None]
    if missing:
        raise ValueError(f"decision_model guardrail check(s) {missing} are not presets and must carry instructions")
    return resolved


def _chunk_text(text: str, max_chars: int) -> tuple[str, ...]:
    """Split text into chunks of at most max_chars that overlap by
    min(2000, max_chars // 4) and together cover the whole text."""
    if len(text) <= max_chars:
        return (text,)
    step: Final = max_chars - min(2000, max_chars // 4)
    return tuple(text[start : start + max_chars] for start in range(0, len(text) - max_chars + step, step))


class DecisionModelGuardrail(CustomGuardrail):
    """Runs predicate checks on request/response text via the Decisions API and blocks on flagged checks."""

    records_own_guardrail_information: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str,
        decision_model: str,
        checks: Sequence[DecisionModelCheck],
        event_hook: GuardrailEventHooks | tuple[GuardrailEventHooks, ...] | Mode | None = None,
        default_on: bool = False,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        max_input_chars: int = 24000,
        router_provider: Callable[[], Router | None] | None = None,
        timeout: float | None = None,
        max_concurrent_decision_calls: int = 8,
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # base init takes untyped **kwargs
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            event_hook=list(event_hook)
            if isinstance(event_hook, tuple)
            else (event_hook or GuardrailEventHooks.pre_call),
            default_on=default_on,
        )
        self.decision_model = decision_model
        self.checks = _resolve_checks(tuple(checks))
        self.unreachable_fallback = unreachable_fallback
        self.max_input_chars = max_input_chars
        self._router_provider = router_provider or default_router_provider
        self.timeout = timeout
        self._max_concurrent_decision_calls = max_concurrent_decision_calls
        self._semaphores_by_loop: dict[AbstractEventLoop, Semaphore] = {}  # mutable-ok: per-loop semaphore registry

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: base signature returns list
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    @staticmethod
    def get_config_model() -> type[DecisionModelGuardrailConfigModel] | None:
        return DecisionModelGuardrailConfigModel

    async def _call_decisions(self, text: str, questions: Sequence[Mapping[str, object]]) -> object:
        router: Final = self._router_provider()
        if judge_target(router, self.decision_model).via == "router" and router is not None:
            return await router.adecisions(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]  # factory_function-generated member
                model=self.decision_model,
                input=text,
                questions=questions,
                num_retries=0,
                fallbacks=[],
                timeout=self.timeout,
            )
        return await litellm.adecisions(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]  # SDK dispatch returns a broad union
            model=self.decision_model,
            input=text,
            questions=questions,
            num_retries=0,
            timeout=self.timeout,
        )

    def _get_decision_semaphore(self) -> asyncio.Semaphore:
        """Per-event-loop semaphore shared by every decisions call on this instance."""
        loop: Final = asyncio.get_running_loop()
        existing: Final = self._semaphores_by_loop.get(loop)
        if existing is not None:
            return existing
        created: Final = asyncio.Semaphore(self._max_concurrent_decision_calls)
        self._semaphores_by_loop[loop] = created
        return created

    async def _run_checks(
        self, text: str, questions: Sequence[Mapping[str, object]], semaphore: asyncio.Semaphore
    ) -> OpenAIDecisionResponse:
        async with semaphore:
            raw_response: Final = await self._call_decisions(text, questions)
        if not isinstance(raw_response, OpenAIDecisionResponse):
            raise TypeError(f"decisions call returned {type(raw_response).__name__}, expected OpenAI format")
        return raw_response

    def _check_verdict(
        self, check: DecisionModelCheck, answers_by_call: Sequence[Mapping[str | None, object]]
    ) -> _CheckVerdict:
        """Aggregate one check across calls: probability is the max over predicate answers;
        a call with no predicate answer counts as unanswered, which blocks a Block check under
        fail_closed and only warns under fail_open or for log-only checks."""
        answers: Final = tuple(call_answers.get(check.name) for call_answers in answers_by_call)
        probabilities: Final = tuple(
            answer.probability for answer in answers if isinstance(answer, OpenAIPredicateAnswer)
        )
        missing: Final = len(probabilities) < len(answers)
        if missing:
            verbose_logger.warning(
                "decision_model guardrail %s: check '%s' got no predicate answer in %d of %d decisions calls",
                self.guardrail_name,
                check.name,
                len(answers) - len(probabilities),
                len(answers),
            )
        probability: Final = max(probabilities) if probabilities else None
        unanswered_block: Final = missing and check.action == "block" and self.unreachable_fallback == "fail_closed"
        return {
            "name": check.name,
            "probability": probability,
            "threshold": check.threshold,
            "action": check.action,
            "flagged": unanswered_block or (probability is not None and probability >= check.threshold),
            "reason": "no_answer" if missing else None,
        }

    def _verdicts(self, responses: Sequence[OpenAIDecisionResponse]) -> tuple[_CheckVerdict, ...]:
        answers_by_call: Final = tuple({answer.name: answer for answer in response.answers} for response in responses)
        return tuple(self._check_verdict(check, answers_by_call) for check in self.checks)

    def _handle_call_failure(self, error: Exception) -> None:
        """fail_open logs and returns; fail_closed raises a generic 502 (upstream detail stays in server logs)."""
        if self.unreachable_fallback == "fail_open":
            verbose_logger.warning(
                "decision_model guardrail %s: decisions call failed; fail_open configured, allowing the request. Error: %s",
                self.guardrail_name,
                error,
            )
            return
        verbose_logger.error(
            "decision_model guardrail %s: decisions call failed. Error: %s", self.guardrail_name, error
        )
        raise HTTPException(
            status_code=502,
            detail={"error": "Decision model guardrail failed to respond"},
        )

    def _log(
        self,
        request_data: dict[str, object],  # mutable-ok: base helper writes into the live request dict
        verdicts: tuple[_CheckVerdict, ...],
        status: GuardrailStatus,
        start_time: float,
    ) -> None:
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]  # untyped base helper
            guardrail_provider="decision_model",
            guardrail_json_response={
                "model": self.decision_model,
                "checks": [
                    {
                        "name": v["name"],
                        "probability": v["probability"],
                        "threshold": v["threshold"],
                        "action": v["action"],
                        "flagged": v["flagged"],
                        "reason": v["reason"],
                    }
                    for v in verdicts
                ],
            },
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time,
            end_time=time(),
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: base signature passes the live request dict
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(text for text in inputs.get("texts") or [] if text and text.strip())
        if not texts:
            return inputs
        start_time: Final = time()

        segments: Final = tuple(chain.from_iterable(_chunk_text(text, self.max_input_chars) for text in texts))
        questions: Final = [
            {"type": "predicate", "name": check.name, "instructions": check.instructions or ""} for check in self.checks
        ]
        semaphore: Final = self._get_decision_semaphore()
        try:
            responses: Final = await asyncio.gather(
                *(self._run_checks(segment, questions, semaphore) for segment in segments)
            )
        except HTTPException:
            raise
        except Exception as call_error:  # noqa: BLE001  # any provider failure is governed by unreachable_fallback
            self._log(request_data, (), "guardrail_failed_to_respond", start_time)
            self._handle_call_failure(call_error)
            return inputs

        verdicts: Final = self._verdicts(responses)
        flagged: Final = tuple(v for v in verdicts if v["flagged"])
        status: Final = (
            "guardrail_intervened"
            if any(v["action"] == "block" for v in flagged)
            else ("guardrail_flagged" if flagged else "success")
        )
        self._log(request_data, verdicts, status, start_time)
        if status == "guardrail_intervened":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated decision model guardrail policy",
                    "guardrail_name": self.guardrail_name,
                    "flagged_checks": [v["name"] for v in flagged if v["action"] == "block"],
                },
            )
        return inputs


__all__ = ["DecisionModelGuardrail"]
