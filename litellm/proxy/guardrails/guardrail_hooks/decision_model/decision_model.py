"""Decision-model guardrail: a decisions-API model (e.g. Jev) scores the request or
response against predicate checks and blocks when any check crosses its threshold."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import time
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
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


_ELISION_MARKER: Final = "\n[... middle of input omitted ...]\n"


def _elide_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_ELISION_MARKER):
        return text[:max_chars]
    budget: Final = max_chars - len(_ELISION_MARKER)
    head: Final = budget // 2
    return text[:head] + _ELISION_MARKER + text[len(text) - (budget - head) :]


def _predicate_verdict(answer: object | None, check: DecisionModelCheck) -> _CheckVerdict:
    if isinstance(answer, OpenAIPredicateAnswer):
        return {
            "name": check.name,
            "probability": answer.probability,
            "threshold": check.threshold,
            "action": check.action,
            "flagged": answer.probability >= check.threshold,
        }
    return {
        "name": check.name,
        "probability": None,
        "threshold": check.threshold,
        "action": check.action,
        "flagged": False,
    }


class DecisionModelGuardrail(CustomGuardrail):
    """Runs predicate checks on request/response text via the Decisions API and blocks on flagged checks."""

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

    async def _run_checks(self, text: str) -> OpenAIDecisionResponse:
        questions: Final = [
            {"type": "predicate", "name": check.name, "instructions": check.instructions or ""} for check in self.checks
        ]
        raw_response: Final = await self._call_decisions(text, questions)
        if not isinstance(raw_response, OpenAIDecisionResponse):
            raise TypeError(f"decisions call returned {type(raw_response).__name__}, expected OpenAI format")
        return raw_response

    def _verdicts(self, response: OpenAIDecisionResponse) -> tuple[_CheckVerdict, ...]:
        answers: Final = {answer.name: answer for answer in response.answers}
        for check in self.checks:
            answer = answers.get(check.name)
            if not isinstance(answer, OpenAIPredicateAnswer):
                verbose_logger.warning(
                    "decision_model guardrail %s: check '%s' got %s, treating as not flagged",
                    self.guardrail_name,
                    check.name,
                    "no answer" if answer is None else f"a {answer.type} answer",
                )
        return tuple(_predicate_verdict(answers.get(check.name), check) for check in self.checks)

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
                    }
                    for v in verdicts
                ],
            },
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time,
            end_time=time(),
        )

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: base signature passes the live request dict
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        joined: Final = "\n".join(inputs.get("texts") or [])
        if not joined:
            return inputs
        text: Final = _elide_middle(joined, self.max_input_chars)
        start_time: Final = time()

        try:
            response: Final = await self._run_checks(text)
        except HTTPException:
            raise
        except Exception as call_error:  # noqa: BLE001  # any provider failure is governed by unreachable_fallback
            self._log(request_data, (), "guardrail_failed_to_respond", start_time)
            self._handle_call_failure(call_error)
            return inputs

        verdicts: Final = self._verdicts(response)
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
                    "flagged_checks": [
                        {"name": v["name"], "probability": v["probability"], "threshold": v["threshold"]}
                        for v in flagged
                        if v["action"] == "block"
                    ],
                },
            )
        return inputs


__all__ = ["DecisionModelGuardrail"]
