"""LLM-as-a-Judge guardrail: uses an LLM to score responses against weighted criteria."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

from fastapi import HTTPException
from typing_extensions import NotRequired, ReadOnly, TypedDict, Unpack

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.llm_judge import (
    default_router_provider,
    extract_text_from_content,
    judge_acompletion,
    parse_json_verdict,
)
from litellm.types.guardrails import GuardrailEventHooks, SupportedGuardrailIntegrations
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm import Router
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.guardrails import Guardrail, LitellmParams
    from litellm.types.utils import StandardLoggingEvalInformation

JUDGE_SYSTEM_PROMPT = """You are a quality judge. Evaluate the assistant's response against the criteria provided.
For each criterion, assign a score from 0 to 100 and provide concise reasoning.
Return ONLY valid JSON in this exact format:
{
  "verdicts": [
    {"criterion_name": "<name>", "score": <0-100>, "reasoning": "<one sentence>", "passed": <true|false>, "weight": <weight>}
  ],
  "overall_score": <weighted average 0-100>
}"""

_VALID_ON_FAILURE: Final = frozenset({"block", "log"})

_default_router_provider: Final = default_router_provider
_parse_judge_verdict: Final = parse_json_verdict
_extract_text_from_content: Final = extract_text_from_content


class _JudgeMessage(TypedDict):
    """Chat message, as far as the judge prompt builder reads it."""

    role: ReadOnly[NotRequired[str]]
    content: ReadOnly[NotRequired[object]]


class _GuardrailOptions(TypedDict, total=False):
    """Base :class:`CustomGuardrail` options forwarded untouched."""

    mask_request_content: ReadOnly[bool]
    mask_response_content: ReadOnly[bool]
    violation_message_template: ReadOnly[str | None]
    end_session_after_n_fails: ReadOnly[int | None]
    on_violation: ReadOnly[str | None]
    realtime_violation_message: ReadOnly[str | None]
    on_sensitive_data: ReadOnly[str | None]
    sensitive_data_route_to_model: ReadOnly[str | None]
    sticky_session_routing: ReadOnly[bool]
    run_in_parallel: ReadOnly[bool]
    only_scan_new_messages: ReadOnly[bool]


class _RequestMessagesView(TypedDict):
    messages: ReadOnly[Sequence[_JudgeMessage]]


class _RequestMetadataView(TypedDict):
    metadata: ReadOnly[MutableMapping[str, object]]


class _OverallScoreView(TypedDict):
    overall_score: ReadOnly[str | float]


class _JudgeModelView(TypedDict):
    judge_model: ReadOnly[str]


class _CriteriaView(TypedDict):
    criteria: ReadOnly[Sequence[Mapping[str, str | float]]]


class _OnFailureView(TypedDict):
    on_failure: ReadOnly[Literal["block", "log"]]


class _ThresholdView(TypedDict):
    overall_threshold: ReadOnly[str | float]


class _ModeView(TypedDict):
    mode: ReadOnly[object]


class _DefaultOnView(TypedDict):
    default_on: ReadOnly[object]


def _get_litellm_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: str | float | bool | None = None,
) -> Any:
    val: Final[object] = getattr(litellm_params, key, None)
    if val is not None:
        return val
    raw: Final = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    if raw is not None and not isinstance(raw, dict):
        attr: Final[object] = getattr(raw, key, None)
        if attr is not None:
            return attr
    return default


def _build_judge_prompt(
    criteria: Sequence[Mapping[str, object]],
    messages: Sequence[_JudgeMessage],
    response_text: str,
) -> str:
    criteria_block: Final = "\n".join(
        f"- {c.get('name', '')} (weight {c.get('weight', 0)}%): {c.get('description', '')}" for c in criteria
    )
    conversation: Final = "\n".join(
        f"{m.get('role', 'user').upper()}: {_extract_text_from_content(m.get('content', ''))}"
        for m in messages
        if m.get("content") is not None
    )
    return (
        f"Criteria to evaluate:\n{criteria_block}\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"Assistant response to evaluate:\n{response_text}"
    )


class LLMAsAJudgeGuardrail(CustomGuardrail):
    """Post-call guardrail that judges response quality via an LLM."""

    def __init__(
        self,
        guardrail_name: str,
        judge_model: str,
        criteria: Sequence[Mapping[str, object]],
        overall_threshold: float = 80.0,
        on_failure: Literal["block", "log"] = "block",
        event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | None = None,
        default_on: bool = False,
        router_provider: "Callable[[], Router | None] | None" = None,
        **kwargs: Unpack[_GuardrailOptions],
    ) -> None:
        _event_hook: GuardrailEventHooks | list[GuardrailEventHooks] | None = None
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [GuardrailEventHooks(h) if isinstance(h, str) else h for h in event_hook]
            else:
                _event_hook = GuardrailEventHooks(event_hook) if isinstance(event_hook, str) else event_hook

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            event_hook=_event_hook or GuardrailEventHooks.post_call,
            default_on=default_on,
            **kwargs,
        )
        self.judge_model = judge_model
        self.criteria = criteria
        self.overall_threshold = overall_threshold
        self.on_failure = on_failure
        self._router_provider = router_provider or _default_router_provider

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [GuardrailEventHooks.post_call]

    async def _run_judge(
        self,
        messages: Sequence[_JudgeMessage],
        response_text: str,
    ) -> dict[str, object]:
        judge_messages: Final = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_judge_prompt(self.criteria, messages, response_text),
            },
        ]
        response: Final = await judge_acompletion(
            self._router_provider(),
            self.judge_model,
            judge_messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw: Final = response.choices[0].message.content or "{}"
        return _parse_judge_verdict(raw)

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        # Only evaluate post-call (response text). Fail open on pre-call.
        if input_type != "response":
            return inputs

        texts: Final = inputs.get("texts") or []
        response_text: Final = " ".join(texts)
        if not response_text:
            return inputs

        start_time: Final = datetime.now()
        status: GuardrailStatus = "success"
        judge_result: dict[str, Any] = {}

        try:
            request_messages: Final[_RequestMessagesView] = {"messages": request_data.get("messages") or []}

            try:
                judge_result = await self._run_judge(request_messages["messages"], response_text)
            except Exception as judge_err:
                verbose_logger.warning(
                    "llm_as_a_judge guardrail: judge call failed, failing open. Error: %s", judge_err
                )
                status = "guardrail_failed_to_respond"
                return inputs

            try:
                raw_score: Final[_OverallScoreView] = {"overall_score": judge_result.get("overall_score", 100)}
                overall_score: Final = max(0.0, min(100.0, float(raw_score["overall_score"])))
            except (TypeError, ValueError):
                verbose_logger.warning("llm_as_a_judge: invalid overall_score from judge, failing open")
                return inputs

            passed: Final = overall_score >= self.overall_threshold

            eval_info: Final[StandardLoggingEvalInformation] = {
                "eval_name": self.guardrail_name or "",
                "overall_score": overall_score,
                "passed": passed,
                "judge_model": self.judge_model,
                "threshold": self.overall_threshold,
                "verdicts": judge_result.get("verdicts", []),
            }
            request_metadata: Final[_RequestMetadataView] = {"metadata": request_data.setdefault("metadata", {})}
            _metadata: Final = request_metadata["metadata"]
            existing: Final = _metadata.get("eval_information")
            if isinstance(existing, list):
                existing.append(eval_info)
            elif existing is not None:
                _metadata["eval_information"] = [existing, eval_info]
            else:
                _metadata["eval_information"] = eval_info

            if not passed:
                status = "guardrail_intervened"
                if self.on_failure == "block":
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error": "LLM judge rejected response: score below threshold",
                            "overall_score": overall_score,
                            "threshold": self.overall_threshold,
                            "verdicts": judge_result.get("verdicts", []),
                        },
                    )

            return inputs

        except HTTPException:
            raise
        except Exception as e:
            verbose_logger.warning("llm_as_a_judge guardrail unexpected error: %s", e)
            return inputs
        finally:
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="llm_as_a_judge",
                guardrail_json_response=judge_result,
                request_data=request_data,
                guardrail_status=status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                event_type=GuardrailEventHooks.post_call,
            )


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> LLMAsAJudgeGuardrail:
    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("llm_as_a_judge guardrail requires a guardrail_name")

    judge_model: Final[_JudgeModelView] = {"judge_model": _get_litellm_param(litellm_params, guardrail, "judge_model")}
    if not judge_model["judge_model"]:
        raise ValueError("llm_as_a_judge guardrail requires judge_model in litellm_params")

    criteria: Final[_CriteriaView] = {"criteria": _get_litellm_param(litellm_params, guardrail, "criteria") or []}
    if not criteria["criteria"]:
        raise ValueError("llm_as_a_judge guardrail requires at least one criterion")

    weight_total: Final = sum(float(c.get("weight", 0)) for c in criteria["criteria"])
    if abs(weight_total - 100) > 0.5:
        raise ValueError(f"llm_as_a_judge criterion weights must sum to 100 (got {weight_total})")

    on_failure: Final[_OnFailureView] = {
        "on_failure": _get_litellm_param(litellm_params, guardrail, "on_failure", "block")
    }
    if on_failure["on_failure"] not in _VALID_ON_FAILURE:
        raise ValueError(f"llm_as_a_judge on_failure must be 'block' or 'log', got '{on_failure['on_failure']}'")

    threshold: Final[_ThresholdView] = {
        "overall_threshold": _get_litellm_param(litellm_params, guardrail, "overall_threshold", 80.0)
    }
    overall_threshold: Final = float(threshold["overall_threshold"])

    mode: Final[_ModeView] = {"mode": _get_litellm_param(litellm_params, guardrail, "mode")}
    event_hook: GuardrailEventHooks | None = None
    if isinstance(mode["mode"], str) and mode["mode"] in {e.value for e in GuardrailEventHooks}:
        event_hook = GuardrailEventHooks(mode["mode"])

    default_on: Final[_DefaultOnView] = {
        "default_on": _get_litellm_param(litellm_params, guardrail, "default_on", False)
    }
    instance: Final = LLMAsAJudgeGuardrail(
        guardrail_name=guardrail_name,
        judge_model=judge_model["judge_model"],
        criteria=criteria["criteria"],
        overall_threshold=overall_threshold,
        on_failure=on_failure["on_failure"],
        event_hook=event_hook,
        default_on=bool(default_on["default_on"]),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.LLM_AS_A_JUDGE.value: LLMAsAJudgeGuardrail,
}


__all__ = [
    "LLMAsAJudgeGuardrail",
    "guardrail_class_registry",
    "initialize_guardrail",
]
