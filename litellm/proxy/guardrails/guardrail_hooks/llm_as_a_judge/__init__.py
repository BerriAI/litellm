"""LLM-as-a-Judge guardrail: uses an LLM to score requests or responses against weighted criteria."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, Optional, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.llm_judge import (
    default_router_provider,
    extract_text_from_content,
    judge_acompletion,
    parse_json_verdict,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import get_last_user_message
from litellm.types.guardrails import GuardrailEventHooks, Mode, SupportedGuardrailIntegrations
from litellm.types.utils import LLM_AS_A_JUDGE_GUARDRAIL_CALL_ORIGIN, GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm import Router
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.guardrails import Guardrail, LitellmParams
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.utils import StandardLoggingEvalInformation

JudgeInputType = Literal["request", "response"]
JudgeEventHook = GuardrailEventHooks | list[GuardrailEventHooks] | Mode
JudgeModeParam = str | list[str] | Mode | GuardrailEventHooks | list[GuardrailEventHooks] | None

_JUDGE_SYSTEM_PROMPT_TEMPLATE: Final = """You are a quality judge. Evaluate the {subject} against the criteria provided.
{focus}For each criterion, assign a score from 0 to 100 and provide concise reasoning.
Return ONLY valid JSON in this exact format:
{{
  "verdicts": [
    {{"criterion_name": "<name>", "score": <0-100>, "reasoning": "<one sentence>", "passed": <true|false>, "weight": <weight>}}
  ],
  "overall_score": <weighted average 0-100>
}}"""

JUDGE_SYSTEM_PROMPTS: Final[MappingProxyType[JudgeInputType, str]] = MappingProxyType(
    {
        "request": _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
            subject="request",
            focus="Judge the most recent user turn; treat earlier turns in the conversation only as context.\n",
        ),
        "response": _JUDGE_SYSTEM_PROMPT_TEMPLATE.format(subject="assistant's response", focus=""),
    }
)

_JUDGE_SUBJECT_LABELS: Final[MappingProxyType[JudgeInputType, str]] = MappingProxyType(
    {"request": "Latest request turn to evaluate", "response": "Assistant response to evaluate"}
)

_LIFECYCLE_HOOKS: Final[MappingProxyType[JudgeInputType, tuple[GuardrailEventHooks, ...]]] = MappingProxyType(
    {
        "request": (GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.logging_only),
        "response": (GuardrailEventHooks.post_call, GuardrailEventHooks.logging_only),
    }
)

_VALID_ON_FAILURE: Final = frozenset({"block", "log"})

_JUDGE_CALL_METADATA: Final = MappingProxyType(
    {INTERNAL_CALL_ORIGIN_METADATA_KEY: LLM_AS_A_JUDGE_GUARDRAIL_CALL_ORIGIN}
)


class _LoggedCallParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: Mapping[str, object] | None = None


def _is_logged_judge_call(data: Mapping[str, object], event_type: GuardrailEventHooks) -> bool:
    """logging_only is the only event whose ``data`` is the SDK's model_call_details rather than the client body."""
    if event_type is not GuardrailEventHooks.logging_only:
        return False
    try:
        params: Final = _LoggedCallParams.model_validate(data.get("litellm_params") or {})
    except ValidationError:
        return False
    return (params.metadata or {}).get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == LLM_AS_A_JUDGE_GUARDRAIL_CALL_ORIGIN


_default_router_provider: Final = default_router_provider
_parse_judge_verdict: Final = parse_json_verdict
_extract_text_from_content: Final = extract_text_from_content

_ParamT = TypeVar("_ParamT")


class _LitellmParamView(TypedDict, Generic[_ParamT]):
    """Typed read of a single entry in an untyped ``litellm_params`` mapping."""

    value: ReadOnly[_ParamT]


class JudgeCriterion(TypedDict):
    """A single weighted criterion the judge scores the response against."""

    name: ReadOnly[NotRequired[str]]
    description: ReadOnly[NotRequired[str]]
    weight: ReadOnly[NotRequired[float]]


class JudgeMessage(TypedDict):
    """The parts of a conversation message the judge prompt renders."""

    role: ReadOnly[NotRequired[str]]
    content: ReadOnly[NotRequired[object]]


def _get_litellm_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: _ParamT,
) -> _ParamT:
    val: Final[_ParamT | None] = getattr(litellm_params, key, None)
    if val is not None:
        return val
    raw: Final = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        entry: Final[_LitellmParamView[_ParamT]] = {"value": raw[key]}
        return entry["value"]
    if raw is not None and not isinstance(raw, dict):
        attr: Final[_ParamT | None] = getattr(raw, key, None)
        if attr is not None:
            return attr
    return default


def _coerce_event_hook(mode: JudgeModeParam) -> JudgeEventHook:
    if mode is None:
        return GuardrailEventHooks.post_call
    if isinstance(mode, Mode):
        return mode
    if isinstance(mode, list):
        return [GuardrailEventHooks(hook) for hook in mode]
    return GuardrailEventHooks(mode)


def _text_under_review(inputs: GenericGuardrailAPIInputs, input_type: JudgeInputType) -> str:
    all_text: Final = "\n".join(inputs.get("texts") or [])
    if input_type == "response":
        return all_text
    latest_user_turn: Final = get_last_user_message(inputs.get("structured_messages") or [])
    return latest_user_turn if latest_user_turn is not None else all_text


def _judge_conversation(
    inputs: GenericGuardrailAPIInputs,
    request_data: Mapping[str, object],
    input_type: JudgeInputType,
) -> Sequence[JudgeMessage]:
    """The context turns; on a response scan the reply is the subject under review, not context."""
    scan_messages: Final = inputs.get("structured_messages")
    if not scan_messages:
        fallback: Final = request_data.get("messages")
        return fallback if isinstance(fallback, list) else ()
    if input_type == "response" and scan_messages[-1].get("role") == "assistant":
        return scan_messages[:-1]
    return scan_messages


def _build_judge_prompt(
    criteria: Sequence[JudgeCriterion],
    messages: Sequence[JudgeMessage],
    text_under_review: str,
    input_type: JudgeInputType = "response",
) -> str:
    criteria_block: Final = "\n".join(
        f"- {c.get('name', '')} (weight {c.get('weight', 0)}%): {c.get('description', '')}" for c in criteria
    )
    conversation: Final = "\n".join(
        f"{m.get('role', 'user').upper()}: {_extract_text_from_content(m.get('content', ''))}"
        for m in messages
        if m.get("content") is not None
    )
    conversation_block: Final = f"Conversation:\n{conversation}\n\n" if conversation or input_type == "response" else ""
    return (
        f"Criteria to evaluate:\n{criteria_block}\n\n"
        f"{conversation_block}"
        f"{_JUDGE_SUBJECT_LABELS[input_type]}:\n{text_under_review}"
    )


class LLMAsAJudgeGuardrail(CustomGuardrail):
    """Guardrail that judges request (pre_call/during_call) or response (post_call) quality via an LLM."""

    reads_conversation_as_context = True

    def __init__(
        self,
        guardrail_name: str,
        judge_model: str,
        criteria: Sequence[JudgeCriterion],
        overall_threshold: float = 80.0,
        on_failure: Literal["block", "log"] = "block",
        event_hook: JudgeModeParam = None,
        default_on: bool = False,
        router_provider: "Callable[[], Router | None] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            event_hook=_coerce_event_hook(event_hook),
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
        return [GuardrailEventHooks.pre_call, GuardrailEventHooks.during_call, GuardrailEventHooks.post_call]

    def should_run_guardrail(self, data: Mapping[str, object], event_type: GuardrailEventHooks) -> bool:
        if _is_logged_judge_call(data, event_type):
            return False
        return super().should_run_guardrail(data, event_type)

    async def _run_judge(
        self,
        messages: Sequence[JudgeMessage],
        text_under_review: str,
        input_type: JudgeInputType = "response",
    ) -> dict[str, object]:
        judge_messages: Final[list[AllMessageValues]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPTS[input_type]},
            {
                "role": "user",
                "content": _build_judge_prompt(self.criteria, messages, text_under_review, input_type),
            },
        ]
        response: Final = await judge_acompletion(
            self._router_provider(),
            self.judge_model,
            judge_messages,
            response_format={"type": "json_object"},
            temperature=0,
            metadata=dict(_JUDGE_CALL_METADATA),
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
        text_under_review: Final = _text_under_review(inputs, input_type)
        if not text_under_review:
            return inputs

        start_time: Final = datetime.now()
        status: GuardrailStatus = "success"
        judge_result: dict[str, object] = {}

        try:
            messages: Final = _judge_conversation(inputs, request_data, input_type)

            try:
                judge_result = await self._run_judge(messages, text_under_review, input_type)
            except Exception as judge_err:
                verbose_logger.warning(
                    "llm_as_a_judge guardrail: judge call failed, failing open. Error: %s", judge_err
                )
                status = "guardrail_failed_to_respond"
                return inputs

            try:
                overall_score: Final = max(0.0, min(100.0, float(judge_result.get("overall_score", 100))))
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
            _metadata: Final = request_data.setdefault("metadata", {})
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
                            "error": f"LLM judge rejected {input_type}: score below threshold",
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
                event_type=self._event_type_for(input_type),
            )

    def _event_type_for(self, input_type: JudgeInputType) -> GuardrailEventHooks | None:
        configured: Final = tuple(hook for hook in _LIFECYCLE_HOOKS[input_type] if self._event_hook_is_event_type(hook))
        return configured[0] if len(configured) == 1 else None


def initialize_guardrail(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
) -> LLMAsAJudgeGuardrail:
    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("llm_as_a_judge guardrail requires a guardrail_name")

    judge_model: Final[str] = _get_litellm_param(litellm_params, guardrail, "judge_model", "")
    if not judge_model:
        raise ValueError("llm_as_a_judge guardrail requires judge_model in litellm_params")

    criteria: Final[Sequence[JudgeCriterion]] = _get_litellm_param(litellm_params, guardrail, "criteria", ()) or ()
    if not criteria:
        raise ValueError("llm_as_a_judge guardrail requires at least one criterion")

    weight_total: Final = sum(float(c.get("weight", 0)) for c in criteria)
    if abs(weight_total - 100) > 0.5:
        raise ValueError(f"llm_as_a_judge criterion weights must sum to 100 (got {weight_total})")

    on_failure: Final[Literal["block", "log"]] = _get_litellm_param(litellm_params, guardrail, "on_failure", "block")
    if on_failure not in _VALID_ON_FAILURE:
        raise ValueError(f"llm_as_a_judge on_failure must be 'block' or 'log', got '{on_failure}'")

    overall_threshold: Final = float(_get_litellm_param(litellm_params, guardrail, "overall_threshold", 80.0))

    mode: Final[JudgeModeParam] = _get_litellm_param(litellm_params, guardrail, "mode", None)

    instance: Final = LLMAsAJudgeGuardrail(
        guardrail_name=guardrail_name,
        judge_model=judge_model,
        criteria=criteria,
        overall_threshold=overall_threshold,
        on_failure=on_failure,
        event_hook=mode,
        default_on=bool(_get_litellm_param(litellm_params, guardrail, "default_on", False)),
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
