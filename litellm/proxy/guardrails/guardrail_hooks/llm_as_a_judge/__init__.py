"""LLM-as-a-Judge guardrail: uses an LLM to score responses against weighted criteria."""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import litellm
from fastapi import HTTPException

from litellm._logging import verbose_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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

_VALID_ON_FAILURE = frozenset({"block", "log"})


def _extract_text_from_content(content: Any) -> str:
    """Return plain text from a message content field (str or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


def _get_litellm_param(
    litellm_params: "LitellmParams",
    guardrail: "Guardrail",
    key: str,
    default: Any = None,
) -> Any:
    val = getattr(litellm_params, key, None)
    if val is not None:
        return val
    raw = guardrail.get("litellm_params")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    if raw is not None and not isinstance(raw, dict):
        attr = getattr(raw, key, None)
        if attr is not None:
            return attr
    return default


def _build_judge_prompt(
    criteria: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    response_text: str,
) -> str:
    criteria_block = "\n".join(
        f'- {c.get("name", "")} (weight {c.get("weight", 0)}%): {c.get("description", "")}'
        for c in criteria
    )
    conversation = "\n".join(
        f'{m.get("role", "user").upper()}: {_extract_text_from_content(m.get("content", ""))}'
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
        criteria: List[Dict[str, Any]],
        overall_threshold: float = 80.0,
        on_failure: Literal["block", "log"] = "block",
        event_hook: Optional[
            Union[GuardrailEventHooks, List[GuardrailEventHooks]]
        ] = None,
        default_on: bool = False,
        **kwargs: Any,
    ) -> None:
        _event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]] = (
            None
        )
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [
                    GuardrailEventHooks(h) if isinstance(h, str) else h
                    for h in event_hook
                ]
            else:
                _event_hook = (
                    GuardrailEventHooks(event_hook)
                    if isinstance(event_hook, str)
                    else event_hook
                )

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[GuardrailEventHooks.post_call],
            event_hook=_event_hook or GuardrailEventHooks.post_call,
            default_on=default_on,
            **kwargs,
        )
        self.judge_model = judge_model
        self.criteria = criteria
        self.overall_threshold = overall_threshold
        self.on_failure = on_failure

    async def _run_judge(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
    ) -> Dict[str, Any]:
        judge_messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_judge_prompt(self.criteria, messages, response_text),
            },
        ]
        response = await litellm.acompletion(
            model=self.judge_model,
            messages=judge_messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"  # type: ignore[union-attr]
        return json.loads(raw)

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

        texts = inputs.get("texts") or []
        response_text = " ".join(texts)
        if not response_text:
            return inputs

        start_time = datetime.now()
        status: GuardrailStatus = "success"
        judge_result: Dict[str, Any] = {}

        try:
            messages: List[Dict[str, Any]] = request_data.get("messages") or []

            try:
                judge_result = await self._run_judge(messages, response_text)
            except Exception as judge_err:
                verbose_logger.warning(
                    f"llm_as_a_judge guardrail: judge call failed, failing open. Error: {judge_err}"
                )
                status = "guardrail_failed_to_respond"
                return inputs

            try:
                overall_score = max(
                    0.0, min(100.0, float(judge_result.get("overall_score", 100)))
                )
            except (TypeError, ValueError):
                verbose_logger.warning(
                    "llm_as_a_judge: invalid overall_score from judge, failing open"
                )
                return inputs

            passed = overall_score >= self.overall_threshold

            eval_info: "StandardLoggingEvalInformation" = {
                "eval_name": self.guardrail_name or "",
                "overall_score": overall_score,
                "passed": passed,
                "judge_model": self.judge_model,
                "threshold": self.overall_threshold,
                "verdicts": judge_result.get("verdicts", []),
            }
            _metadata = request_data.setdefault("metadata", {})
            existing = _metadata.get("eval_information")
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
            verbose_logger.warning(f"llm_as_a_judge guardrail unexpected error: {e}")
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
    guardrail_name = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("llm_as_a_judge guardrail requires a guardrail_name")

    judge_model = _get_litellm_param(litellm_params, guardrail, "judge_model")
    if not judge_model:
        raise ValueError(
            "llm_as_a_judge guardrail requires judge_model in litellm_params"
        )

    criteria = _get_litellm_param(litellm_params, guardrail, "criteria") or []
    if not criteria:
        raise ValueError("llm_as_a_judge guardrail requires at least one criterion")

    weight_total = sum(float(c.get("weight", 0)) for c in criteria)
    if abs(weight_total - 100) > 0.5:
        raise ValueError(
            f"llm_as_a_judge criterion weights must sum to 100 (got {weight_total})"
        )

    on_failure = _get_litellm_param(litellm_params, guardrail, "on_failure", "block")
    if on_failure not in _VALID_ON_FAILURE:
        raise ValueError(
            f"llm_as_a_judge on_failure must be 'block' or 'log', got '{on_failure}'"
        )

    overall_threshold = float(
        _get_litellm_param(litellm_params, guardrail, "overall_threshold", 80.0)
    )

    mode = _get_litellm_param(litellm_params, guardrail, "mode")
    event_hook: Optional[GuardrailEventHooks] = None
    if isinstance(mode, str) and mode in {e.value for e in GuardrailEventHooks}:
        event_hook = GuardrailEventHooks(mode)

    instance = LLMAsAJudgeGuardrail(
        guardrail_name=guardrail_name,
        judge_model=judge_model,
        criteria=criteria,
        overall_threshold=overall_threshold,
        on_failure=on_failure,
        event_hook=event_hook,
        default_on=bool(
            _get_litellm_param(litellm_params, guardrail, "default_on", False)
        ),
    )
    litellm.logging_callback_manager.add_litellm_callback(instance)
    return instance


__all__ = [
    "LLMAsAJudgeGuardrail",
    "initialize_guardrail",
]
