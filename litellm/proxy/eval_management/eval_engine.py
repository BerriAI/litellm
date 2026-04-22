"""
Runtime eval engine for the LiteLLM proxy.
Fires after agent responses, gates bad outputs (hard-gate) or logs them (soft-observer).
"""

import json
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.eval_utils import aevaluate
from litellm.types.utils import EvalCriterion, StandardLoggingEvalInformation


async def run_evals_for_agent(
    agent_id: str,
    response: Any,
    data: Dict[str, Any],
    user_api_key_dict: Any,
    prisma_client: Any,
) -> Any:
    """
    Run all evals attached to an agent against its response.

    - Hard-gate (on_failure="block"): raises HTTPException(422) if eval fails.
    - Soft-observer (on_failure="log"): logs only, never blocks.
    - Fail-open: judge errors never block.

    Returns the response unchanged (gating happens via exception).
    """
    from litellm.proxy.eval_management.eval_registry import get_evals_for_agent

    try:
        attached_evals = await get_evals_for_agent(agent_id, prisma_client)
    except Exception as e:
        verbose_proxy_logger.warning(
            f"[Evals] Failed to load evals for agent {agent_id}: {e}"
        )
        return response

    if not attached_evals:
        return response

    eval_information: List[StandardLoggingEvalInformation] = []

    for attachment in attached_evals:
        eval_row = attachment.get("eval") or {}
        on_failure = attachment.get("on_failure", "block")
        threshold = (
            attachment.get("overall_threshold_override")
            or eval_row.get("overall_threshold")
            or 0.0
        )
        max_iterations = eval_row.get("max_iterations", 1)
        criteria_raw = eval_row.get("criteria", [])

        # Deserialize criteria if stored as JSON string
        if isinstance(criteria_raw, str):
            try:
                criteria_raw = json.loads(criteria_raw)
            except Exception:
                criteria_raw = []

        criteria: List[EvalCriterion] = [EvalCriterion(**c) if not isinstance(c, dict) else c for c in criteria_raw]  # type: ignore[misc]
        judge_model = eval_row.get("judge_model", "")

        if not judge_model or not criteria:
            verbose_proxy_logger.warning(
                f"[Evals] Skipping eval {attachment.get('eval_id')} — missing judge_model or criteria"
            )
            continue

        eval_result = await _run_single_eval_with_retry(
            response=response,
            criteria=criteria,
            judge_model=judge_model,
            max_iterations=max_iterations,
            data=data,
        )

        # Determine pass/fail against threshold
        passed = eval_result.overall_score >= threshold

        # Log eval info into request metadata for spend logs
        eval_info: StandardLoggingEvalInformation = {
            "eval_id": attachment.get("eval_id"),
            "eval_name": attachment.get("eval_name", ""),
            "overall_score": eval_result.overall_score,
            "passed": passed,
            "judge_model": judge_model,
            "iteration": eval_result.iteration,
            "eval_error": eval_result.eval_error,
        }
        eval_information.append(eval_info)

        # Write to request metadata so spend logs pick it up
        metadata = data.get("metadata") or {}
        existing = metadata.get("eval_information", [])
        metadata["eval_information"] = existing + [eval_info]
        data["metadata"] = metadata

        verbose_proxy_logger.debug(
            f"[Evals] eval={attachment.get('eval_name')} score={eval_result.overall_score:.1f} "
            f"threshold={threshold} passed={passed} on_failure={on_failure}"
        )

        if not passed and on_failure == "block":
            from fastapi import HTTPException

            raise HTTPException(
                status_code=422,
                detail={
                    "error": "eval_failed",
                    "eval_name": attachment.get("eval_name", ""),
                    "overall_score": eval_result.overall_score,
                    "threshold": threshold,
                    "verdict": (
                        eval_result.raw_judge_response.get("verdict", "")
                        if eval_result.raw_judge_response
                        else ""
                    ),
                    "scores": [
                        {
                            "criterion": v.get("criterion_name"),
                            "score": v.get("score"),
                            "comment": v.get("reasoning"),
                        }
                        for v in (eval_result.verdicts or [])
                    ],
                    "weakest": (
                        eval_result.raw_judge_response.get("weakest", "")
                        if eval_result.raw_judge_response
                        else ""
                    ),
                    "iteration": eval_result.iteration,
                    "eval_error": eval_result.eval_error,
                },
            )

    return response


async def _run_single_eval_with_retry(
    response: Any,
    criteria: List[EvalCriterion],
    judge_model: str,
    max_iterations: int,
    data: Dict[str, Any],
    iteration: int = 0,
) -> Any:
    """Run aevaluate with retry-on-fail up to max_iterations."""
    eval_result = await aevaluate(
        response=response,
        criteria=criteria,
        judge_model=judge_model,
        iteration=iteration,
    )

    # If eval_error is set the judge failed — return as-is (fail-open)
    if eval_result.eval_error:
        return eval_result

    # No retry configured or already at max — return result
    if max_iterations <= 1 or iteration >= max_iterations - 1:
        return eval_result

    # Determine threshold from eval result to check if retry needed
    # (threshold check happens in caller; here we retry if score < 100 and iterations remain)
    # Inject judge feedback into conversation for retry
    feedback_msg = _build_feedback_message(eval_result)
    messages = data.get("messages", [])
    if messages and feedback_msg:
        data["messages"] = messages + [{"role": "user", "content": feedback_msg}]

        # Re-invoke the agent's LLM with the feedback
        try:
            import litellm

            model = data.get("model", "")
            if model:
                retry_response = await litellm.acompletion(
                    model=model,
                    messages=data["messages"],
                )
                return await _run_single_eval_with_retry(
                    response=retry_response,
                    criteria=criteria,
                    judge_model=judge_model,
                    max_iterations=max_iterations,
                    data=data,
                    iteration=iteration + 1,
                )
        except Exception as e:
            verbose_proxy_logger.warning(
                f"[Evals] Retry attempt {iteration + 1} failed: {e}"
            )

    return eval_result


def _build_feedback_message(eval_result: Any) -> str:
    """Build judge feedback message to inject as user message for retry."""
    if not eval_result.raw_judge_response:
        return ""
    verdict = eval_result.raw_judge_response.get("verdict", "")
    weakest = eval_result.raw_judge_response.get("weakest", "")
    scores = eval_result.raw_judge_response.get("scores", [])
    weakest_comment = next(
        (s.get("comment", "") for s in scores if s.get("criterion") == weakest), ""
    )
    parts = [f"Your previous response scored {eval_result.overall_score:.0f}/100."]
    if verdict:
        parts.append(f"Feedback: {verdict}")
    if weakest and weakest_comment:
        parts.append(f"Weakest area — {weakest}: {weakest_comment}")
    parts.append("Please improve your response.")
    return " ".join(parts)
