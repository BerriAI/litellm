"""
SDK primitive for LLM-judge evaluation.

Usage:
    result = await litellm.aevaluate(
        response=model_response,
        criteria=[{"name": "Accuracy", "weight": 100, "description": "Is the answer correct?"}],
        judge_model="claude-sonnet-4-6",
    )
    result.check_thresholds(overall_min=80)
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from litellm.types.utils import EvalCriterion, EvalResult, EvalVerdict


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Grade the agent response against each criterion "
    "and return ONLY valid JSON — no markdown, no backticks, no extra text.\n\n"
    "Response format:\n"
    '{"scores": [{"criterion": "criterion name", "score": 0-100, "comment": "brief reason"}], '
    '"overall": 0-100, "verdict": "2-3 sentence summary", '
    '"strongest": "criterion name", "weakest": "criterion name"}'
)


def _extract_response_text(response: Any) -> str:
    """Extract plain text from a ModelResponse or raw string."""
    if isinstance(response, str):
        return response
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        try:
            return str(response)
        except Exception:
            return ""


def _build_judge_messages(
    response_text: str,
    criteria: List[EvalCriterion],
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """Build the judge messages (system + user) for the LLM-judge call."""
    criteria_lines = "\n".join(
        f"{i+1}. {c.get('name', f'Criterion {i+1}')} ({c.get('weight', 0)}%): {c.get('description', '')}"
        for i, c in enumerate(criteria)
    )
    user_content = (
        f"Grade the following agent response against the rubric.\n\n"
        f"AGENT RESPONSE:\n{response_text}\n\n"
        f"RUBRIC (score each criterion 0-100):\n{criteria_lines}\n\n"
        f"Compute overall score as the weighted average: "
        f"sum(score * weight / 100) for each criterion."
    )
    return [
        {"role": "system", "content": system_prompt or _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def aevaluate(
    response: Any,
    criteria: List[EvalCriterion],
    judge_model: str,
    system_prompt: Optional[str] = None,
    iteration: int = 0,
    litellm_kwargs: Optional[Dict] = None,
) -> EvalResult:
    """
    Async SDK primitive: run an LLM-judge eval on a response.

    Args:
        response: ModelResponse or plain string to evaluate.
        criteria: List of EvalCriterion dicts (name, weight, description).
        judge_model: Model name to use as judge (e.g. "claude-sonnet-4-6").
        system_prompt: Override the default judge system prompt.
        iteration: Which retry iteration this is (0-indexed). Used for logging.
        litellm_kwargs: Extra kwargs forwarded to litellm.acompletion (e.g. api_base, api_key).

    Returns:
        EvalResult — always returns (never raises). Check .eval_error for judge failures.
    """
    import litellm

    response_text = _extract_response_text(response)
    messages = _build_judge_messages(response_text, criteria, system_prompt)

    try:
        judge_response = await litellm.acompletion(
            model=judge_model,
            messages=messages,
            response_format={"type": "json_object"},
            **(litellm_kwargs or {}),
        )
        raw_content = judge_response.choices[0].message.content or "{}"  # type: ignore[union-attr]
        raw_json: Dict = json.loads(raw_content)
    except Exception as e:
        # Fail open: judge error never blocks the response
        return EvalResult(
            overall_score=100.0,
            passed=True,
            judge_model=judge_model,
            iteration=iteration,
            eval_error=f"Judge call failed: {str(e)}",
        )

    try:
        scores_raw = raw_json.get("scores", [])
        verdicts: List[EvalVerdict] = []

        # Build a weight map for computing overall score
        weight_map: Dict[str, int] = {
            c.get("name", ""): c.get("weight", 0) for c in criteria
        }
        total_weight = sum(weight_map.values()) or 1

        weighted_sum = 0.0
        for s in scores_raw:
            cname = s.get("criterion", "")
            raw_score = s.get("score", 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            weight = weight_map.get(cname, 0)
            weighted_sum += score * (weight / total_weight)
            verdicts.append(
                EvalVerdict(
                    criterion_name=cname,
                    score=score,
                    reasoning=s.get("comment", ""),
                    passed=True,  # per-criterion pass/fail evaluated by caller if threshold set
                    weight=weight_map.get(cname, 0),
                )
            )

        overall_score = min(max(weighted_sum, 0.0), 100.0)

        return EvalResult(
            overall_score=overall_score,
            verdicts=verdicts,
            passed=True,  # no threshold applied at SDK level; caller gates via check_thresholds()
            judge_model=judge_model,
            iteration=iteration,
            raw_judge_response=raw_json,
        )

    except Exception as e:
        return EvalResult(
            overall_score=100.0,
            passed=True,
            judge_model=judge_model,
            iteration=iteration,
            eval_error=f"Verdict parsing failed: {str(e)}",
        )


def evaluate(
    response: Any,
    criteria: List[EvalCriterion],
    judge_model: str,
    system_prompt: Optional[str] = None,
    iteration: int = 0,
    litellm_kwargs: Optional[Dict] = None,
) -> EvalResult:
    """Sync wrapper for aevaluate(). See aevaluate() for full docs."""
    import concurrent.futures

    coro = aevaluate(
        response=response,
        criteria=criteria,
        judge_model=judge_model,
        system_prompt=system_prompt,
        iteration=iteration,
        litellm_kwargs=litellm_kwargs,
    )
    try:
        asyncio.get_running_loop()
        # Already inside a running event loop — run in a fresh thread to avoid deadlock.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)
