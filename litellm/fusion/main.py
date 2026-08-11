"""
litellm.fusion — Multi-model Fusion with optional judge synthesis.

Sends the same prompt to N models in parallel. Without a judge model, returns
all panel responses as a list. With a judge model, synthesizes a final answer
and includes panel responses by default (set include_panel=False to suppress).

Inspired by OpenRouter Fusion / Sakana Fugu / Self-Consistency (Wang et al., 2022).

Usage:
    # Panel only — get all responses, pick yourself
    responses = litellm.fusion(
        models=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
        messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    )
    # returns: list[ModelResponse]

    # Judge synthesis — synthesized answer + panel included by default
    response = litellm.fusion(
        models=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
        judge_model="gpt-4o",
        messages=[{"role": "user", "content": "Explain quantum entanglement"}],
    )
    # returns: ModelResponse
    # response._hidden_params["fusion"]["panel_responses"]  ← individual responses

    # Judge synthesis — panel excluded
    response = litellm.fusion(
        models=["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"],
        judge_model="gpt-4o",
        messages=[{"role": "user", "content": "Explain quantum entanglement"}],
        include_panel=False,
    )
    # returns: ModelResponse  (no panel_responses in metadata)

    # Async variants
    responses = await litellm.afusion(models=[...], messages=[...])
    response  = await litellm.afusion(models=[...], judge_model="...", messages=[...])
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Literal, overload

import litellm
from litellm.types.utils import ModelResponse, Usage

# ---------------------------------------------------------------------------
# Judge prompt templates
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_SINGLE = """\
You received the following responses from {n} AI models for this user request:

{responses}

Your task: synthesize a single, comprehensive, and accurate final answer.
- Identify the points of agreement and the best reasoning across responses.
- Resolve contradictions by selecting the most well-supported position.
- Fill in gaps where some models provided information others missed.
- Write the final answer as if you generated it directly — no meta-commentary about the synthesis process.
"""

_JUDGE_PROMPT_MAJORITY = """\
You received the following responses from {n} AI models for this user request:

{responses}

Select the single best response. Output ONLY the text of the chosen response, verbatim.
"""

_JUDGE_PROMPT_BEST_OF_N = """\
You received the following responses from {n} AI models for this user request:

{responses}

Score each response on a scale of 1-10 for accuracy, completeness, and clarity.
Then output the highest-scored response verbatim.
"""

_JUDGE_PROMPTS: dict[str, str] = {
    "single_judge": _JUDGE_PROMPT_SINGLE,
    "majority_vote": _JUDGE_PROMPT_MAJORITY,
    "best_of_n": _JUDGE_PROMPT_BEST_OF_N,
}

FusionStrategy = Literal["single_judge", "majority_vote", "best_of_n"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_judge_messages(
    original_messages: list[dict],
    panel_responses: list[ModelResponse],
    panel_models: list[str],
    strategy: FusionStrategy,
) -> list[dict]:
    """Build the message list for the judge call."""
    user_request = next(
        (m["content"] for m in reversed(original_messages) if m.get("role") == "user"),
        "",
    )

    response_blocks = []
    for i, (model, resp) in enumerate(zip(panel_models, panel_responses), start=1):
        content = ""
        if resp and resp.choices:
            content = resp.choices[0].message.content or ""
        response_blocks.append(f"[Response {i} — {model}]\n{content}")

    responses_text = "\n\n".join(response_blocks)
    template = _JUDGE_PROMPTS[strategy]
    judge_system = template.format(n=len(panel_responses), responses=responses_text)

    messages: list[dict] = []

    # Preserve any existing system prompt
    for m in original_messages:
        if m.get("role") == "system":
            messages.append(m)
            break

    messages.append({"role": "system", "content": judge_system})
    messages.append({"role": "user", "content": user_request})
    return messages


def _sum_usage(responses: list[ModelResponse]) -> Usage:
    """Sum token usage across a list of ModelResponse objects."""
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for r in responses:
        if r and r.usage:
            total_prompt += r.usage.prompt_tokens or 0
            total_completion += r.usage.completion_tokens or 0
            total_tokens += r.usage.total_tokens or 0
    return Usage(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_tokens,
    )


def _merge_usage(
    panel_responses: list[ModelResponse],
    judge_response: ModelResponse,
) -> Usage:
    """Sum token usage across all panel calls + judge call."""
    return _sum_usage(panel_responses + [judge_response])


def _build_fusion_response(
    judge_response: ModelResponse,
    panel_responses: list[ModelResponse],
    panel_models: list[str],
    judge_model: str,
    include_panel: bool,
    original_model_tag: str,
) -> ModelResponse:
    """Wrap judge response in a ModelResponse tagged with fusion metadata."""
    merged_usage = _merge_usage(panel_responses, judge_response)

    response = ModelResponse(
        id=f"fusion-{uuid.uuid4().hex}",
        choices=judge_response.choices,
        created=int(time.time()),
        model=original_model_tag,
        usage=merged_usage,
        object="chat.completion",
    )

    fusion_meta: dict = {
        "panel_models": panel_models,
        "judge_model": judge_model,
    }
    if include_panel:
        fusion_meta["panel_responses"] = panel_responses

    response._hidden_params = {"fusion": fusion_meta}
    return response


async def _run_panel(
    models: list[str],
    messages: list,
    panel_kwargs: dict,
) -> tuple[list[ModelResponse], list[str]]:
    """Fan out to all panel models in parallel; filter failed calls."""
    tasks = [litellm.acompletion(model=m, messages=messages, stream=False, **panel_kwargs) for m in models]
    raw: list = list(await asyncio.gather(*tasks, return_exceptions=True))

    valid_responses: list[ModelResponse] = []
    valid_models: list[str] = []
    for model, resp in zip(models, raw):
        if isinstance(resp, Exception):
            litellm.utils.print_verbose(f"fusion: panel model {model!r} failed: {resp}")
        else:
            valid_responses.append(resp)
            valid_models.append(model)

    if not valid_responses:
        raise RuntimeError(f"fusion: all panel models failed. Errors: {raw}")

    return valid_responses, valid_models


# ---------------------------------------------------------------------------
# Overloads for precise return-type inference
# ---------------------------------------------------------------------------


@overload
async def afusion(
    models: list[str],
    messages: list,
    *,
    judge_model: None = ...,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,
) -> list[ModelResponse]: ...


@overload
async def afusion(
    models: list[str],
    messages: list,
    *,
    judge_model: str,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,
) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def afusion(
    models: list[str],
    messages: list,
    *,
    judge_model: str | None = None,
    strategy: FusionStrategy = "single_judge",
    include_panel: bool = True,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> list[ModelResponse] | ModelResponse:
    """
    Async fusion: call panel models in parallel, optionally synthesize with judge.

    Args:
        models: List of panel model identifiers (≥ 2 recommended).
        messages: OpenAI-compatible message list.
        judge_model: If provided, synthesizes a final answer from panel responses.
            If omitted, returns the raw list of panel responses.
        strategy: How the judge processes responses (only used when judge_model is set).
            - "single_judge": synthesize a new combined answer (default)
            - "majority_vote": pick the single best response verbatim
            - "best_of_n": score and select highest-scored response
        include_panel: When judge_model is set, whether to include individual panel
            responses in response._hidden_params["fusion"]["panel_responses"].
            Defaults to True. Ignored when judge_model is None.
        timeout: Per-call timeout in seconds (applied to panel + judge calls).
        temperature: Forwarded to panel models. Judge always uses temperature=0.
        max_tokens: Forwarded to panel models.
        **kwargs: Any additional litellm.acompletion kwargs forwarded to panel calls.

    Returns:
        - list[ModelResponse] when judge_model is None
        - ModelResponse (synthesized) when judge_model is provided
    """
    if not models:
        raise ValueError("fusion: `models` must be a non-empty list")
    if strategy not in _JUDGE_PROMPTS:
        raise ValueError(f"fusion: unknown strategy {strategy!r}. Choose from {list(_JUDGE_PROMPTS)}")

    panel_kwargs: dict = dict(kwargs)
    if temperature is not None:
        panel_kwargs["temperature"] = temperature
    if max_tokens is not None:
        panel_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        panel_kwargs["timeout"] = timeout

    # 1. Fan out to all panel models in parallel
    valid_responses, valid_models = await _run_panel(models, messages, panel_kwargs)

    # 2. No judge — return panel responses as-is
    if judge_model is None:
        return valid_responses

    # 3. Judge synthesis
    judge_messages = _build_judge_messages(
        original_messages=messages,
        panel_responses=valid_responses,
        panel_models=valid_models,
        strategy=strategy,
    )

    judge_kwargs: dict = {}
    if timeout is not None:
        judge_kwargs["timeout"] = timeout

    judge_response: ModelResponse = await litellm.acompletion(
        model=judge_model,
        messages=judge_messages,
        stream=False,
        temperature=0,  # deterministic synthesis
        **judge_kwargs,
    )

    # 4. Wrap into a single ModelResponse with merged metadata
    return _build_fusion_response(
        judge_response=judge_response,
        panel_responses=valid_responses,
        panel_models=valid_models,
        judge_model=judge_model,
        include_panel=include_panel,
        original_model_tag=f"fusion/{'+'.join(valid_models)}",
    )


# ---------------------------------------------------------------------------
# Sync overloads
# ---------------------------------------------------------------------------


@overload
def fusion(
    models: list[str],
    messages: list,
    *,
    judge_model: None = ...,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,
) -> list[ModelResponse]: ...


@overload
def fusion(
    models: list[str],
    messages: list,
    *,
    judge_model: str,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,
) -> ModelResponse: ...


def fusion(
    models: list[str],
    messages: list,
    *,
    judge_model: str | None = None,
    strategy: FusionStrategy = "single_judge",
    include_panel: bool = True,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> list[ModelResponse] | ModelResponse:
    """
    Sync fusion: call panel models in parallel, optionally synthesize with judge.

    Thin sync wrapper around :func:`afusion`. For async contexts, prefer
    :func:`afusion` directly.

    Args:
        models: List of panel model identifiers (≥ 2 recommended).
        messages: OpenAI-compatible message list.
        judge_model: If provided, synthesizes a final answer from panel responses.
            If omitted, returns the raw list of panel responses.
        strategy: Synthesis strategy ("single_judge", "majority_vote", "best_of_n").
            Only used when judge_model is provided.
        include_panel: When judge_model is set, whether to attach individual panel
            responses to response._hidden_params["fusion"]["panel_responses"].
            Defaults to True. Ignored when judge_model is None.
        timeout: Per-call timeout in seconds.
        temperature: Panel model temperature (judge always uses 0).
        max_tokens: Forwarded to panel models.
        **kwargs: Any additional litellm.completion kwargs.

    Returns:
        - list[ModelResponse] when judge_model is None
        - ModelResponse (synthesized) when judge_model is provided
    """
    return asyncio.run(
        afusion(
            models=models,
            messages=messages,
            judge_model=judge_model,
            strategy=strategy,
            include_panel=include_panel,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    )


__all__ = ["FusionStrategy", "afusion", "fusion"]
