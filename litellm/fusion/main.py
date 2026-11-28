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
    # response._hidden_params["fusion"]["panel_responses"]  <- individual responses

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
from collections.abc import Sequence
from typing import Final, Literal, TypeAlias, overload

import litellm
from litellm.types.utils import ModelResponse, Usage

# ---------------------------------------------------------------------------
# Judge prompt templates
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_SINGLE: Final = (  # rebind-ok: module-level constant, never rebound
    "You received the following responses from {n} AI models for this user request:\n\n"
    "{responses}\n\n"
    "Your task: synthesize a single, comprehensive, and accurate final answer.\n"
    "- Identify the points of agreement and the best reasoning across responses.\n"
    "- Resolve contradictions by selecting the most well-supported position.\n"
    "- Fill in gaps where some models provided information others missed.\n"
    "- Write the final answer as if you generated it directly"
    " — no meta-commentary about the synthesis process.\n"
)

_JUDGE_PROMPT_MAJORITY: Final = (  # rebind-ok: module-level constant, never rebound
    "You received the following responses from {n} AI models for this user request:\n\n"
    "{responses}\n\n"
    "Select the single best response. Output ONLY the text of the chosen response, verbatim.\n"
)

_JUDGE_PROMPT_BEST_OF_N: Final = (  # rebind-ok: module-level constant, never rebound
    "You received the following responses from {n} AI models for this user request:\n\n"
    "{responses}\n\n"
    "Score each response on a scale of 1-10 for accuracy, completeness, and clarity.\n"
    "Then output the highest-scored response verbatim.\n"
)

_JUDGE_PROMPTS: Final = {  # mutable-ok: module-level lookup table, never replaced
    "single_judge": _JUDGE_PROMPT_SINGLE,
    "majority_vote": _JUDGE_PROMPT_MAJORITY,
    "best_of_n": _JUDGE_PROMPT_BEST_OF_N,
}

FusionStrategy: TypeAlias = Literal["single_judge", "majority_vote", "best_of_n"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_judge_messages(
    original_messages: Sequence[dict],  # mutable-ok: OpenAI message dicts, caller-owned
    panel_responses: Sequence[ModelResponse],  # mutable-ok: panel results, caller-owned
    panel_models: Sequence[str],  # mutable-ok: model names, caller-owned
    strategy: FusionStrategy,
) -> list[dict]:  # mutable-ok: message list consumed directly by litellm.acompletion
    """Build the message list for the judge call."""
    user_request: Final = next(
        (
            m["content"]
            for m in reversed(list(original_messages))  # mutable-ok: ephemeral list for reversed(), never stored
            if m.get("role") == "user"
        ),
        "",
    )

    response_blocks: list[str] = []  # mutable-ok: accumulated then joined, never escapes  # rebind-ok: never rebound
    for i, (model, resp) in enumerate(zip(panel_models, panel_responses), start=1):
        content = ""
        if resp and resp.choices:
            content = resp.choices[0].message.content or ""
        response_blocks.append(f"[Response {i} - {model}]\n{content}")

    responses_text: Final = "\n\n".join(response_blocks)
    template: Final = _JUDGE_PROMPTS[strategy]
    judge_system: Final = template.format(n=len(panel_responses), responses=responses_text)

    messages: list[dict] = []  # mutable-ok: accumulated then returned immediately  # rebind-ok: never rebound

    # Preserve any existing system prompt
    for m in original_messages:
        if m.get("role") == "system":
            messages.append(m)
            break

    messages.append({"role": "system", "content": judge_system})  # mutable-ok: dict consumed by litellm
    messages.append({"role": "user", "content": user_request})  # mutable-ok: dict consumed by litellm
    return messages


def _sum_usage(responses: Sequence[ModelResponse]) -> Usage:
    """Sum token usage across a list of ModelResponse objects."""
    total_prompt = 0  # rebind-ok: accumulator, augmented in loop
    total_completion = 0  # rebind-ok: accumulator, augmented in loop
    total_tokens = 0  # rebind-ok: accumulator, augmented in loop
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
    panel_responses: Sequence[ModelResponse],
    judge_response: ModelResponse,
) -> Usage:
    """Sum token usage across all panel calls + judge call."""
    return _sum_usage(
        (*panel_responses, judge_response)
    )  # mutable-ok: ephemeral tuple, passed immediately to _sum_usage


def _build_fusion_response(
    judge_response: ModelResponse,
    panel_responses: Sequence[ModelResponse],  # mutable-ok: stored in metadata, caller-owned
    panel_models: Sequence[str],  # mutable-ok: stored in metadata, caller-owned
    judge_model: str,
    include_panel: bool,
    original_model_tag: str,
) -> ModelResponse:
    """Wrap judge response in a ModelResponse tagged with fusion metadata."""
    merged_usage: Final = _merge_usage(panel_responses, judge_response)

    response: Final = ModelResponse(
        id=f"fusion-{uuid.uuid4().hex}",
        choices=judge_response.choices,
        created=int(time.time()),
        model=original_model_tag,
        usage=merged_usage,
        object="chat.completion",
    )

    fusion_meta: dict = {  # mutable-ok: conditionally extended before being stored  # rebind-ok: never rebound
        "panel_models": panel_models,
        "judge_model": judge_model,
    }
    if include_panel:
        fusion_meta["panel_responses"] = panel_responses  # mutable-ok: stored in _hidden_params

    response._hidden_params = {"fusion": fusion_meta}  # mutable-ok: ModelResponse._hidden_params is the SDK extension point  # fmt: skip
    return response


async def _run_panel(
    models: Sequence[str],  # mutable-ok: iterated read-only
    messages: Sequence[dict],  # mutable-ok: forwarded to litellm.acompletion
    panel_kwargs: dict,  # mutable-ok: forwarded kwargs dict, caller-owned
) -> tuple[list[ModelResponse], list[str]]:  # mutable-ok: results consumed immediately by afusion
    """Fan out to all panel models in parallel; filter failed calls."""
    tasks: Final = [  # mutable-ok: list of coroutines passed to asyncio.gather
        litellm.acompletion(
            model=m,
            messages=list(messages),  # mutable-ok: ephemeral list copy for acompletion, never stored
            stream=False,
            **panel_kwargs,  # kwargs-ok: forwarded to litellm.acompletion, varies per call
        )
        for m in models
    ]
    raw: Final = list(await asyncio.gather(*tasks, return_exceptions=True))  # mutable-ok: gather results consumed below

    valid_responses: list[ModelResponse] = []  # mutable-ok: accumulated then returned  # rebind-ok: never rebound
    valid_models: list[str] = []  # mutable-ok: accumulated then returned  # rebind-ok: never rebound
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
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: None = ...,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> list[ModelResponse]: ...  # mutable-ok: panel responses list


@overload
async def afusion(
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: str,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Core async implementation
# ---------------------------------------------------------------------------


async def afusion(
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: str | None = None,
    strategy: FusionStrategy = "single_judge",
    include_panel: bool = True,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> list[ModelResponse] | ModelResponse:  # mutable-ok: list[ModelResponse] is the panel-only branch
    """
    Async fusion: call panel models in parallel, optionally synthesize with judge.

    Args:
        models: List of panel model identifiers (>= 2 recommended).
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
        raise ValueError(
            f"fusion: unknown strategy {strategy!r}. Choose from {list(_JUDGE_PROMPTS)}"  # mutable-ok: ephemeral list for error message
        )

    panel_kwargs: dict = dict(kwargs)  # mutable-ok: working copy of kwargs, extended below  # rebind-ok: never rebound
    if temperature is not None:
        panel_kwargs["temperature"] = temperature
    if max_tokens is not None:
        panel_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        panel_kwargs["timeout"] = timeout

    # 1. Fan out to all panel models in parallel
    valid_responses, valid_models = await _run_panel(
        list(models),  # mutable-ok: ephemeral list, converts Sequence for _run_panel
        list(messages),  # mutable-ok: ephemeral list, converts Sequence for _run_panel
        panel_kwargs,
    )

    # 2. No judge — return panel responses as-is
    if judge_model is None:
        return valid_responses

    # 3. Judge synthesis
    judge_messages: Final = _build_judge_messages(
        original_messages=messages,
        panel_responses=valid_responses,
        panel_models=valid_models,
        strategy=strategy,
    )

    judge_kwargs: dict = {}  # mutable-ok: conditionally extended before use  # rebind-ok: never rebound
    if timeout is not None:
        judge_kwargs["timeout"] = timeout

    judge_response: Final = await litellm.acompletion(
        model=judge_model,
        messages=judge_messages,
        stream=False,
        temperature=0,  # deterministic synthesis
        **judge_kwargs,  # kwargs-ok: forwarded to litellm.acompletion
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
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: None = ...,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> list[ModelResponse]: ...  # mutable-ok: panel responses list


@overload
def fusion(
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: str,
    strategy: FusionStrategy = ...,
    include_panel: bool = ...,
    timeout: float | None = ...,
    temperature: float | None = ...,
    max_tokens: int | None = ...,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> ModelResponse: ...


def fusion(
    models: Sequence[str],  # mutable-ok: OpenAI-compatible, matches litellm API
    messages: Sequence[dict],  # mutable-ok: OpenAI-compatible message list
    *,
    judge_model: str | None = None,
    strategy: FusionStrategy = "single_judge",
    include_panel: bool = True,
    timeout: float | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: object,  # kwargs-ok: forwarded to litellm.acompletion, varies per provider
) -> list[ModelResponse] | ModelResponse:  # mutable-ok: list[ModelResponse] is the panel-only branch
    """
    Sync fusion: call panel models in parallel, optionally synthesize with judge.

    Thin sync wrapper around :func:`afusion`. For async contexts, prefer
    :func:`afusion` directly.

    Args:
        models: List of panel model identifiers (>= 2 recommended).
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
            **kwargs,  # kwargs-ok: forwarded to afusion
        )
    )


__all__ = ["FusionStrategy", "afusion", "fusion"]  # mutable-ok: module __all__, Python convention
