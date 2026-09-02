"""Shared primitives for LLM-judge features (llm_as_a_judge guardrail, shadow eval)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

import litellm

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.utils import ModelResponse

JSON_FENCE_RE: Final = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def default_router_provider() -> Router | None:
    try:
        from litellm.proxy.proxy_server import llm_router
    except ImportError:
        return None

    return llm_router


def parse_json_verdict(raw: str) -> dict[str, object]:  # mutable-ok: plain parsed-JSON payload
    """Parse a judge's JSON verdict, tolerating markdown fences and surrounding prose."""
    text = raw.strip()  # rebind-ok: progressively narrowed to the JSON payload
    fenced: Final = JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group(1).strip()  # rebind-ok: progressively narrowed to the JSON payload
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start: Final = text.find("{")
        end: Final = text.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not a JSON object")
    return {str(k): v for k, v in parsed.items()}  # mutable-ok: plain parsed-JSON payload


def extract_text_from_content(content: object) -> str:
    """Return plain text from a message content field (str or multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


@lru_cache(maxsize=512)
def _provider_qualified(model: str) -> str | None:
    """`model` in the one spelling litellm itself resolves it to, or None if it maps to no
    provider.

    A deployment may be configured as `openai/gpt-4o` and a judge given as `gpt-4o`; both
    reach the same model, so an identity that keeps them apart reports two models where
    there is one. None is a different answer from "unchanged": a name that is already
    provider-qualified normalises to itself, and reading that as a failure would call every
    correctly-spelled public model unresolvable.
    """
    try:
        stripped, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:  # noqa: BLE001  # an unmapped name has no provider, which is the answer
        return None
    return f"{provider}/{stripped}" if provider and stripped else None


@dataclass(frozen=True, slots=True)
class JudgeTarget:
    """Where a call to one model name goes for one caller, and what answers it.

    The single answer to that question: the resolvability gate, the judge-vs-candidate
    gate and the dispatch all read it, so none of them can decide it differently. Splitting
    it is what let start-time validation accept a team's own model while dispatch sent the
    literal name to the SDK.
    """

    via: Literal["router", "sdk", "nothing"]
    models: frozenset[str]


def judge_target(router: Router | None, model: str, team_id: str | None = None) -> JudgeTarget:
    """Resolve `model` the way a call from `team_id` would be.

    Three outcomes and no others: the router serves it (a deployment, a team-public name,
    an alias, a routing group or a wildcard, exactly the channels `get_model_list`
    composes); the SDK serves it because litellm recognises the provider; or nothing does,
    which is the only case a caller may refuse on.

    `team_id` is part of the question, not a refinement of it. A team-public name resolves
    only for its own team and a team's own deployment resolves for nobody else, so asking
    without it answers for a caller who does not exist.
    """
    served: Final = router.resolved_litellm_models(model, team_id=team_id) if router is not None else ()
    if served:
        return JudgeTarget("router", frozenset(_provider_qualified(m) or m for m in served))
    qualified: Final = _provider_qualified(model)
    return JudgeTarget("sdk", frozenset({qualified})) if qualified is not None else JudgeTarget("nothing", frozenset())


async def judge_acompletion(
    router: Router | None,
    judge_model: str,
    messages: list[AllMessageValues],  # mutable-ok: the SDK acompletion signature takes a list
    team_id: str | None = None,
    **params: object,
) -> ModelResponse:
    """Dispatch a judge call through the proxy's router when the judge model is a
    configured deployment (DB-stored credentials work), through the SDK for
    provider-qualified public names. The router path never retries or falls back:
    a failed judge call is the caller's counted failure, not a spend multiplier.
    Sampling preferences are advisory: models that removed sampling params (e.g.
    claude-sonnet-5) drop them instead of rejecting the judge call.

    The arm is chosen by `judge_target` under the caller's own team, the same call
    start-time validation makes, so a judge a team can reach cannot be validated as a
    deployment and then dispatched as a public name the SDK has never heard of."""
    if judge_target(router, judge_model, team_id).via == "router":
        return await router.acompletion(  # pyright: ignore[reportOptionalMemberAccess]  # a router target implies router is not None
            model=judge_model,
            messages=messages,
            num_retries=0,
            fallbacks=[],
            drop_params=True,
            **params,
        )
    return await litellm.acompletion(model=judge_model, messages=messages, num_retries=0, drop_params=True, **params)
