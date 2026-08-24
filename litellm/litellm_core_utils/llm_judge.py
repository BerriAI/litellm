"""Shared primitives for LLM-judge features (llm_as_a_judge guardrail, shadow eval)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Final

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


def router_resolves_model(router: Router | None, model: str) -> bool:
    """Whether the model name resolves through the proxy's router (configured deployment
    or model-group alias), the same check the judge dispatch itself makes, so start-time
    validation cannot accept a name the call path then fails on."""
    return router is not None and bool(model in router.model_group_alias or router.get_model_list(model_name=model))


async def judge_acompletion(
    router: Router | None,
    judge_model: str,
    messages: list[AllMessageValues],  # mutable-ok: the SDK acompletion signature takes a list
    **params: object,
) -> ModelResponse:
    """Dispatch a judge call through the proxy's router when the judge model is a
    configured deployment (DB-stored credentials work), through the SDK for
    provider-qualified public names. The router path never retries or falls back:
    a failed judge call is the caller's counted failure, not a spend multiplier.
    Sampling preferences are advisory: models that removed sampling params (e.g.
    claude-sonnet-5) drop them instead of rejecting the judge call."""
    if router_resolves_model(router, judge_model):
        return await router.acompletion(  # pyright: ignore[reportOptionalMemberAccess]  # router_resolves_model implies router is not None
            model=judge_model,
            messages=messages,
            num_retries=0,
            fallbacks=[],
            drop_params=True,
            **params,
        )
    return await litellm.acompletion(model=judge_model, messages=messages, num_retries=0, drop_params=True, **params)
