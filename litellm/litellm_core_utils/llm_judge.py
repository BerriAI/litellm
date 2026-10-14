"""Shared primitives for LLM-judge features (llm_as_a_judge guardrail, shadow eval)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm import Router
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.utils import ModelResponse

JSON_FENCE_RE: Final = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


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
    return {str(k): v for k, v in parsed.items()}


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


def _object_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT_MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def _bucket_auth(request_data: Mapping[str, object], bucket_name: str) -> UserAPIKeyAuth | None:
    bucket: Final = _object_mapping(request_data.get(bucket_name))
    auth: Final = bucket.get("user_api_key_auth") if bucket is not None else None
    return auth if isinstance(auth, UserAPIKeyAuth) else None


def attached_auth(request_data: object) -> UserAPIKeyAuth | None:
    """The `UserAPIKeyAuth` the proxy attached to `request_data`'s metadata, or None.

    Only an instance counts. A `user_api_key_auth` dict or a plain `user_api_key_team_id`
    in the request is JSON the caller wrote, so neither is read.
    """
    data: Final = _object_mapping(request_data)
    if data is None:
        return None
    return next(
        (
            auth
            for bucket_name in ("metadata", "litellm_metadata")
            if (auth := _bucket_auth(data, bucket_name)) is not None
        ),
        None,
    )


def _named_route(router: Router, deployment_id: str) -> bool:
    deployment: Final = router.get_deployment(model_id=deployment_id)
    if deployment is None:
        return False
    public_names: Final = (deployment.model_name, deployment.model_info.team_public_model_name or "")
    return all("*" not in name for name in public_names)


def _served_without_a_team(router: Router, model: str) -> bool:
    """Whether a caller without a team reaches a model under `model`: the SDK knows the name,
    a deployment is listed under it, or a wildcard turns it into a name the SDK knows."""
    names: Final = (model, *router.resolved_litellm_models(model))
    return any(_provider_qualified(name) is not None for name in names) or any(
        _named_route(router, deployment_id) for deployment_id in router.get_candidate_model_ids_for_route(model)
    )


def caller_team_for(router: Router | None, model: str, auth: UserAPIKeyAuth | None) -> str | None:
    """The team a guardrail call for the caller behind `auth` resolves `model` under, or None.

    It is the caller's team only when a caller without a team reaches no model under
    `model`, and the router's route to `model` for that team reaches deployments that are
    each listed under a name rather than a wildcard pattern. A name a shared deployment or
    the SDK already serves keeps resolving exactly as it does without a team, so a team model
    given that name cannot take over the guardrail's calls.
    """
    team_id: Final = auth.team_id if auth is not None else None
    if team_id is None or router is None or _served_without_a_team(router, model):
        return None
    team_route: Final = router.get_candidate_model_ids_for_route(model, team_id)
    return team_id if team_route and all(_named_route(router, deployment_id) for deployment_id in team_route) else None


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
