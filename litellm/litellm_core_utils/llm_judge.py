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


def _callable_by_id(router: Router, deployment_id: str) -> bool:
    deployment: Final = router.get_deployment(model_id=deployment_id)
    return deployment is not None and "*" not in deployment.litellm_params.model


def caller_team_for(router: Router | None, model: str, auth: UserAPIKeyAuth | None) -> str | None:
    """The team a guardrail call for the caller behind `auth` resolves `model` under, or None.

    It is the caller's team only when the router's route to `model` for that team reaches a
    deployment a caller without a team cannot reach, which means the team has its own
    deployment behind the name, and every deployment on that route is a concrete model rather
    than a wildcard. Names every caller shares, and names only a team wildcard covers, keep
    resolving exactly as they do without a team.
    """
    team_id: Final = auth.team_id if auth is not None else None
    if team_id is None or router is None:
        return None
    team_route: Final = router.get_candidate_model_ids_for_route(model, team_id)
    if team_route <= router.get_candidate_model_ids_for_route(model):
        return None
    return team_id if all(_callable_by_id(router, deployment_id) for deployment_id in team_route) else None


def _field(value: object, key: str) -> object:
    mapping: Final = _object_mapping(value)
    return mapping.get(key) if mapping is not None else None


async def team_routed_model(router: Router, model: str, team_id: str | None) -> str:
    """`model`, or the id of the deployment the router picks for `team_id` on `model`.

    The guardrail call then reaches the team's deployment without carrying the team, so,
    like every other guardrail call, it is charged to no team and no key. A wildcard pick
    keeps `model`, since its id alone would send the pattern to the provider.
    """
    if team_id is None:
        return model
    picked: Final = _object_mapping(
        await router.async_get_available_deployment(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # untyped router return
            model=model, request_kwargs={"litellm_metadata": {"user_api_key_team_id": team_id}}
        )
    )
    deployment_id: Final = _field(_field(picked, "model_info"), "id")
    return deployment_id if isinstance(deployment_id, str) and _callable_by_id(router, deployment_id) else model


async def judge_acompletion(
    router: Router | None,
    judge_model: str,
    messages: list[AllMessageValues],  # mutable-ok: the SDK acompletion signature takes a list
    team_id: str | None = None,
    pin_team_deployment: bool = False,
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
    deployment and then dispatched as a public name the SDK has never heard of.

    With `pin_team_deployment`, a router call goes to the id of the deployment `team_id`'s
    route picks, so it reaches that team's model without carrying the team."""
    if router is not None and judge_target(router, judge_model, team_id).via == "router":
        return await router.acompletion(
            model=await team_routed_model(router, judge_model, team_id if pin_team_deployment else None),
            messages=messages,
            num_retries=0,
            fallbacks=[],
            drop_params=True,
            **params,
        )
    return await litellm.acompletion(model=judge_model, messages=messages, num_retries=0, drop_params=True, **params)
