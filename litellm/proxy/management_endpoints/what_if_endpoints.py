"""Prototype endpoints behind the #what-if concepts: control canaries, cache-miss explainer, plan lab."""

import asyncio
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from functools import reduce
from itertools import accumulate, chain, groupby, pairwise, takewhile
from typing import Annotated, Final, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm.llms.custom_httpx.asgi_handler import get_async_asgi_client
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import query_raw_within_statement_timeout

router: Final = APIRouter(prefix="/what_if", tags=["what-if"])

ROUTES: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
CACHE_TTL: Final = timedelta(minutes=5)
BASE_URL: Final = "http://litellm-proxy"


def _require_admin(user: UserAPIKeyAuth) -> None:
    if not user_api_key_has_admin_view(user):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can use what-if endpoints")


_DICT: Final = TypeAdapter(dict[str, object])
_LIST: Final = TypeAdapter(list[object])


def _as_dict(value: object) -> Mapping[str, object]:
    try:
        return _DICT.validate_json(value) if isinstance(value, str) else _DICT.validate_python(value)
    except ValidationError:
        return {}


def _as_list(value: object) -> Sequence[object]:
    try:
        return _LIST.validate_python(value)
    except ValidationError:
        return ()


def _prices(model: str) -> Mapping[str, object]:
    try:
        return _as_dict(dict(litellm.get_model_info(model)))
    except Exception:  # noqa: BLE001  # an unmapped model has no price, which the caller treats as zero
        return {}


def _body(route: str, model: str, prompt: str) -> Mapping[str, object]:
    if route == "/v1/responses":
        return {"model": model, "input": prompt, "max_output_tokens": 16}
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16}


class CanaryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    guardrail_probes: Mapping[str, str] = Field(default_factory=dict)
    forbidden_model: str | None = None


class CanaryCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    route: str
    status_code: int
    enforced: bool
    detail: str


class CanaryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    key_alias: str
    control: Literal["model_access", "rpm_limit", "guardrail"]
    configured: str
    probe: str
    cells: Sequence[CanaryCell]


class CanaryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ran_at: datetime
    results: Sequence[CanaryResult]


async def _probe(
    client: httpx.AsyncClient, key: str, route: str, model: str, prompt: str, repeat: int = 1
) -> CanaryCell:
    responses: Final = [
        await client.post(
            f"{BASE_URL}{route}", json=dict(_body(route, model, prompt)), headers={"Authorization": f"Bearer {key}"}
        )
        for _ in range(repeat)
    ]
    last: Final = responses[-1]
    enforced: Final = last.status_code >= 400
    detail: Final = last.text[:160] if enforced else "request was served"
    return CanaryCell(route=route, status_code=last.status_code, enforced=enforced, detail=detail)


async def _run_canary(
    client: httpx.AsyncClient,
    admin_headers: Mapping[str, str],
    result: CanaryResult,
    shadow: Mapping[str, object],
    model: str,
    prompt: str,
    repeat: int = 1,
) -> CanaryResult:
    created: Final = await client.post(
        f"{BASE_URL}/key/generate", json={**shadow, "duration": "5m", "key_alias": None}, headers=dict(admin_headers)
    )
    created.raise_for_status()
    key: Final = str(created.json()["key"])
    try:
        cells: Final = await asyncio.gather(*(_probe(client, key, route, model, prompt, repeat) for route in ROUTES))
    finally:
        await client.post(f"{BASE_URL}/key/delete", json={"keys": [key]}, headers=dict(admin_headers))
    return result.model_copy(update={"cells": cells})


def _canaries(
    client: httpx.AsyncClient,
    admin_headers: Mapping[str, str],
    body: CanaryRequest,
    all_models: Sequence[str],
    alias: str,
    models: Sequence[str],
    team_id: str | None,
    rpm_limit: int | None,
    guardrails: Sequence[object],
) -> Sequence[Awaitable[CanaryResult]]:
    allowed: Final = tuple(m for m in models if m in all_models)
    if not allowed:
        return ()
    base: Final = {"models": list(models), "team_id": team_id}
    forbidden: Final = body.forbidden_model or next((m for m in all_models if m not in models), None)
    model_access: Final = (
        (
            _run_canary(
                client,
                admin_headers,
                CanaryResult(
                    key_alias=alias,
                    control="model_access",
                    configured=f"models = {', '.join(models)}",
                    probe=f"call {forbidden}",
                    cells=(),
                ),
                base,
                forbidden,
                "ping",
            ),
        )
        if forbidden
        else ()
    )
    rpm: Final = (
        (
            _run_canary(
                client,
                admin_headers,
                CanaryResult(
                    key_alias=alias,
                    control="rpm_limit",
                    configured=f"rpm_limit = {rpm_limit}",
                    probe="shadow key at rpm_limit 1, second request in the same minute",
                    cells=(),
                ),
                {**base, "rpm_limit": 1},
                allowed[0],
                "ping",
                repeat=2,
            ),
        )
        if rpm_limit is not None
        else ()
    )
    guarded: Final = tuple(
        _run_canary(
            client,
            admin_headers,
            CanaryResult(
                key_alias=alias,
                control="guardrail",
                configured=f"guardrail = {name}",
                probe=f'send "{body.guardrail_probes[str(name)]}"',
                cells=(),
            ),
            {**base, "guardrails": [name]},
            allowed[0],
            body.guardrail_probes[str(name)],
        )
        for name in guardrails
        if body.guardrail_probes.get(str(name))
    )
    return (*model_access, *rpm, *guarded)


@router.post("/canaries/run")
async def run_control_canaries(
    body: CanaryRequest, request: Request, user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> CanaryResponse:
    """Fire a real probe at every configured control on every LLM route, using a 5 minute shadow key
    that copies the control from the real key, and report whether the control actually fired."""
    from litellm.proxy.proxy_server import llm_router, prisma_client

    _require_admin(user)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    keys: Final = await prisma_client.db.litellm_verificationtoken.find_many(
        where={"key_alias": {"not": None}}, take=50, order={"created_at": "asc"}
    )
    all_models: Final = tuple(llm_router.registry().names()) if llm_router else ()
    admin_headers: Final = {"Authorization": request.headers.get("authorization", "")}
    with get_async_asgi_client(request.app) as client:
        results: Final = await asyncio.gather(
            *chain.from_iterable(
                _canaries(
                    client,
                    admin_headers,
                    body,
                    all_models,
                    str(row.key_alias),
                    tuple(row.models),
                    row.team_id,
                    row.rpm_limit,
                    _as_list(_as_dict(row.metadata).get("guardrails")),
                )
                for row in keys
            )
        )
    return CanaryResponse(ran_at=datetime.now(timezone.utc), results=results)


class CacheTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    start_time: datetime
    key_alias: str
    session_id: str
    model: str
    provider: str
    prompt_tokens: int
    cache_read_tokens: int
    reusable_tokens: int
    missed_tokens: int
    wasted_usd: float
    verdict: Literal["hit", "client", "gateway", "provider", "first"]
    reason: str
    changed_text: str | None = None


class CacheMissResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: Sequence[CacheTurn]
    wasted_usd_by_cause: Mapping[str, float]
    wasted_usd_per_day: float


def _cache_read(usage: Mapping[str, object]) -> int:
    details: Final = _as_dict(usage.get("prompt_tokens_details"))
    return int(str(usage.get("cache_read_input_tokens") or details.get("cached_tokens") or 0))


def _prompt_text(row: Mapping[str, object]) -> str:
    request: Final = _as_dict(row.get("proxy_server_request"))
    messages: Final = request.get("messages") or request.get("input") or ()
    system: Final = request.get("system") or ""
    return json.dumps([system, messages], ensure_ascii=False)


def _changed_text(previous: str, current: str) -> str | None:
    if not previous.strip('[]" ,') or not current.strip('[]" ,'):
        return None
    offset: Final = next((i for i, (a, b) in enumerate(zip(previous, current)) if a != b), None)
    if offset is None:
        return None
    return f"...{previous[max(0, offset - 30) : offset + 40]}  ->  ...{current[max(0, offset - 30) : offset + 40]}"


def _verdict(
    previous: Mapping[str, object],
    row: Mapping[str, object],
    model: str,
    missed: int,
    block_size: int,
    cache_read: int,
    reusable: int,
    shared: int,
    previous_tokens: int,
    prompt_tokens: int,
    gap: timedelta,
) -> tuple[Literal["hit", "client", "gateway", "provider", "first"], str]:
    if not previous:
        return "first", "First request in the session, nothing to reuse yet"
    if missed < block_size or cache_read >= 0.5 * min(previous_tokens, prompt_tokens):
        return "hit", f"{cache_read:,} tokens read from cache"
    if reusable < 0.8 * min(previous_tokens, prompt_tokens):
        return (
            "client",
            f"Prompt prefix changed at token ~{shared * block_size:,} of {previous_tokens:,}, "
            "so everything after it was re-billed",
        )
    if row.get("model_id") != previous.get("model_id"):
        return (
            "gateway",
            "Prefix was byte-identical, but the router moved the session from "
            f"{previous.get('custom_llm_provider')}/{previous.get('model')} to {row.get('custom_llm_provider')}/{model}",
        )
    if gap > CACHE_TTL:
        return (
            "provider",
            f"Prefix was identical, but {int(gap.total_seconds() // 60)} min passed and the provider cache expired",
        )
    return "provider", "Prefix was identical and routing was sticky, the provider still reported no cache read"


def _explain(previous: Mapping[str, object], row: Mapping[str, object]) -> CacheTurn:
    metadata: Final = _as_dict(row.get("metadata"))
    spend_metadata: Final = _as_dict(metadata.get("spend_logs_metadata"))
    hashes: Final = _as_dict(spend_metadata.get("prefix_block_hashes"))
    previous_hashes: Final = _as_dict(
        _as_dict(_as_dict(previous.get("metadata")).get("spend_logs_metadata")).get("prefix_block_hashes")
    )
    block_size: Final = int(str(hashes.get("block_size") or 0))
    ids: Final = _as_list(hashes.get("hash_ids"))
    previous_ids: Final = _as_list(previous_hashes.get("hash_ids"))
    shared: Final = next(
        (i for i, (a, b) in enumerate(zip(previous_ids, ids)) if a != b), min(len(previous_ids), len(ids))
    )
    prompt_tokens: Final = int(str(row.get("prompt_tokens") or 0))
    cache_read: Final = _cache_read(_as_dict(metadata.get("usage_object")))
    reusable: Final = min(shared * block_size, prompt_tokens)
    previous_tokens: Final = int(str(previous.get("prompt_tokens") or 0))
    missed: Final = max(min(previous_tokens, prompt_tokens) - max(cache_read, 0), 0) if previous else 0
    model: Final = str(row.get("model") or "")
    prices: Final = _prices(model)
    input_price: Final = float(str(prices.get("input_cost_per_token") or 0))
    wasted: Final = missed * (input_price - float(str(prices.get("cache_read_input_token_cost") or input_price)))
    start: Final = (
        row["startTime"] if isinstance(row["startTime"], datetime) else datetime.fromisoformat(str(row["startTime"]))
    )
    previous_start: Final = (
        (
            previous["startTime"]
            if isinstance(previous["startTime"], datetime)
            else datetime.fromisoformat(str(previous["startTime"]))
        )
        if previous
        else start
    )
    changed: Final = _changed_text(_prompt_text(previous), _prompt_text(row)) if previous else None
    verdict, reason = _verdict(
        previous,
        row,
        model,
        missed,
        block_size,
        cache_read,
        reusable,
        shared,
        previous_tokens,
        prompt_tokens,
        start - previous_start,
    )
    return CacheTurn(
        request_id=str(row["request_id"]),
        start_time=start,
        key_alias=str(metadata.get("user_api_key_alias") or ""),
        session_id=str(row.get("session_id") or ""),
        model=model,
        provider=str(row.get("custom_llm_provider") or ""),
        prompt_tokens=prompt_tokens,
        cache_read_tokens=cache_read,
        reusable_tokens=reusable,
        missed_tokens=missed if verdict not in ("hit", "first") else 0,
        wasted_usd=wasted if verdict not in ("hit", "first") else 0.0,
        verdict=verdict,
        reason=reason,
        changed_text=changed if verdict == "client" else None,
    )


@router.get("/cache_miss")
async def explain_cache_misses(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], hours: int = 24
) -> CacheMissResponse:
    """For every turn of every session, say why the prompt cache was or was not read: the client changed
    the prefix, the gateway moved the session to another deployment, or the provider dropped it."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    since: Final = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows: Final = await query_raw_within_statement_timeout(
        prisma_client,
        'SELECT request_id, "startTime", session_id, model, model_id, custom_llm_provider, prompt_tokens, metadata, '
        'proxy_server_request FROM "LiteLLM_SpendLogs" WHERE "startTime" >= $1::timestamptz '
        "AND metadata #> '{spend_logs_metadata,prefix_block_hashes}' IS NOT NULL "
        'ORDER BY session_id, "startTime" LIMIT 2000',
        since,
    )
    turns: Final = tuple(
        chain.from_iterable(
            (_explain(previous, row) for previous, row in pairwise(({}, *session)))
            for _, session in groupby(rows, key=lambda row: row.get("session_id"))
        )
    )
    by_cause: Final = {
        cause: sum(turn.wasted_usd for turn in turns if turn.verdict == cause)
        for cause in ("client", "gateway", "provider")
    }
    return CacheMissResponse(
        turns=turns, wasted_usd_by_cause=by_cause, wasted_usd_per_day=sum(by_cause.values()) * 24 / max(hours, 1)
    )


class DraftPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    days: int = 30
    monthly_price: float
    included_spend: float
    over_budget: Literal["block", "fallback", "allow"] = "block"
    fallback_model: str | None = None


class PlanCustomer(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer: str
    requests: int
    cost_today: float
    cost_under_plan: float
    margin: float
    blocked_requests: int
    rerouted_requests: int


class PlanReplay(BaseModel):
    model_config = ConfigDict(frozen=True)

    customers: Sequence[PlanCustomer]
    requests_replayed: int
    revenue: float
    cost_today: float
    cost_under_plan: float
    margin: float
    margin_pct: float
    customers_losing_money: int
    blocked_requests: int
    rerouted_requests: int


def _replay_customer(customer: str, rows: Sequence[Mapping[str, object]], plan: DraftPlan) -> PlanCustomer:
    fallback: Final = _prices(plan.fallback_model or "")
    cost, blocked, rerouted = 0.0, 0, 0
    for row in rows:
        spend = float(str(row["spend"]))
        if cost + spend <= plan.included_spend or plan.over_budget == "allow":
            cost += spend
        elif plan.over_budget == "fallback" and fallback:
            cost += int(str(row["prompt_tokens"])) * float(str(fallback.get("input_cost_per_token") or 0)) + int(
                str(row["completion_tokens"])
            ) * float(str(fallback.get("output_cost_per_token") or 0))
            rerouted += 1
        else:
            blocked += 1
    return PlanCustomer(
        customer=customer,
        requests=len(rows),
        cost_today=sum(float(str(row["spend"])) for row in rows),
        cost_under_plan=cost,
        margin=plan.monthly_price * plan.days / 30 - cost,
        blocked_requests=blocked,
        rerouted_requests=rerouted,
    )


@router.post("/plan_lab/replay")
async def replay_draft_plan(plan: DraftPlan, user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> PlanReplay:
    """Replay the real spend logs of the last ``days`` against a draft plan and report margin per customer."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    since: Final = datetime.now(timezone.utc) - timedelta(days=plan.days)
    rows: Final = await query_raw_within_statement_timeout(
        prisma_client,
        'SELECT end_user, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" '
        "WHERE \"startTime\" >= $1::timestamptz AND end_user IS NOT NULL AND end_user <> '' "
        'ORDER BY end_user, "startTime" LIMIT 200000',
        since,
    )
    customers: Final = sorted(
        (
            _replay_customer(str(customer), tuple(group), plan)
            for customer, group in groupby(rows, key=lambda row: row["end_user"])
        ),
        key=lambda customer: customer.margin,
    )
    revenue: Final = plan.monthly_price * plan.days / 30 * len(customers)
    cost: Final = sum(customer.cost_under_plan for customer in customers)
    return PlanReplay(
        customers=customers,
        requests_replayed=len(rows),
        revenue=revenue,
        cost_today=sum(customer.cost_today for customer in customers),
        cost_under_plan=cost,
        margin=revenue - cost,
        margin_pct=(revenue - cost) / revenue * 100 if revenue else 0.0,
        customers_losing_money=sum(1 for customer in customers if customer.margin < 0),
        blocked_requests=sum(customer.blocked_requests for customer in customers),
        rerouted_requests=sum(customer.rerouted_requests for customer in customers),
    )


class LimitLayer(BaseModel):
    model_config = ConfigDict(frozen=True)

    scope: str
    name: str
    kind: Literal["budget", "rpm"]
    limit: float
    used: float
    remaining: float
    requests_left: int | None
    resets_at: str | None
    change_with: str


class LimitXray(BaseModel):
    model_config = ConfigDict(frozen=True)

    key_alias: str
    key_aliases: tuple[str, ...]
    avg_request_usd: float
    layers: tuple[LimitLayer, ...]
    unlimited: tuple[str, ...]


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _layers(
    scope: str, name: object, row: Mapping[str, object], used_rpm: float, avg_usd: float, change_with: str
) -> tuple[LimitLayer, ...]:
    budget: Final = _number(row.get("max_budget"))
    rpm: Final = _number(row.get("rpm_limit"))
    spend: Final = _number(row.get("spend")) or 0.0
    reset: Final = row.get("budget_reset_at")
    return (
        *(
            (
                LimitLayer(
                    scope=scope,
                    name=str(name or "-"),
                    kind="budget",
                    limit=budget,
                    used=spend,
                    remaining=max(budget - spend, 0.0),
                    requests_left=int(max(budget - spend, 0.0) / avg_usd) if avg_usd > 0 else None,
                    resets_at=str(reset) if reset else None,
                    change_with=f"{change_with} max_budget",
                ),
            )
            if budget is not None
            else ()
        ),
        *(
            (
                LimitLayer(
                    scope=scope,
                    name=str(name or "-"),
                    kind="rpm",
                    limit=rpm,
                    used=used_rpm,
                    remaining=max(rpm - used_rpm, 0.0),
                    requests_left=int(max(rpm - used_rpm, 0.0)),
                    resets_at=None,
                    change_with=f"{change_with} rpm_limit",
                ),
            )
            if rpm is not None
            else ()
        ),
    )


@router.get("/limit_xray")
async def xray_limits(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], key_alias: str = "") -> LimitXray:
    """Every budget and rate limit that applies to one key, across the key, its team, its team membership,
    its user and its organization, ordered by which one blocks the next request first."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin(user)
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)

    async def one(sql: str, *args: object) -> Mapping[str, object]:
        if any(arg is None for arg in args):
            return {}
        found: Final = await query_raw_within_statement_timeout(prisma_client, sql, *args)
        return found[0] if found else {}

    aliases: Final = tuple(
        str(row["key_alias"])
        for row in await query_raw_within_statement_timeout(
            prisma_client,
            'SELECT key_alias FROM "LiteLLM_VerificationToken" WHERE key_alias IS NOT NULL ORDER BY created_at DESC '
            "LIMIT 50",
        )
    )
    alias: Final = key_alias or (aliases[0] if aliases else "")
    key: Final = await one('SELECT * FROM "LiteLLM_VerificationToken" WHERE key_alias = $1 LIMIT 1', alias)
    if not key:
        raise HTTPException(status_code=404, detail=f"No key with alias {alias}")
    team: Final = await one('SELECT * FROM "LiteLLM_TeamTable" WHERE team_id = $1', key.get("team_id"))
    member: Final = await one(
        'SELECT m.spend, b.max_budget, b.rpm_limit, b.budget_reset_at FROM "LiteLLM_TeamMembership" m '
        'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id WHERE m.team_id = $1 AND m.user_id = $2',
        key.get("team_id"),
        key.get("user_id"),
    )
    owner: Final = await one('SELECT * FROM "LiteLLM_UserTable" WHERE user_id = $1', key.get("user_id"))
    org_id: Final = key.get("organization_id") or team.get("organization_id")
    org: Final = await one(
        "SELECT o.organization_alias, o.spend, b.max_budget, b.rpm_limit, b.budget_reset_at "
        'FROM "LiteLLM_OrganizationTable" o JOIN "LiteLLM_BudgetTable" b ON b.budget_id = o.budget_id '
        "WHERE o.organization_id = $1",
        org_id,
    )
    usage: Final = await one(
        "SELECT COALESCE(AVG(spend) FILTER (WHERE spend > 0), 0) AS avg_usd, "
        "COUNT(*) FILTER (WHERE \"startTime\" >= NOW() - INTERVAL '60 seconds') AS key_rpm "
        'FROM "LiteLLM_SpendLogs" WHERE api_key = $1 AND "startTime" >= NOW() - INTERVAL \'24 hours\'',
        key.get("token"),
    )
    team_usage: Final = await one(
        'SELECT COUNT(*) AS rpm FROM "LiteLLM_SpendLogs" WHERE team_id = $1 '
        "AND \"startTime\" >= NOW() - INTERVAL '60 seconds'",
        key.get("team_id"),
    )
    avg_usd: Final = _number(usage.get("avg_usd")) or 0.0
    key_rpm: Final = _number(usage.get("key_rpm")) or 0.0
    team_rpm: Final = _number(team_usage.get("rpm")) or 0.0
    scopes: Final = (
        ("Key", alias, key, key_rpm, "/key/update"),
        ("Team member", owner.get("user_email") or key.get("user_id"), member, key_rpm, "/team/member_update"),
        ("Team", team.get("team_alias"), team, team_rpm, "/team/update"),
        ("User", owner.get("user_email") or key.get("user_id"), owner, key_rpm, "/user/update"),
        ("Organization", org.get("organization_alias"), org, team_rpm, "/organization/update"),
        ("Proxy", "global", {"max_budget": litellm.max_budget or None, "spend": 0.0}, 0.0, "litellm_settings"),
    )
    layers: Final = tuple(
        chain.from_iterable(
            _layers(scope, name, row, used_rpm, avg_usd, change_with)
            for scope, name, row, used_rpm, change_with in scopes
        )
    )
    limited: Final = frozenset(layer.scope for layer in layers)
    return LimitXray(
        key_alias=alias,
        key_aliases=aliases,
        avg_request_usd=avg_usd,
        layers=tuple(
            sorted(layers, key=lambda layer: layer.requests_left if layer.requests_left is not None else 10**12)
        ),
        unlimited=tuple(scope[0] for scope in scopes if scope[0] not in limited),
    )


MAX_TRACE_MESSAGES: Final = 500
MAX_TRACE_TOTAL_MESSAGES: Final = 100_000
PR_URL: Final = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")


class GhostStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    start_time: str
    model: str
    spend: float
    role: str
    summary: str


class GhostRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    key_alias: str
    task: str
    models: tuple[str, ...]
    spend: float
    total_tokens: int
    tool_calls: int
    duration_seconds: float
    pr_urls: tuple[str, ...]
    steps: tuple[GhostStep, ...]


class GhostTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    loose_requests: int
    client_session_ids: int
    runs: tuple[GhostRun, ...]


class PrReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str
    repo: str
    number: int
    spend: float
    total_tokens: int
    requests: int
    tool_calls: int
    agent_minutes: float
    key_aliases: tuple[str, ...]
    models: tuple[str, ...]
    runs: tuple[str, ...]


class PrReceipts(BaseModel):
    model_config = ConfigDict(frozen=True)

    receipts: tuple[PrReceipt, ...]
    spend_with_pr: float
    spend_without_pr: float


def _messages(row: Mapping[str, object]) -> Sequence[object]:
    return _as_list(_as_dict(row.get("proxy_server_request")).get("messages"))


def _prefix_digests(key: object, messages: Sequence[object]) -> tuple[bytes, ...]:
    hashed: Final = (hashlib.sha256(json.dumps(m, sort_keys=True, default=str).encode()).digest() for m in messages)
    return tuple(accumulate(hashed, lambda seen, m: hashlib.sha256(seen + m).digest(), initial=str(key).encode()))[1:]


def _text(content: object) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _step(row: Mapping[str, object]) -> GhostStep:
    messages: Final = _messages(row)
    last: Final = _as_dict(messages[-1] if messages else None)
    return GhostStep(
        request_id=str(row.get("request_id")),
        start_time=str(row.get("startTime")),
        model=str(row.get("model")),
        spend=_number(row.get("spend")) or 0.0,
        role=str(last.get("role", "")),
        summary=_text(last.get("content"))[:220],
    )


def _run(rows: Sequence[Mapping[str, object]]) -> GhostRun:
    first: Final = rows[0]
    final_messages: Final = _messages(rows[-1])
    users: Final = tuple(_as_dict(m) for m in final_messages if _as_dict(m).get("role") == "user")
    transcript: Final = json.dumps(final_messages, default=str) + json.dumps(rows[-1].get("response"), default=str)
    started: Final = datetime.fromisoformat(str(first.get("startTime")).replace("Z", "+00:00"))
    ended: Final = datetime.fromisoformat(str(rows[-1].get("endTime")).replace("Z", "+00:00"))
    return GhostRun(
        run_id=str(first.get("request_id"))[-12:],
        key_alias=str(_as_dict(first.get("metadata")).get("user_api_key_alias") or "-"),
        task=_text(users[0].get("content"))[:160] if users else "",
        models=tuple(sorted({str(row.get("model")) for row in rows})),
        spend=sum(_number(row.get("spend")) or 0.0 for row in rows),
        total_tokens=sum(int(_number(row.get("total_tokens")) or 0) for row in rows),
        tool_calls=sum(1 for m in final_messages if _as_dict(m).get("role") == "tool"),
        duration_seconds=(ended - started).total_seconds(),
        pr_urls=tuple(sorted(set(PR_URL.findall(transcript)))),
        steps=tuple(_step(row) for row in rows),
    )


def _stitch(logged: Sequence[Mapping[str, object]]) -> GhostTrace:
    sized: Final = tuple(row for row in logged if 0 < len(_messages(row)) <= MAX_TRACE_MESSAGES)
    totals: Final = accumulate(len(_messages(row)) for row in sized)
    rows: Final = tuple(
        row for row, _ in takewhile(lambda pair: pair[1] <= MAX_TRACE_TOTAL_MESSAGES, zip(sized, totals, strict=False))
    )
    prefixes: Final = tuple(_prefix_digests(row.get("api_key"), _messages(row)) for row in rows)
    first_with: Final = {digests[-1]: index for index, digests in reversed(tuple(enumerate(prefixes)))}

    def parent(index: int) -> int | None:
        found: Final = (first_with.get(digest, index) for digest in reversed(prefixes[index][:-1]))
        return next((earlier for earlier in found if earlier < index), None)

    def with_root(roots: tuple[int, ...], index: int) -> tuple[int, ...]:
        above: Final = parent(index)
        return (*roots, index if above is None else roots[above])

    roots: Final = reduce(with_root, range(len(rows)), ())
    runs: Final = tuple(
        _run(tuple(row for row, owner in zip(rows, roots, strict=True) if owner == run_root))
        for run_root in dict.fromkeys(roots)
    )
    counts: Final = tuple(len(tuple(group)) for _, group in groupby(sorted(str(row.get("session_id")) for row in rows)))
    return GhostTrace(
        loose_requests=len(rows),
        client_session_ids=sum(1 for count in counts if count > 1),
        runs=tuple(sorted((run for run in runs if len(run.steps) > 1), key=lambda run: -run.spend)),
    )


async def _ghost_trace(hours: int) -> GhostTrace:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return await asyncio.to_thread(
        _stitch,
        await query_raw_within_statement_timeout(
            prisma_client,
            'SELECT request_id, "startTime", "endTime", session_id, api_key, model, spend, total_tokens, metadata, '
            'proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE "startTime" >= $1::timestamptz '
            'ORDER BY "startTime" LIMIT 3000',
            datetime.now(timezone.utc) - timedelta(hours=hours),
        ),
    )


@router.get("/ghost_trace")
async def ghost_trace(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], hours: int = 24) -> GhostTrace:
    """Stitch requests that carry no session id into agent runs: a request continues an earlier one from
    the same key when its messages start with that request's full message list."""
    _require_admin(user)
    return await _ghost_trace(hours)


@router.get("/pr_receipts")
async def pr_receipts(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], hours: int = 24) -> PrReceipts:
    """Attribute each stitched agent run to the GitHub pull requests its tool calls opened or pushed to,
    and total the spend per pull request."""
    _require_admin(user)
    runs: Final = (await _ghost_trace(hours)).runs
    shares: Final = sorted(
        chain.from_iterable(((url, run, 1 / len(run.pr_urls)) for url in run.pr_urls) for run in runs),
        key=lambda share: share[0],
    )
    receipts: Final = tuple(
        PrReceipt(
            url=url,
            repo="/".join(url.split("/")[3:5]),
            number=int(url.rsplit("/", 1)[1]),
            spend=sum(run.spend * part for _, run, part in group),
            total_tokens=int(sum(run.total_tokens * part for _, run, part in group)),
            requests=sum(len(run.steps) for _, run, _ in group),
            tool_calls=sum(run.tool_calls for _, run, _ in group),
            agent_minutes=sum(run.duration_seconds for _, run, _ in group) / 60,
            key_aliases=tuple(sorted({run.key_alias for _, run, _ in group})),
            models=tuple(sorted(set(chain.from_iterable(run.models for _, run, _ in group)))),
            runs=tuple(run.run_id for _, run, _ in group),
        )
        for url, group in ((url, tuple(group)) for url, group in groupby(shares, key=lambda share: share[0]))
    )
    return PrReceipts(
        receipts=tuple(sorted(receipts, key=lambda receipt: -receipt.spend)),
        spend_with_pr=sum(receipt.spend for receipt in receipts),
        spend_without_pr=sum(run.spend for run in runs if not run.pr_urls),
    )


class RecallRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: str
    model: str
    first_day: str
    last_day: str
    days: int
    requests: int
    prompt_tokens: int
    completion_tokens: int
    logged: float
    corrected: float
    delta: float


class BillRecall(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_days: int
    rows: tuple[RecallRow, ...]
    logged: float
    corrected: float
    delta: float
    days_checked: int


def _int(value: object) -> int:
    return int(_number(value) or 0)


def _at_todays_price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices: Final = _prices(model)
    return prompt_tokens * (_number(prices.get("input_cost_per_token")) or 0.0) + completion_tokens * (
        _number(prices.get("output_cost_per_token")) or 0.0
    )


_DAILY_TEAM_SQL: Final = (
    "SELECT COALESCE(t.team_alias, d.team_id, 'no team') AS team, d.model AS model, d.date AS day, "
    "SUM(d.prompt_tokens) AS prompt_tokens, SUM(d.completion_tokens) AS completion_tokens, "
    "SUM(d.api_requests) AS requests, SUM(d.spend) AS spend, MAX(t.max_budget) AS max_budget "
    'FROM "LiteLLM_DailyTeamSpend" d LEFT JOIN "LiteLLM_TeamTable" t ON t.team_id = d.team_id '
    "WHERE d.model IS NOT NULL AND d.date >= $1 GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"
)


async def _daily_team_rows(window_days: int) -> Sequence[Mapping[str, object]]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    since: Final = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime("%Y-%m-%d")
    return await query_raw_within_statement_timeout(prisma_client, _DAILY_TEAM_SQL, since)


def _recall_row(team: str, model: str, days: Sequence[Mapping[str, object]]) -> RecallRow | None:
    priced: Final = tuple(
        (day, _at_todays_price(model, _int(day.get("prompt_tokens")), _int(day.get("completion_tokens"))))
        for day in days
    )
    wrong: Final = tuple(
        (day, corrected)
        for day, corrected in priced
        if abs(corrected - (_number(day.get("spend")) or 0.0)) > max(0.01 * corrected, 1e-6)
    )
    if not wrong:
        return None
    logged: Final = sum(_number(day.get("spend")) or 0.0 for day, _ in wrong)
    corrected_total: Final = sum(corrected for _, corrected in wrong)
    return RecallRow(
        team=team,
        model=model,
        first_day=str(wrong[0][0].get("day")),
        last_day=str(wrong[-1][0].get("day")),
        days=len(wrong),
        requests=sum(_int(day.get("requests")) for day, _ in wrong),
        prompt_tokens=sum(_int(day.get("prompt_tokens")) for day, _ in wrong),
        completion_tokens=sum(_int(day.get("completion_tokens")) for day, _ in wrong),
        logged=logged,
        corrected=corrected_total,
        delta=corrected_total - logged,
    )


@router.get("/bill_recall")
async def bill_recall(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], window_days: int = 30) -> BillRecall:
    """Re-prices every stored team, model and day at today's cost map and returns the days whose logged spend
    is off by more than one percent, with what they were logged at and what they cost at the corrected price."""
    _require_admin(user)
    daily: Final = await _daily_team_rows(window_days)
    rows: Final = tuple(
        row
        for (team, model), days in groupby(daily, key=lambda day: (str(day.get("team")), str(day.get("model"))))
        if (row := _recall_row(team, model, tuple(days))) is not None
    )
    return BillRecall(
        window_days=window_days,
        rows=tuple(sorted(rows, key=lambda row: -abs(row.delta))),
        logged=sum(row.logged for row in rows),
        corrected=sum(row.corrected for row in rows),
        delta=sum(row.delta for row in rows),
        days_checked=len(daily),
    )


class SwapRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: str
    model: str
    successor: str
    retires: str | None
    prompt_tokens: int
    completion_tokens: int
    now: float
    after: float
    delta: float


class SwapTeam(BaseModel):
    model_config = ConfigDict(frozen=True)

    team: str
    budget: float | None
    now: float
    after: float
    budget_lasts_days_now: float | None
    budget_lasts_days_after: float | None


class SwapBill(BaseModel):
    model_config = ConfigDict(frozen=True)

    window_days: int
    rows: tuple[SwapRow, ...]
    teams: tuple[SwapTeam, ...]
    now: float
    after: float


def _swap_row(team: str, model: str, days: Sequence[Mapping[str, object]]) -> SwapRow | None:
    catalog: Final = _as_dict(litellm.model_cost.get(model, {}))
    successor: Final = catalog.get("successor_model")
    if not isinstance(successor, str) or not _prices(successor):
        return None
    prompt_tokens: Final = sum(_int(day.get("prompt_tokens")) for day in days)
    completion_tokens: Final = sum(_int(day.get("completion_tokens")) for day in days)
    now: Final = _at_todays_price(model, prompt_tokens, completion_tokens)
    after: Final = _at_todays_price(successor, prompt_tokens, completion_tokens)
    retires: Final = catalog.get("deprecation_date")
    return SwapRow(
        team=team,
        model=model,
        successor=successor,
        retires=str(retires) if retires else None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        now=now,
        after=after,
        delta=after - now,
    )


def _swap_team(team: str, days: Sequence[Mapping[str, object]], rows: Sequence[SwapRow], window_days: int) -> SwapTeam:
    now: Final = sum(_number(day.get("spend")) or 0.0 for day in days)
    after: Final = now + sum(row.delta for row in rows if row.team == team)
    budget: Final = next((found for day in days if (found := _number(day.get("max_budget"))) is not None), None)
    return SwapTeam(
        team=team,
        budget=budget,
        now=now,
        after=after,
        budget_lasts_days_now=budget / (now / window_days) if budget is not None and now > 0 else None,
        budget_lasts_days_after=budget / (after / window_days) if budget is not None and after > 0 else None,
    )


@router.get("/swap_bill")
async def swap_bill(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], window_days: int = 30) -> SwapBill:
    """Prices the stored traffic of every model the cost map marks with a successor at that successor's price,
    per team, and says how many days each team's budget lasts before and after the swap."""
    _require_admin(user)
    daily: Final = await _daily_team_rows(window_days)
    rows: Final = tuple(
        row
        for (team, model), days in groupby(daily, key=lambda day: (str(day.get("team")), str(day.get("model"))))
        if (row := _swap_row(team, model, tuple(days))) is not None
    )
    teams: Final = tuple(
        _swap_team(team, tuple(days), rows, window_days)
        for team, days in groupby(daily, key=lambda day: str(day.get("team")))
    )
    return SwapBill(
        window_days=window_days,
        rows=tuple(sorted(rows, key=lambda row: -row.delta)),
        teams=tuple(team for team in teams if any(row.team == team.team for row in rows)),
        now=sum(row.now for row in rows),
        after=sum(row.after for row in rows),
    )


class LocalPatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    added: int
    removed: int
    first_change: str
    found_in: str | None
    verdict: str


class PatchDrop(BaseModel):
    model_config = ConfigDict(frozen=True)

    tree: str
    installed: str
    checked_against: tuple[str, ...]
    patches: tuple[LocalPatch, ...]
    droppable: int


def _git(tree: str, *args: str) -> str | None:
    done: Final = subprocess.run(  # arguments are fixed git subcommands plus refs read from the tree itself
        ("git", "-C", tree, *args), capture_output=True, text=True, timeout=60, check=False
    )
    return done.stdout if done.returncode == 0 else None


Git = Callable[..., str | None]


def _already_in(git: Git, tree: str, ref: str, path: str, patch: str) -> bool:
    upstream: Final = git(tree, "show", f"{ref}:{path}")
    if upstream is None:
        return False
    upstream_lines: Final = frozenset(line.strip() for line in upstream.splitlines())
    added: Final = tuple(
        line[1:].strip() for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    return all(line in upstream_lines for line in added if line)


def _local_patch(git: Git, tree: str, path: str, base: str, refs: Sequence[tuple[str, str]]) -> LocalPatch:
    patch: Final = git(tree, "diff", base, "--", path) or ""
    changed: Final = tuple(
        line for line in patch.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )
    found: Final = next(((ref, verdict) for ref, verdict in refs if _already_in(git, tree, ref, path, patch)), None)
    return LocalPatch(
        path=path,
        added=sum(line.startswith("+") for line in changed),
        removed=sum(line.startswith("-") for line in changed),
        first_change=next((line[1:].strip() for line in changed if line.startswith("+") and line[1:].strip()), ""),
        found_in=found[0] if found else None,
        verdict=found[1] if found else "Only in your tree. No upstream release or branch has it.",
    )


def _patch_drop(tree: str, lts_branch: str, main_branch: str, git: Git = _git) -> PatchDrop:
    base: Final = (git(tree, "describe", "--tags", "--abbrev=0") or "").strip()
    if not base:
        raise HTTPException(status_code=404, detail=f"No release tag found for the tree at {tree}")
    newest_tag: Final = (git(tree, "tag", "--merged", lts_branch, "--sort=-v:refname") or "").split("\n")[0]
    refs: Final = (
        (newest_tag, f"Drop it when you upgrade to {newest_tag}."),
        (lts_branch, f"Merged on {lts_branch.removeprefix('origin/')}. Drop it at the next patch release."),
        (main_branch, "Fixed on main only. It will not reach your LTS line, keep carrying it."),
    )
    paths: Final = tuple(path for path in (git(tree, "diff", "--name-only", base) or "").splitlines() if path)
    patches: Final = tuple(_local_patch(git, tree, path, base, refs) for path in paths)
    return PatchDrop(
        tree=tree,
        installed=base,
        checked_against=tuple(ref for ref, _ in refs),
        patches=patches,
        droppable=sum(patch.found_in in (newest_tag, lts_branch) for patch in patches),
    )


@router.get("/patch_drop")
async def patch_drop(
    user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    lts_branch: str = "origin/lts/1.108.x",
    main_branch: str = "origin/main",
) -> PatchDrop:
    """Lists every file the tree at PATCH_DROP_TREE changed since its release tag and checks each local patch
    against the newest LTS tag, the LTS branch and main, by looking for its added lines in the upstream file."""
    _require_admin(user)
    tree: Final = os.environ.get("PATCH_DROP_TREE", "")
    if not tree:
        raise HTTPException(status_code=404, detail="Set PATCH_DROP_TREE to the checkout that carries local patches")
    return await asyncio.to_thread(_patch_drop, tree, lts_branch, main_branch)
