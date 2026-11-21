"""Prototype endpoints behind the #what-if concepts: control canaries, cache-miss explainer, plan lab."""

import asyncio
import json
from collections.abc import Awaitable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from itertools import groupby, pairwise
from typing import Final, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import query_raw_within_statement_timeout

router: Final = APIRouter(prefix="/what_if", tags=["what-if"])

ROUTES: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
CACHE_TTL: Final = timedelta(minutes=5)


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
    guardrail_probes: Mapping[str, str] = Field(default_factory=dict)
    forbidden_model: str | None = None


class CanaryCell(BaseModel):
    route: str
    status_code: int
    enforced: bool
    detail: str


class CanaryResult(BaseModel):
    key_alias: str
    control: Literal["model_access", "rpm_limit", "guardrail"]
    configured: str
    probe: str
    cells: Sequence[CanaryCell]


class CanaryResponse(BaseModel):
    ran_at: datetime
    results: Sequence[CanaryResult]


async def _probe(
    client: httpx.AsyncClient, key: str, route: str, model: str, prompt: str, repeat: int = 1
) -> CanaryCell:
    responses: Final = [
        await client.post(route, json=dict(_body(route, model, prompt)), headers={"Authorization": f"Bearer {key}"})
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
        "/key/generate", json={**shadow, "duration": "5m", "key_alias": None}, headers=dict(admin_headers)
    )
    created.raise_for_status()
    key: Final = str(created.json()["key"])
    try:
        cells: Final = await asyncio.gather(*(_probe(client, key, route, model, prompt, repeat) for route in ROUTES))
    finally:
        await client.post("/key/delete", json={"keys": [key]}, headers=dict(admin_headers))
    return result.model_copy(update={"cells": cells})


@router.post("/canaries/run")
async def run_control_canaries(
    body: CanaryRequest, request: Request, user: UserAPIKeyAuth = Depends(user_api_key_auth)
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
    all_models: Final = tuple(llm_router.get_model_names()) if llm_router else ()
    admin_headers: Final = {"Authorization": request.headers.get("authorization", "")}
    jobs: Final[
        list[Awaitable[CanaryResult]]
    ] = []  # mutable-ok  # heterogeneous canaries collected per key before one gather
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=request.app), base_url="http://litellm-proxy", timeout=60
    ) as client:
        for row in keys:
            alias = str(row.key_alias)
            allowed = [m for m in row.models if m in all_models]
            guardrails = _as_list(_as_dict(row.metadata).get("guardrails"))
            if not allowed:
                continue
            base = {"models": row.models, "team_id": row.team_id}
            forbidden = body.forbidden_model or next((m for m in all_models if m not in row.models), None)
            if forbidden:
                jobs.append(
                    _run_canary(
                        client,
                        admin_headers,
                        CanaryResult(
                            key_alias=alias,
                            control="model_access",
                            configured=f"models = {', '.join(row.models)}",
                            probe=f"call {forbidden}",
                            cells=(),
                        ),
                        base,
                        forbidden,
                        "ping",
                    )
                )
            if row.rpm_limit is not None:
                jobs.append(
                    _run_canary(
                        client,
                        admin_headers,
                        CanaryResult(
                            key_alias=alias,
                            control="rpm_limit",
                            configured=f"rpm_limit = {row.rpm_limit}",
                            probe="shadow key at rpm_limit 1, second request in the same minute",
                            cells=(),
                        ),
                        {**base, "rpm_limit": 1},
                        allowed[0],
                        "ping",
                        repeat=2,
                    )
                )
            for name in guardrails:
                prompt = body.guardrail_probes.get(str(name))
                if prompt:
                    jobs.append(
                        _run_canary(
                            client,
                            admin_headers,
                            CanaryResult(
                                key_alias=alias,
                                control="guardrail",
                                configured=f"guardrail = {name}",
                                probe=f'send "{prompt}"',
                                cells=(),
                            ),
                            {**base, "guardrails": [name]},
                            allowed[0],
                            prompt,
                        )
                    )
        results: Final = await asyncio.gather(*jobs)
    return CanaryResponse(ran_at=datetime.now(timezone.utc), results=results)


class CacheTurn(BaseModel):
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
    verdict: Literal["hit", "client", "gateway", "provider", "first"]
    if not previous:
        verdict, reason = "first", "First request in the session, nothing to reuse yet"
    elif missed < block_size or cache_read >= 0.5 * min(previous_tokens, prompt_tokens):
        verdict, reason = "hit", f"{cache_read:,} tokens read from cache"
    elif reusable < 0.8 * min(previous_tokens, prompt_tokens):
        verdict = "client"
        reason = f"Prompt prefix changed at token ~{shared * block_size:,} of {previous_tokens:,}, so everything after it was re-billed"
    elif row.get("model_id") != previous.get("model_id"):
        verdict = "gateway"
        reason = (
            f"Prefix was byte-identical, but the router moved the session from "
            f"{previous.get('custom_llm_provider')}/{previous.get('model')} to {row.get('custom_llm_provider')}/{model}"
        )
    elif start - previous_start > CACHE_TTL:
        verdict = "provider"
        reason = f"Prefix was identical, but {int((start - previous_start).total_seconds() // 60)} min passed and the provider cache expired"
    else:
        verdict, reason = (
            "provider",
            "Prefix was identical and routing was sticky, the provider still reported no cache read",
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
async def explain_cache_misses(hours: int = 24, user: UserAPIKeyAuth = Depends(user_api_key_auth)) -> CacheMissResponse:
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
        turn
        for _, session in groupby(rows, key=lambda row: row.get("session_id"))
        for turn in (_explain(previous, row) for previous, row in pairwise(({}, *session)))
    )
    by_cause: Final = {
        cause: sum(turn.wasted_usd for turn in turns if turn.verdict == cause)
        for cause in ("client", "gateway", "provider")
    }
    return CacheMissResponse(
        turns=turns, wasted_usd_by_cause=by_cause, wasted_usd_per_day=sum(by_cause.values()) * 24 / max(hours, 1)
    )


class DraftPlan(BaseModel):
    days: int = 30
    monthly_price: float
    included_spend: float
    over_budget: Literal["block", "fallback", "allow"] = "block"
    fallback_model: str | None = None


class PlanCustomer(BaseModel):
    customer: str
    requests: int
    cost_today: float
    cost_under_plan: float
    margin: float
    blocked_requests: int
    rerouted_requests: int


class PlanReplay(BaseModel):
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
async def replay_draft_plan(plan: DraftPlan, user: UserAPIKeyAuth = Depends(user_api_key_auth)) -> PlanReplay:
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
