"""Prototype endpoints behind the #what-if concepts: lost fields, key split, tape check."""

import asyncio
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from itertools import chain, groupby
from typing import Annotated, Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth, user_api_key_has_admin_view
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import query_raw_within_statement_timeout
from litellm.types.utils import CallTypes

router: Final = APIRouter(prefix="/what_if", tags=["what-if"])

_DICT: Final = TypeAdapter(dict[str, object])
_LIST: Final = TypeAdapter(list[object])
_NOT_PARAMS: Final = frozenset(
    {"model", "messages", "metadata", "litellm_metadata", "litellm_call_id", "stream", "user"}
)
_ROWS_SQL: Final = """
SELECT s.request_id, s.api_key, s.spend, s.status, s.proxy_server_request, v.key_alias
FROM "LiteLLM_SpendLogs" s LEFT JOIN "LiteLLM_VerificationToken" v ON v.token = s.api_key
WHERE s."startTime" >= $1::timestamp AND s.call_type IN ('acompletion', 'completion')
ORDER BY s.api_key, s."startTime" DESC LIMIT 5000
"""

Transform: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


def _require_admin(user: UserAPIKeyAuth) -> None:
    if not user_api_key_has_admin_view(user):
        raise HTTPException(status_code=403, detail="Only proxy admin roles can use what-if endpoints")


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


def _short(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)[:80]


async def _logged_requests(hours: int) -> Sequence[Mapping[str, object]]:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    since: Final = datetime.now(timezone.utc) - timedelta(hours=hours)
    return await query_raw_within_statement_timeout(prisma_client, _ROWS_SQL, since.replace(tzinfo=None).isoformat())


class FieldFate(_Frozen):
    field: str
    fate: Literal["dropped", "rewritten"]
    became: tuple[str, ...]
    sent_value: str


class LostField(_Frozen):
    model: str
    field: str
    fate: Literal["dropped", "rewritten"]
    became: tuple[str, ...]
    sent_value: str
    requests: int
    example_request_id: str


class LostFields(_Frozen):
    window_hours: int
    requests_scanned: int
    requests_with_dropped_fields: int
    fields: tuple[LostField, ...]


def _provider_request(body: Mapping[str, object]) -> Mapping[str, object]:
    from litellm.utils import return_raw_request

    raw: Final = dict[str, object](return_raw_request(CallTypes.completion, body))
    return _as_dict(raw.get("raw_request_body"))


def _client_params(body: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(key for key in body if key not in _NOT_PARAMS))


def _fate(body: Mapping[str, object], sent: Mapping[str, object], field: str, transform: Transform) -> FieldFate | None:
    if field in sent:
        return None
    without: Final = transform({key: value for key, value in body.items() if key != field})
    became: Final = tuple(sorted(key for key in sent.keys() | without.keys() if sent.get(key) != without.get(key)))
    return FieldFate(
        field=field,
        fate="rewritten" if became else "dropped",
        became=became,
        sent_value=_short(body[field]),
    )


def field_fates(body: Mapping[str, object], transform: Transform) -> tuple[FieldFate, ...]:
    """What the provider request looks like with and without each client field tells whether the field had any effect."""
    sent: Final = transform(body)
    if not sent:
        return ()
    fates: Final = (_fate(body, sent, field, transform) for field in _client_params(body))
    return tuple(fate for fate in fates if fate is not None)


def _replayable(row: Mapping[str, object]) -> Mapping[str, object]:
    body: Final = _as_dict(row.get("proxy_server_request"))
    deployment: Final = _as_dict(body.get("metadata")).get("deployment")
    if not isinstance(deployment, str) or not body.get("messages"):
        return {}
    return {**{key: value for key, value in body.items() if key != "metadata"}, "model": deployment}


def _shape(body: Mapping[str, object]) -> str:
    return json.dumps({key: value for key, value in body.items() if key != "messages"}, sort_keys=True, default=str)


def _seen(
    group: Sequence[tuple[str, str, Mapping[str, object]]], fates: Sequence[FieldFate]
) -> tuple[tuple[str, FieldFate, int, str], ...]:
    return tuple((str(group[0][2]["model"]), fate, len(group), group[0][1]) for fate in fates)


def lost_fields(rows: Sequence[Mapping[str, object]], hours: int, transform: Transform) -> LostFields:
    bodies: Final = tuple((str(row.get("request_id")), _replayable(row)) for row in rows)
    usable: Final = sorted(
        ((_shape(body), request_id, body) for request_id, body in bodies if body), key=lambda b: b[0]
    )
    shapes: Final = tuple(tuple(group) for _, group in groupby(usable, key=lambda b: b[0]))
    fated: Final = tuple((group, field_fates(group[0][2], transform)) for group in shapes)
    seen: Final = tuple(chain.from_iterable(_seen(group, fates) for group, fates in fated))
    fields: Final = tuple(
        LostField(
            model=model,
            **fate.model_dump(),
            requests=sum(count for m, f, count, _ in seen if (m, f) == (model, fate)),
            example_request_id=next(request_id for m, f, _, request_id in seen if (m, f) == (model, fate)),
        )
        for model, fate in dict.fromkeys((m, f) for m, f, _, _ in seen)
    )
    return LostFields(
        window_hours=hours,
        requests_scanned=len(usable),
        requests_with_dropped_fields=sum(
            len(group) for group, fates in fated if any(f.fate == "dropped" for f in fates)
        ),
        fields=tuple(sorted(fields, key=lambda f: (f.fate != "dropped", -f.requests, f.model, f.field))),
    )


@router.get("/lost_fields")
async def get_lost_fields(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], hours: int = 24) -> LostFields:
    """Replay the provider translation of logged requests and report client fields that never reached the provider."""
    _require_admin(user)
    rows: Final = await _logged_requests(hours)
    return await asyncio.to_thread(lost_fields, rows, hours, _provider_request)


class InferredAgent(_Frozen):
    fingerprint: str
    suggested_name: str
    system_prompt: str
    tools: tuple[str, ...]
    models: tuple[str, ...]
    user_agents: tuple[str, ...]
    requests: int
    failures: int
    spend: float


class SharedKey(_Frozen):
    key_hash: str
    key_alias: str | None
    requests: int
    spend: float
    agents: tuple[InferredAgent, ...]


class KeySplit(_Frozen):
    window_hours: int
    keys_scanned: int
    shared_keys: tuple[SharedKey, ...]


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    return " ".join(str(_as_dict(part).get("text", "")) for part in _as_list(content))


def _system_prompt(body: Mapping[str, object]) -> str:
    messages: Final = (_as_dict(message) for message in _as_list(body.get("messages")))
    return next((_text(message.get("content")) for message in messages if message.get("role") == "system"), "")


def _tool_names(body: Mapping[str, object]) -> tuple[str, ...]:
    tools: Final = (_as_dict(tool) for tool in _as_list(body.get("tools")))
    names: Final = (_as_dict(tool.get("function")).get("name") or tool.get("name") for tool in tools)
    return tuple(sorted(str(name) for name in names if name))


def fingerprint(body: Mapping[str, object]) -> str:
    """Two requests come from the same agent when they carry the same system prompt and the same tool set."""
    identity: Final = _system_prompt(body) + "\x00" + ",".join(_tool_names(body))
    return hashlib.sha256(identity.encode()).hexdigest()[:8]


def _suggested_name(system_prompt: str, tools: Sequence[str], digest: str) -> str:
    role: Final = re.search(
        r"you are (?:the |an |a )?([a-z0-9 \-]{3,40}?)(?: agent| assistant| bot)?[.,\n]", system_prompt.lower() + "."
    )
    words: Final = role.group(1) if role else (tools[0] if tools else "")
    slug: Final = re.sub(r"[^a-z0-9]+", "-", words).strip("-")
    return f"{slug}-agent" if slug else f"agent-{digest}"


def _spend(row: Mapping[str, object]) -> float:
    spend: Final = row.get("spend")
    return float(spend) if isinstance(spend, (int, float)) else 0.0


def _agent(digest: str, rows: Sequence[Mapping[str, object]]) -> InferredAgent:
    bodies: Final = tuple(_as_dict(row.get("proxy_server_request")) for row in rows)
    headers: Final = (_as_dict(_as_dict(body.get("metadata")).get("headers")).get("user-agent") for body in bodies)
    tools: Final = _tool_names(bodies[0])
    return InferredAgent(
        fingerprint=digest,
        suggested_name=_suggested_name(_system_prompt(bodies[0]), tools, digest),
        system_prompt=_system_prompt(bodies[0])[:160],
        tools=tools,
        models=tuple(sorted({str(body.get("model")) for body in bodies})),
        user_agents=tuple(sorted({str(header) for header in headers if header})),
        requests=len(rows),
        failures=sum(1 for row in rows if row.get("status") not in (None, "success")),
        spend=sum(_spend(row) for row in rows),
    )


def _shared_key(rows: Sequence[Mapping[str, object]]) -> SharedKey:
    def by_agent(row: Mapping[str, object]) -> str:
        return fingerprint(_as_dict(row.get("proxy_server_request")))

    agents: Final = tuple(
        _agent(digest, tuple(group)) for digest, group in groupby(sorted(rows, key=by_agent), key=by_agent)
    )
    alias: Final = rows[0].get("key_alias")
    return SharedKey(
        key_hash=str(rows[0].get("api_key")),
        key_alias=alias if isinstance(alias, str) else None,
        requests=len(rows),
        spend=sum(_spend(row) for row in rows),
        agents=tuple(sorted(agents, key=lambda agent: -agent.spend)),
    )


def key_split(rows: Sequence[Mapping[str, object]], hours: int) -> KeySplit:
    def by_key(row: Mapping[str, object]) -> str:
        return str(row.get("api_key"))

    prompted: Final = tuple(row for row in rows if _as_dict(row.get("proxy_server_request")).get("messages"))
    keys: Final = tuple(_shared_key(tuple(group)) for _, group in groupby(sorted(prompted, key=by_key), key=by_key))
    shared: Final = (key for key in keys if len(key.agents) > 1)
    return KeySplit(
        window_hours=hours,
        keys_scanned=len(keys),
        shared_keys=tuple(sorted(shared, key=lambda key: -len(key.agents))),
    )


@router.get("/key_split")
async def get_key_split(user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)], hours: int = 24) -> KeySplit:
    """Find keys whose logged traffic comes from more than one agent, told apart by system prompt and tool set."""
    _require_admin(user)
    return key_split(await _logged_requests(hours), hours)


class ToolCall(_Frozen):
    server_id: str
    name: str
    arguments: Mapping[str, object] = {}


class Exchange(_Frozen):
    name: str
    method: Literal["GET", "POST"]
    path: str
    upstream_credential: bool
    body: Mapping[str, object] | None = None
    status: int
    answer: Mapping[str, object]


class Tape(_Frozen):
    recorded_from: str
    version: str
    exchanges: tuple[Exchange, ...]


class RecordRequest(_Frozen):
    gateway_url: str
    api_key: str
    upstream_headers: Mapping[str, str] = {}
    read_only_calls: tuple[ToolCall, ...] = ()


class ReplayRequest(_Frozen):
    tape: Tape
    gateway_url: str
    api_key: str
    upstream_headers: Mapping[str, str] = {}


class ReplayedExchange(_Frozen):
    name: str
    verdict: Literal["same", "changed"]
    changes: tuple[str, ...]


class TapeCheck(_Frozen):
    recorded_from: str
    recorded_version: str
    candidate: str
    candidate_version: str
    changed: int
    exchanges: tuple[ReplayedExchange, ...]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:12]


def summarize_answer(status: int, payload: Mapping[str, object]) -> Mapping[str, object]:
    """Keep what an MCP client depends on (tools, their schemas, errors) and drop ids and timings that vary per call."""
    tools: Final = tuple(_as_dict(tool) for tool in _as_list(payload.get("tools")))
    content: Final = tuple(_as_dict(part) for part in _as_list(payload.get("content")))
    return {
        "status": status,
        "tools": {str(tool.get("name")): _digest(tool.get("inputSchema")) for tool in tools},
        "error": payload.get("error") or _as_dict(payload.get("detail")).get("error"),
        "is_error": payload.get("isError"),
        "content": _digest(tuple((part.get("type"), part.get("text")) for part in content)) if content else None,
    }


def answer_changes(recorded: Mapping[str, object], replayed: Mapping[str, object]) -> tuple[str, ...]:
    was: Final = _as_dict(recorded.get("tools"))
    now: Final = _as_dict(replayed.get("tools"))
    scalar: Final = tuple(
        f"{field}: {recorded.get(field)} -> {replayed.get(field)}"
        for field in ("status", "error", "is_error", "content")
        if recorded.get(field) != replayed.get(field)
    )
    return (
        *scalar,
        *(f"tool missing: {name}" for name in sorted(was.keys() - now.keys())),
        *(f"tool added: {name}" for name in sorted(now.keys() - was.keys())),
        *(f"schema changed: {name}" for name in sorted(was.keys() & now.keys()) if was[name] != now[name]),
    )


def compare_tape(tape: Tape, candidate: str, candidate_version: str, replayed: Sequence[Exchange]) -> TapeCheck:
    changes: Final = tuple(
        answer_changes(old.answer, new.answer) for old, new in zip(tape.exchanges, replayed, strict=True)
    )
    return TapeCheck(
        recorded_from=tape.recorded_from,
        recorded_version=tape.version,
        candidate=candidate,
        candidate_version=candidate_version,
        changed=sum(1 for change in changes if change),
        exchanges=tuple(
            ReplayedExchange(name=old.name, verdict="changed" if change else "same", changes=change)
            for old, change in zip(tape.exchanges, changes, strict=True)
        ),
    )


def _planned(calls: Sequence[ToolCall]) -> tuple[Exchange, ...]:
    def listing(name: str, credential: bool) -> Exchange:
        return Exchange(
            name=name, method="GET", path="/mcp-rest/tools/list", upstream_credential=credential, status=0, answer={}
        )

    return (
        listing("list tools, no upstream credential", False),
        listing("list tools, with upstream credential", True),
        *(
            Exchange(
                name=f"call {call.name}",
                method="POST",
                path="/mcp-rest/tools/call",
                upstream_credential=True,
                body=call.model_dump(),
                status=0,
                answer={},
            )
            for call in calls
        ),
    )


async def _play(
    client: httpx.AsyncClient, base: str, api_key: str, upstream_headers: Mapping[str, str], exchange: Exchange
) -> Exchange:
    response: Final = await client.request(
        exchange.method,
        base + exchange.path,
        headers={"Authorization": f"Bearer {api_key}", **(upstream_headers if exchange.upstream_credential else {})},
        json=exchange.body,
        timeout=30.0,
    )
    return exchange.model_copy(
        update={
            "status": response.status_code,
            "answer": summarize_answer(response.status_code, _as_dict(response.text)),
        }
    )


async def _play_all(
    base: str, api_key: str, upstream_headers: Mapping[str, str], exchanges: Sequence[Exchange]
) -> tuple[str, tuple[Exchange, ...]]:
    client: Final = litellm.module_level_aclient.client
    spec: Final = await client.get(base + "/openapi.json", timeout=30.0)
    played: Final = await asyncio.gather(*(_play(client, base, api_key, upstream_headers, e) for e in exchanges))
    return str(_as_dict(_as_dict(spec.text).get("info")).get("version", "unknown")), tuple(played)


_TAPE_ROUTES: Final = frozenset({("GET", "/mcp-rest/tools/list"), ("POST", "/mcp-rest/tools/call")})


def gateway_base(gateway_url: str) -> str:
    """Only a plain http(s) gateway address, with no embedded credentials, may be recorded from or replayed against."""
    parsed: Final = urlsplit(gateway_url)
    plain: Final = parsed.scheme in ("http", "https") and bool(parsed.hostname)
    if not plain or "@" in parsed.netloc:
        raise HTTPException(status_code=400, detail="gateway_url must be an http(s) URL without credentials")
    return gateway_url.rstrip("/")


def replayable_exchanges(tape: Tape) -> tuple[Exchange, ...]:
    """A tape may only replay the MCP routes that recording produces."""
    if any((exchange.method, exchange.path) not in _TAPE_ROUTES for exchange in tape.exchanges):
        raise HTTPException(status_code=400, detail="tape contains a request that tape check does not record")
    return tape.exchanges


def _require_full_admin(user: UserAPIKeyAuth) -> None:
    if user.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="Only the proxy admin can record or replay a tape")


@router.post("/tape_check/record")
async def record_tape(request: RecordRequest, user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> Tape:
    """Record what MCP clients get from a running gateway: tool lists with and without the upstream credential, and read-only calls."""
    _require_full_admin(user)
    base: Final = gateway_base(request.gateway_url)
    version, exchanges = await _play_all(
        base, request.api_key, request.upstream_headers, _planned(request.read_only_calls)
    )
    return Tape(recorded_from=base, version=version, exchanges=exchanges)


@router.post("/tape_check/replay")
async def replay_tape(request: ReplayRequest, user: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]) -> TapeCheck:
    """Replay a recorded tape against a candidate gateway and report every answer an MCP client would see change."""
    _require_full_admin(user)
    base: Final = gateway_base(request.gateway_url)
    version, replayed = await _play_all(
        base, request.api_key, request.upstream_headers, replayable_exchanges(request.tape)
    )
    return compare_tape(request.tape, base, version, replayed)
