"""Agent traces read from LiteLLM_SpendLogs, so Lens works without its own trace store.

A trace is every spend log row sharing a session id. A row's agent comes from an `agent:<name>` tag (with an optional
`parent_agent:<name>` tag), the row's agent_id, a Claude Agent SDK subagent whose first prompt matches a parent's
Task/Agent tool call, or the calling framework read from the user agent."""

import base64
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import reduce
from typing import Final, Protocol

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import AGENT_TRACING_LIST_PAGE_SIZE
from litellm.tracing.generated.models import TraceAgentRow, TraceAgentsParams
from litellm.tracing.generated.types import (
    AgentNode,
    ChatRole,
    Span,
    SpanDetail,
    SpanErrorPage,
    SpanStatus,
    Trace,
    TracePage,
    TraceScope,
    TraceSummary,
    UIFields,
    UIMessage,
    UIMessages,
    UIText,
    UIToolCall,
)

AGENT_TAG_PREFIX: Final = "agent:"
PARENT_AGENT_TAG_PREFIX: Final = "parent_agent:"
SPEND_LOGS_SERVICE: Final = "litellm"
TRACE_ROW_LIMIT: Final = 500
AGENT_LIST_TRACE_LIMIT: Final = 200
PREVIEW_CHARS: Final = 300
ERROR_PREVIEW_CHARS: Final = 500
TRACE_KEY_SQL: Final = "COALESCE(NULLIF(session_id, ''), request_id)"
FRAMEWORK_USER_AGENTS: Final = (
    ("agent-sdk", "claude-agent-sdk"),
    ("claude-cli", "claude-code"),
    ("pydantic-ai", "pydantic-ai"),
    ("openai-agents", "openai-agents"),
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("crewai", "crewai"),
    ("codex", "codex"),
)
AGENT_USER_AGENT_SQL: Final = "|".join(pattern for pattern, _ in FRAMEWORK_USER_AGENTS)
DEFAULT_FRAMEWORK: Final = "litellm"
SUBAGENT_TOOL_NAMES: Final = frozenset(("Task", "Agent"))


class _SupportsQueryRaw(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class _TraceKeyRow(TypedDict):
    trace_key: ReadOnly[str]
    last_activity: ReadOnly[str]


class SpendLogRow(TypedDict):
    request_id: ReadOnly[str]
    trace_key: ReadOnly[str]
    call_type: ReadOnly[str]
    model: ReadOnly[str]
    status: ReadOnly[str | None]
    spend: ReadOnly[float]
    prompt_tokens: ReadOnly[int]
    completion_tokens: ReadOnly[int]
    start_time: ReadOnly[str]
    end_time: ReadOnly[str]
    agent_id: ReadOnly[str | None]
    request_tags: ReadOnly[str | None]
    metadata: ReadOnly[str | None]
    messages: ReadOnly[str | None]
    proxy_server_request: ReadOnly[str | None]
    response: ReadOnly[str | None]


_TRACE_KEY_ROWS: Final = TypeAdapter(tuple[_TraceKeyRow, ...])
_SPEND_LOG_ROWS: Final = TypeAdapter(tuple[SpendLogRow, ...])
_JSON: Final = TypeAdapter(object)
_OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])
_OBJECT_SEQUENCE: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class _ToolCall:
    name: str
    arguments: str
    subagent: str | None
    subagent_prompt: str | None


@dataclass(frozen=True, slots=True)
class _Call:
    row: SpendLogRow
    start_ms: float
    end_ms: float
    framework: str
    explicit_agent: str | None
    explicit_parent: str | None
    prompts: frozenset[str]
    input_messages: tuple[UIMessage, ...]
    output_text: str
    tool_calls: tuple[_ToolCall, ...]
    error: str | None

    @property
    def failed(self) -> bool:
        return self.row["status"] == "failure" or self.error is not None


@dataclass(frozen=True, slots=True)
class _Placed:
    call: _Call
    agent: str
    parent_agent: str | None


def _loads(raw: str | None) -> object:
    if not raw:
        return None
    try:
        return _JSON.validate_python(json.loads(raw))
    except ValueError:
        return None


def _mapping(value: object) -> Mapping[str, object]:
    return _OBJECT_MAPPING.validate_python(value) if isinstance(value, dict) else {}


def _sequence(value: object) -> Sequence[object]:
    return _OBJECT_SEQUENCE.validate_python(value) if isinstance(value, (list, tuple)) else ()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _parse_ms(timestamp: str) -> float:
    return datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc).timestamp() * 1000


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _framework(user_agent: str) -> str:
    lowered: Final = user_agent.lower()
    return next((name for pattern, name in FRAMEWORK_USER_AGENTS if pattern in lowered), DEFAULT_FRAMEWORK)


def _tag_value(tags: Sequence[object], prefix: str) -> str | None:
    return next(
        (tag[len(prefix) :].strip() for tag in tags if isinstance(tag, str) and tag.startswith(prefix)),
        None,
    )


def _content_texts(content: object) -> tuple[str, ...]:
    if isinstance(content, str):
        return (content,)
    return tuple(
        text
        for block in _sequence(content)
        for text in (
            (_text(_mapping(block).get("text")),)
            if _mapping(block).get("type") in ("text", "input_text", "output_text")
            else _content_texts(_mapping(block).get("content"))
            if _mapping(block).get("type") == "tool_result"
            else ()
        )
        if text
    )


def _visible_text(content: object) -> str:
    return "\n".join(text for text in _content_texts(content) if not text.lstrip().startswith("<system-reminder>"))


def _role(value: object) -> ChatRole:
    match value:
        case "system":
            return "system"
        case "assistant":
            return "assistant"
        case "tool":
            return "tool"
        case _:
            return "user"


def _block_tool_calls(content: object) -> tuple[UIToolCall, ...]:
    return tuple(
        UIToolCall(name=_text(_mapping(block).get("name")), arguments=json.dumps(_mapping(block).get("input")))
        for block in _sequence(content)
        if _mapping(block).get("type") == "tool_use"
    )


def _ui_messages(messages: object) -> tuple[UIMessage, ...]:
    return tuple(
        UIMessage(
            role=_role(_mapping(message).get("role")),
            content=_visible_text(_mapping(message).get("content")),
            tool_calls=_block_tool_calls(_mapping(message).get("content")),
        )
        for message in _sequence(messages)
    )


def _first_user_prompts(messages: object) -> frozenset[str]:
    first_user: Final = next(
        (_mapping(message) for message in _sequence(messages) if _mapping(message).get("role") == "user"), None
    )
    if first_user is None:
        return frozenset()
    return frozenset(text.strip() for text in _content_texts(first_user.get("content")) if text.strip())


def _response_tool_calls(response: Mapping[str, object]) -> tuple[_ToolCall, ...]:
    anthropic: Final = tuple(
        (_text(block.get("name")), _mapping(block.get("input")))
        for block in map(_mapping, _sequence(response.get("content")))
        if block.get("type") == "tool_use"
    )
    openai: Final = tuple(
        (
            _text(_mapping(call.get("function")).get("name")),
            _mapping(_loads(_text(_mapping(call.get("function")).get("arguments")))),
        )
        for choice in map(_mapping, _sequence(response.get("choices")))
        for call in map(_mapping, _sequence(_mapping(choice.get("message")).get("tool_calls")))
    )
    responses_api: Final = tuple(
        (_text(item.get("name")), _mapping(_loads(_text(item.get("arguments")))))
        for item in map(_mapping, _sequence(response.get("output")))
        if item.get("type") == "function_call"
    )
    return tuple(
        _ToolCall(
            name=name,
            arguments=json.dumps(arguments),
            subagent=_text(arguments.get("subagent_type")) or None if name in SUBAGENT_TOOL_NAMES else None,
            subagent_prompt=_text(arguments.get("prompt")).strip() or None if name in SUBAGENT_TOOL_NAMES else None,
        )
        for name, arguments in (*anthropic, *openai, *responses_api)
        if name
    )


def _response_text(response: Mapping[str, object]) -> str:
    anthropic: Final = _visible_text(response.get("content"))
    openai: Final = "\n".join(
        _visible_text(_mapping(choice.get("message")).get("content"))
        for choice in map(_mapping, _sequence(response.get("choices")))
    )
    responses_api: Final = "\n".join(
        _visible_text(item.get("content")) for item in map(_mapping, _sequence(response.get("output")))
    )
    return next((text for text in (anthropic, openai.strip(), responses_api.strip()) if text), "")


def _request_messages(row: SpendLogRow) -> object:
    stored: Final = _loads(row["messages"])
    if _sequence(stored):
        return stored
    return _mapping(_loads(row["proxy_server_request"])).get("messages")


def _call(row: SpendLogRow) -> _Call:
    metadata: Final = _mapping(_loads(row["metadata"]))
    tags: Final = _sequence(_loads(row["request_tags"]))
    messages: Final = _request_messages(row)
    response: Final = _mapping(_loads(row["response"]))
    error_message: Final = _text(_mapping(metadata.get("error_information")).get("error_message"))
    return _Call(
        row=row,
        start_ms=_parse_ms(row["start_time"]),
        end_ms=_parse_ms(row["end_time"]),
        framework=_framework(_text(metadata.get("user_agent"))),
        explicit_agent=_tag_value(tags, AGENT_TAG_PREFIX) or row["agent_id"] or None,
        explicit_parent=_tag_value(tags, PARENT_AGENT_TAG_PREFIX),
        prompts=_first_user_prompts(messages),
        input_messages=_ui_messages(messages),
        output_text=_response_text(response),
        tool_calls=_response_tool_calls(response),
        error=error_message or ("Request failed" if row["status"] == "failure" else None),
    )


def _root_agent(calls: Sequence[_Call]) -> str:
    return next((call.framework for call in calls if call.framework != DEFAULT_FRAMEWORK), "agent")


def _placed(call: _Call, earlier: tuple[_Placed, ...], root: str) -> _Placed:
    spawner: Final = next(
        (
            (tool.subagent, placed)
            for placed in reversed(earlier)
            for tool in placed.call.tool_calls
            if tool.subagent and tool.subagent_prompt in call.prompts
        ),
        None,
    )
    if spawner is not None and call.explicit_agent in (None, spawner[1].call.explicit_agent):
        return _Placed(call=call, agent=spawner[0], parent_agent=spawner[1].agent)
    if call.explicit_agent is not None:
        parent: Final = call.explicit_parent if call.explicit_parent != call.explicit_agent else None
        return _Placed(call=call, agent=call.explicit_agent, parent_agent=parent)
    return _Placed(call=call, agent=root, parent_agent=None)


def _place(calls: Sequence[_Call]) -> tuple[_Placed, ...]:
    root: Final = _root_agent(calls)

    def step(earlier: tuple[_Placed, ...], call: _Call) -> tuple[_Placed, ...]:
        return (*earlier, _placed(call, earlier, root))

    initial: tuple[_Placed, ...] = ()
    return reduce(step, calls, initial)


def _agent_span_id(agent: str) -> str:
    return f"agent:{agent}"


def _tool_span_id(call: _Call, index: int) -> str:
    return f"{call.row['request_id']}:tool:{index}"


def _preview(messages: Sequence[UIMessage]) -> str:
    return next(
        (message["content"] for message in reversed(messages) if message["role"] == "user" and message["content"]), ""
    )[:PREVIEW_CHARS]


@dataclass(frozen=True, slots=True)
class _AgentRun:
    name: str
    parent: str | None
    placed: tuple[_Placed, ...]
    start_ms: float
    end_ms: float


def _agent_runs(placed: Sequence[_Placed]) -> tuple[_AgentRun, ...]:
    names: Final = tuple(dict.fromkeys(item.agent for item in placed))
    return tuple(
        _AgentRun(
            name=name,
            parent=next((item.parent_agent for item in own if item.parent_agent), None),
            placed=own,
            start_ms=min(item.call.start_ms for item in own),
            end_ms=max(item.call.end_ms for item in own),
        )
        for name in names
        for own in (tuple(item for item in placed if item.agent == name),)
    )


def _next_start(placed: Sequence[_Placed], index: int) -> float | None:
    current: Final = placed[index]
    return next(
        (item.call.start_ms for item in placed[index + 1 :] if item.agent == current.agent),
        None,
    )


def _status(failed: bool) -> SpanStatus:
    return "error" if failed else "ok"


def _llm_span(item: _Placed, trace_start: float) -> Span:
    call: Final = item.call
    error: Final = call.error
    return Span(
        span_id=call.row["request_id"],
        parent_span_id=_agent_span_id(item.agent),
        name=call.row["model"] or call.row["call_type"],
        type="llm",
        agent=item.agent,
        framework=call.framework,
        model=call.row["model"] or None,
        start_offset_ms=call.start_ms - trace_start,
        duration_ms=max(call.end_ms - call.start_ms, 0.0),
        input_preview=_preview(call.input_messages),
        input_tokens=call.row["prompt_tokens"],
        output_tokens=call.row["completion_tokens"],
        spend=call.row["spend"],
        status=_status(call.failed),
        error=error[:ERROR_PREVIEW_CHARS] if error else None,
        error_truncated=error is not None and len(error) > ERROR_PREVIEW_CHARS,
        litellm_request_id=call.row["request_id"],
        spend_log_request_id=call.row["request_id"],
        spend_match="matched",
    )


def _tool_spans(placed: Sequence[_Placed], index: int, trace_start: float) -> tuple[Span, ...]:
    item: Final = placed[index]
    call: Final = item.call
    resumed_at: Final = _next_start(placed, index)
    duration: Final = max((resumed_at if resumed_at is not None else call.end_ms) - call.end_ms, 0.0)
    return tuple(
        Span(
            span_id=_tool_span_id(call, position),
            parent_span_id=call.row["request_id"],
            name=f"{tool.name} ({tool.subagent})" if tool.subagent else tool.name,
            type="agent" if tool.subagent else "tool",
            agent=item.agent,
            framework=call.framework,
            model=None,
            start_offset_ms=call.end_ms - trace_start,
            duration_ms=duration,
            input_preview=tool.arguments[:PREVIEW_CHARS],
            input_tokens=0,
            output_tokens=0,
            spend=None,
            status="ok",
            error=None,
            error_truncated=False,
            litellm_request_id=None,
            spend_log_request_id=None,
            spend_match=None,
        )
        for position, tool in enumerate(call.tool_calls)
    )


def _agent_span(run: _AgentRun, trace_start: float) -> Span:
    calls: Final = tuple(item.call for item in run.placed)
    return Span(
        span_id=_agent_span_id(run.name),
        parent_span_id=_agent_span_id(run.parent) if run.parent else None,
        name=run.name,
        type="agent",
        agent=run.name,
        framework=calls[0].framework,
        model=None,
        start_offset_ms=run.start_ms - trace_start,
        duration_ms=run.end_ms - run.start_ms,
        input_preview=_preview(calls[0].input_messages),
        input_tokens=sum(call.row["prompt_tokens"] for call in calls),
        output_tokens=sum(call.row["completion_tokens"] for call in calls),
        spend=sum(call.row["spend"] for call in calls),
        status=_status(any(call.failed for call in calls)),
        error=None,
        error_truncated=False,
        litellm_request_id=None,
        spend_log_request_id=None,
        spend_match=None,
    )


def _agent_node(run: _AgentRun) -> AgentNode:
    calls: Final = tuple(item.call for item in run.placed)
    return AgentNode(
        name=run.name,
        parent_agent=run.parent,
        duration_ms=run.end_ms - run.start_ms,
        invocations=1,
        llm_calls=len(calls),
        priced_calls=sum(1 for call in calls if call.row["spend"] > 0),
        spend=sum(call.row["spend"] for call in calls),
        tool_calls=sum(len(call.tool_calls) for call in calls),
    )


def build_trace(rows: Sequence[SpendLogRow]) -> Trace:
    calls: Final = tuple(sorted(map(_call, rows), key=lambda call: (call.start_ms, call.row["request_id"])))
    placed: Final = _place(calls)
    runs: Final = _agent_runs(placed)
    trace_start: Final = calls[0].start_ms
    trace_end: Final = max(call.end_ms for call in calls)
    spans: Final = (
        *(_agent_span(run, trace_start) for run in runs),
        *(
            span
            for index, item in enumerate(placed)
            for span in (_llm_span(item, trace_start), *_tool_spans(placed, index, trace_start))
        ),
    )
    errors: Final = sum(1 for call in calls if call.failed)
    root: Final = next((run for run in runs if run.parent is None), runs[0])
    return Trace(
        agents=tuple(_agent_node(run) for run in runs),
        spans=spans,
        next_cursor=None,
        summary=TraceSummary(
            trace_id=calls[0].row["trace_key"],
            name=root.name,
            service=SPEND_LOGS_SERVICE,
            start_time=_iso(trace_start),
            duration_ms=trace_end - trace_start,
            status=_status(errors > 0),
            error_count=errors,
            span_count=len(spans),
            llm_calls=len(calls),
            tool_calls=sum(len(call.tool_calls) for call in calls),
            priced_calls=sum(1 for call in calls if call.row["spend"] > 0),
            spend=sum(call.row["spend"] for call in calls),
            input_tokens=sum(call.row["prompt_tokens"] for call in calls),
            output_tokens=sum(call.row["completion_tokens"] for call in calls),
            models=tuple(dict.fromkeys(call.row["model"] for call in calls if call.row["model"])),
            agent_count=len(runs),
            agent_invocations=len(runs),
            agent_names=tuple(run.name for run in runs),
            frameworks=tuple(dict.fromkeys(call.framework for call in calls)),
            input_preview=_preview(calls[0].input_messages),
        ),
    )


def _span_detail(trace_rows: Sequence[SpendLogRow], span_id: str) -> SpanDetail | None:
    calls: Final = tuple(sorted(map(_call, trace_rows), key=lambda call: (call.start_ms, call.row["request_id"])))
    llm: Final = next((call for call in calls if call.row["request_id"] == span_id), None)
    if llm is not None:
        return SpanDetail(
            span_id=span_id,
            attributes={
                "litellm.request_id": llm.row["request_id"],
                "litellm.call_type": llm.row["call_type"],
                "litellm.status": llm.row["status"] or "success",
                "gen_ai.request.model": llm.row["model"],
                "agent.framework": llm.framework,
            },
            input=json.dumps(_request_messages(llm.row)),
            input_ui=UIMessages(kind="messages", messages=llm.input_messages),
            output=llm.output_text,
            output_ui=UIText(kind="text", text=llm.output_text or (llm.error or "")),
        )
    tool: Final = next(
        (
            tool
            for call in calls
            for position, tool in enumerate(call.tool_calls)
            if _tool_span_id(call, position) == span_id
        ),
        None,
    )
    if tool is not None:
        return SpanDetail(
            span_id=span_id,
            attributes={"tool.name": tool.name},
            input=tool.arguments,
            input_ui=UIText(kind="text", text=tool.arguments),
            output="",
            output_ui=UIText(kind="text", text=""),
        )
    trace: Final = build_trace(trace_rows)
    agent: Final = next((span for span in trace["spans"] if span["span_id"] == span_id), None)
    if agent is None:
        return None
    return SpanDetail(
        span_id=span_id,
        attributes={"agent.name": agent["agent"], "agent.framework": agent["framework"]},
        input=agent["input_preview"],
        input_ui=UIText(kind="text", text=agent["input_preview"]),
        output="",
        output_ui=UIFields(
            kind="fields",
            fields=(
                {
                    "key": "LLM calls",
                    "value": str(
                        sum(1 for span in trace["spans"] if span["agent"] == agent["agent"] and span["type"] == "llm")
                    ),
                },
            ),
        ),
    )


def _encode_cursor(row: _TraceKeyRow) -> str:
    return base64.urlsafe_b64encode(json.dumps((row["last_activity"], row["trace_key"])).encode()).decode()


def _decode_cursor(cursor: str | None) -> tuple[str, str] | None:
    if not cursor:
        return None
    decoded: Final = TypeAdapter(tuple[str, str]).validate_json(base64.urlsafe_b64decode(cursor.encode()))
    return decoded


def _scope_sql(scope: TraceScope, first_index: int) -> tuple[str, tuple[object, ...]]:
    if scope["all_teams"] == 1:
        return "", ()
    return (
        f'AND ("user" = ${first_index} OR team_id = ANY(${first_index + 1}::text[]))',
        (scope["user_id"], list(scope["team_ids"])),
    )


class SpendLogTraceStorage:
    def __init__(self, database: Callable[[], _SupportsQueryRaw | None]) -> None:
        self._database: Final = database

    async def _query(self, sql: str, *args: object) -> Sequence[Mapping[str, object]]:
        db: Final = self._database()
        return await db.query_raw(sql, *args) if db is not None else ()

    async def _trace_keys(
        self, scope: TraceScope, start_ms: int, end_ms: int, cursor: tuple[str, str] | None, limit: int
    ) -> tuple[_TraceKeyRow, ...]:
        scope_sql, scope_args = _scope_sql(scope, 3)
        cursor_index: Final = 3 + len(scope_args)
        after: Final = (
            f"WHERE (last_activity, trace_key) < (${cursor_index}, ${cursor_index + 1})" if cursor is not None else ""
        )
        rows: Final = await self._query(
            f"""
            SELECT trace_key, last_activity FROM (
                SELECT {TRACE_KEY_SQL} AS trace_key, MAX("startTime")::text AS last_activity
                FROM "LiteLLM_SpendLogs"
                WHERE "startTime" >= to_timestamp($1 / 1000.0) AT TIME ZONE 'UTC'
                  AND "startTime" < to_timestamp($2 / 1000.0) AT TIME ZONE 'UTC'
                  {scope_sql}
                GROUP BY 1
                HAVING COUNT(*) > 1
                    OR bool_or(COALESCE(agent_id, '') <> '')
                    OR bool_or(COALESCE(request_tags::text, '') LIKE '%"{AGENT_TAG_PREFIX}%')
                    OR bool_or(COALESCE(metadata->>'user_agent', '') ~* '{AGENT_USER_AGENT_SQL}')
            ) AS traces
            {after}
            ORDER BY last_activity DESC, trace_key DESC
            LIMIT {limit}
            """,
            start_ms,
            end_ms,
            *scope_args,
            *(cursor if cursor is not None else ()),
        )
        return _TRACE_KEY_ROWS.validate_python(rows)

    async def _rows(self, trace_keys: Sequence[str], scope: TraceScope) -> tuple[SpendLogRow, ...]:
        if not trace_keys:
            return ()
        scope_sql, scope_args = _scope_sql(scope, 2)
        rows: Final = await self._query(
            f"""
            SELECT request_id, {TRACE_KEY_SQL} AS trace_key, call_type, model, status, spend, prompt_tokens,
                   completion_tokens, "startTime"::text AS start_time, "endTime"::text AS end_time, agent_id,
                   request_tags::text AS request_tags, metadata::text AS metadata, messages::text AS messages,
                   proxy_server_request::text AS proxy_server_request, response::text AS response
            FROM "LiteLLM_SpendLogs"
            WHERE (session_id = ANY($1::text[]) OR request_id = ANY($1::text[]))
              AND {TRACE_KEY_SQL} = ANY($1::text[])
              {scope_sql}
            ORDER BY "startTime", request_id
            LIMIT {TRACE_ROW_LIMIT * len(trace_keys)}
            """,
            list(trace_keys),
            *scope_args,
        )
        return _SPEND_LOG_ROWS.validate_python(rows)

    async def _traces(self, keys: Sequence[str], scope: TraceScope) -> tuple[Trace, ...]:
        rows: Final = await self._rows(keys, scope)
        return tuple(
            build_trace(own) for key in keys for own in (tuple(row for row in rows if row["trace_key"] == key),) if own
        )

    async def list_traces(
        self,
        scope: TraceScope,
        start_ms: int,
        end_ms: int,
        cursor: str | None = None,
        limit: int = AGENT_TRACING_LIST_PAGE_SIZE,
    ) -> TracePage:
        keys: Final = await self._trace_keys(scope, start_ms, end_ms, _decode_cursor(cursor), limit + 1)
        page: Final = keys[:limit]
        traces: Final = await self._traces(tuple(row["trace_key"] for row in page), scope)
        return TracePage(
            data=tuple(trace["summary"] for trace in traces),
            next_cursor=_encode_cursor(page[-1]) if len(keys) > limit else None,
        )

    async def get_trace(
        self,
        trace_id: str,
        scope: TraceScope,
        trace_ref: str = "",
        cursor: str | None = None,
        page_size: int | None = None,
    ) -> Trace | None:
        traces: Final = await self._traces((trace_id,), scope)
        return traces[0] if traces else None

    async def get_span(self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "") -> SpanDetail | None:
        rows: Final = await self._rows((trace_id,), scope)
        return _span_detail(rows, span_id) if rows else None

    async def get_span_error(
        self, trace_id: str, span_id: str, scope: TraceScope, trace_ref: str = "", cursor: str | None = None
    ) -> SpanErrorPage | None:
        rows: Final = await self._rows((trace_id,), scope)
        call: Final = next((_call(row) for row in rows if row["request_id"] == span_id), None)
        if call is None or call.error is None:
            return None
        return SpanErrorPage(span_id=span_id, message=call.error, next_cursor=None, total_chars=len(call.error))

    async def trace_agents(self, parameters: TraceAgentsParams) -> tuple[TraceAgentRow, ...]:
        scope: Final = TraceScope(
            all_teams=parameters.all_teams, user_id=parameters.user_id, team_ids=parameters.team_ids
        )
        keys: Final = await self._trace_keys(
            scope, parameters.start_ms, parameters.end_ms, None, AGENT_LIST_TRACE_LIMIT
        )
        traces: Final = await self._traces(tuple(row["trace_key"] for row in keys), scope)
        names: Final = tuple(
            dict.fromkeys(name for trace in traces for name in trace["summary"].get("agent_names", ()))
        )
        return tuple(
            TraceAgentRow(
                agent_name=name,
                runs=len(own),
                failed_runs=sum(
                    1
                    for trace in own
                    if any(span["status"] == "error" for span in trace["spans"] if span["agent"] == name)
                ),
                last_seen_ms=int(max(_parse_ms(trace["summary"]["start_time"].replace("Z", "")) for trace in own)),
                frameworks=tuple(
                    dict.fromkeys(
                        span["framework"] for trace in own for span in trace["spans"] if span["agent"] == name
                    )
                ),
            )
            for name in names[: parameters.limit]
            for own in (tuple(trace for trace in traces if name in trace["summary"].get("agent_names", ())),)
        )
