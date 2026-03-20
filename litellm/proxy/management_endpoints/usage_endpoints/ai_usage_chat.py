"""
AI Usage Chat - uses LLM tool calling to answer questions about
usage/spend data by querying the aggregated daily activity endpoints.
"""

import json
from datetime import date
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional, cast

from typing_extensions import TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_COMPETITOR_DISCOVERY_MODEL
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USAGE_AI_TEMPERATURE = 0.2

TABLE_DAILY_USER_SPEND = "litellm_dailyuserspend"
TABLE_DAILY_TEAM_SPEND = "litellm_dailyteamspend"
TABLE_DAILY_TAG_SPEND = "litellm_dailytagspend"

ENTITY_FIELD_USER = "user_id"
ENTITY_FIELD_TEAM = "team_id"
ENTITY_FIELD_TAG = "tag"

PAGINATED_PAGE_SIZE = 200
MAX_CHAT_MESSAGES = 20
TOP_N_MODELS = 15
TOP_N_PROVIDERS = 10
TOP_N_KEYS = 10

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class SSEStatusEvent(TypedDict):
    type: Literal["status"]
    message: str


class SSEToolCallEvent(TypedDict, total=False):
    type: Literal["tool_call"]
    tool_name: str
    tool_label: str
    arguments: Dict[str, str]
    status: Literal["running", "complete", "error"]
    error: str


class SSEChunkEvent(TypedDict):
    type: Literal["chunk"]
    content: str


class SSEDoneEvent(TypedDict):
    type: Literal["done"]


class SSEErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


SSEEvent = (
    SSEStatusEvent | SSEToolCallEvent | SSEChunkEvent | SSEDoneEvent | SSEErrorEvent
)


class ToolHandler(TypedDict):
    fetch: Callable[..., Any]
    summarise: Callable[[Dict[str, Any]], str]
    label: str


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

_DATE_PARAMS = {
    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
}

_TOOL_USAGE = {
    "type": "function",
    "function": {
        "name": "get_usage_data",
        "description": (
            "Fetch aggregated global usage/spend data. Returns daily spend, "
            "token counts, request counts, and breakdowns by model, provider, "
            "and API key. Use for overall spend, top models, top providers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "user_id": {
                    "type": "string",
                    "description": "Optional user ID filter. Omit for global view.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

_TOOL_TEAM = {
    "type": "function",
    "function": {
        "name": "get_team_usage_data",
        "description": (
            "Fetch usage/spend data broken down by team. Use for questions "
            "like 'which team spends the most' or 'show me team X usage'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "team_ids": {
                    "type": "string",
                    "description": "Optional comma-separated team IDs. Omit for all teams.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

_TOOL_TAG = {
    "type": "function",
    "function": {
        "name": "get_tag_usage_data",
        "description": (
            "Fetch usage/spend data broken down by tag. Tags are labels "
            "attached to requests (features, environments, credentials)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tag names. Omit for all tags.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

TOOLS_BASE = [_TOOL_USAGE]
TOOLS_ADMIN = [_TOOL_USAGE, _TOOL_TEAM, _TOOL_TAG]


def get_tools_for_role(is_admin: bool) -> List[Dict[str, Any]]:
    """Return the tool list appropriate for the user's role."""
    return TOOLS_ADMIN if is_admin else TOOLS_BASE


_SYSTEM_PROMPT_BASE = (
    "You are an AI assistant embedded in the LiteLLM Usage dashboard. "
    "You help users understand their LLM API spend and usage data.\n\n"
    "ALWAYS call the appropriate tool(s) first to fetch data before answering. "
    "You may call multiple tools if the question spans different dimensions.\n\n"
    "Guidelines:\n"
    "- Be concise and specific. Use exact numbers from the data.\n"
    "- Format costs as dollar amounts (e.g. $12.34).\n"
    "- When comparing entities, show a ranked list.\n"
    "- If data is empty or no results found, say so clearly.\n"
    "- Do not hallucinate data — only use what the tools return.\n"
    "- Today's date will be provided below. Use it to interpret relative dates "
    "like 'this week', 'this month', 'last 7 days', etc."
)

_TOOL_DESCRIPTIONS_ADMIN = (
    "You have access to these tools:\n"
    "- `get_usage_data`: Global/user-level usage (spend, models, providers, API keys)\n"
    "- `get_team_usage_data`: Team-level usage breakdown\n"
    "- `get_tag_usage_data`: Tag-level usage breakdown\n\n"
)

_TOOL_DESCRIPTIONS_BASE = (
    "You have access to this tool:\n"
    "- `get_usage_data`: Your usage data (spend, models, providers, API keys)\n\n"
)


def _build_system_prompt(is_admin: bool) -> str:
    """Build role-appropriate system prompt with today's date."""
    tool_desc = _TOOL_DESCRIPTIONS_ADMIN if is_admin else _TOOL_DESCRIPTIONS_BASE
    return (
        f"{_SYSTEM_PROMPT_BASE}\n\n{tool_desc}"
        f"Today's date: {date.today().isoformat()}"
    )


# keep a public reference for test assertions
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE

# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _parse_csv_ids(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


async def _query_activity(
    table_name: str,
    entity_id_field: str,
    entity_id: Optional[Any],
    start_date: str,
    end_date: str,
    *,
    use_aggregated: bool = False,
) -> SpendAnalyticsPaginatedResponse:
    """Shared helper that calls the daily activity query layer."""
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity,
        get_daily_activity_aggregated,
    )
    from litellm.proxy.proxy_server import prisma_client

    if use_aggregated:
        return await get_daily_activity_aggregated(
            prisma_client=prisma_client,
            table_name=table_name,
            entity_id_field=entity_id_field,
            entity_id=entity_id,
            entity_metadata_field=None,
            start_date=start_date,
            end_date=end_date,
            model=None,
            api_key=None,
        )
    return await get_daily_activity(
        prisma_client=prisma_client,
        table_name=table_name,
        entity_id_field=entity_id_field,
        entity_id=entity_id,
        entity_metadata_field=None,
        start_date=start_date,
        end_date=end_date,
        model=None,
        api_key=None,
        page=1,
        page_size=PAGINATED_PAGE_SIZE,
    )


async def _fetch_usage_data(
    start_date: str, end_date: str, user_id: Optional[str] = None
) -> Dict[str, Any]:
    resp = await _query_activity(
        TABLE_DAILY_USER_SPEND,
        ENTITY_FIELD_USER,
        user_id,
        start_date,
        end_date,
        use_aggregated=True,
    )
    return resp.model_dump(mode="json")


async def _fetch_team_usage_data(
    start_date: str, end_date: str, team_ids: Optional[str] = None
) -> Dict[str, Any]:
    resp = await _query_activity(
        TABLE_DAILY_TEAM_SPEND,
        ENTITY_FIELD_TEAM,
        _parse_csv_ids(team_ids),
        start_date,
        end_date,
    )
    return resp.model_dump(mode="json")


async def _fetch_tag_usage_data(
    start_date: str, end_date: str, tags: Optional[str] = None
) -> Dict[str, Any]:
    resp = await _query_activity(
        TABLE_DAILY_TAG_SPEND,
        ENTITY_FIELD_TAG,
        _parse_csv_ids(tags),
        start_date,
        end_date,
    )
    return resp.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Summarisers — convert raw JSON to concise text the LLM can reason over
# ---------------------------------------------------------------------------


def _accumulate_breakdown(
    results: List[Dict[str, Any]], dimension: str, fields: List[str]
) -> Dict[str, Dict[str, float]]:
    """Aggregate a single breakdown dimension across days."""
    totals: Dict[str, Dict[str, float]] = {}
    for day in results:
        for key, entry in day.get("breakdown", {}).get(dimension, {}).items():
            if key not in totals:
                totals[key] = {f: 0.0 for f in fields}
            m = entry.get("metrics", {})
            for f in fields:
                totals[key][f] += m.get(f, 0)
    return totals


def _ranked_lines(
    totals: Dict[str, Dict[str, float]],
    fmt: Callable[[str, Dict[str, float]], str],
    limit: int,
) -> List[str]:
    """Sort by spend descending, format each entry, and truncate."""
    return [
        fmt(name, vals)
        for name, vals in sorted(totals.items(), key=lambda x: -x[1].get("spend", 0))[
            :limit
        ]
    ]


def _summarise_usage_data(data: Dict[str, Any]) -> str:
    meta = data.get("metadata", {})
    results = data.get("results", [])

    header = (
        f"Total Spend: ${meta.get('total_spend', 0):.4f}\n"
        f"Total Requests: {meta.get('total_api_requests', 0)}\n"
        f"Successful: {meta.get('total_successful_requests', 0)} | "
        f"Failed: {meta.get('total_failed_requests', 0)}\n"
        f"Total Tokens: {meta.get('total_tokens', 0)}"
    )

    models = _accumulate_breakdown(
        results, "models", ["spend", "api_requests", "total_tokens"]
    )
    providers = _accumulate_breakdown(results, "providers", ["spend", "api_requests"])

    model_lines = _ranked_lines(
        models,
        lambda n, d: f"  - {n}: ${d['spend']:.4f} ({int(d['api_requests'])} reqs, {int(d['total_tokens'])} tokens)",
        TOP_N_MODELS,
    )
    provider_lines = _ranked_lines(
        providers,
        lambda n, d: f"  - {n}: ${d['spend']:.4f} ({int(d['api_requests'])} reqs)",
        TOP_N_PROVIDERS,
    )

    sections = [header, ""]
    sections += ["Top Models by Spend:"] + (model_lines or ["  (no data)"]) + [""]
    sections += ["Top Providers by Spend:"] + (provider_lines or ["  (no data)"])
    return "\n".join(sections)


def _summarise_entity_data(data: Dict[str, Any], entity_label: str) -> str:
    """Summarise team/tag entity usage data."""
    results = data.get("results", [])
    if not results:
        return f"No {entity_label} usage data found for the given date range."

    totals: Dict[str, Dict[str, Any]] = {}
    for day in results:
        for eid, entry in day.get("breakdown", {}).get("entities", {}).items():
            if eid not in totals:
                alias = entry.get("metadata", {}).get("alias", eid)
                totals[eid] = {"alias": alias, "spend": 0.0, "requests": 0, "tokens": 0}
            m = entry.get("metrics", {})
            totals[eid]["spend"] += m.get("spend", 0)
            totals[eid]["requests"] += m.get("api_requests", 0)
            totals[eid]["tokens"] += m.get("total_tokens", 0)

    lines = [f"{entity_label} Usage ({len(totals)} {entity_label.lower()}s):", ""]
    for eid, d in sorted(totals.items(), key=lambda x: -x[1]["spend"]):
        label = d["alias"] if d["alias"] != eid else eid
        lines.append(
            f"- {label} (ID: {eid}): ${d['spend']:.4f} | "
            f"{int(d['requests'])} reqs | {int(d['tokens'])} tokens"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dispatch registry
# ---------------------------------------------------------------------------

TOOL_HANDLERS: Dict[str, ToolHandler] = {
    "get_usage_data": ToolHandler(
        fetch=_fetch_usage_data,
        summarise=_summarise_usage_data,
        label="global usage data",
    ),
    "get_team_usage_data": ToolHandler(
        fetch=_fetch_team_usage_data,
        summarise=lambda data: _summarise_entity_data(data, "Team"),
        label="team usage data",
    ),
    "get_tag_usage_data": ToolHandler(
        fetch=_fetch_tag_usage_data,
        summarise=lambda data: _summarise_entity_data(data, "Tag"),
        label="tag usage data",
    ),
}


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


def _sse(event: SSEEvent) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _resolve_fetch_kwargs(
    fn_name: str,
    fn_args: Dict[str, str],
    user_id: Optional[str],
    is_admin: bool,
) -> Dict[str, Any]:
    """Build keyword arguments for a tool's fetch function."""
    start_date = fn_args.get("start_date", "")
    end_date = fn_args.get("end_date", "")
    if not start_date or not end_date:
        raise ValueError("Missing required start_date or end_date from tool arguments")
    kwargs: Dict[str, Any] = {"start_date": start_date, "end_date": end_date}
    if fn_name == "get_usage_data":
        if not is_admin:
            kwargs["user_id"] = user_id
        elif fn_args.get("user_id"):
            kwargs["user_id"] = fn_args["user_id"]
    elif fn_name == "get_team_usage_data" and fn_args.get("team_ids"):
        kwargs["team_ids"] = fn_args["team_ids"]
    elif fn_name == "get_tag_usage_data" and fn_args.get("tags"):
        kwargs["tags"] = fn_args["tags"]
    return kwargs


async def _execute_tool_call(
    handler: ToolHandler,
    fn_name: str,
    fn_args: Dict[str, str],
    user_id: Optional[str],
    is_admin: bool,
) -> str:
    """Run a single tool and return the summarised result text."""
    kwargs = _resolve_fetch_kwargs(fn_name, fn_args, user_id, is_admin)
    raw_data = await handler["fetch"](**kwargs)
    return handler["summarise"](raw_data)


async def _process_tool_call(
    tc: Any,
    chat_messages: List[Dict[str, Any]],
    user_id: Optional[str],
    is_admin: bool,
) -> AsyncIterator[str]:
    """Execute a single tool call, yielding SSE events for status."""
    fn_name = tc.function.name
    fn_args = json.loads(tc.function.arguments)

    allowed_names = {t["function"]["name"] for t in get_tools_for_role(is_admin)}
    handler = TOOL_HANDLERS.get(fn_name)

    if fn_name not in allowed_names or not handler:
        chat_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Tool not available: {fn_name}",
            }
        )
        return

    tool_event_base = {
        "type": "tool_call",
        "tool_name": fn_name,
        "tool_label": handler["label"],
        "arguments": fn_args,
    }
    yield _sse(cast(SSEToolCallEvent, {**tool_event_base, "status": "running"}))

    try:
        tool_result = await _execute_tool_call(
            handler, fn_name, fn_args, user_id, is_admin
        )
        yield _sse(cast(SSEToolCallEvent, {**tool_event_base, "status": "complete"}))
    except Exception as e:
        verbose_proxy_logger.error("Tool %s failed: %s", fn_name, e)
        tool_result = f"Error fetching {handler['label']}. Please try again."
        yield _sse(cast(SSEToolCallEvent, {**tool_event_base, "status": "error"}))

    chat_messages.append(
        {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
    )


async def _stream_final_response(
    model: str, chat_messages: List[Dict[str, Any]]
) -> AsyncIterator[str]:
    """Stream the final LLM response after tool results are appended."""
    yield _sse({"type": "status", "message": "Analyzing results..."})

    response = await litellm.acompletion(
        model=model,
        messages=chat_messages,
        stream=True,
        temperature=USAGE_AI_TEMPERATURE,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield _sse({"type": "chunk", "content": delta})


async def stream_usage_ai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    user_id: Optional[str] = None,
    is_admin: bool = False,
) -> AsyncIterator[str]:
    """Stream SSE events: status → tool_call → chunk → done."""
    resolved_model = (model or "").strip() or DEFAULT_COMPETITOR_DISCOVERY_MODEL
    truncated = (
        messages[-MAX_CHAT_MESSAGES:] if len(messages) > MAX_CHAT_MESSAGES else messages
    )
    chat_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt(is_admin)},
        *truncated,
    ]

    try:
        yield _sse({"type": "status", "message": "Thinking..."})
        tools = get_tools_for_role(is_admin)
        response = await litellm.acompletion(
            model=resolved_model,
            messages=chat_messages,
            tools=tools,
            temperature=USAGE_AI_TEMPERATURE,
        )
        choice = response.choices[0]  # type: ignore

        if not choice.message.tool_calls:
            if choice.message.content:
                yield _sse({"type": "chunk", "content": choice.message.content})
            yield _sse({"type": "done"})
            return

        chat_messages.append(choice.message.model_dump())
        for tc in choice.message.tool_calls:
            async for event in _process_tool_call(tc, chat_messages, user_id, is_admin):
                yield event
        async for event in _stream_final_response(resolved_model, chat_messages):
            yield event
        yield _sse({"type": "done"})

    except Exception as e:
        verbose_proxy_logger.error("AI usage chat failed: %s", e)
        yield _sse(
            {
                "type": "error",
                "message": "An internal error occurred. Please try again.",
            }
        )
