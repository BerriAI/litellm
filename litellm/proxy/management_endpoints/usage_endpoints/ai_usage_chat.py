"""
AI Usage Chat - uses LLM tool calling to answer questions about
usage/spend data by querying the aggregated daily activity endpoints.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_COMPETITOR_DISCOVERY_MODEL

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

GET_USAGE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_usage_data",
        "description": (
            "Fetch aggregated global usage/spend data for the LiteLLM proxy. "
            "Returns daily spend, token usage, request counts, and breakdowns "
            "by model, provider, and API key for the given date range. "
            "Use this for questions about overall spend, top models, top providers, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                },
                "user_id": {
                    "type": "string",
                    "description": "Optional user ID to filter by a specific user. Omit for global view.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

GET_TEAM_USAGE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_team_usage_data",
        "description": (
            "Fetch usage/spend data broken down by team. "
            "Returns each team's spend, requests, tokens, model breakdown, and provider breakdown. "
            "Use this for questions like 'which team is spending the most' or 'show me team X usage'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                },
                "team_ids": {
                    "type": "string",
                    "description": "Optional comma-separated team IDs to filter by. Omit for all teams.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

GET_TAG_USAGE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_tag_usage_data",
        "description": (
            "Fetch usage/spend data broken down by tag. "
            "Tags are labels attached to requests (e.g. feature names, environments, credentials). "
            "Use this for questions about tag-level spend or 'top tags for team X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tag names to filter. Omit for all tags.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
}

ALL_TOOLS = [GET_USAGE_DATA_TOOL, GET_TEAM_USAGE_DATA_TOOL, GET_TAG_USAGE_DATA_TOOL]

SYSTEM_PROMPT = (
    "You are an AI assistant embedded in the LiteLLM Usage dashboard. "
    "You help users understand their LLM API spend and usage data.\n\n"
    "You have access to these tools:\n"
    "- `get_usage_data`: Global/user-level usage (spend, models, providers, API keys)\n"
    "- `get_team_usage_data`: Team-level usage breakdown\n"
    "- `get_tag_usage_data`: Tag-level usage breakdown\n\n"
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


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

async def _fetch_usage_data(
    start_date: str,
    end_date: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity_aggregated,
    )
    from litellm.proxy.proxy_server import prisma_client

    response = await get_daily_activity_aggregated(
        prisma_client=prisma_client,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=user_id,
        entity_metadata_field=None,
        start_date=start_date,
        end_date=end_date,
        model=None,
        api_key=None,
    )
    return response.model_dump(mode="json")


async def _fetch_team_usage_data(
    start_date: str,
    end_date: str,
    team_ids: Optional[str] = None,
) -> Dict[str, Any]:
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity,
    )
    from litellm.proxy.proxy_server import prisma_client

    team_ids_list: Optional[List[str]] = None
    if team_ids:
        team_ids_list = [t.strip() for t in team_ids.split(",") if t.strip()]

    response = await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailyteamspend",
        entity_id_field="team_id",
        entity_id=team_ids_list,
        entity_metadata_field=None,
        start_date=start_date,
        end_date=end_date,
        model=None,
        api_key=None,
        page=1,
        page_size=200,
    )
    return response.model_dump(mode="json")


async def _fetch_tag_usage_data(
    start_date: str,
    end_date: str,
    tags: Optional[str] = None,
) -> Dict[str, Any]:
    from litellm.proxy.management_endpoints.common_daily_activity import (
        get_daily_activity,
    )
    from litellm.proxy.proxy_server import prisma_client

    tag_list: Optional[List[str]] = None
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    response = await get_daily_activity(
        prisma_client=prisma_client,
        table_name="litellm_dailytagspend",
        entity_id_field="tag",
        entity_id=tag_list,
        entity_metadata_field=None,
        start_date=start_date,
        end_date=end_date,
        model=None,
        api_key=None,
        page=1,
        page_size=200,
    )
    return response.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Summarisers
# ---------------------------------------------------------------------------

def _summarise_usage_data(data: Dict[str, Any]) -> str:
    meta = data.get("metadata", {})
    results = data.get("results", [])

    lines = [
        f"Date Range: {results[0]['date'] if results else 'N/A'} to {results[-1]['date'] if results else 'N/A'}",
        f"Total Spend: ${meta.get('total_spend', 0):.4f}",
        f"Total Requests: {meta.get('total_api_requests', 0)}",
        f"Successful Requests: {meta.get('total_successful_requests', 0)}",
        f"Failed Requests: {meta.get('total_failed_requests', 0)}",
        f"Total Tokens: {meta.get('total_tokens', 0)}",
        "",
    ]

    model_spend: Dict[str, Dict[str, float]] = {}
    provider_spend: Dict[str, Dict[str, float]] = {}
    key_spend: Dict[str, Dict[str, Any]] = {}

    for day in results:
        breakdown = day.get("breakdown", {})
        for model, metrics in breakdown.get("models", {}).items():
            if model not in model_spend:
                model_spend[model] = {"spend": 0, "requests": 0, "tokens": 0}
            m = metrics.get("metrics", {})
            model_spend[model]["spend"] += m.get("spend", 0)
            model_spend[model]["requests"] += m.get("api_requests", 0)
            model_spend[model]["tokens"] += m.get("total_tokens", 0)

        for provider, metrics in breakdown.get("providers", {}).items():
            if provider not in provider_spend:
                provider_spend[provider] = {"spend": 0, "requests": 0}
            m = metrics.get("metrics", {})
            provider_spend[provider]["spend"] += m.get("spend", 0)
            provider_spend[provider]["requests"] += m.get("api_requests", 0)

        for key, metrics in breakdown.get("api_keys", {}).items():
            if key not in key_spend:
                alias = metrics.get("metadata", {}).get("key_alias")
                key_spend[key] = {"spend": 0, "alias": alias}
            key_spend[key]["spend"] += metrics.get("metrics", {}).get("spend", 0)

    if model_spend:
        lines.append("Top Models by Spend:")
        for name, d in sorted(model_spend.items(), key=lambda x: -x[1]["spend"])[:15]:
            lines.append(f"  - {name}: ${d['spend']:.4f} ({int(d['requests'])} reqs, {int(d['tokens'])} tokens)")
    else:
        lines.append("Models: (no data)")

    lines.append("")

    if provider_spend:
        lines.append("Top Providers by Spend:")
        for name, d in sorted(provider_spend.items(), key=lambda x: -x[1]["spend"])[:10]:
            lines.append(f"  - {name}: ${d['spend']:.4f} ({int(d['requests'])} reqs)")
    else:
        lines.append("Providers: (no data)")

    lines.append("")

    if key_spend:
        lines.append("Top API Keys by Spend:")
        for key, d in sorted(key_spend.items(), key=lambda x: -x[1]["spend"])[:10]:
            label = d["alias"] or key
            lines.append(f"  - {label}: ${d['spend']:.4f}")
    else:
        lines.append("API Keys: (no data)")

    lines.append("")

    if results:
        lines.append("Daily Spend:")
        sorted_days = sorted(results, key=lambda x: x["date"])
        for day in sorted_days:
            m = day.get("metrics", {})
            lines.append(f"  - {day['date']}: ${m.get('spend', 0):.4f} ({m.get('api_requests', 0)} reqs)")

    return "\n".join(lines)


def _summarise_entity_data(data: Dict[str, Any], entity_label: str) -> str:
    """Summarise team/tag/org/customer entity usage data."""
    results = data.get("results", [])
    if not results:
        return f"No {entity_label} usage data found for the given date range."

    entity_totals: Dict[str, Dict[str, Any]] = {}
    for day in results:
        breakdown = day.get("breakdown", {})
        for entity_id, entity_data in breakdown.get("entities", {}).items():
            if entity_id not in entity_totals:
                alias = entity_data.get("metadata", {}).get("alias", entity_id)
                entity_totals[entity_id] = {
                    "alias": alias,
                    "spend": 0,
                    "requests": 0,
                    "tokens": 0,
                    "models": {},
                }
            m = entity_data.get("metrics", {})
            entity_totals[entity_id]["spend"] += m.get("spend", 0)
            entity_totals[entity_id]["requests"] += m.get("api_requests", 0)
            entity_totals[entity_id]["tokens"] += m.get("total_tokens", 0)

            for model_name, model_data in entity_data.get("api_key_breakdown", {}).items():
                models_dict = entity_totals[entity_id]["models"]
                if model_name not in models_dict:
                    models_dict[model_name] = 0
                models_dict[model_name] += model_data.get("metrics", {}).get("spend", 0)

    lines = [f"{entity_label} Usage Summary ({len(entity_totals)} {entity_label.lower()}s):", ""]

    for eid, d in sorted(entity_totals.items(), key=lambda x: -x[1]["spend"]):
        label = d["alias"] if d["alias"] != eid else eid
        lines.append(f"- {label} (ID: {eid}): ${d['spend']:.4f} | {int(d['requests'])} reqs | {int(d['tokens'])} tokens")
        if d["models"]:
            for model, spend in sorted(d["models"].items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    Model: {model}: ${spend:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "get_usage_data": {
        "fetch": _fetch_usage_data,
        "summarise": _summarise_usage_data,
        "label": "global usage data",
    },
    "get_team_usage_data": {
        "fetch": _fetch_team_usage_data,
        "summarise": lambda data: _summarise_entity_data(data, "Team"),
        "label": "team usage data",
    },
    "get_tag_usage_data": {
        "fetch": _fetch_tag_usage_data,
        "summarise": lambda data: _summarise_entity_data(data, "Tag"),
        "label": "tag usage data",
    },
}


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---------------------------------------------------------------------------
# Main streaming function
# ---------------------------------------------------------------------------

async def stream_usage_ai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    user_id: Optional[str] = None,
    is_admin: bool = False,
) -> AsyncIterator[str]:
    """
    Stream an AI chat response about usage data.

    Yields SSE events:
      {"type": "status", "message": "..."}   - thinking/tool status
      {"type": "chunk",  "content": "..."}   - streamed response text
      {"type": "done"}                        - stream finished
      {"type": "error",  "message": "..."}   - error
    """
    model = model.strip() if model else ""
    model = model or DEFAULT_COMPETITOR_DISCOVERY_MODEL

    from datetime import date as date_type

    today = date_type.today().isoformat()
    system_content = f"{SYSTEM_PROMPT}\n\nToday's date: {today}"

    chat_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        *messages,
    ]

    try:
        yield _sse({"type": "status", "message": "Thinking..."})

        response = await litellm.acompletion(
            model=model,
            messages=chat_messages,
            tools=ALL_TOOLS,
            temperature=0.2,
        )

        choice = response.choices[0]  # type: ignore
        tool_calls = choice.message.tool_calls

        if tool_calls:
            chat_messages.append(choice.message.model_dump())

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                handler = TOOL_HANDLERS.get(fn_name)
                if not handler:
                    yield _sse({"type": "status", "message": f"Unknown tool: {fn_name}"})
                    chat_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Unknown tool: {fn_name}",
                    })
                    continue

                yield _sse({
                    "type": "status",
                    "message": f"Fetching {handler['label']} ({fn_args.get('start_date', '')} to {fn_args.get('end_date', '')})..."
                })

                try:
                    fetch_kwargs: Dict[str, Any] = {
                        "start_date": fn_args["start_date"],
                        "end_date": fn_args["end_date"],
                    }

                    if fn_name == "get_usage_data":
                        if not is_admin:
                            fetch_kwargs["user_id"] = user_id
                        elif fn_args.get("user_id"):
                            fetch_kwargs["user_id"] = fn_args["user_id"]
                    elif fn_name == "get_team_usage_data":
                        if fn_args.get("team_ids"):
                            fetch_kwargs["team_ids"] = fn_args["team_ids"]
                    elif fn_name == "get_tag_usage_data":
                        if fn_args.get("tags"):
                            fetch_kwargs["tags"] = fn_args["tags"]

                    raw_data = await handler["fetch"](**fetch_kwargs)
                    tool_result = handler["summarise"](raw_data)
                except Exception as e:
                    tool_result = f"Error fetching {handler['label']}: {str(e)}"

                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            yield _sse({"type": "status", "message": "Analyzing results..."})

            final_response = await litellm.acompletion(
                model=model,
                messages=chat_messages,
                stream=True,
                temperature=0.2,
            )

            async for chunk in final_response:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield _sse({"type": "chunk", "content": delta_content})
        else:
            content = choice.message.content or ""
            if content:
                yield _sse({"type": "chunk", "content": content})

        yield _sse({"type": "done"})

    except Exception as e:
        verbose_proxy_logger.error("AI usage chat failed: %s", e)
        yield _sse({"type": "error", "message": str(e)})
