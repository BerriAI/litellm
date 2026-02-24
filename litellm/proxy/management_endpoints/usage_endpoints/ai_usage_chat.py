"""
AI Usage Chat - uses LLM tool calling to answer questions about
usage/spend data from the /user/daily/activity endpoints.
"""

import json
from datetime import date, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_COMPETITOR_DISCOVERY_MODEL

GET_USAGE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "get_usage_data",
        "description": (
            "Fetch aggregated usage/spend data for the LiteLLM proxy. "
            "Returns daily spend, token usage, request counts, and breakdowns "
            "by model, provider, and API key for the given date range. "
            "Always call this tool first to get data before answering."
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

SYSTEM_PROMPT = (
    "You are an AI assistant embedded in the LiteLLM Usage dashboard. "
    "You help users understand their LLM API spend and usage data.\n\n"
    "You have access to a tool called `get_usage_data` which fetches "
    "aggregated usage data from the LiteLLM database. You MUST call "
    "this tool first to retrieve the data, then answer the user's question "
    "based on the returned data.\n\n"
    "Guidelines:\n"
    "- Be concise and specific. Use exact numbers from the data.\n"
    "- Format costs as dollar amounts (e.g. $12.34).\n"
    "- When comparing models or providers, show a ranked list.\n"
    "- If data is empty, say so clearly.\n"
    "- Do not hallucinate data — only use what the tool returns."
)


async def _fetch_usage_data(
    start_date: str,
    end_date: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the aggregated daily activity query and return serialisable dict."""
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


def _summarise_usage_data(data: Dict[str, Any]) -> str:
    """Convert the raw aggregated response into a concise text summary for the LLM."""
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
            lines.append(f"  - {name}: ${d['spend']:.4f} ({int(d['requests'])} requests, {int(d['tokens'])} tokens)")
    else:
        lines.append("Models: (no data)")

    lines.append("")

    if provider_spend:
        lines.append("Top Providers by Spend:")
        for name, d in sorted(provider_spend.items(), key=lambda x: -x[1]["spend"])[:10]:
            lines.append(f"  - {name}: ${d['spend']:.4f} ({int(d['requests'])} requests)")
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
            lines.append(f"  - {day['date']}: ${m.get('spend', 0):.4f} ({m.get('api_requests', 0)} requests)")

    return "\n".join(lines)


async def stream_usage_ai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    user_id: Optional[str] = None,
    is_admin: bool = False,
) -> AsyncIterator[str]:
    """
    Stream an AI chat response about usage data.

    Yields SSE-formatted events:
      data: {"type": "chunk", "content": "..."}
      data: {"type": "done"}
      data: {"type": "error", "message": "..."}
    """
    model = model or DEFAULT_COMPETITOR_DISCOVERY_MODEL

    chat_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *messages,
    ]

    try:
        response = await litellm.acompletion(
            model=model,
            messages=chat_messages,
            tools=[GET_USAGE_DATA_TOOL],
            temperature=0.2,
        )

        choice = response.choices[0]  # type: ignore
        tool_calls = choice.message.tool_calls

        if tool_calls:
            chat_messages.append(choice.message.model_dump())

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name == "get_usage_data":
                    effective_user = None
                    if not is_admin:
                        effective_user = user_id
                    elif fn_args.get("user_id"):
                        effective_user = fn_args["user_id"]

                    try:
                        raw_data = await _fetch_usage_data(
                            start_date=fn_args["start_date"],
                            end_date=fn_args["end_date"],
                            user_id=effective_user,
                        )
                        tool_result = _summarise_usage_data(raw_data)
                    except Exception as e:
                        tool_result = f"Error fetching usage data: {str(e)}"
                else:
                    tool_result = f"Unknown tool: {fn_name}"

                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

            final_response = await litellm.acompletion(
                model=model,
                messages=chat_messages,
                stream=True,
                temperature=0.2,
            )

            async for chunk in final_response:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    event = json.dumps({"type": "chunk", "content": delta_content})
                    yield f"data: {event}\n\n"
        else:
            content = choice.message.content or ""
            if content:
                event = json.dumps({"type": "chunk", "content": content})
                yield f"data: {event}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        verbose_proxy_logger.error("AI usage chat failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
