"""Ask AI agent: answers usage/spend questions by driving an LLM tool-calling
loop over the scoped usage data provider.

The LLM call goes through the proxy's own ``llm_router`` (not the bare
``litellm`` SDK), so UI-selected model groups / aliases resolve exactly as they
do on ``/chat/completions``, and the call is credentialed, logged, budgeted,
rate-limited, and guardrailed like any other proxy request.
"""

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Set, Union, cast

from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict, assert_never

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.router import Router
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Message,
    ModelResponse,
)
from litellm.proxy.management_endpoints.usage_endpoints.scoped_data import (
    ScopedUsageDataProvider,
    summarise_entity_data,
    summarise_usage_data,
)

USAGE_AI_TEMPERATURE = 0.2
MAX_CHAT_MESSAGES = 20
MAX_TOOL_ROUNDS = 5
USAGE_AI_MODEL_SETTING = "usage_ai_model"


# ---------------------------------------------------------------------------
# SSE wire events (kept identical to the v1 contract the frontend consumes)
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


class SSEChunkEvent(TypedDict):
    type: Literal["chunk"]
    content: str


class SSEDoneEvent(TypedDict):
    type: Literal["done"]


class SSEErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


SSEEvent = Union[SSEStatusEvent, SSEToolCallEvent, SSEChunkEvent, SSEDoneEvent, SSEErrorEvent]


def _sse(event: SSEEvent) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ---------------------------------------------------------------------------
# Terminal errors modelled as values, mapped to the SSE error contract once
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelNotConfigured:
    pass


@dataclass(frozen=True, slots=True)
class RouterUnavailable:
    pass


@dataclass(frozen=True, slots=True)
class LLMCallError:
    detail: str


UsageAiError = Union[ModelNotConfigured, RouterUnavailable, LLMCallError]


def _error_event(err: UsageAiError) -> SSEErrorEvent:
    match err:
        case ModelNotConfigured():
            message = (
                "No model is configured for Ask AI. Pick a model in the selector, "
                f"or set '{USAGE_AI_MODEL_SETTING}' under general_settings."
            )
        case RouterUnavailable():
            message = "The proxy has no model router initialized yet. Try again in a moment."
        case LLMCallError():
            message = "The AI request failed. Confirm the selected model is configured on this proxy and reachable."
        case _:
            assert_never(err)
    return {"type": "error", "message": message}


class _RouterUnavailableError(Exception):
    pass


def _require_router() -> Router:
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise _RouterUnavailableError()
    return llm_router


def _assembled_message(chunks: List[object]) -> Optional[Message]:
    """Reassemble streamed chunks into a single message (content + tool_calls)."""
    built = litellm.stream_chunk_builder(chunks)
    if not isinstance(built, ModelResponse) or not built.choices:
        return None
    choice = built.choices[0]
    return choice.message if isinstance(choice, Choices) else None  # pyright: ignore[reportUnnecessaryIsInstance]  # choices[0] can be StreamingChoices at runtime


def resolve_model(requested: Optional[str]) -> Union[str, ModelNotConfigured]:
    """Resolve the model group to use: explicit request wins, then the
    configured ``usage_ai_model`` setting, else an actionable error value."""
    explicit = (requested or "").strip()
    if explicit:
        return explicit

    from litellm.proxy.proxy_server import general_settings

    configured = TypeAdapter(Optional[str]).validate_python(general_settings.get(USAGE_AI_MODEL_SETTING))
    stripped = (configured or "").strip()
    return stripped or ModelNotConfigured()


# ---------------------------------------------------------------------------
# Tools (OpenAI function-calling schema) + typed argument validation
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
            "Fetch aggregated usage/spend data. Returns daily spend, token counts, "
            "request counts, and breakdowns by model and provider. Use for overall "
            "spend, top models, and top providers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "user_id": {
                    "type": "string",
                    "description": "Optional user ID filter (admin only). Omit for global view.",
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
            "Fetch usage/spend data broken down by team. Use for questions like "
            "'which team spends the most' or 'show me team X usage'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "team_ids": {"type": "string", "description": "Optional comma-separated team IDs. Omit for all teams."},
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
            "Fetch usage/spend data broken down by tag. Tags are labels attached to "
            "requests (features, environments, credentials)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **_DATE_PARAMS,
                "tags": {"type": "string", "description": "Optional comma-separated tag names. Omit for all tags."},
            },
            "required": ["start_date", "end_date"],
        },
    },
}


def tools_for_role(is_admin: bool) -> List[Dict[str, Any]]:
    return [_TOOL_USAGE, _TOOL_TEAM, _TOOL_TAG] if is_admin else [_TOOL_USAGE]


_TOOL_LABELS = {
    "get_usage_data": "global usage data",
    "get_team_usage_data": "team usage data",
    "get_tag_usage_data": "tag usage data",
}


class _UsageArgs(BaseModel):
    start_date: str
    end_date: str
    user_id: Optional[str] = None


class _TeamArgs(BaseModel):
    start_date: str
    end_date: str
    team_ids: Optional[str] = None


class _TagArgs(BaseModel):
    start_date: str
    end_date: str
    tags: Optional[str] = None


async def _dispatch_tool(name: str, raw_args: Dict[str, Any], provider: ScopedUsageDataProvider) -> str:
    if name == "get_usage_data":
        args = _UsageArgs.model_validate(raw_args)
        data = await provider.usage(args.start_date, args.end_date, args.user_id)
        return summarise_usage_data(data)
    if name == "get_team_usage_data":
        team_args = _TeamArgs.model_validate(raw_args)
        team_data = await provider.team(team_args.start_date, team_args.end_date, team_args.team_ids)
        return summarise_entity_data(team_data, "Team")
    if name == "get_tag_usage_data":
        tag_args = _TagArgs.model_validate(raw_args)
        tag_data = await provider.tag(tag_args.start_date, tag_args.end_date, tag_args.tags)
        return summarise_entity_data(tag_data, "Tag")
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = (
    "You are an AI assistant embedded in the LiteLLM Usage dashboard. "
    "You help users understand their LLM API spend and usage data.\n\n"
    "ALWAYS call the appropriate tool(s) first to fetch data before answering. "
    "You may call multiple tools, across multiple turns, if the question spans "
    "different dimensions or needs follow-up lookups.\n\n"
    "Guidelines:\n"
    "- Be concise and specific. Use exact numbers from the data.\n"
    "- Format costs as dollar amounts (e.g. $12.34).\n"
    "- When comparing entities, show a ranked list.\n"
    "- If data is empty or no results found, say so clearly.\n"
    "- Do not hallucinate data; only use what the tools return.\n"
    "- Today's date is provided below; use it to interpret relative dates like "
    "'this week', 'this month', or 'last 7 days'."
)

_TOOL_DESCRIPTIONS_ADMIN = (
    "You have access to these tools:\n"
    "- `get_usage_data`: Global/user-level usage (spend, models, providers)\n"
    "- `get_team_usage_data`: Team-level usage breakdown\n"
    "- `get_tag_usage_data`: Tag-level usage breakdown\n\n"
)

_TOOL_DESCRIPTIONS_BASE = (
    "You have access to this tool:\n- `get_usage_data`: Your usage data (spend, models, providers)\n\n"
)


def _system_prompt(is_admin: bool) -> str:
    tool_desc = _TOOL_DESCRIPTIONS_ADMIN if is_admin else _TOOL_DESCRIPTIONS_BASE
    return f"{_SYSTEM_PROMPT_BASE}\n\n{tool_desc}Today's date: {date.today().isoformat()}"


# ---------------------------------------------------------------------------
# Streaming agent loop
# ---------------------------------------------------------------------------


async def _run_tool_call(
    tc: ChatCompletionMessageToolCall,
    provider: ScopedUsageDataProvider,
    allowed_names: Set[str],
    convo: List[Dict[str, Any]],
) -> AsyncIterator[str]:
    """Execute one tool call, yield status events, and append its result to convo."""
    name = tc.function.name
    try:
        parsed = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        parsed = {}
    raw_args = parsed if isinstance(parsed, dict) else {}

    if name not in allowed_names:
        convo.append({"role": "tool", "tool_call_id": tc.id, "content": f"Tool not available: {name}"})
        return

    label = _TOOL_LABELS.get(name, name)
    base: Dict[str, Any] = {"type": "tool_call", "tool_name": name, "tool_label": label, "arguments": raw_args}
    yield _sse(cast(SSEToolCallEvent, {**base, "status": "running"}))

    try:
        result = await _dispatch_tool(name, raw_args, provider)
        yield _sse(cast(SSEToolCallEvent, {**base, "status": "complete"}))
    except Exception as e:
        verbose_proxy_logger.error("Usage AI tool %s failed: %s", name, e)
        result = f"Error fetching {label}. Please try again."
        yield _sse(cast(SSEToolCallEvent, {**base, "status": "error"}))

    convo.append({"role": "tool", "tool_call_id": tc.id, "content": result})


async def stream_usage_ai_chat(
    provider: ScopedUsageDataProvider,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream SSE events: status -> tool_call -> chunk -> done (or a single error)."""
    resolved = resolve_model(model)
    if isinstance(resolved, ModelNotConfigured):
        yield _sse(_error_event(resolved))
        return

    tools = tools_for_role(provider.is_admin)
    allowed_names = {t["function"]["name"] for t in tools}
    history = messages[-MAX_CHAT_MESSAGES:]
    convo: List[Dict[str, Any]] = [{"role": "system", "content": _system_prompt(provider.is_admin)}, *history]

    try:
        router = _require_router()
        yield _sse({"type": "status", "message": "Thinking..."})

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            use_tools = tools if round_index < MAX_TOOL_ROUNDS else None
            chunks: List[object] = []
            response = await router.acompletion(
                model=resolved,
                messages=cast(List[AllMessageValues], convo),
                tools=use_tools,
                stream=True,
                temperature=USAGE_AI_TEMPERATURE,
                metadata={"feature": "usage_ai"},
            )
            async for chunk in response:
                choices = getattr(chunk, "choices", None)
                delta = choices[0].delta if choices else None
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    yield _sse({"type": "chunk", "content": content})
                chunks.append(chunk)

            message = _assembled_message(chunks)
            tool_calls = message.tool_calls if message is not None else None

            if not message or not tool_calls:
                yield _sse({"type": "done"})
                return

            convo.append(message.model_dump())
            for tc in tool_calls:
                async for event in _run_tool_call(tc, provider, allowed_names, convo):
                    yield event

        yield _sse({"type": "done"})

    except _RouterUnavailableError:
        yield _sse(_error_event(RouterUnavailable()))
    except Exception as e:
        verbose_proxy_logger.error("Usage AI chat failed: %s", e)
        yield _sse(_error_event(LLMCallError(detail=str(e))))
