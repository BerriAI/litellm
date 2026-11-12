from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from itertools import count
from typing import Final
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.constants import request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.exceptions import APIError, AuthenticationError, BadRequestError, RateLimitError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines
from litellm.types.interactions import Error, InteractionsAPIResponse, InteractionsAPIStreamingResponse

QCA_DEFAULT_API_BASE: Final = "https://api.qoder.com/api/v1/cloud"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, object]])


class QCAEventContent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str
    text: str | None = None


class QCAStopReason(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str


class QCAEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    type: str
    content: tuple[QCAEventContent, ...] = ()
    stop_reason: QCAStopReason | None = None
    model_credits: float | None = None
    sandbox_runtime_credits: float | None = None
    total_credits: float | None = None


class QCASessionAgent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str


class QCASession(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    agent: str | QCASessionAgent | None = None
    environment_id: str | None = None


class QCAEventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: QCAEvent | None = None
    event: QCAEvent | None = None


class QCAEventsPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[QCAEvent, ...]


class QCAValidationError(ValueError):
    pass


def _require_identifier(value: str | None, prefix: str, name: str) -> str:
    if value is None or not value.startswith(prefix):
        raise QCAValidationError(f"QCA {name} must start with {prefix!r}")
    return value


def _api_base(value: str | None) -> str:
    base: Final = (value or QCA_DEFAULT_API_BASE).rstrip("/")
    parsed: Final = urlsplit(base)
    invalid: Final = (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
    )
    if invalid:
        raise QCAValidationError("QCA api_base must be an absolute HTTP URL without credentials, query, or fragment")
    return base


def _text_input(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise QCAValidationError("QCA interactions currently require a non-empty text input")
    return value


def _previous_session_id(previous_interaction_id: str | None) -> str | None:
    if previous_interaction_id is None:
        return None
    parts: Final = previous_interaction_id.split(".")
    if len(parts) != 3 or parts[0] != "qca1":
        raise QCAValidationError("QCA previous_interaction_id is invalid")
    _require_identifier(parts[2], "evt_", "event id")
    return _require_identifier(parts[1], "sess_", "session id")


def _event_from_payload(payload: str) -> QCAEvent:
    decoded: Final = _JSON_OBJECT.validate_json(payload)
    envelope: Final = QCAEventEnvelope.model_validate(decoded)
    return envelope.data or envelope.event or QCAEvent.model_validate(decoded)


def _accepted_event(payload: object) -> QCAEvent:
    page: Final = QCAEventsPage.model_validate(payload)
    if not page.data:
        raise ValueError("QCA accepted the user message without returning its event")
    return page.data[0]


def _provider_error(error: httpx.HTTPStatusError, model: str) -> Exception:
    response: Final = error.response
    message: Final = response.text or str(error)
    match response.status_code:
        case 401 | 403:
            return AuthenticationError(message=message, llm_provider="qca", model=model, response=response)
        case 429:
            return RateLimitError(message=message, llm_provider="qca", model=model, response=response)
        case status_code:
            return APIError(
                status_code=status_code,
                message=message,
                llm_provider="qca",
                model=model,
                request=response.request,
                response=response,
            )


def _interaction_id(session_id: str, input_event_id: str) -> str:
    return f"qca1.{session_id}.{input_event_id}"


def _usage(event: QCAEvent) -> Mapping[str, object] | None:
    if event.total_credits is None and event.model_credits is None and event.sandbox_runtime_credits is None:
        return None
    return {
        "qca_session_credits": {
            "model_credits": event.model_credits,
            "sandbox_runtime_credits": event.sandbox_runtime_credits,
            "total_credits": event.total_credits,
        },
        "qca_usage_scope": "session_cumulative",
        "qca_usage_is_final": event.type == "session.status_idle",
    }


def _text_step(event: QCAEvent) -> Mapping[str, object] | None:
    text: Final = "".join(content.text or "" for content in event.content if content.type == "text")
    if event.type != "agent.message" or not text:
        return None
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def _text_delta(step: Mapping[str, object]) -> Mapping[str, object]:
    try:
        content: Final = _JSON_OBJECTS.validate_python(step.get("content"))
    except ValidationError:
        return {}
    return content[0] if content else {}


def _step_from_delta(delta: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "model_output", "content": [dict(delta)]}


def _terminal_status(event: QCAEvent) -> str | None:
    if event.type in {"session.error", "session.deleted", "session.status_terminated"}:
        return "failed"
    if event.type != "session.status_idle":
        return None
    if event.stop_reason is not None and event.stop_reason.type == "requires_action":
        return "requires_action"
    return "completed"


def _normalized_event(event: QCAEvent) -> tuple[Mapping[str, object] | None, str | None]:
    return _text_step(event), _terminal_status(event)


async def _stream_events(response: httpx.Response) -> AsyncIterator[QCAEvent]:
    data_lines: tuple[str, ...] = ()  # rebind-ok: immutable SSE record accumulation
    async for line in aiter_sse_lines(response.aiter_text()):
        if line == "":
            if data_lines:
                yield _event_from_payload("\n".join(data_lines))
                data_lines = ()
            continue
        if line.startswith("data:"):
            data_lines = (*data_lines, line[5:].lstrip())
    if data_lines:
        yield _event_from_payload("\n".join(data_lines))


async def _normalized_stream(
    response: httpx.Response, interaction_id: str, agent: str
) -> AsyncIterator[InteractionsAPIStreamingResponse]:
    created: Final = datetime.now(timezone.utc).isoformat()
    step_indexes: Final = count()
    try:
        yield InteractionsAPIStreamingResponse(
            event_type="interaction.created",
            interaction_id=interaction_id,
            interaction={
                "id": interaction_id,
                "agent": agent,
                "status": "in_progress",
                "created": created,
                "updated": created,
                "steps": [],
            },
        )
        async for event in _stream_events(response):
            step, status = _normalized_event(event)
            if step is not None:
                (step_index,) = (next(step_indexes),)
                yield InteractionsAPIStreamingResponse(
                    event_type="step.start",
                    interaction_id=interaction_id,
                    index=step_index,
                    step={"type": "model_output", "content": []},
                )
                yield InteractionsAPIStreamingResponse(
                    event_type="step.delta",
                    interaction_id=interaction_id,
                    index=step_index,
                    delta=dict(_text_delta(step)),
                )
                yield InteractionsAPIStreamingResponse(
                    event_type="step.stop",
                    interaction_id=interaction_id,
                    index=step_index,
                    status="done",
                )
            if status is not None:
                if status == "failed":
                    yield InteractionsAPIStreamingResponse(
                        event_type="error",
                        interaction_id=interaction_id,
                        error=Error(
                            code="qca_session_failed",
                            message=f"Qoder Cloud Agents session ended with {event.type}",
                        ),
                    )
                    return
                yield InteractionsAPIStreamingResponse(
                    event_type=f"interaction.{status}",
                    interaction_id=interaction_id,
                    interaction={
                        "id": interaction_id,
                        "agent": agent,
                        "status": status,
                        "created": created,
                        "updated": datetime.now(timezone.utc).isoformat(),
                        "usage": _usage(event),
                    },
                )
                return
        raise httpx.RemoteProtocolError("QCA event stream ended before the session reached a terminal state")
    finally:
        await response.aclose()


async def _closing_stream(
    stream: AsyncIterator[InteractionsAPIStreamingResponse], client: httpx.AsyncClient
) -> AsyncIterator[InteractionsAPIStreamingResponse]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await client.aclose()


def _response_from_terminal(
    terminal: InteractionsAPIStreamingResponse,
    interaction_id: str,
    agent: str,
    steps: tuple[Mapping[str, object], ...],
) -> InteractionsAPIResponse:
    if terminal.interaction is None:
        raise httpx.RemoteProtocolError("QCA terminal event is missing its interaction")
    status: Final = terminal.interaction.get("status")
    usage_value: Final = terminal.interaction.get("usage")
    created: Final = terminal.interaction.get("created")
    updated: Final = terminal.interaction.get("updated")
    usage: Final = _JSON_OBJECT.validate_python(usage_value) if isinstance(usage_value, dict) else None
    return InteractionsAPIResponse(
        id=interaction_id,
        agent=agent,
        status=status if isinstance(status, str) else None,
        created=created if isinstance(created, str) else None,
        updated=updated if isinstance(updated, str) else None,
        steps=[dict(step) for step in steps],
        usage=usage,
    )


def _terminal_for_logging(
    terminal: InteractionsAPIStreamingResponse,
    interaction_id: str,
    agent: str,
    steps: tuple[Mapping[str, object], ...],
) -> InteractionsAPIStreamingResponse:
    response: Final = _response_from_terminal(terminal, interaction_id, agent, steps)
    return terminal.model_copy(update={"interaction": response.model_dump()})


async def _observed_stream(
    stream: AsyncIterator[InteractionsAPIStreamingResponse],
    interaction_id: str,
    agent: str,
    logging_obj: LiteLLMLoggingObj,
    start_time: datetime,
) -> AsyncIterator[InteractionsAPIStreamingResponse]:
    steps: tuple[Mapping[str, object], ...] = ()  # rebind-ok: immutable response accumulation
    async for chunk in stream:
        if chunk.event_type == "step.delta" and chunk.delta is not None:
            steps = (*steps, _step_from_delta(chunk.delta))
        if chunk.event_type in {"interaction.completed", "interaction.requires_action"}:
            await logging_obj.dispatch_success_handlers(  # pyright: ignore[reportUnknownMemberType]  # legacy callback API is untyped
                result=_terminal_for_logging(chunk, interaction_id, agent, steps),
                start_time=start_time,
                end_time=datetime.now(timezone.utc),
                cache_hit=None,
                prefer_async_handlers=True,
            )
        yield chunk


async def _collect_response(
    stream: AsyncIterator[InteractionsAPIStreamingResponse], interaction_id: str, agent: str
) -> InteractionsAPIResponse:
    steps: tuple[Mapping[str, object], ...] = ()  # rebind-ok: immutable response accumulation
    terminal: InteractionsAPIStreamingResponse | None = None  # rebind-ok: captures the terminal chunk
    async for chunk in stream:
        if chunk.event_type == "step.delta" and chunk.delta is not None:
            steps = (*steps, _step_from_delta(chunk.delta))
        if chunk.event_type == "error":
            raise APIError(
                status_code=502,
                message=(
                    chunk.error.message
                    if chunk.error is not None and chunk.error.message is not None
                    else "Qoder Cloud Agents session failed"
                ),
                llm_provider="qca",
                model=agent,
            )
        if chunk.event_type in {"interaction.completed", "interaction.requires_action"}:
            terminal = chunk
    if terminal is None:
        raise httpx.RemoteProtocolError("QCA event stream did not produce a terminal interaction")
    return _response_from_terminal(terminal, interaction_id, agent, steps)


async def _session_id(
    client: httpx.AsyncClient,
    base: str,
    headers: Mapping[str, str],
    agent_id: str,
    environment_id: str,
    previous_session_id: str | None,
) -> str:
    if previous_session_id is not None:
        existing_response: Final = await client.get(f"{base}/sessions/{previous_session_id}", headers=headers)
        existing_response.raise_for_status()
        session: Final = QCASession.model_validate(existing_response.json())
        session_agent: Final = session.agent.id if isinstance(session.agent, QCASessionAgent) else session.agent
        if session.id != previous_session_id or session_agent != agent_id or session.environment_id != environment_id:
            raise QCAValidationError("QCA previous_interaction_id does not match the configured agent and environment")
        return previous_session_id
    create_response: Final = await client.post(
        f"{base}/sessions",
        headers=headers,
        json={"agent": agent_id, "environment_id": environment_id, "metadata": {"source": "litellm"}},
    )
    create_response.raise_for_status()
    return _require_identifier(QCASession.model_validate(create_response.json()).id, "sess_", "session id")


async def create_qca_interaction(
    *,
    agent: str | None,
    input: object,
    environment: object,
    previous_interaction_id: str | None,
    stream: bool | None,
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    unsupported_params: Mapping[str, object],
    logging_obj: LiteLLMLoggingObj,
    client: httpx.AsyncClient | None = None,
) -> InteractionsAPIResponse | AsyncIterator[InteractionsAPIStreamingResponse]:
    try:
        unsupported: Final = tuple(
            name
            for name, value in unsupported_params.items()
            if value not in (None, (), {}) and (name == "store" or value is not False)
        )
        if unsupported:
            raise QCAValidationError(f"QCA interactions do not support {', '.join(sorted(unsupported))}")
        agent_id: Final = _require_identifier(agent, "agent_", "agent id")
        environment_id: Final = _require_identifier(
            environment if isinstance(environment, str) else None,
            "env_",
            "environment id",
        )
        token: Final = api_key
        if not token:
            raise QCAValidationError("QCA api_key is required")
        base: Final = _api_base(api_base)
        text: Final = _text_input(input)
        previous_session_id: Final = _previous_session_id(previous_interaction_id)
    except QCAValidationError as error:
        raise BadRequestError(message=str(error), model=agent or "", llm_provider="qca") from error
    owns_client: Final = client is None
    http_client: Final = client or httpx.AsyncClient(
        timeout=timeout if timeout is not None else DEFAULT_REQUEST_TIMEOUT
    )
    start_time: Final = datetime.now(timezone.utc)
    try:
        headers: Final = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        session_id: Final = await _session_id(
            http_client,
            base,
            headers,
            agent_id,
            environment_id,
            previous_session_id,
        )
        event_response: Final = await http_client.post(
            f"{base}/sessions/{session_id}/events",
            headers=headers,
            json={"events": [{"type": "user.message", "content": [{"type": "text", "text": text}]}]},
        )
        event_response.raise_for_status()
        input_event: Final = _accepted_event(_JSON_OBJECT.validate_json(event_response.content))
        input_event_id: Final = _require_identifier(input_event.id, "evt_", "event id")
        interaction_id: Final = _interaction_id(session_id, input_event_id)
        stream_request: Final = http_client.build_request(
            "GET",
            f"{base}/sessions/{session_id}/events/stream",
            headers={**headers, "Accept": "text/event-stream", "Last-Event-ID": input_event_id},
        )
        stream_response: Final = await http_client.send(stream_request, stream=True)
        if stream_response.is_error:
            await stream_response.aread()
        stream_response.raise_for_status()
        normalized: Final = _normalized_stream(stream_response, interaction_id, agent_id)
        if stream is True:
            observed: Final = _observed_stream(normalized, interaction_id, agent_id, logging_obj, start_time)
            return _closing_stream(observed, http_client) if owns_client else observed
        result: Final = await _collect_response(normalized, interaction_id, agent_id)
    except httpx.HTTPStatusError as error:
        if owns_client:
            await http_client.aclose()
        raise _provider_error(error, agent_id) from error
    except QCAValidationError as error:
        if owns_client:
            await http_client.aclose()
        raise BadRequestError(message=str(error), model=agent_id, llm_provider="qca") from error
    except BaseException:
        if owns_client:
            await http_client.aclose()
        raise
    else:
        if owns_client:
            await http_client.aclose()
        return result
