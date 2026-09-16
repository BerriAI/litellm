"""Anthropic Managed Agents sessions behind the Interactions API (beta managed-agents-2026-04-01).

A create starts a session with the prompt as its first event and returns ``in_progress``; the
session settles on Anthropic's side and get reads it back. The session object only learns its
cost when a turn ends, so usage comes from the newest ``session.usage`` event, and its
``list_cost`` (whole US cents, the provider's own price) travels in the response-cost header so
the settlement is billed as reported instead of repriced from tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.anthropic.common_utils import AnthropicError
from litellm.llms.anthropic.managed_agents import (
    invalid_request,
    managed_agents_api_base,
    managed_agents_headers,
)
from litellm.llms.base_llm.interactions.session_transformation import BaseSessionInteractionsConfig
from litellm.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_RESPONSE_COST_HEADER: Final = "llm_provider-x-litellm-response-cost"
_TRANSCRIPT_PAGE_SIZE: Final = 100
_TERMINAL_RETRY_STATUSES: Final = frozenset(("exhausted", "terminal"))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _TextBlock(_Frozen):
    type: Literal["text"] = "text"
    text: str


class _UserTurn(_Frozen):
    role: Literal["user"]
    content: str | tuple[_TextBlock, ...]


class _Input(_Frozen):
    value: str | tuple[_TextBlock, ...] | tuple[_UserTurn, ...]


class _AgentRef(_Frozen):
    type: Literal["agent"] = "agent"
    id: str


class _UserMessage(_Frozen):
    type: Literal["user.message"] = "user.message"
    content: tuple[_TextBlock, ...]


class _CreateSessionRequest(_Frozen):
    agent: _AgentRef
    environment_id: str
    initial_events: tuple[_UserMessage, ...]


class _Interrupt(_Frozen):
    type: Literal["user.interrupt"] = "user.interrupt"


class _SendEventsRequest(_Frozen):
    events: tuple[_Interrupt, ...]


class _ModelRef(_Frozen):
    id: str | None = None


class _AgentSnapshot(_Frozen):
    id: str | None = None
    model: _ModelRef | None = None


class _MonetaryAmount(_Frozen):
    amount: str
    currency: str


class _CacheCreation(_Frozen):
    ephemeral_1h_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0


class _SessionUsage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation: _CacheCreation | None = None
    list_cost: _MonetaryAmount | None = None


class _Session(_Frozen):
    id: str
    status: Literal["rescheduling", "running", "idle", "terminated"]
    agent: _AgentSnapshot | None = None
    created_at: str | None = None
    updated_at: str | None = None
    usage: _SessionUsage | None = None


class _StopReason(_Frozen):
    type: str


class _RetryStatus(_Frozen):
    type: str


class _SessionError(_Frozen):
    retry_status: _RetryStatus | None = None


class _Event(_Frozen):
    id: str | None = None
    type: str
    content: tuple[Mapping[str, object], ...] | None = None
    name: str | None = None
    input: Mapping[str, object] | None = None
    mcp_server_name: str | None = None
    tool_use_id: str | None = None
    custom_tool_use_id: str | None = None
    mcp_tool_use_id: str | None = None
    is_error: bool | None = None
    stop_reason: _StopReason | None = None
    usage: _SessionUsage | None = None
    error: _SessionError | None = None


class _EventsPage(_Frozen):
    data: tuple[_Event, ...] = ()


class _GoogleUsage(_Frozen):
    total_input_tokens: int
    total_cached_tokens: int
    total_output_tokens: int
    total_tokens: int


class _Text(_Frozen):
    type: Literal["text"] = "text"
    text: str


class _FunctionCall(_Frozen):
    type: Literal["function_call"] = "function_call"
    id: str
    name: str
    arguments: Mapping[str, object]


class _FunctionResult(_Frozen):
    type: Literal["function_result"] = "function_result"
    call_id: str
    result: str
    is_error: bool


class _McpCall(_Frozen):
    type: Literal["mcp_server_tool_call"] = "mcp_server_tool_call"
    id: str
    name: str
    server_name: str
    arguments: Mapping[str, object]


class _McpResult(_Frozen):
    type: Literal["mcp_server_tool_result"] = "mcp_server_tool_result"
    call_id: str
    result: str


class _CodeArguments(_Frozen):
    code: str


class _CodeCall(_Frozen):
    type: Literal["code_execution_call"] = "code_execution_call"
    id: str
    arguments: _CodeArguments


class _CodeResult(_Frozen):
    type: Literal["code_execution_result"] = "code_execution_result"
    call_id: str
    result: str
    is_error: bool


_StepContent: TypeAlias = _Text | _FunctionCall | _FunctionResult | _McpCall | _McpResult | _CodeCall | _CodeResult


class _Step(_Frozen):
    type: Literal["user_input", "model_output"]
    content: tuple[_StepContent, ...]


class _Items(_Frozen):
    items: tuple[object, ...]


def _plain(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, str):
        return value
    try:
        items: Final = _Items.model_validate(MappingProxyType({"items": value})).items
    except ValidationError:
        return value
    return tuple(_plain(item) for item in items)


def _input_texts(input: InteractionInput | None) -> tuple[str, ...] | None:
    try:
        value: Final = _Input.model_validate(MappingProxyType({"value": _plain(input)})).value
    except ValidationError:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(item.text if isinstance(item, _TextBlock) else _turn_text(item) for item in value if _has_text(item))


def _turn_text(turn: _UserTurn) -> str:
    return turn.content if isinstance(turn.content, str) else "".join(block.text for block in turn.content)


def _has_text(item: _TextBlock | _UserTurn) -> bool:
    return bool(item.text) if isinstance(item, _TextBlock) else bool(_turn_text(item))


def _text_of(content: Sequence[Mapping[str, object]] | None) -> str:
    return "".join(
        text for block in content or () if block.get("type") == "text" and isinstance(text := block.get("text"), str)
    )


def _google_usage(usage: _SessionUsage) -> _GoogleUsage:
    cache_creation: Final = usage.cache_creation or _CacheCreation()
    total_input: Final = (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + cache_creation.ephemeral_1h_input_tokens
        + cache_creation.ephemeral_5m_input_tokens
    )
    return _GoogleUsage(
        total_input_tokens=total_input,
        total_cached_tokens=usage.cache_read_input_tokens,
        total_output_tokens=usage.output_tokens,
        total_tokens=total_input + usage.output_tokens,
    )


def _list_cost_usd(usage: _SessionUsage | None) -> float | None:
    if usage is None or usage.list_cost is None or usage.list_cost.currency != "USD":
        return None
    return int(usage.list_cost.amount) / 100


def _session_only_status(session: _Session) -> str:
    match session.status:
        case "running" | "rescheduling":
            return "in_progress"
        case "terminated":
            return "failed"
        case "idle":
            return "completed"


def _newest(events: Sequence[_Event], event_type: str) -> int | None:
    return next((index for index, event in enumerate(events) if event.type == event_type), None)


def _newest_user_event(events: Sequence[_Event]) -> int | None:
    return next((index for index, event in enumerate(events) if event.type.startswith("user.")), None)


def _newest_dead_error(events: Sequence[_Event]) -> int | None:
    return next(
        (
            index
            for index, event in enumerate(events)
            if event.type == "session.error"
            and event.error is not None
            and event.error.retry_status is not None
            and event.error.retry_status.type in _TERMINAL_RETRY_STATUSES
        ),
        None,
    )


def _status(session: _Session, events: Sequence[_Event]) -> str:
    """``events`` is newest first. An idle session is judged by the events after its last idle:
    a newer user event means a turn we started is still queued, a newer dead error means the
    turn died, otherwise the idle's stop reason says how the turn ended.
    """
    if session.status != "idle":
        return _session_only_status(session)
    idle: Final = _newest(events, "session.status_idle")
    if idle is None:
        return "completed"
    user_input: Final = _newest_user_event(events)
    if user_input is not None and user_input < idle:
        return "in_progress"
    dead: Final = _newest_dead_error(events)
    if dead is not None and dead < idle:
        return "failed"
    stop_reason: Final = events[idle].stop_reason
    match stop_reason.type if stop_reason else "end_turn":
        case "requires_action":
            return "requires_action"
        case "budget_reached":
            return "budget_exceeded"
        case "retries_exhausted":
            return "failed"
        case _:
            return "completed"


def _settled_usage(session: _Session, events: Sequence[_Event]) -> _SessionUsage | None:
    return next((event.usage for event in events if event.type == "session.usage" and event.usage), session.usage)


def _step(event: _Event, bash_call_ids: frozenset[str]) -> _Step | None:
    match event.type:
        case "user.message":
            return _Step(type="user_input", content=(_Text(text=_text_of(event.content)),))
        case "agent.message":
            return _Step(type="model_output", content=(_Text(text=_text_of(event.content)),))
        case "agent.custom_tool_use" if event.id and event.name:
            return _Step(
                type="model_output",
                content=(_FunctionCall(id=event.id, name=event.name, arguments=event.input or MappingProxyType({})),),
            )
        case "user.custom_tool_result" if event.custom_tool_use_id:
            return _Step(
                type="user_input",
                content=(
                    _FunctionResult(
                        call_id=event.custom_tool_use_id,
                        result=_text_of(event.content),
                        is_error=event.is_error is True,
                    ),
                ),
            )
        case "agent.mcp_tool_use" if event.id and event.name and event.mcp_server_name:
            return _Step(
                type="model_output",
                content=(
                    _McpCall(
                        id=event.id,
                        name=event.name,
                        server_name=event.mcp_server_name,
                        arguments=event.input or MappingProxyType({}),
                    ),
                ),
            )
        case "agent.mcp_tool_result" if event.mcp_tool_use_id:
            return _Step(
                type="model_output",
                content=(_McpResult(call_id=event.mcp_tool_use_id, result=_text_of(event.content)),),
            )
        case "agent.tool_use" if event.id and event.name == "bash":
            command: Final = (event.input or MappingProxyType({})).get("command")
            return _Step(
                type="model_output",
                content=(_CodeCall(id=event.id, arguments=_CodeArguments(code=str(command or ""))),),
            )
        case "agent.tool_result" if event.tool_use_id in bash_call_ids:
            return _Step(
                type="model_output",
                content=(
                    _CodeResult(
                        call_id=str(event.tool_use_id),
                        result=_text_of(event.content),
                        is_error=event.is_error is True,
                    ),
                ),
            )
        case _:
            return None


def _steps(events: Sequence[_Event]) -> tuple[Mapping[str, object], ...]:
    """``events`` is newest first; steps are oldest first."""
    bash_call_ids: Final = frozenset(
        event.id for event in events if event.type == "agent.tool_use" and event.name == "bash" and event.id
    )
    return tuple(
        step.model_dump(mode="json", exclude_none=True)
        for event in reversed(events)
        if (step := _step(event, bash_call_ids)) is not None
    )


class AnthropicSessionsInteractionsConfig(BaseSessionInteractionsConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.ANTHROPIC

    def get_supported_params(self, model: str) -> list[str]:  # mutable-ok: BaseInteractionsAPIConfig signature
        return ["agent", "input", "environment", "background"]  # mutable-ok: BaseInteractionsAPIConfig signature

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> AnthropicError:
        return AnthropicError(status_code=status_code, message=error_message, headers=httpx.Headers(headers))

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict[str, str]:  # mutable-ok: BaseInteractionsAPIConfig.validate_environment signature
        return managed_agents_headers(
            headers,
            api_key=litellm_params.api_key if litellm_params else None,
            api_base=litellm_params.api_base if litellm_params else None,
        )

    def _session_url(self, api_base: str | None, interaction_id: str) -> str:
        session_id: Final = encode_url_path_segment(interaction_id, field_name="interaction_id")
        return f"{managed_agents_api_base(api_base or None)}/v1/sessions/{session_id}"

    def get_complete_url(
        self,
        api_base: str | None,
        model: str | None,
        agent: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        stream: bool | None = None,
    ) -> str:
        return f"{managed_agents_api_base(api_base or None)}/v1/sessions"

    def transform_request(
        self,
        model: str | None,
        agent: str | None,
        input: InteractionInput | None,
        optional_params: InteractionsAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: BaseInteractionsAPIConfig.transform_request signature
        if optional_params.get("stream"):
            raise invalid_request("streaming an Anthropic managed agent session is not supported yet; poll with get")
        if optional_params.get("previous_interaction_id"):
            raise invalid_request(
                "continuing an Anthropic managed agent session (previous_interaction_id) is not supported yet"
            )
        if not agent:
            raise invalid_request(
                "Anthropic managed agent sessions need agent=<agent id>; model= interactions go to the Responses bridge"
            )
        environment: Final = optional_params.get("environment")
        if not isinstance(environment, str):
            raise invalid_request(
                "Anthropic managed agent sessions need environment=<environment id>, created with POST /v1/environments"
            )
        texts: Final = _input_texts(input)
        if not texts:
            raise invalid_request("Anthropic managed agent sessions start from user text input")
        request: Final = _CreateSessionRequest(
            agent=_AgentRef(id=agent),
            environment_id=environment,
            initial_events=(_UserMessage(content=tuple(_TextBlock(text=text) for text in texts)),),
        )
        return request.model_dump(mode="json", exclude_none=True)

    def _session(self, raw_response: httpx.Response) -> _Session:
        if not 200 <= raw_response.status_code < 300:
            raise AnthropicError(
                status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers
            )
        try:
            return _Session.model_validate_json(raw_response.content)
        except ValidationError as e:
            raise AnthropicError(
                status_code=raw_response.status_code,
                message=f"response does not match the Anthropic session schema: {e}",
                headers=raw_response.headers,
            )

    def _events(self, raw_response: httpx.Response) -> tuple[_Event, ...]:
        if not 200 <= raw_response.status_code < 300:
            raise AnthropicError(
                status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers
            )
        try:
            return _EventsPage.model_validate_json(raw_response.content).data
        except ValidationError as e:
            raise AnthropicError(
                status_code=raw_response.status_code,
                message=f"response does not match the Anthropic session events schema: {e}",
                headers=raw_response.headers,
            )

    def _interaction(
        self,
        session: _Session,
        status: str,
        usage: _SessionUsage | None,
        steps: tuple[Mapping[str, object], ...],
        raw_response: httpx.Response,
    ) -> InteractionsAPIResponse:
        response: Final = InteractionsAPIResponse.model_validate(
            MappingProxyType(
                {
                    "id": session.id,
                    "object": "interaction",
                    "agent": session.agent.id if session.agent else None,
                    "model": session.agent.model.id if session.agent and session.agent.model else None,
                    "status": status,
                    "created": session.created_at,
                    "updated": session.updated_at,
                    "steps": steps,
                    "usage": _google_usage(usage).model_dump() if usage else None,
                }
            )
        )
        cost: Final = _list_cost_usd(usage)
        if cost is not None:
            response._hidden_params["additional_headers"] = {  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # mutable-ok: response_cost_calculator reads the provider price from the response's hidden params
                _RESPONSE_COST_HEADER: cost
            }
        return response

    def transform_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        logging_obj.post_call(original_response=raw_response.text)  # pyright: ignore[reportUnknownMemberType]  # Logging.post_call is untyped
        session: Final = self._session(raw_response)
        verbose_logger.debug("Anthropic managed agent session created: %s %s", session.id, session.status)
        return self._interaction(
            session, status=_session_only_status(session), usage=None, steps=(), raw_response=raw_response
        )

    def transform_streaming_response(
        self,
        model: str | None,
        parsed_chunk: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIStreamingResponse:
        raise invalid_request("streaming an Anthropic managed agent session is not supported yet; poll with get")

    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseInteractionsAPIConfig signature
        return self._session_url(api_base, interaction_id), {}  # mutable-ok: BaseInteractionsAPIConfig signature

    def transform_get_transcript_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
    ) -> tuple[str, Mapping[str, object]]:
        return (
            f"{self._session_url(api_base, interaction_id)}/events",
            MappingProxyType({"order": "desc", "limit": _TRANSCRIPT_PAGE_SIZE}),
        )

    def transform_get_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        session: Final = self._session(raw_response)
        return self._interaction(
            session, status=_session_only_status(session), usage=session.usage, steps=(), raw_response=raw_response
        )

    def assemble_get_response(
        self,
        session_response: httpx.Response,
        transcript_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        session: Final = self._session(session_response)
        events: Final = self._events(transcript_response)
        return self._interaction(
            session,
            status=_status(session, events),
            usage=_settled_usage(session, events),
            steps=_steps(events),
            raw_response=session_response,
        )

    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseInteractionsAPIConfig signature
        return self._session_url(api_base, interaction_id), {}  # mutable-ok: BaseInteractionsAPIConfig signature

    def transform_delete_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        interaction_id: str,
    ) -> DeleteInteractionResult:
        if not 200 <= raw_response.status_code < 300:
            raise AnthropicError(
                status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers
            )
        return DeleteInteractionResult(success=True, id=interaction_id)

    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseInteractionsAPIConfig signature
        return (
            f"{self._session_url(api_base, interaction_id)}/events",
            _SendEventsRequest(events=(_Interrupt(),)).model_dump(mode="json"),
        )

    def transform_cancel_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelInteractionResult:
        """The interrupt is queued, the session stays running until the agent applies it."""
        if not 200 <= raw_response.status_code < 300:
            raise AnthropicError(
                status_code=raw_response.status_code, message=raw_response.text, headers=raw_response.headers
            )
        return CancelInteractionResult(id=None, status="in_progress")
