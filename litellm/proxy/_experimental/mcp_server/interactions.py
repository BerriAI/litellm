import asyncio
import hashlib
import json
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, TypeVar
from uuid import uuid4

from mcp import MCPError
from mcp.client._input_required import run_input_required_driver
from mcp.client.session import ClientRequestContext, ClientSession
from mcp.types import (
    CallToolRequest,
    ElicitRequest,
    ElicitRequestURLParams,
    ErrorData,
    GetPromptRequest,
    InputRequest,
    InputRequests,
    InputRequiredResult,
    InputResponse,
    InputResponses,
    ReadResourceRequest,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.proxy._experimental.mcp_server.contracts import OperationContext
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error
from litellm.proxy._experimental.mcp_server.state_tokens import StateTokenError, open_state, seal_state
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@dataclass(frozen=True, slots=True)
class LegacyClientInteraction:
    session: ClientSession

    async def request(self, key: str, request: InputRequest) -> InputResponse | ErrorData:
        context: Final = ClientRequestContext(
            session=self.session, request_id=key, meta=request.params.meta if request.params else None
        )
        legacy_request: Final = (
            request.model_copy(update={"params": request.params.model_copy(update={"elicitation_id": str(uuid4())})})
            if isinstance(request, ElicitRequest)
            and isinstance(request.params, ElicitRequestURLParams)
            and request.params.elicitation_id is None
            else request
        )
        return await self.session.dispatch_input_request(context, legacy_request)


InteractionOperation: TypeAlias = CallToolRequest | GetPromptRequest | ReadResourceRequest
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_PURPOSE: Final = "mcp:interaction:repeatable:v1"


class BoundInputRequiredResult(InputRequiredResult):
    target_id: str | None = Field(default=None, exclude=True)
    target_digest: str | None = Field(default=None, exclude=True)
    gateway_responses: InputResponses | None = Field(default=None, exclude=True)


class ContinuationState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal: str
    operation: str
    target_id: str
    target_digest: str
    upstream_state: str | None
    gateway_responses: InputResponses | None = None
    policy: Literal["repeatable"] = "repeatable"
    expires_at: int
    nonce: str


def _digest(value: JsonValue) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def target_digest(server: MCPServer) -> str:
    return _digest(
        _JSON.validate_python(
            {
                "id": server.server_id,
                "url": server.url,
                "transport": server.transport,
                "protocol": server.protocol_version,
                "command": server.command,
                "args": server.args,
            }
        )
    )


def bind_target(result: InputRequiredResult, server: MCPServer) -> BoundInputRequiredResult:
    bound: Final = (
        result
        if isinstance(result, BoundInputRequiredResult)
        else BoundInputRequiredResult.model_validate(result.model_dump())
    )
    return bound.model_copy(update={"target_id": server.server_id, "target_digest": target_digest(server)})


def _principal(context: OperationContext) -> str:
    caller: Final = context.user_api_key_auth
    if caller is None or not caller.user_id:
        raise MCPError(code=-32602, message="MCP continuations require an authenticated caller identity")
    return _digest(
        _JSON.validate_python(
            {
                "user": caller.user_id,
                "team": caller.team_id,
                "org": caller.org_id,
                "end_user": caller.end_user_id,
                "servers": sorted(context.mcp_servers) if context.mcp_servers is not None else None,
            }
        )
    )


def _operation(operation: InteractionOperation) -> str:
    return _digest(
        _JSON.validate_python(
            {
                "method": operation.method,
                "params": operation.params.model_dump(
                    mode="json", by_alias=True, exclude={"meta", "input_responses", "request_state"}
                ),
            }
        )
    )


def _state_error(error: StateTokenError) -> MCPError:
    return MCPError(
        code=-32602,
        message=(
            "Set the same LITELLM_SALT_KEY on every replica to enable MCP continuations"
            if error is StateTokenError.MISSING_KEY
            else "Invalid or expired MCP continuation; start a fresh request"
        ),
    )


def open_continuation(
    operation: InteractionOperation, context: OperationContext, *, now: int
) -> ContinuationState | None:
    token: Final = operation.params.request_state
    if token is None:
        if operation.params.input_responses:
            raise MCPError(code=-32602, message="Input responses require a gateway continuation")
        return None
    opened: Final = open_state(token, purpose=_PURPOSE, now=now)
    if isinstance(opened, Error):
        raise _state_error(opened.error)
    try:
        state: Final = ContinuationState.model_validate(opened.ok)
    except ValidationError as error:
        raise _state_error(StateTokenError.INVALID) from error
    if state.principal != _principal(context) or state.operation != _operation(operation) or state.expires_at <= now:
        raise _state_error(StateTokenError.INVALID)
    return state


def seal_continuation(
    result: BoundInputRequiredResult,
    operation: InteractionOperation,
    context: OperationContext,
    *,
    now: int,
    previous: ContinuationState | None = None,
) -> InputRequiredResult:
    if result.target_id is None or result.target_digest is None:
        raise MCPError(code=-32602, message="MCP continuation target is unavailable")
    state: Final = ContinuationState(
        principal=_principal(context),
        operation=_operation(operation),
        target_id=result.target_id,
        target_digest=result.target_digest,
        upstream_state=result.request_state,
        gateway_responses=result.gateway_responses,
        expires_at=previous.expires_at if previous is not None else now + 600,
        nonce=previous.nonce if previous is not None else secrets.token_urlsafe(24),
    )
    sealed: Final = seal_state(
        _JSON.validate_json(state.model_dump_json()), purpose=_PURPOSE, expires_at=state.expires_at, now=now
    )
    if isinstance(sealed, Error):
        raise _state_error(sealed.error)
    return InputRequiredResult(input_requests=result.input_requests, request_state=sealed.ok, _meta=result.meta)


_Terminal: Final = TypeVar("_Terminal")


@dataclass(frozen=True, slots=True)
class _DeferredInteraction:
    result: BoundInputRequiredResult


@dataclass(frozen=True, slots=True)
class ModernClientInteraction:
    session: ClientSession
    allow_elicitation: bool

    async def request(self, key: str, request: InputRequest) -> InputResponse | ErrorData:
        if isinstance(request, ElicitRequest):
            return ErrorData(code=-32602, message="Modern elicitation requires a continuation")
        return await LegacyClientInteraction(self.session).request(key, request)

    async def prepare(
        self, result: _Terminal | InputRequiredResult
    ) -> _Terminal | InputRequiredResult | _DeferredInteraction:
        if not isinstance(result, InputRequiredResult):
            return result
        pending: Final[InputRequests] = {
            key: request for key, request in (result.input_requests or {}).items() if isinstance(request, ElicitRequest)
        }
        if pending and not self.allow_elicitation:
            raise MCPError(code=-32602, message="Elicitation is disabled for this MCP server")
        if result.input_requests and not pending:
            return result
        local: Final = tuple(
            (key, request) for key, request in (result.input_requests or {}).items() if key not in pending
        )
        responses: Final = await asyncio.gather(*(self.request(key, request) for key, request in local))
        for response in responses:
            if isinstance(response, ErrorData):
                raise MCPError(code=response.code, message=response.message)
        return _DeferredInteraction(
            BoundInputRequiredResult(
                input_requests=pending or None,
                request_state=result.request_state,
                _meta=result.meta,
                gateway_responses={
                    key: response for (key, _), response in zip(local, responses) if not isinstance(response, ErrorData)
                }
                or None,
            )
        )

    async def complete(
        self,
        first: _Terminal | InputRequiredResult,
        retry: Callable[[InputResponses | None, str | None], Awaitable[_Terminal | InputRequiredResult]],
    ) -> _Terminal | InputRequiredResult:
        prepared: Final = await self.prepare(first)
        if isinstance(prepared, _DeferredInteraction):
            return prepared.result
        if not isinstance(prepared, InputRequiredResult):
            return prepared

        async def resume(
            responses: InputResponses | None, state: str | None
        ) -> _Terminal | InputRequiredResult | _DeferredInteraction:
            return await self.prepare(await retry(responses, state))

        completed: Final = await run_input_required_driver(prepared, dispatch=self.request, retry=resume)
        return completed.result if isinstance(completed, _DeferredInteraction) else completed
