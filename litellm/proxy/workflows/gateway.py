import asyncio
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from contextvars import Context
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final, Literal, Protocol

from fastapi import Request, Response
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import from_json
from starlette.types import Message

from litellm.exceptions import BudgetExceededError
from litellm.models.workflow_definition import ChatCompletionNode
from litellm.models.workflow_execution import ChildCallResult, ExecutionError, workflow_call_id
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import authorize_internal_virtual_key
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    safe_set_request_parsed_body,  # pyright: ignore[reportUnknownVariableType]  # legacy helper accepts an untyped body
)
from litellm.proxy.spend_tracking.budget_reservation import (
    release_budget_reservation_on_cancel,  # pyright: ignore[reportUnknownVariableType]  # legacy reservation dictionary
    release_unbound_budget_reservation,
)
from litellm.proxy.workflows.monty.models import MontyErrorKind
from litellm.proxy.workflows.monty.validation import ValidatorProcess
from litellm.repositories.table_repositories import SpendLogsRepository

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout

_ROUTE: Final = "/v1/chat/completions"
_MESSAGES: Final = TypeAdapter(list[ChatCompletionMessageParam])
_JSON: Final = TypeAdapter[JsonValue](JsonValue, config=ConfigDict(allow_inf_nan=False))
_MAX_OUTPUT_BYTES: Final = 1024 * 1024
_RECEIPT_LOOKUP_SECONDS: Final = 0.1


@dataclass(frozen=True, slots=True)
class _NodeCall:
    key_hash: str
    team_id: str
    node: ChatCompletionNode
    messages: JsonValue
    workflow_id: str
    run_id: str
    session_id: str
    attempt: int
    deadline: datetime
    end_user_id: str | None


@dataclass(frozen=True, slots=True)
class _RequestContext:
    request: Request
    data: dict[str, object]  # mutable-ok: the existing auth and processor APIs enrich this attempt-owned body
    response: Response
    billing: asyncio.Future[ChildCallResult | None]


class _AssistantMessage(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    role: Literal["assistant"]
    content: str | None
    refusal: str | None = None


class _Choice(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    message: _AssistantMessage
    finish_reason: str | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)


class _ChatResult(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)
    usage: _Usage


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    usage: _Usage


class _FinalizedBill(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    cost_finalized: Literal[True]
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    response_cost: float | None = Field(ge=0)


class _FailureData(BaseModel):
    model_config = ConfigDict(frozen=True)

    standard_logging_object: _FinalizedBill


class _ProcessorFailure(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    data: _FailureData


class _ProxySettings(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    general_settings: Mapping[str, object]
    user_model: str | None
    user_temperature: float | None
    user_request_timeout: float | None
    user_max_tokens: int | None
    user_api_base: str | None
    version: str | None


class _Gateway(Protocol):
    async def authorize(self, call: _NodeCall, context: _RequestContext) -> UserAPIKeyAuth: ...

    async def process(self, context: _RequestContext, auth: UserAPIKeyAuth) -> object: ...

    async def receipts(self, call: _NodeCall, call_id: str) -> tuple[str, ...]: ...

    async def validate_output(
        self, output: JsonValue, schema: Mapping[str, JsonValue], deadline: float
    ) -> ExecutionError | None: ...

    async def release(self, context: _RequestContext) -> None: ...


@dataclass(frozen=True, slots=True)
class _ProxyGateway:
    async def authorize(self, call: _NodeCall, context: _RequestContext) -> UserAPIKeyAuth:
        return await authorize_internal_virtual_key(
            key_hash=call.key_hash,
            request=context.request,
            request_data=context.data,
            route=_ROUTE,
            expected_team_id=call.team_id,
        )

    async def process(self, context: _RequestContext, auth: UserAPIKeyAuth) -> object:
        from litellm.proxy import proxy_server

        processor: Final = ProxyBaseLLMRequestProcessing(data=context.data)
        settings: Final = _ProxySettings.model_validate(vars(proxy_server))
        try:
            try:
                return await processor.base_process_llm_request(
                    request=context.request,
                    fastapi_response=context.response,
                    user_api_key_dict=auth,
                    route_type="acompletion",
                    proxy_logging_obj=proxy_server.proxy_logging_obj,
                    general_settings=dict(settings.general_settings),
                    proxy_config=proxy_server.proxy_config,
                    llm_router=proxy_server.llm_router,
                    user_model=settings.user_model,
                    user_temperature=settings.user_temperature,
                    user_request_timeout=settings.user_request_timeout,
                    user_max_tokens=settings.user_max_tokens,
                    user_api_base=settings.user_api_base,
                    version=settings.version,
                )
            except Exception as exc:  # noqa: BLE001  # route arbitrary provider and hook failures through accounting
                try:
                    await processor.handle_llm_api_exception(
                        e=exc, user_api_key_dict=auth, proxy_logging_obj=proxy_server.proxy_logging_obj
                    )
                except Exception as failure:  # noqa: BLE001  # sanitize errors raised by failure hooks
                    return _execution_error(failure)
                finally:
                    context.billing.set_result(_failure_receipt(processor))
        except asyncio.CancelledError:
            if not context.billing.done():
                context.billing.set_result(_failure_receipt(processor))
            await release_budget_reservation_on_cancel(auth.budget_reservation)
            await proxy_server.proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
            raise

    async def receipts(self, call: _NodeCall, call_id: str) -> tuple[str, ...]:
        from litellm.proxy import proxy_server

        if proxy_server.prisma_client is None:
            return ()
        return await SpendLogsRepository(proxy_server.prisma_client, use_writer=True).request_ids_for_call(
            call_id=call_id, key_hash=call.key_hash, team_id=call.team_id, session_id=call.session_id
        )

    async def validate_output(
        self, output: JsonValue, schema: Mapping[str, JsonValue], deadline: float
    ) -> ExecutionError | None:
        validator: Final = ValidatorProcess()
        try:
            failure: Final = await validator.validate(output, dict(schema), deadline=deadline, output=True)
            if failure is None:
                return None
            if failure.kind is MontyErrorKind.TIMEOUT:
                return ExecutionError("timeout", "The node deadline expired")
            return ExecutionError("output_invalid", "The model output did not satisfy the node schema")
        finally:
            await validator.close()

    async def release(self, context: _RequestContext) -> None:
        reservation: Final[object] = getattr(context.request.state, "budget_reservation", None)
        if isinstance(reservation, Mapping):
            await release_unbound_budget_reservation(reservation)  # pyright: ignore[reportUnknownArgumentType]  # preserve the reservation identity shared with accounting


def _node_data(call: _NodeCall, remaining: float) -> Mapping[str, object]:
    messages: Final = _MESSAGES.validate_python(call.messages, strict=True)
    if not messages:
        raise ValueError("Chat messages must not be empty")
    trace: Final = {
        "workflow_id": call.workflow_id,
        "run_id": call.run_id,
        "node_id": call.node.id,
        "attempt": call.attempt,
        "session_id": call.session_id,
    }
    return {
        **call.node.model_dump(
            mode="json",
            by_alias=True,
            exclude={"id", "type", "messages", "timeout_seconds", "retry"},
            exclude_none=True,
        ),
        "messages": messages,
        "stream": False,
        "num_retries": 0,
        "timeout": remaining,
        "metadata": {**trace, "spend_logs_metadata": trace},
        **({"user": call.end_user_id} if call.end_user_id is not None else {}),
    }


def _internal_request(data: Mapping[str, object], call_id: str, session_id: str) -> _RequestContext:
    """Adapt a server-owned call to request-aware gateway hooks without dispatching HTTP or inheriting caller state."""
    from litellm.proxy import proxy_server

    body: Final = dict(data)
    encoded: Final = json.dumps(body, allow_nan=False).encode()
    incoming: Final[asyncio.Queue[Message]] = asyncio.Queue(maxsize=1)
    incoming.put_nowait({"type": "http.request", "body": encoded, "more_body": False})

    async def receive() -> Message:
        return await incoming.get()

    request: Final = Request(
        {
            "type": "http",
            "app": proxy_server.app,
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": _ROUTE,
            "root_path": "",
            "query_string": b"",
            "headers": (
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode()),
                (b"x-litellm-call-id", call_id.encode()),
                (b"x-litellm-session-id", session_id.encode()),
            ),
            "server": ("localhost", 80),
            "client": None,
            "state": {},
        },
        receive,
    )
    safe_set_request_parsed_body(request, body)
    return _RequestContext(
        request=request, data=body, response=Response(), billing=asyncio.get_running_loop().create_future()
    )


def _execution_error(exc: Exception) -> ExecutionError:
    if isinstance(exc, BudgetExceededError) or (
        isinstance(exc, ProxyException) and exc.type == ProxyErrorTypes.budget_exceeded
    ):
        return ExecutionError("budget_exceeded", "The execution identity has insufficient budget")
    status: Final = exc.code if isinstance(exc, ProxyException) else getattr(exc, "status_code", None)
    if str(status) in {"401", "403"}:
        return ExecutionError("authorization_failed", "The execution identity is not authorized for this node")
    if str(status) == "429":
        return ExecutionError("rate_limit", "The gateway could not admit this node")
    if str(status) in {"400", "422"}:
        return ExecutionError("request_rejected", "The gateway rejected this node")
    if str(status) == "408" or str(status).startswith("5"):
        return ExecutionError("provider_error", "The provider could not complete this node")
    return ExecutionError("runtime_error", "The gateway could not complete this node")


def _response_cost(response: Response) -> float | None:
    try:
        cost: Final = float(response.headers["x-litellm-response-cost"])
        return cost if math.isfinite(cost) and cost >= 0 else None
    except (KeyError, ValueError):
        return None


def _failure_receipt(processor: object) -> ChildCallResult | None:
    try:
        billing: Final = _ProcessorFailure.model_validate(processor).data.standard_logging_object
    except ValidationError:
        return None
    return ChildCallResult(
        output=None,
        prompt_tokens=billing.prompt_tokens,
        completion_tokens=billing.completion_tokens,
        request_id=None,
        response_cost=billing.response_cost,
    )


def _response_receipt(raw: object, response: Response) -> ChildCallResult | None:
    try:
        usage: Final = _ResponseUsage.model_validate(raw).usage
    except ValidationError:
        return None
    return ChildCallResult(
        output=None,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        request_id=None,
        response_cost=_response_cost(response),
    )


async def _receipt(call: _NodeCall, call_id: str, gateway: _Gateway, deadline: float) -> str | None:
    try:
        async with timeout(min(_RECEIPT_LOOKUP_SECONDS, max(0, deadline - time.monotonic()))):
            ids: Final = await gateway.receipts(call, call_id)
            return ids[0] if len(ids) == 1 else None
    except Exception:  # noqa: BLE001  # receipt lookup failure must not discard an already billed result
        return None


async def _read_output(
    result: _ChatResult, call: _NodeCall, gateway: _Gateway, deadline: float
) -> JsonValue | ExecutionError:
    message: Final = result.choices[0].message
    if message.refusal or result.choices[0].finish_reason == "content_filter":
        return ExecutionError("content_rejected", "The gateway rejected the node output")
    if message.content is None:
        return ExecutionError("invalid_response", "The model returned no assistant content")
    if len(message.content.encode()) > _MAX_OUTPUT_BYTES:
        return ExecutionError("output_too_large", "The model output exceeded the workflow limit")
    if call.node.response_format is None:
        return message.content
    try:
        output: Final = _JSON.validate_python(from_json(message.content, allow_inf_nan=False))
    except ValueError:
        return ExecutionError("output_invalid", "The model did not return valid JSON")
    failure: Final = await gateway.validate_output(output, call.node.response_format.json_schema.schema_, deadline)
    return failure if failure is not None else output


async def _admitted_call(
    call: _NodeCall, gateway: _Gateway, context: _RequestContext, deadline: float
) -> ChildCallResult | ExecutionError:
    auth: Final = await gateway.authorize(call, context)
    raw: Final = await gateway.process(context, auth)
    if isinstance(raw, ExecutionError):
        return raw
    context.billing.set_result(_response_receipt(raw, context.response))
    try:
        result: Final = _ChatResult.model_validate(raw)
    except ValidationError:
        return ExecutionError("invalid_response", "The gateway returned an invalid chat response")
    output: Final = await _read_output(result, call, gateway, deadline)
    if isinstance(output, ExecutionError):
        return output
    return ChildCallResult(
        output=output,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        request_id=None,
        response_cost=_response_cost(context.response),
    )


async def _timed_call(
    call: _NodeCall, gateway: _Gateway, context: _RequestContext, deadline: float
) -> ChildCallResult | ExecutionError:
    try:
        async with timeout(max(0, deadline - time.monotonic())):
            return await _admitted_call(call, gateway, context, deadline)
    except asyncio.TimeoutError:
        return ExecutionError("timeout", "The node deadline expired")
    except Exception as exc:  # noqa: BLE001  # auth, provider and validator failures need sanitized error values
        return _execution_error(exc)


async def _execute(call: _NodeCall, gateway: _Gateway) -> ChildCallResult | ExecutionError:
    if call.deadline.tzinfo is None or call.deadline.utcoffset() is None:
        return ExecutionError("invalid_deadline", "The node deadline must include a timezone")
    if re.fullmatch(r"[0-9a-f]{64}", call.key_hash) is None or not call.team_id:
        return ExecutionError("authorization_failed", "A persisted team virtual key is required")
    remaining: Final = min(call.node.timeout_seconds, (call.deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        return ExecutionError("timeout", "The node deadline expired")
    deadline: Final = time.monotonic() + remaining
    try:
        data: Final = _node_data(call, remaining)
        call_id: Final = workflow_call_id(call.run_id, call.node.id, call.attempt)
        context: Final = _internal_request(data, call_id, call.session_id)
    except (ValueError, TypeError):
        return ExecutionError("invalid_messages", "The node input must contain valid chat messages")
    try:
        result: Final = await _timed_call(call, gateway, context, deadline)
        billing: Final = (
            result
            if isinstance(result, ChildCallResult)
            else context.billing.result()
            if context.billing.done()
            else result.receipt
        )
        if billing is None:
            return result
        receipt: Final = await _receipt(call, call_id, gateway, deadline)
        resolved: Final = replace(billing, request_id=receipt)
        return (
            ExecutionError(result.code, result.message, receipt=resolved)
            if isinstance(result, ExecutionError)
            else resolved
        )
    finally:
        await gateway.release(context)


async def _execute_isolated(call: _NodeCall, gateway: _Gateway) -> ChildCallResult | ExecutionError:
    task: Final = Context().run(asyncio.create_task, _execute(call, gateway))
    return await task


async def execute_chat_node(
    *,
    key_hash: str,
    team_id: str,
    node: ChatCompletionNode,
    messages: JsonValue,
    workflow_id: str,
    run_id: str,
    session_id: str,
    attempt: int,
    deadline: datetime,
    end_user_id: str | None,
) -> ChildCallResult | ExecutionError:
    call: Final = _NodeCall(
        key_hash, team_id, node, messages, workflow_id, run_id, session_id, attempt, deadline, end_user_id
    )
    return await _execute_isolated(call, _ProxyGateway())
