import asyncio
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from contextvars import Context
from dataclasses import replace
from datetime import datetime, timezone
from typing import Annotated, Final

from fastapi import Request, Response
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from pydantic_core import from_json
from starlette.types import Message

from litellm.exceptions import BudgetExceededError
from litellm.models.workflow_definition import ChatCompletionNode
from litellm.models.workflow_execution import ChildCallResult, EngineRun, ExecutionError, StepTrace
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import authorize_internal_virtual_key
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    safe_set_request_parsed_body,  # pyright: ignore[reportUnknownVariableType]  # legacy parsed-body API
)
from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers, redact_credential_headers
from litellm.proxy.spend_tracking.budget_reservation import (
    release_budget_reservation_on_cancel,  # pyright: ignore[reportUnknownVariableType]  # legacy reservation dictionary
    release_unbound_budget_reservation,
)
from litellm.proxy.workflows.monty.models import MontyErrorKind
from litellm.proxy.workflows.monty.validation import ValidatorProcess
from litellm.types.utils import Usage

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout

_ROUTE: Final = "/v1/chat/completions"
_MESSAGES: Final = TypeAdapter[list[ChatCompletionMessageParam]](
    Annotated[list[ChatCompletionMessageParam], Field(min_length=1)]
)
_JSON: Final = TypeAdapter[JsonValue](JsonValue, config=ConfigDict(allow_inf_nan=False))
_FIELDS: Final = TypeAdapter(dict[str, object])
_STRING: Final = TypeAdapter[str | None](str | None)
_NUMBER: Final = TypeAdapter[float | None](float | None)


def _child_request(
    parent: Request,
    data: dict[str, object],  # mutable-ok: shared request body mutated by the existing proxy pipeline
    call_id: str,
    session_id: str,
) -> Request:
    encoded: Final = json.dumps(data, allow_nan=False).encode()
    incoming: Final[asyncio.Queue[Message]] = asyncio.Queue(maxsize=1)
    incoming.put_nowait({"type": "http.request", "body": encoded, "more_body": False})

    async def receive() -> Message:
        return await incoming.get()

    session_headers: Final = (
        ()
        if get_chain_id_from_headers(dict(parent.headers)) or parent.headers.get("traceparent")
        else ((b"x-litellm-session-id", session_id.encode()),)
    )
    original_headers: Final = dict(parent.headers)
    headers: Final = tuple(
        (key.encode("latin-1"), value.encode("latin-1"))
        for key, value in redact_credential_headers(original_headers).items()
        if key.lower() not in {"content-type", "content-length", "x-litellm-call-id"} and value == original_headers[key]
    )
    shared_state: Final = _FIELDS.validate_python(parent.scope.get("state", {}))
    request: Final = Request(
        {
            **parent.scope,
            "method": "POST",
            "path": _ROUTE,
            "raw_path": _ROUTE.encode(),
            "query_string": b"",
            "path_params": {},
            "state": {key: value for key, value in shared_state.items() if key == "database"},
            "headers": (
                *headers,
                *session_headers,
                (b"content-type", b"application/json"),
                (b"content-length", str(len(encoded)).encode()),
                (b"x-litellm-call-id", call_id.encode()),
            ),
        },
        receive,
    )
    safe_set_request_parsed_body(request, data)
    return request


def _execution_error(exc: Exception, receipt: ChildCallResult | None = None) -> ExecutionError:
    if isinstance(exc, BudgetExceededError) or (
        isinstance(exc, ProxyException) and exc.type == ProxyErrorTypes.budget_exceeded
    ):
        return ExecutionError("budget_exceeded", "The execution identity has insufficient budget", receipt)
    status: Final = str(exc.code if isinstance(exc, ProxyException) else getattr(exc, "status_code", None))
    if status in {"401", "403"}:
        return ExecutionError("authorization_failed", "The execution identity is not authorized for this node", receipt)
    if status == "429":
        return ExecutionError("rate_limit", "The gateway could not admit this node", receipt)
    if status in {"400", "422"}:
        return ExecutionError("request_rejected", "The gateway rejected this node", receipt)
    if status == "408" or status.startswith("5"):
        return ExecutionError("provider_error", "The provider could not complete this node", receipt)
    return ExecutionError("runtime_error", "The gateway could not complete this node", receipt)


async def _authorize_child(
    request: Request,
    run: EngineRun,
    data: dict[str, object],  # mutable-ok: authentication annotates the shared proxy request body
) -> UserAPIKeyAuth | ExecutionError:
    try:
        return await authorize_internal_virtual_key(
            key_hash=run.created_by, request=request, request_data=data, route=_ROUTE, expected_team_id=run.team_id
        )
    except Exception as exc:  # noqa: BLE001  # authentication failures are known to precede dispatch
        failure: Final = _execution_error(exc)
        return replace(failure, code="admission_rate_limit") if failure.code == "rate_limit" else failure


def _receipt(
    processor: ProxyBaseLLMRequestProcessing, response: Response, raw: object = None
) -> ChildCallResult | None:
    logging: Final = processor.logging_obj
    prepared: Final = logging.prepared_cost if logging is not None else None
    finalized: Final = logging.finalized_cost if logging is not None else None
    fields: Final = _FIELDS.validate_python(raw.model_dump() if isinstance(raw, BaseModel) else raw or {})
    reported: Final = fields.get("usage")
    usage: Final = (
        prepared.usage if prepared is not None else Usage.model_validate(reported) if reported is not None else None
    )
    if usage is None or any(
        type(value) is not int or value < 0 for value in (usage.prompt_tokens, usage.completion_tokens)
    ):
        return None
    try:
        cost: Final = finalized.amount if finalized is not None else float(response.headers["x-litellm-response-cost"])
    except (KeyError, ValueError):
        return ChildCallResult(None, usage.prompt_tokens, usage.completion_tokens, None, None)
    return ChildCallResult(
        None,
        usage.prompt_tokens,
        usage.completion_tokens,
        None,
        cost if math.isfinite(cost) and cost >= 0 else None,
    )


async def _output(raw: object, node: ChatCompletionNode, deadline: float) -> JsonValue | ExecutionError:
    try:
        result: Final = ChatCompletion.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
    except ValueError:
        return ExecutionError("invalid_response", "The gateway returned an invalid chat response")
    if len(result.choices) != 1:
        return ExecutionError("invalid_response", "The gateway returned an invalid chat response")
    choice: Final = result.choices[0]
    if choice.message.refusal or choice.finish_reason == "content_filter":
        return ExecutionError("content_rejected", "The gateway rejected the node output")
    content: Final = choice.message.content
    if content is None:
        return ExecutionError("invalid_response", "The model returned no assistant content")
    if len(content.encode()) > 1024 * 1024:
        return ExecutionError("output_too_large", "The model output exceeded the workflow limit")
    if node.response_format is None:
        return content
    try:
        output: Final = _JSON.validate_python(from_json(content, allow_inf_nan=False))
    except ValueError:
        return ExecutionError("output_invalid", "The model did not return valid JSON")
    validator: Final = ValidatorProcess()
    try:
        failure: Final = await validator.validate(
            output, dict(node.response_format.json_schema.schema_), deadline=deadline, output=True
        )
        if failure is None:
            return output
        if failure.kind is MontyErrorKind.TIMEOUT:
            return ExecutionError("timeout", "The node deadline expired")
        return ExecutionError("output_invalid", "The model output did not satisfy the node schema")
    finally:
        await validator.close()


async def execute_chat_node(
    request: Request, run: EngineRun, node: ChatCompletionNode, step: StepTrace
) -> ChildCallResult | ExecutionError:
    from litellm.proxy import proxy_server

    if step.deadline.tzinfo is None or step.deadline.utcoffset() is None:
        return ExecutionError("invalid_deadline", "The node deadline must include a timezone")
    if re.fullmatch(r"[0-9a-f]{64}", run.created_by) is None or not run.team_id or not step.call_id:
        return ExecutionError("authorization_failed", "A persisted team virtual key and child call ID are required")
    remaining: Final = min(node.timeout_seconds, (step.deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        return ExecutionError("timeout", "The node deadline expired")
    deadline: Final = time.monotonic() + remaining
    try:
        messages: Final = _MESSAGES.validate_python(step.input, strict=True)
        trace: Final = {
            "workflow_id": run.workflow_id,
            "run_id": run.run_id,
            "node_id": node.id,
            "attempt": step.attempts,
            "session_id": run.session_id,
        }
        data: Final[dict[str, object]] = {  # mutable-ok: proxy processor and parsed-body cache share this object
            **node.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={"id", "type", "messages", "timeout_seconds", "retry"},
            ),
            "messages": messages,
            "stream": False,
            "num_retries": 0,
            "timeout": remaining,
            "metadata": {**trace, "spend_logs_metadata": trace},
            **({"user": run.input.end_user_id} if run.input.end_user_id is not None else {}),
        }
        child: Final = _child_request(request, data, step.call_id, run.session_id)
    except (TypeError, ValueError):
        return ExecutionError("invalid_messages", "The node input must contain valid chat messages")
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    response: Final = Response()

    async def execute() -> ChildCallResult | ExecutionError:
        try:
            async with timeout(remaining):
                auth: Final = await _authorize_child(child, run, data)
                if isinstance(auth, ExecutionError):
                    return auth
                try:
                    try:
                        raw: Final = await processor.base_process_llm_request(
                            request=child,
                            fastapi_response=response,
                            user_api_key_dict=auth,
                            route_type="acompletion",
                            proxy_logging_obj=proxy_server.proxy_logging_obj,
                            general_settings=_FIELDS.validate_python(proxy_server.general_settings),  # pyright: ignore[reportUnknownMemberType]  # legacy proxy global validated at this boundary
                            proxy_config=proxy_server.proxy_config,
                            llm_router=proxy_server.llm_router,
                            user_model=_STRING.validate_python(proxy_server.user_model),  # pyright: ignore[reportUnknownMemberType]  # legacy proxy global validated at this boundary
                            user_temperature=_NUMBER.validate_python(proxy_server.user_temperature),  # pyright: ignore[reportUnknownMemberType]  # legacy proxy global validated at this boundary
                            user_request_timeout=proxy_server.user_request_timeout,
                            user_max_tokens=proxy_server.user_max_tokens,
                            user_api_base=_STRING.validate_python(proxy_server.user_api_base),  # pyright: ignore[reportUnknownMemberType]  # legacy proxy global validated at this boundary
                            version=proxy_server.version,
                        )
                    except Exception as exc:  # noqa: BLE001  # provider and hook failures must run the proxy's accounting
                        try:
                            await processor.handle_llm_api_exception(
                                e=exc, user_api_key_dict=auth, proxy_logging_obj=proxy_server.proxy_logging_obj
                            )
                        except Exception as failure:  # noqa: BLE001  # failures become sanitized workflow values
                            return _execution_error(failure, _receipt(processor, response))
                except asyncio.CancelledError:
                    await release_budget_reservation_on_cancel(auth.budget_reservation)
                    await proxy_server.proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
                    raise
                receipt: Final = _receipt(processor, response, raw)
                output: Final = await _output(raw, node, deadline)
                if isinstance(output, ExecutionError):
                    return replace(output, receipt=receipt)
                if receipt is None:
                    return ExecutionError("invalid_response", "The gateway returned no valid usage")
                return replace(receipt, output=output)
        except asyncio.TimeoutError:
            return ExecutionError("timeout", "The node deadline expired", _receipt(processor, response))
        except Exception as exc:  # noqa: BLE001  # auth and validator failures must not expose credentials or content
            return _execution_error(exc, _receipt(processor, response))
        finally:
            reservation: Final[object] = getattr(child.state, "budget_reservation", None)
            if isinstance(reservation, Mapping):
                await release_unbound_budget_reservation(reservation)  # pyright: ignore[reportUnknownArgumentType]  # preserve callback-shared reservation identity

    return await Context().run(asyncio.create_task, execute())
