import asyncio
import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from fastapi import HTTPException
from prisma.errors import DataError

import litellm
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.common_request_processing import proxy_exception_from_route_error
from litellm.proxy.common_utils.openai_error_payload import proxy_exception_for
from litellm.proxy.utils import get_error_message_str, handle_exception_on_proxy
from litellm.types.router import RouterErrors


def normalize(value):
    return value


def test_get_error_message_str_happy_path_http_exception_with_string_detail():
    exc = HTTPException(status_code=400, detail="something went wrong")
    summary = {
        "result": get_error_message_str(exc),
        "status_code": exc.status_code,
        "is_str": True,
    }
    assert summary == {
        "result": "something went wrong",
        "status_code": 400,
        "is_str": True,
    }


def test_get_error_message_str_happy_path_http_exception_with_dict_detail():
    detail = {"error": "bad input", "code": "invalid_request"}
    exc = HTTPException(status_code=422, detail=detail)
    summary = {
        "result": get_error_message_str(exc),
        "result_parsed": json.loads(get_error_message_str(exc)),
        "status_code": exc.status_code,
    }
    assert summary == {
        "result": json.dumps(detail),
        "result_parsed": detail,
        "status_code": 422,
    }


def test_get_error_message_str_happy_path_generic_exception():
    exc = ValueError("boom")
    summary = {
        "result": get_error_message_str(exc),
        "type": type(exc).__name__,
        "args": list(exc.args),
    }
    assert summary == {
        "result": "boom",
        "type": "ValueError",
        "args": ["boom"],
    }


def test_get_error_message_str_with_runtime_error():
    exc = RuntimeError("runtime explosion")
    summary = {
        "result": get_error_message_str(exc),
        "type": type(exc).__name__,
        "matches_str": str(exc) == get_error_message_str(exc),
    }
    assert summary == {
        "result": "runtime explosion",
        "type": "RuntimeError",
        "matches_str": True,
    }


def test_get_error_message_str_error_path_none_input_returns_string_none():
    summary = {
        "result": get_error_message_str(None),
        "is_str": isinstance(get_error_message_str(None), str),
        "input": None,
    }
    assert summary == {
        "result": "None",
        "is_str": True,
        "input": None,
    }


def test_handle_exception_on_proxy_happy_path_http_exception():
    exc = HTTPException(status_code=403, detail="forbidden")
    result = handle_exception_on_proxy(exc)
    snapshot = {
        "is_proxy_exception": isinstance(result, ProxyException),
        "message": result.message,
        "type": result.type,
        "code": result.code,
    }
    assert snapshot == {
        "is_proxy_exception": True,
        "message": "forbidden",
        "type": "auth_error",
        "code": "403",
    }


def test_handle_exception_on_proxy_happy_path_already_proxy_exception():
    original = ProxyException(
        message="already wrapped",
        type=ProxyErrorTypes.budget_exceeded.value,
        param="key",
        code=402,
    )
    result = handle_exception_on_proxy(original)
    snapshot = {
        "is_same_object": result is original,
        "message": result.message,
        "type": result.type,
        "code": result.code,
    }
    assert snapshot == {
        "is_same_object": True,
        "message": "already wrapped",
        "type": ProxyErrorTypes.budget_exceeded.value,
        "code": "402",
    }


def test_handle_exception_on_proxy_happy_path_generic_exception_defaults_to_500():
    exc = ValueError("kaboom")
    result = handle_exception_on_proxy(exc)
    snapshot = {
        "is_proxy_exception": isinstance(result, ProxyException),
        "message": result.message,
        "type": result.type,
        "code": result.code,
        "param": result.param,
    }
    assert snapshot == {
        "is_proxy_exception": True,
        "message": "kaboom",
        "type": ProxyErrorTypes.internal_server_error.value,
        "code": "500",
        "param": None,
    }


def test_handle_exception_on_proxy_uses_attached_status_code_when_present():
    class _CustomErr(Exception):
        status_code = 418

    exc = _CustomErr("teapot")
    result = handle_exception_on_proxy(exc)
    snapshot = {
        "code": result.code,
        "message": result.message,
        "type": result.type,
    }
    assert snapshot == {
        "code": "418",
        "message": "teapot",
        "type": "invalid_request_error",
    }


def test_handle_exception_on_proxy_error_path_none_input_wraps_as_500():
    result = handle_exception_on_proxy(None)
    snapshot = {
        "is_proxy_exception": isinstance(result, ProxyException),
        "message": result.message,
        "code": result.code,
        "type": result.type,
    }
    assert snapshot == {
        "is_proxy_exception": True,
        "message": "None",
        "code": "500",
        "type": ProxyErrorTypes.internal_server_error.value,
    }


@pytest.mark.parametrize(
    "status_code, expected_type",
    [
        (400, "invalid_request_error"),
        (401, "auth_error"),
        (403, "auth_error"),
        (404, "invalid_request_error"),
        (408, "invalid_request_error"),
        (409, "invalid_request_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, ProxyErrorTypes.internal_server_error.value),
        (503, ProxyErrorTypes.internal_server_error.value),
    ],
)
def test_handle_exception_on_proxy_labels_an_http_exception_by_the_status_it_answers(
    status_code: int, expected_type: str
):
    """A management route that rejects a caller's own argument answers 400, and calling that
    internal_server_error tells the caller's retry loop the gateway is down when nothing but
    their request is wrong."""
    result: Final = handle_exception_on_proxy(HTTPException(status_code=status_code, detail="boom"))

    assert (result.code, result.type) == (str(status_code), expected_type)


def _exception_carrying_status(carried: int | None, message: str = "boom") -> Exception:
    class _CarriedStatus(Exception):
        status_code = carried

    return _CarriedStatus(message)


@pytest.mark.parametrize(
    "status_code, expected_type",
    [
        (400, "invalid_request_error"),
        (404, "invalid_request_error"),
        (408, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, ProxyErrorTypes.internal_server_error.value),
    ],
)
def test_handle_exception_on_proxy_labels_a_carried_status_the_same_way(status_code: int, expected_type: str):
    """Routes also reach this handler with plain exceptions carrying their own status, and
    those must be named by that status rather than all landing on the 500 label."""
    result: Final = handle_exception_on_proxy(_exception_carrying_status(status_code))

    assert (result.code, result.type) == (str(status_code), expected_type)


def test_handle_exception_on_proxy_keeps_the_type_the_exception_already_names():
    """A rate limit litellm raised already names itself, and overwriting that with the status
    map's guess would drop what the client branches on."""
    result: Final = handle_exception_on_proxy(
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-4o")
    )

    assert (result.code, result.type) == ("429", "throttling_error")


@pytest.mark.parametrize(
    "exc",
    [
        openai.InternalServerError(
            message="upstream down",
            response=httpx.Response(500, request=httpx.Request("GET", "https://api.openai.com/v1/batches/b")),
            body={"type": "server_error"},
        ),
        litellm.Timeout(message="took too long", model="gpt-6.1-sol", llm_provider="openai"),
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-6.1-sol"),
        litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-6.1-sol"),
        litellm.BadRequestError(message="bad input", model="gpt-6.1-sol", llm_provider="openai"),
        HTTPException(status_code=404, detail="no such batch"),
    ],
    ids=["provider_5xx_naming_its_own_type", "timeout", "rate_limit", "bad_key", "bad_request", "http_404"],
)
def test_handle_exception_on_proxy_labels_a_failure_the_way_the_chat_route_does(exc: Exception):
    """Batch, file, and management routes answer through this handler while chat answers through the route
    error path, and one client calling both must read the same type for the same failure."""
    shared: Final = handle_exception_on_proxy(exc)
    chat: Final = proxy_exception_from_route_error(exc)

    assert (shared.code, shared.type) == (chat.code, chat.type)


@pytest.mark.parametrize(
    "exc",
    [
        HTTPException(status_code=400, detail="bad budget_duration"),
        HTTPException(status_code=401, detail="not allowed"),
        HTTPException(status_code=403, detail="admin only"),
        HTTPException(status_code=404, detail="no such team"),
        litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-6.1-sol"),
        ValueError(f"{RouterErrors.no_deployments_with_tag_routing.value}=['paid']"),
        _exception_carrying_status(409, "duplicate credential"),
    ],
    ids=["http_400", "http_401", "http_403", "http_404", "provider_401", "tag_denial", "carried_409"],
)
def test_handle_exception_on_proxy_labels_a_failure_the_way_route_catch_alls_do(exc: Exception):
    """Management routes answer some failures through their own catch-all and the rest through this handler,
    so a client reading /key/list and /key/generate must see the same type and status for the same failure."""
    shared: Final = handle_exception_on_proxy(exc)
    catch_all: Final = proxy_exception_for(exc, default_status_code=500)

    assert (shared.code, shared.type, shared.message) == (catch_all.code, catch_all.type, catch_all.message)


@pytest.mark.parametrize(
    "message, expected",
    [
        (f"{RouterErrors.no_deployments_with_provider_budget_routing.value}, try later", ("429", "rate_limit_error")),
        (f"{RouterErrors.no_deployments_with_tag_routing.value}=['paid']", ("401", "auth_error")),
    ],
    ids=["no_deployment_left", "no_deployment_for_tags"],
)
def test_handle_exception_on_proxy_names_a_router_failure_by_the_status_it_answers(
    message: str, expected: tuple[str, str]
):
    """A router failure that carries no status answers the 429 or 401 its message maps to, and a 429 labelled
    internal_server_error reads as an outage to a client that branches on the type."""
    result: Final = handle_exception_on_proxy(ValueError(message))

    assert (result.code, result.type) == expected


def test_handle_exception_on_proxy_keeps_the_path_a_rejected_request_names():
    """A provider rejecting a batch input names the caller's own bucket path, and redacting the /tmp part of it
    leaves the caller nothing to fix their request with."""
    result: Final = handle_exception_on_proxy(
        litellm.BadRequestError(
            message="Invalid input_file_id gs://my-bucket/tmp/batch_input.jsonl",
            model="gemini-3.1-pro-preview",
            llm_provider="vertex_ai",
        )
    )

    assert result.code == "400"
    assert "gs://my-bucket/tmp/batch_input.jsonl" in result.message


def test_handle_exception_on_proxy_keeps_the_message_an_exception_with_a_status_carries():
    """Main sends a status-bearing failure's message verbatim, and relabeling its type must not change that text."""
    result: Final = handle_exception_on_proxy(_exception_carrying_status(503, "Cannot connect to host 10.12.4.7:8443"))

    assert (result.code, result.message) == ("503", "Cannot connect to host 10.12.4.7:8443")


@pytest.mark.parametrize(
    "message",
    [
        "boom",
        "Cannot connect to host 10.12.4.7:8443",
        "Cannot connect to host guardrail.internal.svc.cluster.local:443",
        "Cannot connect to host [fd00::12]:8443",
    ],
    ids=["plain", "ipv4", "cluster_dns", "ipv6"],
)
def test_handle_exception_on_proxy_answers_a_generic_500_for_an_exception_whose_status_is_none(message: str):
    """A guardrail whose upstream refuses the connection raises with status_code=None, which crashed the handler
    into a 503 blaming the auth database, and its text names private hosts no virtual key was ever sent."""
    result: Final = handle_exception_on_proxy(_exception_carrying_status(None, message))

    assert (result.code, result.type, result.message) == (
        "500",
        ProxyErrorTypes.internal_server_error.value,
        "Internal server error",
    )


@pytest.mark.parametrize(
    "exc",
    [
        HTTPException(status_code=401, detail="bad key"),
        ValueError("provider boom"),
        ProxyException(message="already wrapped", type=ProxyErrorTypes.budget_exceeded.value, param="key", code=402),
    ],
    ids=["http_exception", "generic_exception", "already_proxy_exception"],
)
def test_handle_exception_on_proxy_returns_the_litellm_call_id_header(exc: Exception):
    result = handle_exception_on_proxy(exc, "call-7836")

    assert result.headers == {"x-litellm-call-id": "call-7836"}


def test_handle_exception_on_proxy_sends_no_call_id_header_when_the_request_has_none():
    assert handle_exception_on_proxy(ValueError("provider boom")).headers == {}


@pytest.mark.asyncio
async def test_handle_exception_on_proxy_read_only_transaction_forces_writer_recreate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma_client = MagicMock()
    prisma_client.recreate_read_only_writer = AsyncMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)
    exc = DataError(
        data={
            "user_facing_error": {
                "message": 'PostgresError { code: "25006", message: "cannot execute UPDATE in a read-only transaction" }'
            }
        }
    )

    result = handle_exception_on_proxy(exc)
    await asyncio.sleep(0)

    snapshot = {
        "code": result.code,
        "recreate_kwargs": prisma_client.recreate_read_only_writer.await_args.kwargs,
    }
    assert snapshot == {
        "code": "500",
        "recreate_kwargs": {"reason": "postgres_read_only_transaction"},
    }


@pytest.mark.asyncio
async def test_handle_exception_on_proxy_leaves_writer_alone_for_other_db_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prisma_client = MagicMock()
    prisma_client.recreate_read_only_writer = AsyncMock(return_value=True)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma_client)

    handle_exception_on_proxy(DataError(data={"user_facing_error": {"message": "deadlock detected"}}))
    await asyncio.sleep(0)

    assert prisma_client.recreate_read_only_writer.await_count == 0
