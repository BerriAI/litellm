"""Regression tests for LIT-2818 - LiteLLMSendMessageResponse must accept
the full JSON-RPC 2.0 id contract: str, int, or None.

Before LIT-2818, LiteLLMSendMessageResponse declared id: str (required).
Per JSON-RPC 2.0 the response id may be a String, Number, or Null;
error responses that could not be correlated to a request MUST use null.
The upstream a2a SDK (a2a-sdk==0.3.24) reflects this on both
SendMessageSuccessResponse and JSONRPCErrorResponse:
    id: str | int | None = None
Pinning id: str (required) here made the proxy raise a Pydantic
ValidationError whenever an upstream agent returned a legitimate JSON-RPC
error response with a numeric or null id - surfacing as a 500.
"""

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from litellm.types.agents import LiteLLMSendMessageResponse


def test_lit2818_constructor_accepts_string_id():
    r = LiteLLMSendMessageResponse(
        id="req-abc-123",
        result={"kind": "task", "id": "task-1"},
    )
    assert r.id == "req-abc-123"
    assert r.jsonrpc == "2.0"


def test_lit2818_constructor_accepts_integer_id():
    # Per JSON-RPC 2.0 id may be a Number. Pre-fix raised:
    # ValidationError: Input should be a valid string.
    r = LiteLLMSendMessageResponse(
        id=42,
        result={"kind": "message"},
    )
    assert r.id == 42


def test_lit2818_constructor_accepts_null_id():
    # JSON-RPC 2.0: error responses that could not be correlated to a
    # request MUST use null id.
    r = LiteLLMSendMessageResponse(
        id=None,
        error={"code": -32700, "message": "Parse error"},
    )
    assert r.id is None


def test_lit2818_constructor_accepts_missing_id_defaults_to_none():
    r = LiteLLMSendMessageResponse(
        error={"code": -32600, "message": "Invalid Request"},
    )
    assert r.id is None


def _error_response_dict(rpc_id):
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "error": {"code": -32603, "message": "Internal error"},
    }


@pytest.mark.parametrize(
    "rpc_id",
    [
        "req-error-1",
        42,
        None,
    ],
)
def test_lit2818_from_dict_error_response_accepts_all_jsonrpc_id_shapes(rpc_id):
    r = LiteLLMSendMessageResponse.from_dict(_error_response_dict(rpc_id))
    assert r.id == rpc_id
    assert r.error is not None
    assert r.error["code"] == -32603


def test_lit2818_from_dict_omitted_id_does_not_raise():
    r = LiteLLMSendMessageResponse.from_dict(
        {
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": "Parse error"},
        }
    )
    assert r.id is None
    assert r.error["code"] == -32700


def _build_jsonrpc_error_response(rpc_id):
    from a2a.types import JSONRPCError, JSONRPCErrorResponse, SendMessageResponse

    err = JSONRPCErrorResponse(
        id=rpc_id,
        error=JSONRPCError(code=-32603, message="Internal error"),
    )
    return SendMessageResponse(root=err)


@pytest.mark.parametrize("rpc_id", ["req-1", 7, None])
def test_lit2818_from_a2a_response_error_path(rpc_id):
    sdk_response = _build_jsonrpc_error_response(rpc_id)
    wrapped = LiteLLMSendMessageResponse.from_a2a_response(sdk_response)
    assert wrapped.id == rpc_id
    assert wrapped.error is not None
    assert wrapped.error["code"] == -32603


@pytest.mark.parametrize(
    "bad_id",
    [
        12.34,
        ["arr"],
        {"o": "bj"},
    ],
)
def test_lit2818_constructor_still_rejects_invalid_id_shapes(bad_id):
    with pytest.raises(ValidationError):
        LiteLLMSendMessageResponse(id=bad_id)
