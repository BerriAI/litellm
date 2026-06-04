import json

import pytest
from fastapi import HTTPException

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.utils import get_error_message_str, handle_exception_on_proxy


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
        "type": ProxyErrorTypes.internal_server_error.value,
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
        "param": "None",
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
        "type": ProxyErrorTypes.internal_server_error.value,
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
