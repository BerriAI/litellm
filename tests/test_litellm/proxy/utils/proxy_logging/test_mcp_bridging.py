"""Pin ProxyLogging's MCP-LLM bridging helpers.

Covers:
- ``_convert_mcp_to_llm_format``
- ``_convert_llm_result_to_mcp_response``
- ``_extract_modified_arguments_from_content``
- ``_parse_arguments_manually``
- ``_convert_llm_result_to_mcp_during_response``
- ``_parse_pre_mcp_call_hook_response``
- ``_create_mcp_request_object_from_kwargs``
- ``_convert_mcp_hook_response_to_kwargs``
"""

from __future__ import annotations

import pytest

from litellm.types.mcp import (
    MCPDuringCallResponseObject,
    MCPPreCallRequestObject,
    MCPPreCallResponseObject,
)


# ---------------------------------------------------------------------------
# _convert_mcp_to_llm_format
# ---------------------------------------------------------------------------


def test_convert_mcp_to_llm_format_returns_synthetic_data(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="search", arguments={"q": "hello"})
    out = proxy_logging._convert_mcp_to_llm_format(
        request_obj=req,
        kwargs={
            "model": "gpt-4o-mini",
            "user_api_key_user_id": "u-1",
            "user_api_key_team_id": "t-1",
            "user_api_key_end_user_id": "eu-1",
            "user_api_key_hash": "hash",
            "user_api_key_request_route": "/mcp",
            "incoming_bearer_token": "tok",
        },
    )
    snapshot = {
        "model": out["model"],
        "user_id": out["user_api_key_user_id"],
        "mcp_tool_name": out["mcp_tool_name"],
        "mcp_arguments": out["mcp_arguments"],
        "incoming_bearer_token": out["incoming_bearer_token"],
        "message_role": out["messages"][0]["role"],
    }
    assert snapshot == {
        "model": "gpt-4o-mini",
        "user_id": "u-1",
        "mcp_tool_name": "search",
        "mcp_arguments": {"q": "hello"},
        "incoming_bearer_token": "tok",
        "message_role": "user",
    }


def test_convert_mcp_to_llm_format_defaults_model(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    out = proxy_logging._convert_mcp_to_llm_format(request_obj=req, kwargs={})
    snapshot = {
        "model": out["model"],
        "mcp_tool_name": out["mcp_tool_name"],
        "incoming_bearer_token": out["incoming_bearer_token"],
        "user_id": out["user_api_key_user_id"],
    }
    assert snapshot == {
        "model": "mcp-tool-call",
        "mcp_tool_name": "calculator",
        "incoming_bearer_token": None,
        "user_id": None,
    }


def test_convert_mcp_to_llm_format_missing_request_obj_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._convert_mcp_to_llm_format(request_obj=None, kwargs={})


# ---------------------------------------------------------------------------
# _convert_llm_result_to_mcp_response
# ---------------------------------------------------------------------------


def test_convert_llm_result_to_mcp_response_exception_blocks(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    result = proxy_logging._convert_llm_result_to_mcp_response(
        llm_result=ValueError("boom"),
        request_obj=req,
    )
    assert isinstance(result, MCPPreCallResponseObject)
    snapshot = {
        "should_proceed": result.should_proceed,
        "error_message": result.error_message,
        "modified_arguments": result.modified_arguments,
    }
    assert snapshot == {"should_proceed": False, "error_message": "boom", "modified_arguments": None}


def test_convert_llm_result_to_mcp_response_blocked_content(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="t", arguments={"a": 1})
    llm_result = {"messages": [{"content": "this is blocked"}]}
    result = proxy_logging._convert_llm_result_to_mcp_response(llm_result=llm_result, request_obj=req)
    assert isinstance(result, MCPPreCallResponseObject)
    assert result.should_proceed is False
    assert "blocked" in (result.error_message or "").lower()


def test_convert_llm_result_to_mcp_response_modified_content_redacted(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="search", arguments={"q": "ssn 123"})
    llm_result = {"messages": [{"content": "Tool: search\nArguments: {\"q\": \"[REDACTED]\"}"}]}
    result = proxy_logging._convert_llm_result_to_mcp_response(llm_result=llm_result, request_obj=req)
    assert isinstance(result, MCPPreCallResponseObject)
    snapshot = {
        "should_proceed": result.should_proceed,
        "modified_q": (result.modified_arguments or {}).get("q"),
        "error": result.error_message,
    }
    assert snapshot == {"should_proceed": True, "modified_q": "[REDACTED]", "error": None}


def test_convert_llm_result_to_mcp_response_string_blocks(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    result = proxy_logging._convert_llm_result_to_mcp_response(llm_result="bad input", request_obj=req)
    assert isinstance(result, MCPPreCallResponseObject)
    snapshot = {
        "should_proceed": result.should_proceed,
        "error_message": result.error_message,
        "modified_arguments": result.modified_arguments,
    }
    assert snapshot == {"should_proceed": False, "error_message": "bad input", "modified_arguments": None}


def test_convert_llm_result_to_mcp_response_unmodified_returns_none(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="x", arguments={"a": 1})
    same_content = "Tool: x\nArguments: {'a': 1}"
    result = proxy_logging._convert_llm_result_to_mcp_response(
        llm_result={"messages": [{"content": same_content}]},
        request_obj=req,
    )
    assert result is None


def test_convert_llm_result_to_mcp_response_no_request_obj_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._convert_llm_result_to_mcp_response(llm_result={"messages": [{"content": "x"}]}, request_obj=None)


# ---------------------------------------------------------------------------
# _extract_modified_arguments_from_content
# ---------------------------------------------------------------------------


def test_extract_modified_arguments_from_content_parses_json(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    out = proxy_logging._extract_modified_arguments_from_content(
        masked_content="Tool: x\nArguments: {\"a\": 1, \"b\": 2, \"c\": 3}",
        request_obj=req,
    )
    assert out == {"a": 1, "b": 2, "c": 3}


def test_extract_modified_arguments_from_content_no_arguments_line_returns_none(proxy_logging, make_mcp_request_obj):
    out = proxy_logging._extract_modified_arguments_from_content(
        masked_content="random content with no arguments",
        request_obj=make_mcp_request_obj(),
    )
    assert out is None


def test_extract_modified_arguments_from_content_empty_string_returns_none(proxy_logging, make_mcp_request_obj):
    out = proxy_logging._extract_modified_arguments_from_content(
        masked_content="",
        request_obj=make_mcp_request_obj(),
    )
    assert out is None


def test_extract_modified_arguments_from_content_invalid_json_falls_back(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(arguments={"name": "alice"})
    out = proxy_logging._extract_modified_arguments_from_content(
        masked_content="Tool: x\nArguments: {name: REDACTED}",
        request_obj=req,
    )
    assert isinstance(out, dict)
    assert "name" in out


def test_extract_modified_arguments_from_content_error_swallowed_returns_none(proxy_logging):
    """Internal try/except swallows any unexpected error and returns None."""
    out = proxy_logging._extract_modified_arguments_from_content(masked_content=None, request_obj=None)
    assert out is None


# ---------------------------------------------------------------------------
# _parse_arguments_manually
# ---------------------------------------------------------------------------


def test_parse_arguments_manually_applies_overrides(proxy_logging):
    original = {"name": "alice", "ssn": "123-45-6789"}
    out = proxy_logging._parse_arguments_manually(
        args_text='"name": "[REDACTED]", "ssn": "[REDACTED]"',
        original_args=original,
    )
    snapshot = {"name": out["name"], "ssn": out["ssn"], "original_unchanged": original["name"]}
    assert snapshot == {"name": "[REDACTED]", "ssn": "[REDACTED]", "original_unchanged": "alice"}


def test_parse_arguments_manually_returns_original_if_no_match(proxy_logging):
    original = {"foo": "bar"}
    out = proxy_logging._parse_arguments_manually(args_text="nothing here", original_args=original)
    assert out == {"foo": "bar"}


def test_parse_arguments_manually_error_swallowed_returns_none(proxy_logging):
    # Defensive: function catches any exception internally and returns None.
    assert proxy_logging._parse_arguments_manually(args_text="x", original_args=None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _convert_llm_result_to_mcp_during_response
# ---------------------------------------------------------------------------


def test_convert_llm_result_to_mcp_during_response_exception(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    result = proxy_logging._convert_llm_result_to_mcp_during_response(
        llm_result=ValueError("during boom"), request_obj=req
    )
    assert isinstance(result, MCPDuringCallResponseObject)
    snapshot = {
        "should_continue": result.should_continue,
        "error_message": result.error_message,
        "type": type(result).__name__,
    }
    assert snapshot == {
        "should_continue": False,
        "error_message": "during boom",
        "type": "MCPDuringCallResponseObject",
    }


def test_convert_llm_result_to_mcp_during_response_blocked_content(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="t", arguments={"a": 1})
    result = proxy_logging._convert_llm_result_to_mcp_during_response(
        llm_result={"messages": [{"content": "blocked content"}]},
        request_obj=req,
    )
    assert isinstance(result, MCPDuringCallResponseObject)
    assert result.should_continue is False
    assert "blocked" in (result.error_message or "").lower()


def test_convert_llm_result_to_mcp_during_response_modified_stops(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="t", arguments={"a": 1})
    result = proxy_logging._convert_llm_result_to_mcp_during_response(
        llm_result={"messages": [{"content": "Tool: t\nArguments: {\"a\": \"[REDACTED]\"}"}]},
        request_obj=req,
    )
    assert isinstance(result, MCPDuringCallResponseObject)
    assert result.should_continue is False
    assert "modified" in (result.error_message or "").lower()


def test_convert_llm_result_to_mcp_during_response_string_blocks(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj()
    result = proxy_logging._convert_llm_result_to_mcp_during_response(
        llm_result="kill switch", request_obj=req
    )
    assert isinstance(result, MCPDuringCallResponseObject)
    snapshot = {"should_continue": result.should_continue, "error_message": result.error_message}
    assert snapshot == {"should_continue": False, "error_message": "kill switch"}


def test_convert_llm_result_to_mcp_during_response_unmodified_returns_none(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(tool_name="t", arguments={"a": 1})
    same = "Tool: t\nArguments: {'a': 1}"
    assert (
        proxy_logging._convert_llm_result_to_mcp_during_response(
            llm_result={"messages": [{"content": same}]},
            request_obj=req,
        )
        is None
    )


def test_convert_llm_result_to_mcp_during_response_no_request_obj_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._convert_llm_result_to_mcp_during_response(
            llm_result={"messages": [{"content": "x"}]}, request_obj=None
        )


# ---------------------------------------------------------------------------
# _parse_pre_mcp_call_hook_response
# ---------------------------------------------------------------------------


def test_parse_pre_mcp_call_hook_response_with_modified_args(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(arguments={"a": 1})
    resp = MCPPreCallResponseObject(
        should_proceed=True,
        modified_arguments={"a": "x", "b": "y"},
        error_message=None,
    )
    out = proxy_logging._parse_pre_mcp_call_hook_response(response=resp, original_request=req)
    snapshot = {
        "should_proceed": out["should_proceed"],
        "modified_arguments": out["modified_arguments"],
        "error_message": out["error_message"],
        "hidden_params_type": type(out["hidden_params"]).__name__,
    }
    assert snapshot == {
        "should_proceed": True,
        "modified_arguments": {"a": "x", "b": "y"},
        "error_message": None,
        "hidden_params_type": "HiddenParams",
    }


def test_parse_pre_mcp_call_hook_response_no_modifications_uses_original(proxy_logging, make_mcp_request_obj):
    req = make_mcp_request_obj(arguments={"original": True})
    resp = MCPPreCallResponseObject(
        should_proceed=True, modified_arguments=None, error_message=None
    )
    out = proxy_logging._parse_pre_mcp_call_hook_response(response=resp, original_request=req)
    assert out["modified_arguments"] == {"original": True}


def test_parse_pre_mcp_call_hook_response_invalid_response_raises(proxy_logging, make_mcp_request_obj):
    with pytest.raises(AttributeError):
        proxy_logging._parse_pre_mcp_call_hook_response(
            response=None, original_request=make_mcp_request_obj()
        )


# ---------------------------------------------------------------------------
# _create_mcp_request_object_from_kwargs
# ---------------------------------------------------------------------------


def test_create_mcp_request_object_from_kwargs_full(proxy_logging, make_user_api_key_auth):
    auth = make_user_api_key_auth(user_id="u-1")
    obj = proxy_logging._create_mcp_request_object_from_kwargs(
        kwargs={
            "name": "calc",
            "arguments": {"x": 1},
            "server_name": "math",
            "user_api_key_auth": auth,
        }
    )
    assert isinstance(obj, MCPPreCallRequestObject)
    snapshot = {
        "tool_name": obj.tool_name,
        "arguments": obj.arguments,
        "server_name": obj.server_name,
        "auth_user_id": obj.user_api_key_auth.get("user_id"),
    }
    assert snapshot == {"tool_name": "calc", "arguments": {"x": 1}, "server_name": "math", "auth_user_id": "u-1"}


def test_create_mcp_request_object_from_kwargs_empty(proxy_logging):
    obj = proxy_logging._create_mcp_request_object_from_kwargs(kwargs={})
    snapshot = {
        "tool_name": obj.tool_name,
        "arguments": obj.arguments,
        "server_name": obj.server_name,
    }
    assert snapshot == {"tool_name": "", "arguments": {}, "server_name": None}


def test_create_mcp_request_object_from_kwargs_non_dict_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._create_mcp_request_object_from_kwargs(kwargs=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _convert_mcp_hook_response_to_kwargs
# ---------------------------------------------------------------------------


def test_convert_mcp_hook_response_to_kwargs_applies_modified_args(proxy_logging):
    original = {"arguments": {"a": 1}, "name": "old"}
    out = proxy_logging._convert_mcp_hook_response_to_kwargs(
        response_data={"modified_arguments": {"a": 2}, "extra_headers": {"H": "1"}},
        original_kwargs=original,
    )
    snapshot = {
        "arguments": out["arguments"],
        "extra_headers": out["extra_headers"],
        "name": out["name"],
        "original_unmodified": original["arguments"],
    }
    assert snapshot == {
        "arguments": {"a": 2},
        "extra_headers": {"H": "1"},
        "name": "old",
        "original_unmodified": {"a": 1},
    }


def test_convert_mcp_hook_response_to_kwargs_merges_headers(proxy_logging):
    original = {"extra_headers": {"keep": "yes", "overwrite": "old"}}
    out = proxy_logging._convert_mcp_hook_response_to_kwargs(
        response_data={"extra_headers": {"overwrite": "new", "added": "1"}},
        original_kwargs=original,
    )
    assert out["extra_headers"] == {"keep": "yes", "overwrite": "new", "added": "1"}


def test_convert_mcp_hook_response_to_kwargs_no_response_data_returns_original(proxy_logging):
    original = {"a": 1}
    out = proxy_logging._convert_mcp_hook_response_to_kwargs(response_data=None, original_kwargs=original)
    assert out is original


def test_convert_mcp_hook_response_to_kwargs_invalid_original_raises(proxy_logging):
    with pytest.raises(AttributeError):
        proxy_logging._convert_mcp_hook_response_to_kwargs(
            response_data={"modified_arguments": {"a": 1}}, original_kwargs=None  # type: ignore[arg-type]
        )
