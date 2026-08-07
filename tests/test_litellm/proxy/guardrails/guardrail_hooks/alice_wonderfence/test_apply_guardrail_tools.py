"""Tests for ``apply_guardrail`` scanning of tool calls, tool definitions, and legacy functions[]."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException


def _tool_call(arguments, name="send_email"):
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_tool_call_arguments(guardrail_and_client, make_request_data):
    """Bypass regression: blocked content in tool_calls[].function.arguments must
    BLOCK. tool_calls reach the model but were never scanned (texts-only)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["please run the tool"],
        "tool_calls": [_tool_call('{"body": "DISALLOWED payload"}')],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert exc.value.detail["action"] == "BLOCK"


@pytest.mark.asyncio
async def test_apply_guardrail_request_tool_call_args_mask_fails_closed(guardrail_and_client, make_request_data):
    """On the request side, tool-call args are detection-only pieces in the join.
    A MASK that redacts an arg cannot be spliced back into the wire-format
    arguments string, so forwarding the original unredacted value would leak it;
    the request fails closed (block) instead."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK"
        r.action_text = prompt.replace("secret value", "[REDACTED]")
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["benign"],
        "tool_calls": [_tool_call('{"body": "secret value"}')],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_detect_on_tool_call_args_passes_through(guardrail_and_client, make_request_data):
    """A DETECT verdict on a tool-call argument logs but does not block or mutate
    the arguments (symmetric with the text-side DETECT behavior)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "DETECT" if "watch me" in prompt else "NO_ACTION"
        r.action_text = None
        r.detections = []
        r.correlation_id = "corr-detect"
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {"texts": ["benign"], "tool_calls": [_tool_call('{"x": "watch me"}')]}
    out = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=make_request_data(),
        input_type="request",
    )
    assert out["tool_calls"][0]["function"]["arguments"] == '{"x": "watch me"}'
    assert out["texts"] == ["benign"]


@pytest.mark.asyncio
async def test_apply_guardrail_scans_tool_calls_when_no_texts(guardrail_and_client, make_request_data):
    """An assistant message can carry tool_calls with no text content, so texts
    is empty; the hook must still scan the tool-call arguments (the old
    empty-texts early return skipped them)."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tool_calls": [_tool_call('{"x": "DISALLOWED"}')]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_response_tool_call_arguments(guardrail_and_client, make_request_data):
    """Model-generated tool-call arguments on the response side are scanned too."""
    guardrail, client = guardrail_and_client

    def evaluate(response, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in response else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_response.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tool_calls": [_tool_call('{"x": "DISALLOWED"}')]},
            request_data=make_request_data(),
            input_type="response",
        )
    assert exc.value.status_code == 400


def _tool_def(description="a helpful tool", param_desc=None):
    fn = {
        "name": "do_thing",
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }
    if param_desc is not None:
        fn["parameters"]["properties"]["city"] = {
            "type": "string",
            "description": param_desc,
        }
    return {"type": "function", "function": fn}


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_tool_definition_description(guardrail_and_client, make_request_data):
    """Blocked content in tools[].function.description must BLOCK; tool defs are
    forwarded to the model but were previously unscanned."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["use the tool"],
        "tools": [_tool_def(description="DISALLOWED instructions here")],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(inputs=inputs, request_data=make_request_data(), input_type="request")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_tool_parameter_description(guardrail_and_client, make_request_data):
    """Nested parameter descriptions are scanned too, not just the top-level one."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["hi"],
        "tools": [_tool_def(description="benign", param_desc="DISALLOWED payload")],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(inputs=inputs, request_data=make_request_data(), input_type="request")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_tool_definition_mask_fails_closed(guardrail_and_client, make_request_data):
    """Tool definitions are detection-only; a MASK that would redact a
    description cannot be spliced back into the schema, so the request fails
    closed rather than forward the original unredacted description."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK"
        r.action_text = prompt.replace("secret", "[REDACTED]")
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    inputs = {
        "texts": ["hi"],
        "tools": [_tool_def(description="contains secret stuff")],
    }
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(inputs=inputs, request_data=make_request_data(), input_type="request")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_scans_tools_when_no_texts_or_tool_calls(guardrail_and_client, make_request_data):
    """A request carrying only tool definitions must still be scanned."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tools": [_tool_def(description="DISALLOWED")]},
            request_data=make_request_data(),
            input_type="request",
        )
    assert exc.value.status_code == 400


def _legacy_function(description="a function", param_desc=None):
    fn = {
        "name": "do_thing",
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }
    if param_desc is not None:
        fn["parameters"]["properties"]["city"] = {
            "type": "string",
            "description": param_desc,
        }
    return fn


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_legacy_function_description(guardrail_and_client, make_request_data):
    """Blocked content in the deprecated functions[].description (read from
    request_data, not inputs) must BLOCK."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(functions=[_legacy_function(description="DISALLOWED instructions")]),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_on_legacy_function_parameter_description(guardrail_and_client, make_request_data):
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=make_request_data(functions=[_legacy_function(description="ok", param_desc="DISALLOWED")]),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_scans_legacy_functions_when_no_other_content(guardrail_and_client, make_request_data):
    """A request whose only scannable content is functions[] is still scanned."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "BLOCK" if "DISALLOWED" in prompt else "NO_ACTION"
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": []},
            request_data=make_request_data(functions=[_legacy_function(description="DISALLOWED")]),
            input_type="request",
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_guardrail_legacy_function_detect_does_not_mutate(guardrail_and_client, make_request_data):
    """A DETECT verdict on a function definition logs but does not rewrite it."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "DETECT" if "watch" in prompt else "NO_ACTION"
        r.action_text = None
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    request_data = make_request_data(functions=[_legacy_function(description="watch this")])
    out = await guardrail.apply_guardrail(
        inputs={"texts": ["hi"]},
        request_data=request_data,
        input_type="request",
    )
    assert out is not None
    assert request_data["functions"][0]["description"] == "watch this"


@pytest.mark.asyncio
async def test_apply_guardrail_legacy_function_definition_mask_fails_closed(guardrail_and_client, make_request_data):
    """Legacy functions[] descriptions are detection-only; a MASK that would
    redact one fails closed rather than forward the original unredacted value."""
    guardrail, client = guardrail_and_client

    def evaluate(prompt, **kwargs):
        r = Mock()
        r.action = "MASK"
        r.action_text = prompt.replace("secret", "[REDACTED]")
        r.detections = []
        r.correlation_id = None
        return r

    client.evaluate_prompt.side_effect = evaluate

    request_data = make_request_data(functions=[_legacy_function(description="contains secret stuff")])
    with pytest.raises(HTTPException) as exc:
        await guardrail.apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data=request_data,
            input_type="request",
        )
    assert exc.value.status_code == 400
    assert request_data["functions"][0]["description"] == "contains secret stuff"
