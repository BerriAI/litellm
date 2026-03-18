"""
Tests for the Responses API _extract_tool_result_output_items path,
the non-streaming _hidden_params propagation of code_interpreter_results,
and mock end-to-end streaming integration.
"""

from unittest.mock import MagicMock

from litellm.llms.anthropic.chat.handler import ModelResponseIterator
from litellm.main import stream_chunk_builder
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.responses.main import (
    OutputCodeInterpreterCall,
    OutputCodeInterpreterCallLog,
)
from litellm.types.utils import Choices, Message, ModelResponse


def _make_model_response(code_interpreter_results=None, provider_specific_fields=None):
    """Helper to build a ModelResponse with provider_specific_fields on the message."""
    psf = provider_specific_fields or {}
    if code_interpreter_results is not None:
        psf["code_interpreter_results"] = code_interpreter_results
    msg = Message(content="test", provider_specific_fields=psf if psf else None)
    choice = Choices(index=0, message=msg, finish_reason="stop")
    resp = ModelResponse()
    resp.choices = [choice]
    return resp


def test_extract_tool_result_output_items_from_pydantic_objects():
    """Non-streaming path: code_interpreter_results are Pydantic OutputCodeInterpreterCall objects."""
    items = [
        OutputCodeInterpreterCall(
            type="code_interpreter_call",
            id="srvtoolu_01AAA",
            code="echo hello",
            container_id=None,
            status="completed",
            outputs=[OutputCodeInterpreterCallLog(type="logs", logs="hello\n")],
        ),
        OutputCodeInterpreterCall(
            type="code_interpreter_call",
            id="srvtoolu_01BBB",
            code="echo world",
            container_id=None,
            status="completed",
            outputs=[OutputCodeInterpreterCallLog(type="logs", logs="world\n")],
        ),
    ]
    resp = _make_model_response(code_interpreter_results=items)
    result = LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(resp)
    assert len(result) == 2
    assert result[0].id == "srvtoolu_01AAA"
    assert result[1].id == "srvtoolu_01BBB"


def test_extract_tool_result_output_items_from_dicts():
    """Streaming path: after model_dump(), code_interpreter_results are plain dicts."""
    items = [
        {
            "type": "code_interpreter_call",
            "id": "srvtoolu_01AAA",
            "code": "echo hello",
            "container_id": None,
            "status": "completed",
            "outputs": [{"type": "logs", "logs": "hello\n"}],
        },
    ]
    resp = _make_model_response(code_interpreter_results=items)
    result = LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(resp)
    assert len(result) == 1
    assert result[0]["id"] == "srvtoolu_01AAA"


def test_extract_tool_result_output_items_empty():
    """No code_interpreter_results → empty list."""
    resp = _make_model_response()
    result = LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(resp)
    assert result == []


def test_extract_tool_result_output_items_no_provider_specific_fields():
    """Message with no provider_specific_fields → empty list."""
    msg = Message(content="test")
    choice = Choices(index=0, message=msg, finish_reason="stop")
    resp = ModelResponse()
    resp.choices = [choice]
    result = LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(resp)
    assert result == []


def test_in_place_substitution_preserves_ordering():
    """
    function_call items matching code_interpreter_results should be replaced
    in-place, preserving the original output ordering.

    Simulates: [message, function_call(exec1), function_call(regular), function_call(exec2)]
    Expected:  [message, code_interpreter_call(exec1), function_call(regular), code_interpreter_call(exec2)]
    """
    code_results = [
        OutputCodeInterpreterCall(
            type="code_interpreter_call",
            id="srvtoolu_01AAA",
            code="echo first",
            container_id=None,
            status="completed",
            outputs=[OutputCodeInterpreterCallLog(type="logs", logs="first\n")],
        ),
        OutputCodeInterpreterCall(
            type="code_interpreter_call",
            id="srvtoolu_01CCC",
            code="echo third",
            container_id=None,
            status="completed",
            outputs=[OutputCodeInterpreterCallLog(type="logs", logs="third\n")],
        ),
    ]
    resp = _make_model_response(code_interpreter_results=code_results)

    # Build a mock responses_output list with interleaved items
    class MockItem:
        def __init__(self, type, call_id=None):
            self.type = type
            self.call_id = call_id

    msg_item = MockItem(type="message")
    fc_exec1 = MockItem(type="function_call", call_id="srvtoolu_01AAA")
    fc_regular = MockItem(type="function_call", call_id="srvtoolu_01BBB")
    fc_exec2 = MockItem(type="function_call", call_id="srvtoolu_01CCC")

    responses_output = [msg_item, fc_exec1, fc_regular, fc_exec2]

    # Apply the same logic as _transform_chat_completion_choices_to_responses_output
    tool_result_items = (
        LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(resp)
    )
    if tool_result_items:
        result_by_id = {
            (item.get("id") if isinstance(item, dict) else item.id): item
            for item in tool_result_items
        }
        replaced_ids = set(result_by_id.keys())
        responses_output = [
            (
                result_by_id[getattr(item, "call_id", None)]
                if (
                    getattr(item, "type", None) == "function_call"
                    and getattr(item, "call_id", None) in replaced_ids
                )
                else item
            )
            for item in responses_output
        ]

    # Verify ordering: message, code_interpreter(AAA), function_call(BBB), code_interpreter(CCC)
    assert len(responses_output) == 4
    assert responses_output[0].type == "message"
    assert responses_output[1].type == "code_interpreter_call"
    assert responses_output[1].id == "srvtoolu_01AAA"
    assert responses_output[2].type == "function_call"
    assert responses_output[2].call_id == "srvtoolu_01BBB"
    assert responses_output[3].type == "code_interpreter_call"
    assert responses_output[3].id == "srvtoolu_01CCC"


def test_end_to_end_streaming_chunks_to_code_interpreter_output():
    """
    Mock end-to-end test: Anthropic SSE chunks → ModelResponseIterator →
    stream_chunk_builder → _extract_tool_result_output_items → final output
    with code_interpreter_call items replacing function_call items.

    This exercises the full streaming data flow without a live server.
    """
    # Realistic Anthropic streaming chunks for a single code execution
    raw_chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_01XYZ",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01AAA",
                "name": "bash_code_execution",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"command": "echo e2e_test"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "bash_code_execution_tool_result",
                "tool_use_id": "srvtoolu_01AAA",
                "content": {
                    "type": "bash_code_execution_result",
                    "stdout": "e2e_test\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50},
        },
    ]

    # Step 1: Parse chunks through ModelResponseIterator (Anthropic handler)
    iterator = ModelResponseIterator(None, sync_stream=True)
    parsed_chunks = []
    for chunk in raw_chunks:
        parsed = iterator.chunk_parser(chunk)
        d = parsed.model_dump()
        # In production, CustomStreamWrapper sets the model on each chunk;
        # stream_chunk_builder requires it.
        d["model"] = "claude-sonnet-4-20250514"
        parsed_chunks.append(d)

    # Step 2: Assemble via stream_chunk_builder (simulates end-of-stream)
    assembled = stream_chunk_builder(chunks=parsed_chunks)
    assert assembled is not None

    # Verify stream_chunk_builder picked up code_interpreter_results via last-value-wins
    psf = assembled.choices[0].message.provider_specific_fields
    assert psf is not None
    assert "code_interpreter_results" in psf
    code_results = psf["code_interpreter_results"]
    assert len(code_results) == 1
    # After model_dump + stream_chunk_builder, results are plain dicts
    assert code_results[0]["id"] == "srvtoolu_01AAA"
    assert code_results[0]["code"] == "echo e2e_test"

    # Step 3: Extract via _extract_tool_result_output_items (Responses API layer)
    tool_result_items = (
        LiteLLMCompletionResponsesConfig._extract_tool_result_output_items(assembled)
    )
    assert len(tool_result_items) == 1
    item = tool_result_items[0]
    # Items are dicts after the model_dump path
    assert item["type"] == "code_interpreter_call"
    assert item["id"] == "srvtoolu_01AAA"
    assert item["code"] == "echo e2e_test"
    assert item["outputs"][0]["logs"] == "e2e_test\n"
