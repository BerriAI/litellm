"""`/v1/responses` bridged to a Bedrock Converse deployment.

Bedrock has no Responses API for Claude or Nova, so the proxy translates a Responses request into a chat
completion (`litellm/responses/litellm_completion_transformation/`), runs it through the Converse transform,
and translates the answer back into a Responses object. The contract a caller relies on: `instructions`
becomes the Converse `system`, `input` becomes the user turn, `max_output_tokens` becomes
`inferenceConfig.maxTokens`, function tools become Converse tool specs and a Bedrock `toolUse` comes back as a
`function_call` output item, reasoning comes back as a `reasoning` item, and usage is reported as
`input_tokens` and `output_tokens`. Streaming must produce the Responses event sequence from the Converse
event stream. Only the basic text round trip is pinned elsewhere
(`translation/responses/basic/test_responses_basic_bedrock_converse.py`).
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    CLAUDE_SONNET_45,
    bedrock,
    body_of,
    converse_frames,
    converse_reply,
    marker,
    path_of,
)
from pydantic import JsonValue

MODEL: Final = f"bedrock/{CLAUDE_SONNET_45}"
FUNCTION_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Look up the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def _ok(_: Request) -> Reply:
    return converse_reply()


def test_instructions_input_and_max_output_tokens_become_the_converse_request(gateway: Gateway) -> None:
    """A Responses call with `instructions`, a string `input` and `max_output_tokens: 48`.

    Bedrock must receive one `POST /model/<id>/converse` with `system: [{"text": <instructions>}]`, the
    input as the single user text block, and `inferenceConfig.maxTokens == 48`.
    """
    prompt: Final = marker()
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.responses(instructions="Reply in one word.", input=prompt, max_output_tokens=48)
        assert response.status_code == 200, response.text
        sent_request: Final = rig.one_post()
        assert path_of(sent_request) == f"/model/{CLAUDE_SONNET_45}/converse"
        sent: Final = body_of(sent_request)
        assert sent["system"] == [{"text": "Reply in one word."}]
        assert sent["messages"] == [{"role": "user", "content": [{"text": prompt}]}]
        assert sent["inferenceConfig"]["maxTokens"] == 48  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]


def test_the_converse_answer_becomes_a_completed_response_with_output_text_and_usage(gateway: Gateway) -> None:
    """Bedrock answers `pong` with 11 input and 7 output tokens.

    The caller gets `object: response`, `status: completed`, one `message` output item whose content is
    `[{"type": "output_text", "text": "pong", ...}]`, and `usage.input_tokens` 11, `output_tokens` 7,
    `total_tokens` 18.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.responses(input=marker())
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["object"] == "response"
        assert body["status"] == "completed"
        messages: Final = [item for item in body["output"] if item["type"] == "message"]
        assert messages[0]["content"][0]["type"] == "output_text"
        assert messages[0]["content"][0]["text"] == "pong"
        assert (body["usage"]["input_tokens"], body["usage"]["output_tokens"], body["usage"]["total_tokens"]) == (
            11,
            7,
            18,
        )


def test_a_responses_function_tool_becomes_a_converse_tool_and_tool_use_a_function_call_item(gateway: Gateway) -> None:
    """A Responses function tool `get_weather`, answered by a Bedrock `toolUse` (`tooluse_resp`, `{"city": "Oslo"}`).

    Bedrock must receive `toolConfig.tools[0].toolSpec.name == "get_weather"`, and the caller's output must
    contain a `function_call` item with `call_id: tooluse_resp`, `name: get_weather` and `arguments` that parse
    to `{"city": "Oslo"}`.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[{"toolUse": {"toolUseId": "tooluse_resp", "name": "get_weather", "input": {"city": "Oslo"}}}],
            stop_reason="tool_use",
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.responses(input=marker(), tools=[FUNCTION_TOOL])
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        calls: Final = [item for item in response.json()["output"] if item["type"] == "function_call"]
        assert len(calls) == 1, response.text
        assert (calls[0]["call_id"], calls[0]["name"]) == ("tooluse_resp", "get_weather")
        assert json.loads(calls[0]["arguments"]) == {"city": "Oslo"}


def test_a_function_call_output_turn_is_replayed_as_a_converse_tool_result(gateway: Gateway) -> None:
    """The caller replays a `function_call` item and its `function_call_output` (`sunny`) as Responses input.

    Bedrock must receive an assistant `toolUse` block with `toolUseId: tooluse_back` and a following user turn
    whose `toolResult` has the same id and `content: [{"text": "sunny"}]`.
    """
    with bedrock(gateway, _ok, model=MODEL) as rig:
        response: Final = rig.responses(
            tools=[FUNCTION_TOOL],
            input=[
                {"role": "user", "content": marker()},
                {
                    "type": "function_call",
                    "call_id": "tooluse_back",
                    "name": "get_weather",
                    "arguments": '{"city": "Oslo"}',
                },
                {"type": "function_call_output", "call_id": "tooluse_back", "output": "sunny"},
            ],
        )
        assert response.status_code == 200, response.text
        messages: Final = body_of(rig.one_post())["messages"]
        assert isinstance(messages, list)
        flat: Final = json.dumps(messages)
        assert '"toolUseId": "tooluse_back"' in flat
        result_blocks: Final = [
            block["toolResult"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
            for message in messages
            for block in message["content"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript, reportGeneralTypeIssues]
            if isinstance(block, dict) and "toolResult" in block
        ]
        assert result_blocks[0]["toolUseId"] == "tooluse_back"  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]
        assert result_blocks[0]["content"] == [{"text": "sunny"}]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue, reportOptionalSubscript]


def test_responses_reasoning_effort_becomes_a_thinking_budget_and_reasoning_comes_back_as_an_item(
    gateway: Gateway,
) -> None:
    """`reasoning: {"effort": "low"}` on Claude Sonnet 4.5 through the Responses bridge.

    Bedrock must receive `additionalModelRequestFields.thinking == {"type": "enabled", "budget_tokens": 1024}`,
    and a Converse `reasoningContent` answer must come back as an output item of `type: reasoning` ahead of
    the message. UNVERIFIED: the reasoning item's summary layout, so only its presence and order are asserted.
    """

    def respond(_: Request) -> Reply:
        return converse_reply(
            content=[
                {"reasoningContent": {"reasoningText": {"text": "weigh it", "signature": "sig-resp"}}},
                {"text": "done"},
            ]
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.responses(input=marker(), reasoning={"effort": "low"}, max_output_tokens=4096)
        assert response.status_code == 200, response.text
        thinking: Final = body_of(rig.one_post())["additionalModelRequestFields"]["thinking"]  # pyright: ignore[reportIndexIssue, reportArgumentType, reportCallIssue]
        assert thinking == {"type": "enabled", "budget_tokens": 1024}
        kinds: Final = [item["type"] for item in response.json()["output"]]
        assert "reasoning" in kinds and "message" in kinds
        assert kinds.index("reasoning") < kinds.index("message")


def test_a_streamed_response_turns_converse_frames_into_responses_events(gateway: Gateway) -> None:
    """`stream: true` on the Responses bridge calls `/converse-stream` and emits Responses SSE events.

    The event types must start with `response.created`, include `response.output_text.delta` events whose
    deltas join to `hello stream`, and end with `response.completed` whose `response.usage.output_tokens` is 7.
    """

    def respond(_: Request) -> Reply:
        return converse_frames(
            (
                ("messageStart", {"role": "assistant"}),
                ("contentBlockDelta", {"delta": {"text": "hello "}, "contentBlockIndex": 0}),
                ("contentBlockDelta", {"delta": {"text": "stream"}, "contentBlockIndex": 0}),
                ("contentBlockStop", {"contentBlockIndex": 0}),
                ("messageStop", {"stopReason": "end_turn"}),
                (
                    "metadata",
                    {"usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}, "metrics": {"latencyMs": 1}},
                ),
            )
        )

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.responses(input=marker(), stream=True)
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/{CLAUDE_SONNET_45}/converse-stream"
        events: Final = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        kinds: Final = [event["type"] for event in events]
        assert kinds[0] == "response.created", kinds
        assert kinds[-1] == "response.completed", kinds
        deltas: Final = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
        assert deltas == "hello stream"
        assert events[-1]["response"]["usage"]["output_tokens"] == 7
