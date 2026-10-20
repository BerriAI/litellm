import asyncio
import base64
import logging
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from typing import Final, Literal
from unittest.mock import patch

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.anthropic.pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
    truncate_tool_name,
)
from litellm.llms.vertex_ai.count_tokens.context import (
    CountContextEdits,
    InferenceRequestView,
    VertexGeminiCountContext,
    VertexGeminiCountContextError,
    gemini_count_contents,
    inference_request_view,
    unconvertible_messages_error,
    vertex_gemini_count_context,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

_SCHEMA: Final[JsonValue] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_NO_EDITS: Final = CountContextEdits(spec=None, model="gemini/gemini-2.5-flash", tools=None)
_FUNCTION_DECLARATION: Final = {
    "functionDeclarations": [
        {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {"type": "object"},
        }
    ]
}


@pytest.mark.parametrize(
    "tool",
    (
        {
            "name": "lookup",
            "description": "Look up a value",
            "input_schema": _SCHEMA,
        },
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": _SCHEMA,
            },
        },
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": _SCHEMA,
        },
    ),
    ids=("anthropic", "openai-chat", "responses"),
)
def test_function_tool_shapes_map_to_vertex_declarations(tool: JsonValue) -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == (_FUNCTION_DECLARATION,)
    assert _SCHEMA["additionalProperties"] is False


def test_long_anthropic_tool_name_is_truncated_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=None,
        tools=[{"name": name, "description": "Look up a value", "input_schema": _SCHEMA}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == (
        {
            "functionDeclarations": [
                {
                    "name": truncate_tool_name(name),
                    "description": "Look up a value",
                    "parameters": {"type": "object"},
                }
            ]
        },
    )
    assert len(truncate_tool_name(name)) == 64


def test_64_character_anthropic_tool_name_is_unchanged_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 53
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=None,
        tools=[{"name": name, "input_schema": _SCHEMA}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_long_openai_chat_tool_name_is_unchanged_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=None,
        tools=[{"type": "function", "function": {"name": name, "parameters": _SCHEMA}}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_unknown_tools_are_omitted() -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=None,
        tools=[{"type": "web_search_preview"}, {"type": "computer_use"}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is None


def test_text_block_system_instruction_keeps_one_part_per_block() -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=[{"type": "text", "text": "First instruction."}, {"type": "text", "text": "Second instruction."}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "First instruction."}, {"text": "Second instruction."})}


def test_string_system_instruction_maps_to_one_part() -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", system="Follow this instruction.", tools=None)

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "Follow this instruction."},)}


def test_non_text_system_block_returns_an_error_value() -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", 
        system=[{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContextError)
    assert "text blocks" in context.error_message


@pytest.mark.parametrize(("system", "tools"), ((None, None), ("", None), ([], [])))
def test_empty_context_is_omitted(system: JsonValue | None, tools: JsonValue | None) -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", system=system, tools=tools)

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction is None
    assert context.tools is None


@pytest.mark.parametrize(
    "tool",
    (
        {"description": "Look up a value", "input_schema": _SCHEMA},
        {"type": "bash_20250124", "name": "bash"},
    ),
    ids=("unnamed", "anthropic-hosted"),
)
def test_anthropic_tools_count_the_same_declarations_inference_sends(tool: dict[str, JsonValue]) -> None:
    translated_tools, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai([tool])  # pyright: ignore[reportArgumentType]  # raw Anthropic tool dict
    inference_tools: Final = VertexGeminiConfig().map_function(
        value=[dict(translated) for translated in translated_tools],
        optional_params={},
    )

    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is not None
    assert [tool_value["functionDeclarations"] for tool_value in context.tools] == [
        inference_tool["function_declarations"] for inference_tool in inference_tools
    ]


@pytest.mark.parametrize("model", ("gemini-3.8-flash", "gemini-2.5-flash"))
@pytest.mark.parametrize("ids", (range(3), range(10**12, 10**12 + 200)), ids=("small-enum", "past-enum-budget"))
def test_count_context_counts_the_integer_enums_inference_sends(model: str, ids: range) -> None:
    tool: Final = {
        "type": "function",
        "function": {
            "name": "open_record",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "integer", "enum": list(ids)}},
                "required": ["record_id"],
            },
        },
    }
    inference_params: Final = VertexGeminiConfig().map_openai_params(
        non_default_params={"tools": [tool]}, optional_params={}, model=model, drop_params=False
    )

    context: Final = vertex_gemini_count_context(model=model, system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is not None
    assert [tool_value["functionDeclarations"] for tool_value in context.tools] == [
        inference_tool["function_declarations"] for inference_tool in inference_params["tools"]
    ]


@pytest.mark.parametrize(
    "tools",
    (
        "lookup",
        [1],
        [{"name": "lookup", "input_schema": "not-a-schema"}],
        [{"type": "function", "function": {"name": "lookup", "parameters": [1]}}],
        [{"name": "lookup", "input_schema": {"type": "object", "properties": {"query": "not-a-schema"}}}],
    ),
    ids=("not-a-list", "non-object-tool", "string-schema", "list-parameters", "malformed-property"),
)
def test_invalid_tools_return_an_error_value(tools: JsonValue) -> None:
    context: Final = vertex_gemini_count_context(model="gemini-2.5-flash", system=None, tools=tools)

    assert isinstance(context, VertexGeminiCountContextError)
    assert "tools" in context.error_message


_ANTHROPIC_CONVERSATION: Final = (
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Summarize both"},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {"query": "x"}}],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "42"}]},
)


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_anthropic_documents_images_and_tool_turns_become_gemini_parts(
    custom_llm_provider: Literal["vertex_ai", "gemini"],
) -> None:
    contents: Final = await gemini_count_contents(
        _ANTHROPIC_CONVERSATION, "anthropic", "gemini-2.5-flash", custom_llm_provider, {}
    )

    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "Summarize both"},
                {"inline_data": {"data": "JVBERi0=", "mime_type": "application/pdf"}},
                {"inline_data": {"data": "aW1hZ2U=", "mime_type": "image/png"}},
            ],
        },
        {"role": "model", "parts": [{"function_call": {"name": "lookup", "args": {"query": "x"}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "lookup", "response": {"content": "42"}}}]},
    ]


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_openai_messages_keep_their_chat_conversion(custom_llm_provider: Literal["vertex_ai", "gemini"]) -> None:
    contents: Final = await gemini_count_contents(
        ({"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}),
        "openai",
        "gemini-2.5-flash",
        custom_llm_provider,
        {},
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]


_PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_LOOP_PROOF_TIMEOUT_SECONDS: Final = 5


@dataclass(frozen=True, slots=True)
class _LoopProbe:
    request_arrived: Event = field(default_factory=Event)
    loop_ran: Event = field(default_factory=Event)
    loop_ran_while_download_pending: Event = field(default_factory=Event)


def _png_handler_waiting_for_loop(probe: _LoopProbe) -> type[BaseHTTPRequestHandler]:
    class _PngHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            probe.request_arrived.set()
            if probe.loop_ran.wait(_LOOP_PROOF_TIMEOUT_SECONDS):
                probe.loop_ran_while_download_pending.set()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(_PNG_BYTES)))
            self.end_headers()
            self.wfile.write(_PNG_BYTES)

        def log_message(self, format: str, *args: object) -> None:
            return None

    return _PngHandler


@contextmanager
def _png_server_waiting_for_loop(probe: _LoopProbe) -> Generator[str, None, None]:
    with ThreadingHTTPServer(("127.0.0.1", 0), _png_handler_waiting_for_loop(probe)) as server:
        thread: Final = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


async def _run_loop_once_download_starts(probe: _LoopProbe) -> None:
    assert await asyncio.to_thread(probe.request_arrived.wait, _LOOP_PROOF_TIMEOUT_SECONDS)
    probe.loop_ran.set()


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_remote_image_downloads_without_blocking_the_event_loop(
    custom_llm_provider: Literal["vertex_ai", "gemini"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    probe: Final = _LoopProbe()
    with _png_server_waiting_for_loop(probe) as base_url:
        image_url: Final = f"{base_url}/{uuid.uuid4().hex}.png"
        _, contents = await asyncio.gather(
            _run_loop_once_download_starts(probe),
            gemini_count_contents(
                (
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe it"},
                            {"type": "image", "source": {"type": "url", "url": image_url}},
                        ],
                    },
                ),
                "anthropic",
                "gemini-2.5-flash",
                custom_llm_provider,
                {},
            ),
        )

    assert contents == [
        {
            "role": "user",
            "parts": [
                {"text": "Describe it"},
                {"inline_data": {"data": base64.b64encode(_PNG_BYTES).decode(), "mime_type": "image/png"}},
            ],
        }
    ]
    assert probe.loop_ran_while_download_pending.is_set()


class _MissingImageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return None


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_image_that_cannot_be_fetched_returns_an_error_value(
    custom_llm_provider: Literal["vertex_ai", "gemini"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    with ThreadingHTTPServer(("127.0.0.1", 0), _MissingImageHandler) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            contents = await gemini_count_contents(
                (
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": f"http://127.0.0.1:{server.server_address[1]}/{uuid.uuid4().hex}.png",
                                },
                            }
                        ],
                    },
                ),
                "anthropic",
                "gemini-2.5-flash",
                custom_llm_provider,
                {},
            )
        finally:
            server.shutdown()

    assert isinstance(contents, VertexGeminiCountContextError)


@pytest.mark.parametrize(
    "system", ("Be brief.", [{"type": "text", "text": "Be brief."}]), ids=("string-system", "text-block-system")
)
async def test_anthropic_view_keeps_only_the_turns_after_a_client_compaction_block(system: JsonValue) -> None:
    view: Final = await inference_request_view(
        (
            {"role": "user", "content": "first question " * 50},
            {"role": "assistant", "content": [{"type": "compaction", "content": "Summary of the earlier turns."}]},
            {"role": "user", "content": "latest question"},
        ),
        system,
        "anthropic",
        _NO_EDITS,
    )

    assert isinstance(view, InferenceRequestView)
    assert view.messages == [{"role": "user", "content": "latest question"}]
    assert "Summary of the earlier turns." in str(view.system)
    assert "Be brief." in str(view.system)


async def test_anthropic_view_flattens_a_replayed_web_search_result_into_text() -> None:
    snippet: Final = "LiteLLM ships a weekly release candidate and a stable line."
    view: Final = await inference_request_view(
        (
            {"role": "user", "content": "What is new?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "news"}},
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://docs.litellm.ai/release_notes",
                                "title": "Release notes",
                                "encrypted_content": "",
                                "snippet": snippet,
                            }
                        ],
                    },
                ],
            },
        ),
        None,
        "anthropic",
        _NO_EDITS,
    )

    assert isinstance(view, InferenceRequestView)
    assistant_blocks: Final = TypeAdapter(list[dict[str, str]]).validate_python(view.messages[1]["content"])
    assert [block["type"] for block in assistant_blocks] == ["text"]
    assert snippet in assistant_blocks[0]["text"]


async def test_openai_view_is_the_request_unchanged() -> None:
    messages: Final = ({"role": "user", "content": [{"type": "compaction", "content": "kept"}]},)

    view: Final = await inference_request_view(messages, "Be brief.", "openai", _NO_EDITS)

    assert isinstance(view, InferenceRequestView)
    assert view.messages == messages
    assert view.system == "Be brief."


async def test_gcs_media_type_lookup_uses_the_deployment_vertex_project() -> None:
    with patch(
        "litellm.llms.vertex_ai.gemini.transformation._get_gcs_object_content_type",
        return_value="image/png",
    ) as lookup:
        contents: Final = await gemini_count_contents(
            (
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "gs://bucket/image-without-extension"}}
                    ],
                },
            ),
            "anthropic",
            "gemini-2.5-flash",
            "vertex_ai",
            {"vertex_project": "deployment-project"},
        )

    assert lookup.call_args.kwargs.get("vertex_project") == "deployment-project"
    assert not isinstance(contents, VertexGeminiCountContextError)
    assert contents[0]["parts"][0] == {
        "file_data": {"mime_type": "image/png", "file_uri": "gs://bucket/image-without-extension"}
    }


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_anthropic_replayed_web_search_result_is_counted_as_its_text(
    custom_llm_provider: Literal["vertex_ai", "gemini"],
) -> None:
    snippet: Final = "LiteLLM ships a weekly release candidate and a stable line."
    messages: Final = (
        {"role": "user", "content": "What is new?"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "news"}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://docs.litellm.ai/release_notes",
                            "title": "Release notes",
                            "encrypted_content": "",
                            "snippet": snippet,
                        }
                    ],
                },
            ],
        },
        {"role": "user", "content": "Thanks."},
    )

    view: Final = await inference_request_view(messages, None, "anthropic", _NO_EDITS)
    assert isinstance(view, InferenceRequestView)
    contents: Final = await gemini_count_contents(
        view.messages, "anthropic", "gemini-2.5-flash", custom_llm_provider, {}
    )

    assert snippet in str(contents)


_TOOL_OUTPUT: Final = "line of a long file that the agent read earlier. " * 180
_READ_FILE_TOOL: Final[dict[str, object]] = {
    "name": "read_file",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_CLEAR_EDIT: Final[JsonValue] = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 500},
    "keep": {"type": "tool_uses", "value": 1},
}
_CLEAR_OLD_TOOL_RESULTS: Final[JsonValue] = {"edits": [_CLEAR_EDIT]}


def _agent_turns(rounds: int, first_round: int = 0) -> tuple[dict[str, JsonValue], ...]:
    return (
        *(
            turn
            for index in range(first_round, first_round + rounds)
            for turn in (
                {"role": "user", "content": f"Read file {index}."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": f"toolu_{index}", "content": _TOOL_OUTPUT}],
                },
            )
        ),  # comprehension-ok: flattens each round into its three turns
        {"role": "user", "content": "Summarize what you read."},
    )


def _tool_outputs_kept(view: InferenceRequestView) -> tuple[bool, ...]:
    return tuple(
        _TOOL_OUTPUT in str(message["content"]) for message in view.messages if "tool_result" in str(message["content"])
    )


def _edits(spec: JsonValue) -> CountContextEdits:
    return CountContextEdits(spec=spec, model="gemini/gemini-2.5-flash", tools=(_READ_FILE_TOOL,))


async def test_anthropic_view_clears_the_tool_results_inference_clears() -> None:
    view: Final = await inference_request_view(_agent_turns(3), None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (False, False, True)


async def test_anthropic_view_applies_clear_edits_without_calling_the_compaction_summary_model() -> None:
    spec: Final[JsonValue] = {
        "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50000}}, _CLEAR_EDIT]
    }
    turns: Final = (
        {"role": "user", "content": "first question " * 30000},
        {"role": "assistant", "content": "first answer"},
        *_agent_turns(3),
    )

    with (
        patch(
            "litellm.llms.anthropic.pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="gemini/gemini-2.5-flash",
        ),
        patch.object(litellm, "acompletion", side_effect=AssertionError("count_tokens must not summarize")) as summary,
    ):
        view: Final = await inference_request_view(turns, None, "anthropic", _edits(spec))

    assert summary.call_count == 0
    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (False, False, True)


async def test_anthropic_view_counts_the_unedited_body_without_an_error_log_when_an_edit_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch.object(litellm, "token_counter", side_effect=RuntimeError("tokenizer unavailable")),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
    ):
        view: Final = await inference_request_view(_agent_turns(3), None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True, True)
    assert any("skipping edits" in record.getMessage() for record in caplog.records)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_anthropic_view_counts_the_turns_after_a_compaction_block_when_an_edit_before_compact_fails() -> None:
    clear_before_compact: Final[JsonValue] = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "tool_uses", "value": 3},
                "keep": {"type": "tool_uses", "value": 1},
            },
            {"type": "compact_20260112"},
        ]
    }
    turns: Final = (
        *_agent_turns(3)[:-1],
        {"role": "assistant", "content": [{"type": "compaction", "content": "Read files 0 to 2."}]},
        *_agent_turns(2, first_round=3),
    )

    view: Final = await inference_request_view(turns, None, "anthropic", _edits(clear_before_compact))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True)
    assert "Read files 0 to 2." in str(view.system)


async def test_anthropic_view_rejects_a_compact_trigger_inference_rejects() -> None:
    low_trigger: Final[JsonValue] = {
        "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
    }

    view: Final = await inference_request_view(_agent_turns(1), None, "anthropic", _edits(low_trigger))

    assert isinstance(view, VertexGeminiCountContextError)
    assert view.status_code == 400


async def test_anthropic_view_names_the_block_gemini_cannot_count_when_an_edit_trigger_cannot_tokenize_it() -> None:
    turns: Final = (
        *_agent_turns(1)[:-1],
        {
            "role": "user",
            "content": [
                {
                    "type": "search_result",
                    "source": "https://docs.litellm.ai",
                    "title": "Docs",
                    "content": [{"type": "text", "text": "LiteLLM docs"}],
                }
            ],
        },
    )

    view: Final = await inference_request_view(turns, None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    blocks_error: Final = unconvertible_messages_error(view.messages, "anthropic")
    assert blocks_error is not None
    assert blocks_error.status_code == 400
    assert "search_result" in blocks_error.error_message
