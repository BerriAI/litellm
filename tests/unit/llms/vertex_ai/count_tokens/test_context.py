import asyncio
import base64
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
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
    VertexGeminiCountContext,
    VertexGeminiCountContextError,
    gemini_count_contents,
    inference_request_view,
    vertex_gemini_count_context,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

_SCHEMA: Final[JsonValue] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
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
    context: Final = vertex_gemini_count_context(system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == (_FUNCTION_DECLARATION,)
    assert _SCHEMA["additionalProperties"] is False


def test_long_anthropic_tool_name_is_truncated_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(
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
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"name": name, "input_schema": _SCHEMA}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_long_openai_chat_tool_name_is_unchanged_for_vertex() -> None:
    name: Final = "mcp__docs__" + "a" * 70
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"type": "function", "function": {"name": name, "parameters": _SCHEMA}}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools == ({"functionDeclarations": [{"name": name, "parameters": {"type": "object"}}]},)


def test_unknown_tools_are_omitted() -> None:
    context: Final = vertex_gemini_count_context(
        system=None,
        tools=[{"type": "web_search_preview"}, {"type": "computer_use"}],
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is None


def test_text_block_system_instruction_keeps_one_part_per_block() -> None:
    context: Final = vertex_gemini_count_context(
        system=[{"type": "text", "text": "First instruction."}, {"type": "text", "text": "Second instruction."}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "First instruction."}, {"text": "Second instruction."})}


def test_string_system_instruction_maps_to_one_part() -> None:
    context: Final = vertex_gemini_count_context(system="Follow this instruction.", tools=None)

    assert isinstance(context, VertexGeminiCountContext)
    assert context.system_instruction == {"parts": ({"text": "Follow this instruction."},)}


def test_non_text_system_block_returns_an_error_value() -> None:
    context: Final = vertex_gemini_count_context(
        system=[{"type": "image", "source": {"type": "base64", "data": "AAAA"}}],
        tools=None,
    )

    assert isinstance(context, VertexGeminiCountContextError)
    assert "text blocks" in context.error_message


@pytest.mark.parametrize(("system", "tools"), ((None, None), ("", None), ([], [])))
def test_empty_context_is_omitted(system: JsonValue | None, tools: JsonValue | None) -> None:
    context: Final = vertex_gemini_count_context(system=system, tools=tools)

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

    context: Final = vertex_gemini_count_context(system=None, tools=[tool])

    assert isinstance(context, VertexGeminiCountContext)
    assert context.tools is not None
    assert [tool_value["functionDeclarations"] for tool_value in context.tools] == [
        inference_tool["function_declarations"] for inference_tool in inference_tools
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
    context: Final = vertex_gemini_count_context(system=None, tools=tools)

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
_DOWNLOAD_SECONDS: Final = 0.5


class _SlowPngHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        time.sleep(_DOWNLOAD_SECONDS)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(_PNG_BYTES)))
        self.end_headers()
        self.wfile.write(_PNG_BYTES)

    def log_message(self, format: str, *args: object) -> None:
        return None


@contextmanager
def _slow_png_server() -> Generator[str, None, None]:
    with ThreadingHTTPServer(("127.0.0.1", 0), _SlowPngHandler) as server:
        thread: Final = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()


async def _event_loop_wake_delay(nap_seconds: float) -> float:
    started: Final = time.monotonic()
    await asyncio.sleep(nap_seconds)
    return time.monotonic() - started


@pytest.mark.parametrize("custom_llm_provider", ("vertex_ai", "gemini"))
async def test_remote_image_downloads_without_blocking_the_event_loop(
    custom_llm_provider: Literal["vertex_ai", "gemini"], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "user_url_validation", False)
    with _slow_png_server() as base_url:
        image_url: Final = f"{base_url}/{uuid.uuid4().hex}.png"
        wake_delay, contents = await asyncio.gather(
            _event_loop_wake_delay(0.05),
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
    assert wake_delay < _DOWNLOAD_SECONDS / 2


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
def test_anthropic_view_keeps_only_the_turns_after_a_client_compaction_block(system: JsonValue) -> None:
    view: Final = inference_request_view(
        (
            {"role": "user", "content": "first question " * 50},
            {"role": "assistant", "content": [{"type": "compaction", "content": "Summary of the earlier turns."}]},
            {"role": "user", "content": "latest question"},
        ),
        system,
        "anthropic",
    )

    assert view.messages == [{"role": "user", "content": "latest question"}]
    assert "Summary of the earlier turns." in str(view.system)
    assert "Be brief." in str(view.system)


def test_anthropic_view_flattens_a_replayed_web_search_result_into_text() -> None:
    snippet: Final = "LiteLLM ships a weekly release candidate and a stable line."
    view: Final = inference_request_view(
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
    )

    assistant_blocks: Final = TypeAdapter(list[dict[str, str]]).validate_python(view.messages[1]["content"])
    assert [block["type"] for block in assistant_blocks] == ["text"]
    assert snippet in assistant_blocks[0]["text"]


def test_openai_view_is_the_request_unchanged() -> None:
    messages: Final = ({"role": "user", "content": [{"type": "compaction", "content": "kept"}]},)

    view: Final = inference_request_view(messages, "Be brief.", "openai")

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
def test_anthropic_replayed_web_search_result_is_counted_as_its_text(
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

    view: Final = inference_request_view(messages, None, "anthropic")
    contents: Final = asyncio.run(
        gemini_count_contents(view.messages, "anthropic", "gemini-2.5-flash", custom_llm_provider, {})
    )

    assert snippet in str(contents)
