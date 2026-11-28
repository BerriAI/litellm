"""`gemini/` chat: Google's built-in tools, thinking configuration, media resolution and image output.

These are the parts of a Gemini chat request that the Vertex suite (`test_dec1_vertex_*`) does not pin
and that AI Studio users lean on most:

Built-in tools. A request tool `{"googleSearch": {}}`, OpenAI's `web_search_options` or a
`{"type": "web_search"}` tool becomes the Gemini tool `{"googleSearch": {}}`; `{"urlContext": {}}`
becomes `{"url_context": {}}`; `{"codeExecution": {}}` becomes `{"code_execution": {}}`. Each built-in
tool is its own entry of `tools`. Google refuses search tools next to function declarations, so the
proxy drops the search tools in that case unless the caller sets `include_server_side_tool_invocations`
(`_resolve_search_tool_conflict`). On the way back `groundingMetadata` becomes OpenAI `annotations`
plus the raw `vertex_ai_grounding_metadata` list on the response, `urlContextMetadata` becomes
`vertex_ai_url_context_metadata`, and the distinct `webSearchQueries` become
`usage.prompt_tokens_details.web_search_requests` (what search billing is computed from).

Thinking. Gemini 2.5 takes a token budget (`thinkingConfig.thinkingBudget`), Gemini 3 and newer take a
level (`thinkingConfig.thinkingLevel`); `reasoning_effort` maps to whichever the model takes, and an
Anthropic style `thinking.budget_tokens` is dropped on Gemini 3 (`_map_thinking_param`).

Media resolution. An image part's OpenAI `detail` becomes a per-part `media_resolution` on Gemini 3,
and one request-wide `generationConfig.mediaResolution` (the highest detail in the request) on
Gemini 2.x (`_apply_gemini_metadata`, `transformation.py:1228`).

Every test uses a fresh prompt marker because the shared proxy caches responses.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_gemini_support import (
    GEMINI_3_FLASH,
    GEMINI_3_PRO,
    at,
    gemini,
    gemini_echo,
    gemini_reply,
    json_of,
    marker,
    message_of,
    search_grounding,
    usage_metadata,
)
from integration.providers.dec1_vertex_support import PNG_B64
from pydantic import JsonValue

_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_IMAGE: Final = f"data:image/png;base64,{PNG_B64}"


def _tools(body: dict[str, JsonValue]) -> list[JsonValue]:
    return list(list_value(body.get("tools") or []))


def _thinking(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(at(body, "generationConfig", "thinkingConfig"))


def _image_message(*details: str | None) -> list[JsonValue]:
    parts: Final[list[JsonValue]] = [{"type": "text", "text": f"describe {marker()}"}]
    for detail in details:
        image_url: dict[str, JsonValue] = {"url": _IMAGE}
        if detail is not None:
            image_url["detail"] = detail
        parts.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": parts}]


def _image_parts(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    contents: Final = list_value(body["contents"])
    parts: Final = list_value(object_value(contents[0])["parts"])
    return [object_value(part) for part in parts if "inline_data" in object_value(part)]


# ---------------------------------------------------------------- Google Search grounding


def test_a_google_search_tool_is_sent_as_its_own_google_search_entry(gateway: Gateway) -> None:
    """A request tool `{"googleSearch": {}}` reaches AI Studio as `tools: [{"googleSearch": {}}]`, one
    Tool object holding only the search tool."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"googleSearch": {}}]).status_code == 200
        assert _tools(rig.peer.one_body()) == [{"googleSearch": {}}]


def test_web_search_options_become_a_google_search_tool(gateway: Gateway) -> None:
    """OpenAI's `web_search_options: {"search_context_size": "high"}` becomes `{"googleSearch": {}}`;
    Google has no context size or user location knob, so the options are dropped
    (`_map_web_search_options`)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(web_search_options={"search_context_size": "high"}).status_code == 200
        assert _tools(rig.peer.one_body()) == [{"googleSearch": {}}]


@pytest.mark.parametrize("tool_type", ["web_search", "web_search_preview"])
def test_an_openai_web_search_tool_becomes_google_search(gateway: Gateway, tool_type: str) -> None:
    """An OpenAI style tool `{"type": "web_search"}` or `{"type": "web_search_preview"}` is rewritten to
    `{"googleSearch": {}}` instead of being forwarded as an unknown tool."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"type": tool_type}]).status_code == 200
        assert _tools(rig.peer.one_body()) == [{"googleSearch": {}}]


def test_a_search_tool_next_to_a_function_is_dropped(gateway: Gateway) -> None:
    """With both `get_weather` and `{"googleSearch": {}}` the body keeps only
    `tools: [{"function_declarations": [...]}]`; Google answers 400 to a mix, so the proxy drops the search
    tool and keeps the function (#23337)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[_WEATHER_TOOL, {"googleSearch": {}}]).status_code == 200
        tools: Final = _tools(rig.peer.one_body())
        assert len(tools) == 1
        assert "function_declarations" in object_value(tools[0])


def test_server_side_tool_invocations_keep_search_next_to_functions(gateway: Gateway) -> None:
    """With `include_server_side_tool_invocations: true` (Gemini 3 tool combination) the same mix keeps
    both entries, `function_declarations` and `googleSearch`, and the body gains
    `toolConfig.includeServerSideToolInvocations: true`."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        response: Final = rig.chat(
            tools=[_WEATHER_TOOL, {"googleSearch": {}}], include_server_side_tool_invocations=True
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        keys: Final = [next(iter(object_value(tool))) for tool in _tools(body)]
        assert keys == ["function_declarations", "googleSearch"]
        assert at(body, "toolConfig", "includeServerSideToolInvocations") is True


def test_grounding_metadata_is_returned_on_the_response(gateway: Gateway) -> None:
    """A candidate carrying `groundingMetadata` (one query, one web chunk) is answered with the raw list
    under `vertex_ai_grounding_metadata`, whose first item still has `webSearchQueries: ["weather paris"]`
    and the `searchEntryPoint` Google requires apps to render."""
    grounding: Final = search_grounding()
    with gemini(gateway, lambda _r: gemini_reply(text="Sunny in Paris", groundingMetadata=grounding)) as rig:
        response: Final = rig.chat(tools=[{"googleSearch": {}}])
        assert response.status_code == 200, response.text
        metadata: Final = list_value(json_of(response)["vertex_ai_grounding_metadata"])
        assert object_value(metadata[0])["webSearchQueries"] == ["weather paris"]
        assert "searchEntryPoint" in object_value(metadata[0])


def test_grounding_supports_become_url_citation_annotations(gateway: Gateway) -> None:
    """The support covering characters 0..5 with chunk 0 becomes one annotation
    `{"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 5, "url":
    "https://example.com/paris", "title": "example.com"}}` on the assistant message."""
    with gemini(gateway, lambda _r: gemini_reply(text="Sunny in Paris", groundingMetadata=search_grounding())) as rig:
        response: Final = rig.chat(tools=[{"googleSearch": {}}])
        assert message_of(response)["annotations"] == [
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": 0,
                    "end_index": 5,
                    "url": "https://example.com/paris",
                    "title": "example.com",
                },
            }
        ]


def test_distinct_search_queries_are_counted_as_web_search_requests(gateway: Gateway) -> None:
    """Three `webSearchQueries` of which two are identical are reported as
    `usage.prompt_tokens_details.web_search_requests: 2`: duplicates collapse (#36377) and the count is
    what per-query search billing multiplies."""
    grounding: Final = search_grounding(("paris weather", "paris weather", "paris forecast"))
    with gemini(gateway, lambda _r: gemini_reply(text="Sunny", groundingMetadata=grounding)) as rig:
        response: Final = rig.chat(tools=[{"googleSearch": {}}])
        assert at(json_of(response), "usage", "prompt_tokens_details", "web_search_requests") == 2


def test_search_tool_use_tokens_are_not_added_to_the_prompt(gateway: Gateway) -> None:
    """With search grounding Google bills a per-request fee instead of the retrieved tokens, so
    `toolUsePromptTokenCount: 40` is NOT added: `prompt_tokens` stays at `promptTokenCount` 11
    (`billable_tool_use_prompt_tokens` is 0 when `_response_has_search_grounding`)."""
    usage: Final = usage_metadata(11, 7, toolUsePromptTokenCount=40, totalTokenCount=58)
    with gemini(
        gateway, lambda _r: gemini_reply(text="Sunny", groundingMetadata=search_grounding(), usage=usage)
    ) as rig:
        response: Final = rig.chat(tools=[{"googleSearch": {}}])
        assert at(json_of(response), "usage", "prompt_tokens") == 11


# ---------------------------------------------------------------- URL context and code execution


def test_a_url_context_tool_is_sent_as_url_context(gateway: Gateway) -> None:
    """A request tool `{"urlContext": {}}` is sent with Google's snake_case name, `tools: [{"url_context": {}}]`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"urlContext": {}}]).status_code == 200
        assert _tools(rig.peer.one_body()) == [{"url_context": {}}]


def test_url_context_metadata_is_returned_on_the_response(gateway: Gateway) -> None:
    """A candidate with `urlContextMetadata.urlMetadata` (one retrieved URL with
    `URL_RETRIEVAL_STATUS_SUCCESS`) is answered with that metadata under `vertex_ai_url_context_metadata`."""
    url_metadata: Final[dict[str, JsonValue]] = {
        "urlMetadata": [
            {"retrievedUrl": "https://example.com/doc", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}
        ]
    }
    with gemini(gateway, lambda _r: gemini_reply(text="Read it", urlContextMetadata=url_metadata)) as rig:
        response: Final = rig.chat(tools=[{"urlContext": {}}])
        assert response.status_code == 200, response.text
        returned: Final = list_value(json_of(response)["vertex_ai_url_context_metadata"])
        assert object_value(returned[0])["urlMetadata"] == url_metadata["urlMetadata"]


def test_url_context_tool_use_tokens_are_billed_as_prompt_tokens(gateway: Gateway) -> None:
    """URL context is billed at the input token rate, so `toolUsePromptTokenCount: 40` IS added:
    `prompt_tokens = 11 + 40 = 51` and `prompt_tokens_details.tool_use_tokens = 40`."""
    usage: Final = usage_metadata(11, 7, toolUsePromptTokenCount=40, totalTokenCount=58)
    with gemini(gateway, lambda _r: gemini_reply(text="Read it", usage=usage)) as rig:
        body: Final = json_of(rig.chat(tools=[{"urlContext": {}}]))
        assert at(body, "usage", "prompt_tokens") == 51
        assert at(body, "usage", "prompt_tokens_details", "tool_use_tokens") == 40


def test_a_code_execution_tool_is_sent_as_code_execution(gateway: Gateway) -> None:
    """A request tool `{"codeExecution": {}}` reaches AI Studio as `tools: [{"code_execution": {}}]`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"codeExecution": {}}]).status_code == 200
        assert _tools(rig.peer.one_body()) == [{"code_execution": {}}]


def test_code_execution_can_sit_next_to_a_function(gateway: Gateway) -> None:
    """Code execution is not a search tool, so next to `get_weather` both survive:
    `tools` has a `function_declarations` entry and a `code_execution` entry, in that order."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[_WEATHER_TOOL, {"codeExecution": {}}]).status_code == 200
        keys: Final = [next(iter(object_value(tool))) for tool in _tools(rig.peer.one_body())]
        assert keys == ["function_declarations", "code_execution"]


def test_code_execution_parts_are_left_out_of_the_text_content(gateway: Gateway) -> None:
    """An answer whose parts are text "Let me compute.", an `executableCode` part, a
    `codeExecutionResult` part and text " The answer is 4." comes back with `content` equal to the two
    text parts joined; `get_assistant_content_message` only reads `text` and non-media `inlineData`, so
    the code and its output are not in the message today. Pinned as current behaviour."""
    parts: Final[list[JsonValue]] = [
        {"text": "Let me compute."},
        {"executableCode": {"language": "PYTHON", "code": "print(2 + 2)"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "4\n"}},
        {"text": " The answer is 4."},
    ]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        response: Final = rig.chat(tools=[{"codeExecution": {}}])
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "Let me compute. The answer is 4."


# ---------------------------------------------------------------- thinking


@pytest.mark.parametrize(("effort", "budget"), [("low", 1024), ("medium", 2048), ("high", 4096)])
def test_reasoning_effort_is_a_thinking_budget_on_gemini_2_5(gateway: Gateway, effort: str, budget: int) -> None:
    """On `gemini-2.5-flash`, `reasoning_effort` low, medium and high become `thinkingBudget` 1024, 2048 and
    4096 (`DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET` in `litellm/constants.py`) with
    `includeThoughts: true`, and no `thinkingLevel`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(reasoning_effort=effort).status_code == 200
        thinking: Final = _thinking(rig.peer.one_body())
        assert thinking.get("thinkingBudget") == budget
        assert thinking.get("includeThoughts") is True
        assert "thinkingLevel" not in thinking


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_is_a_thinking_level_on_gemini_3(gateway: Gateway, effort: str) -> None:
    """On `gemini-3-flash-preview` the same efforts become `thinkingConfig: {"thinkingLevel": <effort>,
    "includeThoughts": true}` and no `thinkingBudget` (`_map_reasoning_effort_to_thinking_level`)."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(reasoning_effort=effort).status_code == 200
        assert _thinking(rig.peer.one_body()) == {"thinkingLevel": effort, "includeThoughts": True}


def test_minimal_effort_is_minimal_on_a_gemini_3_flash_model(gateway: Gateway) -> None:
    """`reasoning_effort: minimal` on a Gemini 3 flash model is sent as `thinkingLevel: minimal`, a level
    only the flash models accept (`_supports_minimal_thinking_level`)."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(reasoning_effort="minimal").status_code == 200
        assert _thinking(rig.peer.one_body())["thinkingLevel"] == "minimal"


def test_minimal_effort_falls_back_to_low_on_a_gemini_3_pro_model(gateway: Gateway) -> None:
    """`reasoning_effort: minimal` on `gemini-3.1-pro-preview` (not a flash model) is sent as
    `thinkingLevel: low`, since Gemini 3 pro rejects `minimal`."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_PRO) as rig:
        assert rig.chat(reasoning_effort="minimal").status_code == 200
        assert _thinking(rig.peer.one_body())["thinkingLevel"] == "low"


def test_disabling_reasoning_on_gemini_3_sends_the_lowest_level_without_thoughts(gateway: Gateway) -> None:
    """`reasoning_effort: none` cannot switch Gemini 3 thinking off, so it is sent as the lowest level the
    model takes (`minimal` on a flash model) with `includeThoughts: false`."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(reasoning_effort="none").status_code == 200
        assert _thinking(rig.peer.one_body()) == {"thinkingLevel": "minimal", "includeThoughts": False}


def test_xhigh_effort_is_xhigh_only_on_a_model_that_supports_it(gateway: Gateway) -> None:
    """`gemini-4-hydrogen` has `supports_xhigh_reasoning_effort` in the cost map, so `reasoning_effort:
    xhigh` is sent as `thinkingLevel: xhigh` (#52060, merged 2026-11-16)."""
    with gemini(gateway, gemini_echo, backend="gemini-4-hydrogen") as rig:
        response: Final = rig.chat(reasoning_effort="xhigh")
        assert response.status_code == 200, response.text
        assert _thinking(rig.peer.one_body())["thinkingLevel"] == "xhigh"


def test_xhigh_effort_on_a_model_without_it_is_refused_before_google(gateway: Gateway) -> None:
    """On `gemini-3-flash-preview`, which lacks `supports_xhigh_reasoning_effort`, `reasoning_effort: xhigh`
    falls to the unsupported branch and is answered 400 (`UnsupportedParamsError`) with no
    generateContent call. UNVERIFIED: whether a global `drop_params` on the shared proxy drops it instead."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort="xhigh")
        assert response.status_code == 400, response.text
        assert rig.peer.generate_calls() == ()


def test_an_anthropic_thinking_budget_is_dropped_on_gemini_3(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 2048}` on a Gemini 3 model sends no `thinkingBudget`
    (Gemini 3 takes only a level, `_LEVEL_ONLY_BUDGET_REASON`) and no `thinkingLevel` either, leaving the
    provider default, with `includeThoughts: true`."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(thinking={"type": "enabled", "budget_tokens": 2048}).status_code == 200
        assert _thinking(rig.peer.one_body()) == {"includeThoughts": True}


def test_an_anthropic_thinking_budget_is_kept_on_gemini_2_5(gateway: Gateway) -> None:
    """The same `thinking` block on `gemini-2.5-flash` becomes `thinkingBudget: 2048` with
    `includeThoughts: true`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(thinking={"type": "enabled", "budget_tokens": 2048}).status_code == 200
        assert _thinking(rig.peer.one_body()) == {"includeThoughts": True, "thinkingBudget": 2048}


def test_reasoning_effort_and_a_thinking_block_together_are_refused(gateway: Gateway) -> None:
    """Sending both `reasoning_effort` and `thinking` on a Gemini 3 model is refused with 400 before Google
    is called (`_validate_thinking_config_conflicts`). UNVERIFIED: the validator only raises when a
    `thinkingLevel` and a `thinkingBudget` collide; if the second param simply overwrites the first,
    this test documents the gap."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort="high", thinking={"type": "enabled", "budget_tokens": 1024})
        assert response.status_code in (200, 400), response.text
        if response.status_code == 200:
            assert "thinkingBudget" not in _thinking(rig.peer.one_body())


def test_thought_signatures_come_back_on_thinking_blocks(gateway: Gateway) -> None:
    """A `thought: true` part with `thoughtSignature: "c2ln"` comes back as `reasoning_content` and as a
    `thinking_blocks` entry `{"type": "thinking", "thinking": <text>, "signature": "c2ln"}`, which a client
    must send back on the next turn of a Gemini 3 tool loop."""
    parts: Final[list[JsonValue]] = [
        {"text": "plan the answer", "thought": True, "thoughtSignature": "c2ln"},
        {"text": "done"},
    ]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts), backend=GEMINI_3_FLASH) as rig:
        message: Final = message_of(rig.chat(reasoning_effort="low"))
        assert message["content"] == "done"
        assert message["reasoning_content"] == "plan the answer"
        blocks: Final = list_value(message["thinking_blocks"])
        assert object_value(blocks[0])["signature"] == "c2ln"


# ---------------------------------------------------------------- media resolution and image output


def test_image_detail_becomes_a_per_part_media_resolution_on_gemini_3(gateway: Gateway) -> None:
    """On a Gemini 3 model an `image_url` with `detail: "high"` is sent as an `inline_data` part carrying
    `media_resolution: {"level": "MEDIA_RESOLUTION_HIGH"}`, and `generationConfig` gets no
    `mediaResolution`."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(messages=_image_message("high")).status_code == 200
        body: Final = rig.peer.one_body()
        assert _image_parts(body)[0]["media_resolution"] == {"level": "MEDIA_RESOLUTION_HIGH"}
        assert "mediaResolution" not in object_value(body.get("generationConfig") or {})


@pytest.mark.parametrize(
    ("detail", "level"),
    [
        ("low", "MEDIA_RESOLUTION_LOW"),
        ("medium", "MEDIA_RESOLUTION_MEDIUM"),
        ("high", "MEDIA_RESOLUTION_HIGH"),
        ("ultra_high", "MEDIA_RESOLUTION_ULTRA_HIGH"),
    ],
)
def test_each_detail_value_maps_to_its_media_resolution_level(gateway: Gateway, detail: str, level: str) -> None:
    """`detail` low, medium, high and ultra_high map to the four `MEDIA_RESOLUTION_*` levels
    (`_convert_detail_to_media_resolution_enum`)."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(messages=_image_message(detail)).status_code == 200
        assert _image_parts(rig.peer.one_body())[0]["media_resolution"] == {"level": level}


def test_image_detail_auto_sends_no_media_resolution(gateway: Gateway) -> None:
    """`detail: "auto"` (OpenAI's default) maps to nothing, so the part has no `media_resolution` and Google
    picks its own default."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(messages=_image_message("auto")).status_code == 200
        assert "media_resolution" not in _image_parts(rig.peer.one_body())[0]


def test_gemini_2_gets_one_request_wide_media_resolution_at_the_highest_detail(gateway: Gateway) -> None:
    """On `gemini-2.5-flash` two images with `detail` low and high produce one
    `generationConfig.mediaResolution: "MEDIA_RESOLUTION_HIGH"` (the highest detail wins) and no
    per-part `media_resolution`, which 2.x does not accept."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(messages=_image_message("low", "high")).status_code == 200
        body: Final = rig.peer.one_body()
        assert at(body, "generationConfig", "mediaResolution") == "MEDIA_RESOLUTION_HIGH"
        assert all("media_resolution" not in part for part in _image_parts(body))


def test_a_data_uri_image_is_sent_as_inline_data(gateway: Gateway) -> None:
    """A base64 data URI image is sent as `{"inline_data": {"mime_type": "image/png", "data": <base64>}}`
    with the bytes unchanged."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(messages=_image_message(None)).status_code == 200
        inline: Final = object_value(_image_parts(rig.peer.one_body())[0]["inline_data"])
        assert inline == {"mime_type": "image/png", "data": PNG_B64}


def test_modalities_become_response_modalities(gateway: Gateway) -> None:
    """`modalities: ["text", "image"]` becomes `generationConfig.responseModalities: ["TEXT", "IMAGE"]`
    (`map_response_modalities`), which an image capable model needs to answer with an image."""
    with gemini(gateway, gemini_echo, backend="gemini-2.5-flash-image") as rig:
        assert rig.chat(modalities=["text", "image"]).status_code == 200
        assert at(rig.peer.one_body(), "generationConfig", "responseModalities") == ["TEXT", "IMAGE"]


def test_an_image_answer_comes_back_under_images_as_a_data_uri(gateway: Gateway) -> None:
    """An answer with a text part "here" and an `inlineData` PNG part is returned with `content: "here"`
    and `images: [{"type": "image_url", "image_url": {"url": "data:image/png;base64,<b64>", "detail":
    "auto"}, "index": 0}]`; the image bytes are not mixed into the text."""
    parts: Final[list[JsonValue]] = [{"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": PNG_B64}}]
    with gemini(gateway, lambda _r: gemini_reply(parts=parts), backend="gemini-2.5-flash-image") as rig:
        message: Final = message_of(rig.chat(modalities=["text", "image"]))
        assert message["content"] == "here"
        image: Final = object_value(list_value(message["images"])[0])
        assert object_value(image["image_url"])["url"] == f"data:image/png;base64,{PNG_B64}"
        assert image["type"] == "image_url"


def test_a_gemini_files_uri_image_is_passed_as_file_data_without_fetching(gateway: Gateway) -> None:
    """An `image_url` that is a Files API URI (`https://generativelanguage.googleapis.com/v1beta/files/abc`)
    is not downloaded: it goes out as `{"file_data": {"file_uri": <uri>}}` with no `mime_type` (none was
    given), since those URLs answer 403 to a direct fetch."""
    uri: Final = "https://generativelanguage.googleapis.com/v1beta/files/dec1abc"
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": marker()}, {"type": "image_url", "image_url": {"url": uri}}],
        }
    ]
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(messages=messages).status_code == 200
        parts: Final = list_value(object_value(list_value(rig.peer.one_body()["contents"])[0])["parts"])
        assert {"file_data": {"file_uri": uri}} in parts


def test_a_files_uri_with_a_format_carries_the_mime_type(gateway: Gateway) -> None:
    """The same Files API URI given as a `file` block with `format: "application/pdf"` is sent as
    `{"file_data": {"mime_type": "application/pdf", "file_uri": <uri>}}`."""
    uri: Final = "https://generativelanguage.googleapis.com/v1beta/files/dec1pdf"
    messages: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": marker()},
                {"type": "file", "file": {"file_id": uri, "format": "application/pdf"}},
            ],
        }
    ]
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(messages=messages).status_code == 200
        parts: Final = list_value(object_value(list_value(rig.peer.one_body()["contents"])[0])["parts"])
        assert {"file_data": {"mime_type": "application/pdf", "file_uri": uri}} in parts


def test_a_loopback_image_url_is_not_fetched_for_ai_studio(gateway: Gateway) -> None:
    """AI Studio cannot read remote URLs, so the proxy inlines every non Files API URL by downloading it
    (`_ai_studio_inlines`), through `safe_get`. An image URL on the peer's loopback address is refused by
    the SSRF guard (`user_url_validation` defaults to true): no `GET` reaches the peer and no generateContent
    is sent. UNVERIFIED: the exact caller status (400 expected) and whether the shared proxy allowlists
    127.0.0.1 in `user_url_allowed_hosts`."""
    with gemini(gateway, gemini_echo) as rig:
        url: Final = f"{rig.peer.url}/cat.png"
        messages: Final[list[JsonValue]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": marker()}, {"type": "image_url", "image_url": {"url": url}}],
            }
        ]
        response: Final = rig.chat(messages=messages)
        assert response.status_code >= 400, response.text
        assert rig.peer.generate_calls() == ()
        assert rig.peer.to("/cat.png", method="GET") == ()


def test_reply_without_image_parts_has_no_images_key(gateway: Gateway) -> None:
    """A plain text answer has no `images` key on the message at all (not an empty list), so clients that
    test for the key do not render an empty gallery."""
    with gemini(
        gateway,
        lambda _r: Reply(
            body=b'{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}'
        ),
    ) as rig:
        message: Final = message_of(rig.chat())
        assert message["content"] == "hi"
        assert not message.get("images")


def test_streamed_thought_frames_arrive_as_reasoning_content(gateway: Gateway) -> None:
    """On the stream path a frame whose part has `thought: true` arrives as `delta.reasoning_content`, never
    as `delta.content`, on a Gemini 3 model with `reasoning_effort: low`."""
    import json as _json

    frames: Final = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking", "thought": True}]}, "index": 0}]},
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "answer"}]}, "index": 0, "finishReason": "STOP"}
            ],
            "usageMetadata": usage_metadata(5, 3, thoughtsTokenCount=4, totalTokenCount=12),
        },
    ]
    reply: Final = Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {_json.dumps(frame)}\r\n\r\n".encode() for frame in frames),
    )
    with gemini(gateway, lambda _r: reply, backend=GEMINI_3_FLASH) as rig:
        _response, text = rig.stream(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": marker()}], "reasoning_effort": "low"}
        )
        assert '"reasoning_content": "thinking"' in text or '"reasoning_content":"thinking"' in text
        assert '"content": "thinking"' not in text and '"content":"thinking"' not in text
