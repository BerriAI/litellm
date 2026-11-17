"""`POST /v1/chat/completions` served by a Vertex AI Gemini deployment (`vertex_ai/gemini-*`).

The proxy turns an OpenAI chat request into one Vertex `generateContent` call:
`POST {api_base}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`
with `Authorization: Bearer <access token minted from the deployment's service account>`. The body
has no `model`; OpenAI messages become `contents`, system messages become `system_instruction`,
sampling and output options go under `generationConfig` in their snake_case Vertex names
(`max_output_tokens`, `stop_sequences`, `candidate_count`, `response_mime_type`,
`response_json_schema`, `thinkingConfig`), tools become `tools[].function_declarations`, `tool_choice`
becomes `toolConfig.functionCallingConfig`, and `safety_settings` and `labels` pass through at the top
level (`litellm/llms/vertex_ai/gemini/transformation.py::transform_request_body`). On the way back the
candidate's parts become `content`, `reasoning_content` and `tool_calls`, Gemini finish reasons map to
OpenAI ones, and `usageMetadata` becomes `usage` with cached and thinking token details.

All of this is Python today (`vertex_and_google_ai_studio_gemini.py`). A Rust port of the chat path
has to reproduce every field below byte for byte, since Vertex rejects unknown keys and silently
ignores misspelled generation options. Each test uses a fresh prompt marker because the shared proxy
caches responses, and every token is minted by the owned peer, never by a Google host.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.providers.dec1_vertex_support import (
    GEMINI,
    LOCATION,
    PROJECT,
    SA_PROJECT,
    at,
    choice,
    gemini_echo,
    gemini_json,
    gemini_reply,
    json_of,
    marker,
    message_of,
    models_path,
    path_of,
    query_of,
    usage_metadata,
    vertex,
)
from pydantic import JsonValue

_GENERATE: Final = models_path() + ":generateContent"
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_SAFETY: Final[list[JsonValue]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def _generation_config(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(body.get("generationConfig") or {})


# ---------------------------------------------------------------- URL, method and auth


def test_a_gemini_chat_posts_to_generate_content_with_the_minted_token(gateway: Gateway) -> None:
    """A plain chat request reaches the peer as exactly one `POST /v1/projects/dec1-vertex-project/
    locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent` with no query,
    `Authorization: Bearer token-1` (the first token the diverted OAuth endpoint minted) and no
    `x-goog-api-key`, which only the AI Studio route sends."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        call: Final = rig.peer.one_call()
        assert (call.method, path_of(call), query_of(call)) == ("POST", _GENERATE, "")
        assert call.headers["authorization"].startswith("Bearer token-"), dict(call.headers)
        assert call.headers["authorization"] in {f"Bearer {token}" for token in rig.peer.plan.issued}
        assert "x-goog-api-key" not in call.headers


def test_the_gemini_body_has_contents_and_no_model_or_messages(gateway: Gateway) -> None:
    """The user turn reaches Vertex as `contents: [{"role": "user", "parts": [{"text": <prompt>}]}]`;
    the body never carries `model` (the model is in the URL) or the OpenAI `messages` key."""
    prompt: Final = f"hello vertex {marker()}"
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(prompt).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["contents"] == [{"role": "user", "parts": [{"text": prompt}]}]
        assert "model" not in body and "messages" not in body


def test_the_project_comes_from_the_service_account_when_vertex_project_is_unset(gateway: Gateway) -> None:
    """Without `vertex_project`, `load_auth` resolves the project from the service account's
    `project_id` (`dec1-sa-project`), and that project is the one in the Vertex path."""
    with vertex(gateway, gemini_echo, vertex_project=None) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(project=SA_PROJECT) + ":generateContent"


@pytest.mark.parametrize("location", ("global", "us", "europe-west4"))
def test_the_configured_location_is_the_one_in_the_path(gateway: Gateway, location: str) -> None:
    """`vertex_location` is put verbatim in `locations/<location>` for the global endpoint, a
    multi-region (`us`) and a region, since `gemini-2.5-flash` has no `supported_regions` override in the
    cost map (`VertexBase.get_vertex_region`)."""
    with vertex(gateway, gemini_echo, vertex_location=location) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(location=location) + ":generateContent"


def test_a_missing_location_defaults_to_us_central1_for_gemini_chat(gateway: Gateway) -> None:
    """With no `vertex_location` the Gemini chat path uses `locations/us-central1`
    (`get_vertex_region` falls back to `us-central1`). UNVERIFIED: assumes the shared proxy has no
    VERTEXAI_LOCATION env."""
    with vertex(gateway, gemini_echo, vertex_location=None) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == models_path(location=LOCATION) + ":generateContent"


def test_a_numeric_model_id_posts_to_the_tuned_endpoint_path(gateway: Gateway) -> None:
    """A model id made only of digits is a tuned Gemini endpoint, so the path is
    `/v1/projects/<p>/locations/<l>/endpoints/<id>:generateContent` instead of `publishers/google/models`."""
    with vertex(gateway, gemini_echo, backend="4965075652664360960") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.peer.one_call()) == (
            f"/v1/projects/{PROJECT}/locations/{LOCATION}/endpoints/4965075652664360960:generateContent"
        )


# ---------------------------------------------------------------- request mapping


def test_system_messages_become_system_instruction_parts(gateway: Gateway) -> None:
    """Two system messages are lifted out of `contents` into
    `system_instruction: {"parts": [{"text": first}, {"text": second}]}`, in order, and only the user
    turn stays in `contents`."""
    with vertex(gateway, gemini_echo) as rig:
        prompt: Final = f"question {marker()}"
        response: Final = rig.post(
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "system", "content": "answer in French"},
                    {"role": "user", "content": prompt},
                ]
            },
        )
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert body["system_instruction"] == {"parts": [{"text": "be brief"}, {"text": "answer in French"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": prompt}]}]


def test_a_system_message_with_text_blocks_is_joined_into_one_part(gateway: Gateway) -> None:
    """A system message whose content is a list of text blocks becomes one part holding the
    concatenated text (`transform_system_message` joins the blocks without a separator)."""
    with vertex(gateway, gemini_echo) as rig:
        system: Final[list[JsonValue]] = [{"type": "text", "text": "rule one. "}, {"type": "text", "text": "rule two."}]
        response: Final = rig.post(
            "/v1/chat/completions",
            {"messages": [{"role": "system", "content": system}, {"role": "user", "content": f"q {marker()}"}]},
        )
        assert response.status_code == 200, response.text
        assert rig.peer.one_body()["system_instruction"] == {"parts": [{"text": "rule one. rule two."}]}


def test_an_assistant_turn_is_sent_with_the_model_role(gateway: Gateway) -> None:
    """A multi turn history keeps its order and the assistant turn is sent with Gemini's `model` role."""
    with vertex(gateway, gemini_echo) as rig:
        last: Final = f"and now {marker()}"
        response: Final = rig.post(
            "/v1/chat/completions",
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": last},
                ]
            },
        )
        assert response.status_code == 200, response.text
        assert rig.peer.one_body()["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
            {"role": "user", "parts": [{"text": last}]},
        ]


def test_sampling_and_length_options_land_in_generation_config(gateway: Gateway) -> None:
    """`max_tokens`, `temperature`, `top_p`, `stop` (a string), `n` and `seed` reach Vertex as
    `generationConfig.max_output_tokens`, `temperature`, `top_p`, `stop_sequences: [<stop>]`,
    `candidate_count` and `seed` (`VertexGeminiConfig.map_openai_params`)."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(max_tokens=32, temperature=0.25, top_p=0.5, stop="END", n=2, seed=7)
        assert response.status_code == 200, response.text
        config: Final = _generation_config(rig.peer.one_body())
        assert {
            name: config.get(name)
            for name in ("max_output_tokens", "temperature", "top_p", "stop_sequences", "candidate_count", "seed")
        } == {
            "max_output_tokens": 32,
            "temperature": 0.25,
            "top_p": 0.5,
            "stop_sequences": ["END"],
            "candidate_count": 2,
            "seed": 7,
        }


def test_max_completion_tokens_is_sent_as_max_output_tokens(gateway: Gateway) -> None:
    """`max_completion_tokens` is an alias of `max_tokens` and reaches Vertex as
    `generationConfig.max_output_tokens`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(max_completion_tokens=48).status_code == 200
        assert _generation_config(rig.peer.one_body())["max_output_tokens"] == 48


def test_json_object_response_format_sets_only_the_json_mime_type(gateway: Gateway) -> None:
    """`response_format: {"type": "json_object"}` becomes `generationConfig.response_mime_type:
    "application/json"` with no `response_schema` or `response_json_schema`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(response_format={"type": "json_object"}).status_code == 200
        config: Final = _generation_config(rig.peer.one_body())
        assert config["response_mime_type"] == "application/json"
        assert "response_schema" not in config and "response_json_schema" not in config


def test_a_json_schema_response_format_is_sent_as_response_json_schema_on_gemini_2(gateway: Gateway) -> None:
    """On a Gemini 2.x model (`supports_response_json_schema`), `response_format.json_schema.schema`
    is sent as `generationConfig.response_json_schema` in standard lowercase JSON Schema with
    `response_mime_type: "application/json"`, and `strict` never reaches Vertex."""
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(
            response_format={"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        )
        assert response.status_code == 200, response.text
        config: Final = _generation_config(rig.peer.one_body())
        assert config["response_mime_type"] == "application/json"
        sent: Final = object_value(config["response_json_schema"])
        assert sent["type"] == "object"
        assert at(sent, "properties", "answer", "type") == "string"
        assert "strict" not in json.dumps(config)
        assert "response_schema" not in config


@pytest.mark.parametrize(
    ("effort", "thinking_config"),
    (
        ("low", {"thinkingBudget": 1024, "includeThoughts": True}),
        ("medium", {"thinkingBudget": 2048, "includeThoughts": True}),
        ("high", {"thinkingBudget": 4096, "includeThoughts": True}),
        ("none", {"thinkingBudget": 0, "includeThoughts": False}),
    ),
)
def test_reasoning_effort_becomes_a_thinking_budget_on_gemini_2(
    gateway: Gateway, effort: str, thinking_config: dict[str, JsonValue]
) -> None:
    """On Gemini 2.5, `reasoning_effort` maps to a fixed `thinkingConfig.thinkingBudget`
    (low 1024, medium 2048, high 4096 from `litellm/constants.py`, none 0) and `includeThoughts` is true
    unless thinking is off, because `litellm.gemini_include_thoughts_default` is true."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(reasoning_effort=effort).status_code == 200
        assert _generation_config(rig.peer.one_body())["thinkingConfig"] == thinking_config


def test_an_anthropic_thinking_budget_is_sent_as_the_gemini_thinking_budget(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 3000}` on Gemini 2.5 becomes
    `thinkingConfig: {"includeThoughts": true, "thinkingBudget": 3000}` (`_map_thinking_param`)."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(thinking={"type": "enabled", "budget_tokens": 3000})
        assert response.status_code == 200, response.text
        assert _generation_config(rig.peer.one_body())["thinkingConfig"] == {
            "includeThoughts": True,
            "thinkingBudget": 3000,
        }


def test_a_zero_thinking_budget_turns_thoughts_off(gateway: Gateway) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": 0}` is "no thinking" on Gemini 2.5, so the
    config is exactly `{"thinkingBudget": 0}` with no `includeThoughts`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(thinking={"type": "enabled", "budget_tokens": 0}).status_code == 200
        assert _generation_config(rig.peer.one_body())["thinkingConfig"] == {"thinkingBudget": 0}


def test_an_unknown_reasoning_effort_is_refused_before_vertex(gateway: Gateway) -> None:
    """`reasoning_effort: "extreme"` has no Gemini budget, so the caller gets a 400 and Vertex is never
    called (`_unsupported_reasoning_effort`)."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(reasoning_effort="extreme")
        assert response.status_code == 400, response.text
        assert rig.peer.model_calls() == ()


def test_safety_settings_pass_through_unchanged_as_safety_settings(gateway: Gateway) -> None:
    """A caller's `safety_settings` list reaches Vertex unchanged at the top level as `safetySettings`,
    not inside `generationConfig`."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(safety_settings=_SAFETY).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["safetySettings"] == _SAFETY
        assert "safety_settings" not in json.dumps(body.get("generationConfig") or {})


def test_deployment_safety_settings_apply_to_every_request(gateway: Gateway) -> None:
    """`safety_settings` set in the deployment's `litellm_params` is sent on a request that did not
    name any, so an admin can pin safety thresholds for all callers. UNVERIFIED: assumes deployment
    litellm_params flow into optional params like other provider-specific params."""
    with vertex(gateway, gemini_echo, safety_settings=_SAFETY) as rig:
        assert rig.chat().status_code == 200
        assert rig.peer.one_body()["safetySettings"] == _SAFETY


def test_caller_labels_reach_vertex_as_top_level_labels(gateway: Gateway) -> None:
    """With `vertex_request_label_fields` off (the shared proxy default), a caller's
    `labels: {"team": "search", "env": "dev"}` reaches the Vertex body unchanged as `labels`, the field
    Google bills by."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(labels={"team": "search", "env": "dev"}).status_code == 200
        assert rig.peer.one_body()["labels"] == {"team": "search", "env": "dev"}


def test_no_labels_means_no_labels_key(gateway: Gateway) -> None:
    """A request without labels or requester metadata sends no `labels` key at all; an empty `{}`
    would be a different body for a rewrite to get wrong."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat().status_code == 200
        assert "labels" not in rig.peer.one_body()


def test_tools_become_function_declarations(gateway: Gateway) -> None:
    """An OpenAI function tool becomes `tools: [{"function_declarations": [{"name": "get_weather",
    "description": ..., "parameters": ...}]}]`, keeping the name, description and the `city` property."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[_WEATHER_TOOL]).status_code == 200
        declaration: Final = object_value(at(rig.peer.one_body(), "tools", 0, "function_declarations", 0))
        assert declaration["name"] == "get_weather"
        assert declaration["description"] == "Look up the weather for a city"
        schema: Final = declaration.get("parameters") or declaration.get("parametersJsonSchema")
        assert "city" in json.dumps(schema), declaration


@pytest.mark.parametrize(
    ("tool_choice", "config"),
    (
        ("none", {"mode": "NONE"}),
        ("auto", {"mode": "AUTO"}),
        ("required", {"mode": "ANY"}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"mode": "ANY", "allowed_function_names": ["get_weather"]},
        ),
    ),
    ids=("none", "auto", "required", "named"),
)
def test_tool_choice_becomes_a_function_calling_mode(
    gateway: Gateway, tool_choice: JsonValue, config: dict[str, JsonValue]
) -> None:
    """`tool_choice` none/auto/required map to `toolConfig.functionCallingConfig.mode` NONE/AUTO/ANY,
    and a named function becomes mode ANY with `allowed_function_names: [<name>]`
    (`VertexGeminiConfig.map_tool_choice_values`)."""
    with vertex(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[_WEATHER_TOOL], tool_choice=tool_choice).status_code == 200
        assert rig.peer.one_body()["toolConfig"] == {"functionCallingConfig": config}


def test_an_unsupported_tool_choice_string_is_refused_without_a_vertex_call(gateway: Gateway) -> None:
    """`tool_choice: "any"` is not an OpenAI value Vertex can map, so the caller gets HTTP 400 whose
    message says `VertexAI doesn't support tool_choice=any` and the peer sees no model call."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(tools=[_WEATHER_TOOL], tool_choice="any")
        assert response.status_code == 400, response.text
        assert "VertexAI doesn't support tool_choice=any" in response.text
        assert rig.peer.model_calls() == ()


def test_a_tool_result_round_trip_is_sent_as_function_call_and_function_response(gateway: Gateway) -> None:
    """An assistant tool call followed by its `tool` message reaches Vertex as a `model` turn with a
    `functionCall {name, args}` part and a following turn with a `functionResponse` part for the same
    function name."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.post(
            "/v1/chat/completions",
            {
                "tools": [_WEATHER_TOOL],
                "messages": [
                    {"role": "user", "content": f"weather {marker()}"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        contents: Final = list_value(rig.peer.one_body()["contents"])
        assert object_value(contents[1])["role"] == "model"
        call_part: Final = object_value(at(contents[1], "parts", 0))
        assert object_value(call_part["functionCall"])["name"] == "get_weather"
        assert object_value(call_part["functionCall"])["args"] == {"city": "Paris"}
        response_part: Final = object_value(at(contents[2], "parts", 0))
        assert object_value(response_part["functionResponse"])["name"] == "get_weather"


# ---------------------------------------------------------------- response mapping


def test_the_candidate_text_comes_back_as_the_assistant_message(gateway: Gateway) -> None:
    """A `STOP` candidate with one text part is answered as `choices[0].message = {"role": "assistant",
    "content": <text>}` with `finish_reason: "stop"` and `object: "chat.completion"`."""
    with vertex(gateway, lambda _r: gemini_reply(text="bonjour")) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert json_of(response)["object"] == "chat.completion"
        assert message_of(response)["content"] == "bonjour"
        assert message_of(response)["role"] == "assistant"
        assert choice(response)["finish_reason"] == "stop"


@pytest.mark.parametrize(
    ("gemini_reason", "openai_reason"),
    (
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("RECITATION", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "stop"),
    ),
)
def test_gemini_finish_reasons_map_to_openai_ones(gateway: Gateway, gemini_reason: str, openai_reason: str) -> None:
    """`finishReason` maps through `_FINISH_REASON_MAP`: MAX_TOKENS is `length`, SAFETY and RECITATION
    are `content_filter`, and MALFORMED_FUNCTION_CALL is `stop`."""
    with vertex(gateway, lambda _r: gemini_reply(finish=gemini_reason)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert choice(response)["finish_reason"] == openai_reason


def test_thought_parts_come_back_as_reasoning_content(gateway: Gateway) -> None:
    """A part marked `thought: true` is returned in `message.reasoning_content` and kept out of
    `message.content`, which holds only the answer part."""
    parts: Final[list[JsonValue]] = [{"text": "let me think", "thought": True}, {"text": "42"}]
    with vertex(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert message_of(response)["content"] == "42"
        assert message_of(response)["reasoning_content"] == "let me think"


def test_a_function_call_part_comes_back_as_a_tool_call(gateway: Gateway) -> None:
    """A `functionCall {name, args}` part is returned as one `tool_calls` entry of type `function`
    whose `arguments` is the JSON text of `args`, and the finish reason becomes `tool_calls`."""
    parts: Final[list[JsonValue]] = [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]
    with vertex(gateway, lambda _r: gemini_reply(parts=parts)) as rig:
        response: Final = rig.chat(tools=[_WEATHER_TOOL])
        assert response.status_code == 200, response.text
        call: Final = object_value(list_value(message_of(response)["tool_calls"])[0])
        assert call["type"] == "function"
        assert object_value(call["function"])["name"] == "get_weather"
        assert json.loads(str(object_value(call["function"])["arguments"])) == {"city": "Paris"}
        assert choice(response)["finish_reason"] == "tool_calls"


def test_a_blocked_prompt_is_answered_with_content_filter_and_no_content(gateway: Gateway) -> None:
    """When Vertex blocks the prompt (`promptFeedback.blockReason`, no candidates), the caller still
    gets HTTP 200 with `finish_reason: "content_filter"` and `content: null`
    (`_handle_blocked_response`), not an error."""
    blocked: Final[dict[str, JsonValue]] = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 5, "totalTokenCount": 5},
    }
    with vertex(gateway, lambda _r: Reply(body=json.dumps(blocked).encode())) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert choice(response)["finish_reason"] == "content_filter"
        assert message_of(response)["content"] is None


def test_usage_maps_prompt_candidates_and_total_counts(gateway: Gateway) -> None:
    """`usageMetadata {promptTokenCount: 11, candidatesTokenCount: 7, totalTokenCount: 18}` is returned
    as `usage {prompt_tokens: 11, completion_tokens: 7, total_tokens: 18}`."""
    with vertex(gateway, lambda _r: gemini_reply(usage=usage_metadata(11, 7))) as rig:
        usage: Final = object_value(json_of(rig.chat())["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 7, 18)


def test_cached_content_tokens_are_reported_as_cached_tokens(gateway: Gateway) -> None:
    """`cachedContentTokenCount: 40` out of 100 prompt tokens is reported as
    `usage.prompt_tokens_details.cached_tokens = 40` while `prompt_tokens` stays the full 100, and the
    TEXT modality detail drops to the 60 uncached tokens (`_calculate_usage`)."""
    usage_in: Final = usage_metadata(
        100, 20, cachedContentTokenCount=40, promptTokensDetails=[{"modality": "TEXT", "tokenCount": 100}]
    )
    with vertex(gateway, lambda _r: gemini_reply(usage=usage_in)) as rig:
        usage: Final = object_value(json_of(rig.chat())["usage"])
        assert usage["prompt_tokens"] == 100
        details: Final = object_value(usage["prompt_tokens_details"])
        assert details["cached_tokens"] == 40
        assert details["text_tokens"] == 60


def test_thoughts_tokens_outside_the_candidate_count_are_added_to_completion_tokens(gateway: Gateway) -> None:
    """When `promptTokenCount + candidatesTokenCount != totalTokenCount` (100 + 20 vs 150) the 30
    `thoughtsTokenCount` are not inside the candidate count, so `completion_tokens = 20 + 30 = 50`,
    `completion_tokens_details.reasoning_tokens = 30` and `total_tokens = 150`."""
    usage_in: Final = usage_metadata(100, 20, thoughtsTokenCount=30, totalTokenCount=150)
    with vertex(gateway, lambda _r: gemini_reply(usage=usage_in)) as rig:
        usage: Final = object_value(json_of(rig.chat())["usage"])
        assert (usage["completion_tokens"], usage["total_tokens"]) == (50, 150)
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == 30


def test_thoughts_tokens_already_inside_the_candidate_count_are_not_added_twice(gateway: Gateway) -> None:
    """When `promptTokenCount + candidatesTokenCount == totalTokenCount` (100 + 50 = 150) the candidate
    count already includes the 30 thinking tokens, so `completion_tokens` stays 50 and
    `reasoning_tokens` is 30 (`is_candidate_token_count_inclusive`)."""
    usage_in: Final = usage_metadata(100, 50, thoughtsTokenCount=30, totalTokenCount=150)
    with vertex(gateway, lambda _r: gemini_reply(usage=usage_in)) as rig:
        usage: Final = object_value(json_of(rig.chat())["usage"])
        assert usage["completion_tokens"] == 50
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == 30


def test_two_candidates_come_back_as_two_choices(gateway: Gateway) -> None:
    """With `n: 2` Vertex returns two candidates (index 0 and 1) and the caller gets two choices in the
    same order with each candidate's text."""

    def respond(_request: Request) -> Reply:
        body: Final = gemini_json(text="first")
        second: Final = {
            "content": {"role": "model", "parts": [{"text": "second"}]},
            "finishReason": "STOP",
            "index": 1,
        }
        body["candidates"] = [*list_value(body["candidates"]), second]
        return Reply(body=json.dumps(body).encode())

    with vertex(gateway, respond) as rig:
        choices: Final = list_value(json_of(rig.chat(n=2))["choices"])
        assert [object_value(object_value(item)["message"])["content"] for item in choices] == ["first", "second"]


def test_the_gemini_chat_spend_row_names_vertex_ai(gateway: Gateway) -> None:
    """One Gemini chat writes one SpendLogs row for the model group with `custom_llm_provider =
    "vertex_ai"`, 11 prompt and 7 completion tokens, priced at the deployment's custom prices
    (`11 * 0.001 + 7 * 0.002 = 0.025`)."""
    from integration._support.client import eventually
    from integration._support.database import read_rows

    with vertex(gateway, gemini_echo, input_cost_per_token=0.001, output_cost_per_token=0.002) as rig:
        assert rig.chat().status_code == 200
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT custom_llm_provider, prompt_tokens, completion_tokens, spend FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
                (rig.model,),
            ),
            lambda found: len(found) >= 1,
            seconds=90,
        )
        assert len(rows) == 1, rows
        row: Final = rows[0]
        assert (row["custom_llm_provider"], row["prompt_tokens"], row["completion_tokens"]) == ("vertex_ai", 11, 7)
        assert float(str(row["spend"])) == pytest.approx(0.025)


def test_the_upstream_body_carries_no_openai_only_keys(gateway: Gateway) -> None:
    """Vertex rejects unknown top-level fields, so a request with `user`, `metadata` and `stream: false`
    reaches the peer with only Gemini keys: nothing named `user`, `metadata`, `stream` or `messages`."""
    with vertex(gateway, gemini_echo) as rig:
        response: Final = rig.chat(user="dec1-user", metadata={"note": "x"}, stream=False)
        assert response.status_code == 200, response.text
        body: Final = rig.peer.one_body()
        assert not {"user", "metadata", "stream", "messages", "model"} & set(body), sorted(body)
        assert GEMINI not in json.dumps(body)
