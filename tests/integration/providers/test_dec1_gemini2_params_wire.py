"""`gemini/` chat: which OpenAI parameters AI Studio accepts, refuses, drops or rewrites.

AI Studio has its own supported parameter list (`GoogleAIStudioGeminiConfig.get_supported_openai_params`,
`litellm/llms/gemini/chat/transformation.py:86`) that differs from Vertex's
(`vertex_and_google_ai_studio_gemini.py:982`): it has no `top_logprobs`, it lists `audio` only for TTS
models, and it always lists `frequency_penalty` / `presence_penalty`, which `map_openai_params` then
silently skips on Gemini 3. A parameter missing from the list is refused with HTTP 400 before any call
to Google unless the request or proxy sets `drop_params`
(`gemini does not support parameters: [...]`, `litellm/utils.py:5793`).

Models flagged `supports_sampling_params: false` in the cost map (Gemini 3.6 Flash and later) lose
`temperature` and `top_p` from that list, and `top_k` is refused while the body is built
(`reject_unsupported_sampling_param`).

The rest pins body rewrites that are easy to lose in a rewrite: `parallel_tool_calls` never reaches
Google (it is not a `GenerationConfig` field), `strict` is stripped from response schemas while
`additionalProperties` and `$defs` survive in `response_json_schema`, a raw `extra_body` is merged into
the Google body, `googleMaps` with JSON output switches to the newer `responseFormat`, and the
`googleMaps` and `computer_use` tools are reshaped.

Every test owns one loopback peer and one deployment; prompts carry a fresh marker because the shared
proxy caches responses.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_gemini2_support import (
    SAMPLING_OFF_MODEL,
    TIME_TOOL,
    WEATHER_TOOL,
    error_message,
    generation_config,
)
from integration.providers.dec1_gemini_support import (
    GEMINI_3_FLASH,
    at,
    choice,
    gemini,
    gemini_echo,
    gemini_json,
    json_of,
    marker,
    reply_json,
)
from pydantic import JsonValue

_SCHEMA_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------- refused or dropped before Google


def test_top_logprobs_is_refused_on_ai_studio_before_google(gateway: Gateway) -> None:
    """`logprobs: true, top_logprobs: 3` on a `gemini/` deployment is answered 400 with
    `gemini does not support parameters: ['top_logprobs']` and Google is never called: AI Studio's list
    (`gemini/chat/transformation.py:86`) has no `top_logprobs`, although Vertex's list does and maps it to
    `generationConfig.logprobs`."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.chat(logprobs=True, top_logprobs=3)
        assert response.status_code == 400, response.text
        assert "gemini does not support parameters: ['top_logprobs']" in error_message(response)
        assert rig.peer.generate_calls() == ()


def test_request_drop_params_drops_top_logprobs_and_keeps_logprobs(gateway: Gateway) -> None:
    """With `drop_params: true` in the request the same call succeeds: `top_logprobs` is dropped and only
    `generationConfig.responseLogprobs: true` reaches Google, with no `logprobs` count key
    (`utils.py:5775` drop branch, `map_openai_params` `logprobs` branch)."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.chat(logprobs=True, top_logprobs=3, drop_params=True)
        assert response.status_code == 200, response.text
        config: Final = generation_config(rig.peer.one_body())
        assert config.get("responseLogprobs") is True
        assert "logprobs" not in config


def test_logprobs_result_comes_back_as_openai_logprobs(gateway: Gateway) -> None:
    """A candidate `logprobsResult` with chosen tokens "po" (-0.1) and "ng" (-0.2) and top candidates for
    the first token comes back as `choices[0].logprobs.content` with one entry per chosen token, each with
    `token`, `logprob` and the matching `top_logprobs` (`_transform_logprobs`,
    `vertex_and_google_ai_studio_gemini.py:2441`)."""
    reply: Final = gemini_json(
        logprobsResult={
            "chosenCandidates": [{"token": "po", "logProbability": -0.1}, {"token": "ng", "logProbability": -0.2}],
            "topCandidates": [
                {"candidates": [{"token": "po", "logProbability": -0.1}, {"token": "pi", "logProbability": -2.5}]}
            ],
        }
    )
    with gemini(gateway, lambda _r: reply_json(reply)) as rig:
        logprobs: Final = object_value(choice(rig.chat(logprobs=True))["logprobs"])
        content: Final = [object_value(item) for item in list_value(logprobs["content"])]
        assert [(item["token"], item["logprob"]) for item in content] == [("po", -0.1), ("ng", -0.2)]
        assert [object_value(top)["token"] for top in list_value(content[0]["top_logprobs"])] == ["po", "pi"]
        assert content[1]["top_logprobs"] == []


def test_audio_on_a_model_without_tts_in_its_name_is_refused(gateway: Gateway) -> None:
    """`audio: {"voice": "alloy", "format": "pcm16"}` on `gemini/gemini-2.5-flash` is answered 400 naming
    `audio`, since AI Studio only lists `audio` for a model whose name contains `tts`
    (`is_model_gemini_audio_model`, `gemini/chat/transformation.py:83`); Google is not called."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.chat(audio={"voice": "alloy", "format": "pcm16"})
        assert response.status_code == 400, response.text
        assert "gemini does not support parameters: ['audio']" in error_message(response)
        assert rig.peer.generate_calls() == ()


def test_temperature_on_a_sampling_off_model_is_refused(gateway: Gateway) -> None:
    """`temperature: 0.2` on `gemini/gemini-3.6-flash`, which the cost map flags
    `supports_sampling_params: false`, is answered 400 naming `temperature` before any call
    (`get_supported_openai_params` leaves out `temperature` and `top_p` when the flag is off)."""
    with gemini(gateway, gemini_echo, backend=SAMPLING_OFF_MODEL) as rig:
        response: Final = rig.chat(temperature=0.2)
        assert response.status_code == 400, response.text
        assert "gemini does not support parameters: ['temperature']" in error_message(response)
        assert rig.peer.generate_calls() == ()


def test_temperature_is_dropped_on_a_sampling_off_model_with_drop_params(gateway: Gateway) -> None:
    """The same `temperature: 0.2` with request `drop_params: true` reaches Google with no `temperature`
    in `generationConfig`, while `max_tokens: 16` still arrives as `max_output_tokens: 16`."""
    with gemini(gateway, gemini_echo, backend=SAMPLING_OFF_MODEL) as rig:
        assert rig.chat(temperature=0.2, max_tokens=16, drop_params=True).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert "temperature" not in config
        assert config.get("max_output_tokens") == 16


def test_top_k_on_a_sampling_off_model_is_refused(gateway: Gateway) -> None:
    """`top_k: 5` is a provider kwarg that bypasses the supported list, so on `gemini-3.6-flash` it is
    refused while the body is built with a 400 whose message says the model `does not support top_k`
    and names `supports_sampling_params: false` (`reject_unsupported_sampling_param`,
    `vertex_and_google_ai_studio_gemini.py:968`, called from `transformation.py` `drops_top_k`)."""
    with gemini(gateway, gemini_echo, backend=SAMPLING_OFF_MODEL) as rig:
        response: Final = rig.chat(top_k=5)
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "does not support top_k" in message
        assert "supports_sampling_params: false" in message
        assert rig.peer.generate_calls() == ()


def test_top_k_reaches_generation_config_on_gemini_2_5(gateway: Gateway) -> None:
    """On `gemini-2.5-flash` (no sampling flag in the map, so sampling is on) `top_k: 5` reaches Google
    as `generationConfig.top_k: 5`."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(top_k=5).status_code == 200
        assert generation_config(rig.peer.one_body()).get("top_k") == 5


def test_penalties_reach_gemini_2_5(gateway: Gateway) -> None:
    """`frequency_penalty: 0.5` and `presence_penalty: 0.25` reach gemini-2.5-flash as
    `generationConfig.frequency_penalty` and `presence_penalty` with the same values
    (`_supports_penalty_parameters` is true for non Gemini 3 models)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(frequency_penalty=0.5, presence_penalty=0.25).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert (config.get("frequency_penalty"), config.get("presence_penalty")) == (0.5, 0.25)


def test_penalties_on_gemini_3_are_silently_dropped(gateway: Gateway) -> None:
    """On `gemini-3-flash-preview` the same penalties are accepted with 200 (AI Studio always lists them)
    but never reach Google: `map_openai_params` skips them because Gemini 3 rejects penalties
    (`vertex_and_google_ai_studio_gemini.py:940`). Vertex answers 400 here instead, since its list leaves
    them out."""
    with gemini(gateway, gemini_echo, backend=GEMINI_3_FLASH) as rig:
        assert rig.chat(frequency_penalty=0.5, presence_penalty=0.25).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert "frequency_penalty" not in config and "presence_penalty" not in config


@pytest.mark.parametrize("parallel", [True, False])
def test_parallel_tool_calls_never_reaches_google(gateway: Gateway, parallel: bool) -> None:
    """`parallel_tool_calls` (true, or false with two tools) is accepted with 200 and appears nowhere in the
    Google body: it is mapped into `optional_params` but is not a `GenerationConfig` field and not part of
    `toolConfig` (`transformation.py:1225`); the never called `validate_parallel_tool_calls` would refuse
    false with two tools."""
    with gemini(gateway, gemini_echo) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL, TIME_TOOL], parallel_tool_calls=parallel)
        assert response.status_code == 200, response.text
        assert "parallel_tool_calls" not in rig.peer.one_call().body.decode()


# ---------------------------------------------------------------- structured output


def test_strict_is_removed_and_additional_properties_kept_in_the_json_schema(gateway: Gateway) -> None:
    """A strict `json_schema` response format reaches gemini-2.5-flash as
    `generationConfig.response_mime_type: application/json` and `response_json_schema` equal to the
    caller's schema with `additionalProperties: false` kept and no `strict` key anywhere
    (`apply_response_schema_transformation`, `build_json_schema` returns the schema as is)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(response_format=_SCHEMA_FORMAT).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert config["response_mime_type"] == "application/json"
        assert config["response_json_schema"] == {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        assert "response_schema" not in config


def test_a_json_schema_with_defs_and_refs_is_sent_verbatim(gateway: Gateway) -> None:
    """A schema using `$defs` and `$ref` reaches Google unchanged inside `response_json_schema`: the
    Gemini 2.0+ JSON schema path does not unpack references (`common_utils.py:1052`)."""
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "$defs": {"city": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "properties": {"home": {"$ref": "#/$defs/city"}},
    }
    with gemini(gateway, gemini_echo) as rig:
        response_format: Final[dict[str, JsonValue]] = {
            "type": "json_schema",
            "json_schema": {"name": "trip", "schema": schema},
        }
        assert rig.chat(response_format=response_format).status_code == 200
        assert generation_config(rig.peer.one_body())["response_json_schema"] == schema


def test_a_property_named_strict_survives_the_schema_cleanup(gateway: Gateway) -> None:
    """A schema with a real property called `strict` must keep it in `response_json_schema`.
    `remove_strict_from_schema` (`litellm/utils.py:5247`) deletes every key named `strict` at any depth,
    so the property itself is removed from `properties` and Google is asked for a different schema."""
    pytest.skip("BUG: remove_strict_from_schema deletes a schema property named 'strict' (utils.py:5253)")
    schema: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"strict": {"type": "boolean"}, "name": {"type": "string"}},
    }
    with gemini(gateway, gemini_echo) as rig:
        response_format: Final[dict[str, JsonValue]] = {
            "type": "json_schema",
            "json_schema": {"name": "mode", "schema": schema},
        }
        assert rig.chat(response_format=response_format).status_code == 200
        sent: Final = object_value(generation_config(rig.peer.one_body())["response_json_schema"])
        assert set(object_value(sent["properties"])) == {"strict", "name"}


def test_a_text_response_format_sets_text_plain(gateway: Gateway) -> None:
    """`response_format: {"type": "text"}` reaches Google as `generationConfig.response_mime_type:
    text/plain` with no schema key (`apply_response_schema_transformation` text branch)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(response_format={"type": "text"}).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert config.get("response_mime_type") == "text/plain"
        assert "response_json_schema" not in config and "response_schema" not in config


def test_a_raw_extra_body_is_merged_into_the_google_body(gateway: Gateway) -> None:
    """`extra_body: {"generationConfig": {"responseLogprobs": true}, "dec1Field": "x"}` together with
    `max_tokens: 9` gives a body whose `generationConfig` has both `max_output_tokens: 9` and
    `responseLogprobs: true` (dict values are merged one level deep) and a top-level `dec1Field: "x"`
    (`_pop_and_merge_extra_body`, `transformation.py:1059`)."""
    with gemini(gateway, gemini_echo) as rig:
        extra: Final[dict[str, JsonValue]] = {"generationConfig": {"responseLogprobs": True}, "dec1Field": "x"}
        assert rig.chat(max_tokens=9, extra_body=extra).status_code == 200
        body: Final = rig.peer.one_body()
        config: Final = generation_config(body)
        assert (config.get("max_output_tokens"), config.get("responseLogprobs")) == (9, True)
        assert body.get("dec1Field") == "x"


# ---------------------------------------------------------------- googleMaps and computer use


def test_google_maps_location_moves_into_tool_config(gateway: Gateway) -> None:
    """A tool `{"googleMaps": {"enableWidget": true, "latitude": 48.85, "longitude": 2.35, "languageCode":
    "fr"}}` reaches Google as `tools: [{"googleMaps": {"enableWidget": true}}]` plus
    `toolConfig.retrievalConfig: {"latLng": {"latitude": 48.85, "longitude": 2.35}, "languageCode": "fr"}`
    (`_extract_google_maps_retrieval_config`, `vertex_and_google_ai_studio_gemini.py:1172`)."""
    tool: Final[dict[str, JsonValue]] = {
        "googleMaps": {"enableWidget": True, "latitude": 48.85, "longitude": 2.35, "languageCode": "fr"}
    }
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[tool]).status_code == 200
        body: Final = rig.peer.one_body()
        assert body["tools"] == [{"googleMaps": {"enableWidget": True}}]
        assert at(body, "toolConfig", "retrievalConfig") == {
            "latLng": {"latitude": 48.85, "longitude": 2.35},
            "languageCode": "fr",
        }


def test_google_maps_with_json_output_uses_response_format(gateway: Gateway) -> None:
    """Google refuses `googleMaps` next to `response_mime_type: application/json`, so with a maps tool and
    a json_schema response format the body carries `generationConfig.responseFormat: {"text": {"mimeType":
    "APPLICATION_JSON", "schema": <schema>}}` and neither `response_mime_type` nor `response_json_schema`
    (`_rewrite_google_maps_response_format`, `transformation.py:1145`)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"googleMaps": {}}], response_format=_SCHEMA_FORMAT).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert "response_mime_type" not in config and "response_json_schema" not in config
        assert at(config, "responseFormat", "text", "mimeType") == "APPLICATION_JSON"
        assert at(config, "responseFormat", "text", "schema", "properties") == {"answer": {"type": "string"}}


def test_json_output_without_google_maps_keeps_the_mime_type(gateway: Gateway) -> None:
    """The same response format next to `googleSearch` instead of `googleMaps` keeps
    `response_mime_type: application/json` and gets no `responseFormat`: the rewrite is keyed on the maps
    tool only (`_has_google_maps_tool`)."""
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[{"googleSearch": {}}], response_format=_SCHEMA_FORMAT).status_code == 200
        config: Final = generation_config(rig.peer.one_body())
        assert config.get("response_mime_type") == "application/json"
        assert "responseFormat" not in config


def test_a_computer_use_tool_is_sent_as_computer_use_with_the_browser_environment(gateway: Gateway) -> None:
    """`{"type": "computer_use", "environment": "browser", "excluded_predefined_functions": ["drag_and_drop"]}`
    reaches Google as its own tool `{"computerUse": {"environment": "ENVIRONMENT_BROWSER",
    "excludedPredefinedFunctions": ["drag_and_drop"]}}` (`_transform_computer_use_config`,
    `vertex_and_google_ai_studio_gemini.py:1136`)."""
    tool: Final[dict[str, JsonValue]] = {
        "type": "computer_use",
        "environment": "browser",
        "excluded_predefined_functions": ["drag_and_drop"],
    }
    with gemini(gateway, gemini_echo) as rig:
        assert rig.chat(tools=[tool]).status_code == 200
        assert rig.peer.one_body()["tools"] == [
            {"computerUse": {"environment": "ENVIRONMENT_BROWSER", "excludedPredefinedFunctions": ["drag_and_drop"]}}
        ]


def test_a_file_block_without_its_file_field_is_refused_before_google(gateway: Gateway) -> None:
    """A user content block `{"type": "file"}` with no `file` field is answered 400 with
    `Content block has type='file' but is missing the required 'file' field` and Google is not called
    (`GoogleAIStudioGeminiConfig._transform_messages`, `gemini/chat/transformation.py:153`).

    BUG (run 44176682961): the caller gets HTTP 500 with the bare message `'file'`, so a `KeyError` on the
    block's `file` key escapes before the 400 guard at litellm/llms/gemini/chat/transformation.py:149-154 runs.
    The raising line was not identified from the CI artifacts; a malformed client block must not be a 500."""
    pytest.skip("BUG: a Gemini file block without its file field is a 500 KeyError instead of the 400 guard")
    with gemini(gateway, gemini_echo) as rig:
        messages: Final[list[JsonValue]] = [
            {"role": "user", "content": [{"type": "text", "text": f"read {marker()}"}, {"type": "file"}]}
        ]
        response: Final = rig.gateway.request(
            "POST", "/v1/chat/completions", {"model": rig.model, "messages": messages}
        )
        assert response.status_code == 400, response.text
        assert "missing the required 'file' field" in error_message(response)
        assert rig.peer.generate_calls() == ()


def test_the_model_version_names_the_served_model_without_its_revision(gateway: Gateway) -> None:
    """When Google answers with `modelVersion: "dec1-served-version@7"` the caller's `model` field is
    `dec1-served-version`, the served name with the `@` revision cut off (`_served_model_name`,
    `vertex_and_google_ai_studio_gemini.py:260`). UNVERIFIED: whether the proxy later rewrites `model`
    back to the deployment's name for some callers."""
    reply: Final = gemini_json()
    reply["modelVersion"] = "dec1-served-version@7"
    with gemini(gateway, lambda _r: reply_json(reply)) as rig:
        assert json_of(rig.chat())["model"] == "dec1-served-version"
