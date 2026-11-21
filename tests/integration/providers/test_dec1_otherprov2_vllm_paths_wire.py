"""Contract: hosted vLLM (self-hosted OpenAI-compatible servers) base path joining, extra_body and request shapes.

A `hosted_vllm/<served-model>` deployment builds its URL with `OpenAIGPTConfig.get_complete_url`
(`litellm/llms/openai/chat/gpt_transformation.py:925-951`): trailing slashes are stripped, the base is used as is
when the text `chat/completions` appears anywhere in it, and otherwise `/chat/completions` is joined on
(`join_api_base`, `litellm/litellm_core_utils/url_utils.py:151-163`). Self-hosted gateways often put tenants or
versions in the path, so the join is a contract a Rust port must keep.

hosted_vllm is in `openai_compatible_providers`, so vLLM's own sampling params (`top_k`, `repetition_penalty`,
`chat_template_kwargs`, guided decoding) travel in `extra_body` and are merged into the top level of the body,
after the transformed request, so an `extra_body` key wins over a mapped key; `model` inside `extra_body` is never
sent (`EXTRA_BODY_ROUTING_KEYS`, `litellm/utils.py:6341`, `:6366-6398`, `llm_http_handler.py:973-983`).
`HostedVLLMChatConfig` (`litellm/llms/hosted_vllm/chat/transformation.py`) also turns `custom` tools into function
tools and video `file` parts into `video_url` parts.

The fake key, empty tools, strict and closed schemas, `thinking`, `user`, and reasoning replay are covered in
`test_dec1_openai_compatible_chat_wire.py` and `test_hosted_vllm_reasoning_content_wire.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_otherprov2_support import upstream_error
from integration.providers.dec1_otherprov_support import (
    PROVIDER_KEY,
    chat,
    deployment,
    error_message,
    marker,
    peer,
    served,
)
from pydantic import JsonValue

_VLLM: Final = "hosted_vllm/dec1-served-model"


@pytest.mark.parametrize(
    ("suffix", "path"),
    [
        ("/v1", "/v1/chat/completions"),
        ("/v1/", "/v1/chat/completions"),
        ("/tenant-a/vllm/v1", "/tenant-a/vllm/v1/chat/completions"),
        ("/tenant-a/vllm/v1//", "/tenant-a/vllm/v1/chat/completions"),
        ("/v1/chat/completions", "/v1/chat/completions"),
    ],
    ids=["v1", "v1-slash", "tenant-prefix", "tenant-two-slashes", "full-route"],
)
def test_hosted_vllm_joins_chat_completions_onto_the_base_path(gateway: Gateway, suffix: str, path: str) -> None:
    """A hosted vLLM deployment whose base is the peer plus a path. Trailing slashes are stripped and
    `/chat/completions` is joined once; a base that already names the route is used as is
    (`gpt_transformation.py:944-951`). The peer receives exactly one POST at the expected path with the deployment's
    bearer key."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix=suffix)
        served(chat(gateway, model))
        call: Final = provider.only()
        assert call.path == path, call.path
        assert call.headers["authorization"] == f"Bearer {PROVIDER_KEY}", call.headers


def test_hosted_vllm_base_with_a_query_keeps_the_query_after_the_route(gateway: Gateway) -> None:
    """A base `<peer>/v1?tenant=dec1` (gateways that select a tenant by query). The intended request is `POST
    /v1/chat/completions?tenant=dec1`.

    `join_api_base` appends `/chat/completions` to the whole string (`url_utils.py:151-163`), so today the peer
    gets path `/v1` with query `tenant=dec1/chat/completions`.
    """
    pytest.skip("BUG: a hosted_vllm api_base with a query string gets /chat/completions appended after the query")
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1?tenant=dec1")
        served(chat(gateway, model))
        call: Final = provider.only()
        assert (call.path, dict(call.query)) == ("/v1/chat/completions", {"tenant": ["dec1"]}), call


def test_hosted_vllm_base_merely_containing_the_route_text_is_still_joined(gateway: Gateway) -> None:
    """A base `<peer>/chat/completions-gateway/v1`, a path that contains the text `chat/completions` but does not
    end with the route. The intended request is `POST /chat/completions-gateway/v1/chat/completions`.

    `get_complete_url` tests `endpoint in api_base` as a substring (`gpt_transformation.py:947-949`), so today the
    base is sent as is and the peer gets `POST /chat/completions-gateway/v1`.
    """
    pytest.skip("BUG: any api_base containing the text chat/completions is used verbatim instead of joined")
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/chat/completions-gateway/v1")
        served(chat(gateway, model))
        assert provider.only().path == "/chat/completions-gateway/v1/chat/completions"


def test_hosted_vllm_extra_body_fields_reach_the_top_level(gateway: Gateway) -> None:
    """The caller sends `extra_body: {"top_k": 20, "repetition_penalty": 1.1, "chat_template_kwargs":
    {"enable_thinking": false}, "guided_choice": ["yes", "no"]}`. vLLM receives all four at the top level of the
    body with the same values and no `extra_body` key (`llm_http_handler.py:973-983`)."""
    extra: Final[dict[str, JsonValue]] = {
        "top_k": 20,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_choice": ["yes", "no"],
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, extra_body=extra))
        sent: Final = provider.only().body
        assert {key: sent.get(key) for key in extra} == extra, sent
        assert "extra_body" not in sent, sent


def test_hosted_vllm_top_level_unknown_params_are_forwarded(gateway: Gateway) -> None:
    """The caller sends `top_k: 5` and `min_p: 0.05` at the top level of its request (no `extra_body`). They are not
    OpenAI params, so they are collected into `extra_body` (`litellm/utils.py:6378-6381`) and vLLM receives both at
    the top level."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, top_k=5, min_p=0.05))
        sent: Final = provider.only().body
        assert (sent["top_k"], sent["min_p"]) == (5, 0.05), sent


def test_hosted_vllm_extra_body_never_replaces_the_served_model(gateway: Gateway) -> None:
    """The caller's `extra_body` carries `model: "dec1-other-model"`. `model` is an `EXTRA_BODY_ROUTING_KEYS`
    entry and is filtered out (`litellm/utils.py:6341`, `:6388-6396`), so vLLM receives `model:
    "dec1-served-model"`, the deployment's model."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, extra_body={"model": "dec1-other-model", "top_k": 3}))
        sent: Final = provider.only().body
        assert (sent["model"], sent["top_k"]) == ("dec1-served-model", 3), sent


def test_hosted_vllm_extra_body_key_wins_over_the_mapped_param(gateway: Gateway) -> None:
    """The caller sends `temperature: 0.1` and `extra_body: {"temperature": 0.9}`. The handler merges
    `{**transformed, **extra_body}` (`llm_http_handler.py:973-983`), so vLLM receives `temperature: 0.9`."""
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, temperature=0.1, extra_body={"temperature": 0.9}))
        assert provider.only().body["temperature"] == 0.9


def test_hosted_vllm_custom_tool_becomes_a_function_tool(gateway: Gateway) -> None:
    """The caller sends a `custom` tool `{"type": "custom", "custom": {"name": "run_sql", "description": "Run
    SQL"}}` with no input schema. vLLM only accepts function tools, so it receives `{"type": "function",
    "function": {"name": "run_sql", "description": "Run SQL", "parameters": <one required string "input">}}`
    (`_convert_custom_tools_to_function_tools`, `hosted_vllm/chat/transformation.py:32-76`)."""
    custom: Final[dict[str, JsonValue]] = {"type": "custom", "custom": {"name": "run_sql", "description": "Run SQL"}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, tools=[custom]))
        tool: Final = object_value(list_value(provider.only().body["tools"])[0])
        function: Final = object_value(tool["function"])
        assert (tool["type"], function["name"], function["description"]) == ("function", "run_sql", "Run SQL"), tool
        parameters: Final = object_value(function["parameters"])
        assert parameters["required"] == ["input"], parameters
        assert object_value(object_value(parameters["properties"])["input"])["type"] == "string", parameters


def test_hosted_vllm_video_file_part_becomes_a_video_url_part(gateway: Gateway) -> None:
    """A user message with a `file` part whose `file_id` is `https://media.example.com/dec1.mp4` and `format:
    "video/mp4"`. vLLM receives that part as `{"type": "video_url", "video_url": {"url": <the same URL>}}`
    (`_is_video_file`, `_convert_file_to_video_url`, `hosted_vllm/chat/transformation.py:130-158`, `:239-249`)."""
    url: Final = "https://media.example.com/dec1.mp4"
    parts: Final[list[JsonValue]] = [
        {"type": "text", "text": marker()},
        {"type": "file", "file": {"file_id": url, "format": "video/mp4"}},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, messages=[{"role": "user", "content": parts}]))
        content: Final = list_value(object_value(list_value(provider.only().body["messages"])[0])["content"])
        assert object_value(content[1]) == {"type": "video_url", "video_url": {"url": url}}, content


def test_hosted_vllm_anthropic_style_tool_use_blocks_in_history_become_tool_calls(gateway: Gateway) -> None:
    """A replayed assistant turn whose content list is a text block plus an Anthropic-style `tool_use` block
    (`id: toolu_dec1`, `name: get_weather`, `input: {"city": "Oslo"}`). vLLM receives the assistant message with
    `content: "checking"` (text parts joined) and one tool call with that id, name and JSON string arguments
    (`hosted_vllm/chat/transformation.py:187-238`)."""
    messages: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_dec1", "name": "get_weather", "input": {"city": "Oslo"}},
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1", "content": "sunny"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        served(chat(gateway, model, messages=messages))
        assistant: Final = object_value(list_value(provider.only().body["messages"])[1])
        assert assistant["content"] == "checking", assistant
        call: Final = object_value(list_value(assistant["tool_calls"])[0])
        function: Final = object_value(call["function"])
        assert (call["id"], function["name"]) == ("toolu_dec1", "get_weather"), call
        assert function["arguments"] == '{"city": "Oslo"}', call


@pytest.mark.parametrize("status", [400, 404, 500])
def test_hosted_vllm_error_status_reaches_the_caller(gateway: Gateway, status: int) -> None:
    """vLLM answers 400 (bad request), 404 (unknown served model) or 500 with `dec1 vllm failure`. hosted_vllm is
    mapped like OpenAI (`exception_mapping_utils.py:2733-2749`), so the caller gets the same status and the message
    after one upstream call."""
    with peer(lambda call: upstream_error(status, "dec1 vllm failure")) as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, _VLLM, api_base_suffix="/v1")
        response: Final = chat(gateway, model)
        assert response.status_code == status, response.text
        assert "dec1 vllm failure" in error_message(response)
        assert len(provider.posts()) == 1
