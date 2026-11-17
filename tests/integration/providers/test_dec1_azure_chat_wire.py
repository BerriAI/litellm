"""Azure OpenAI `/v1/chat/completions` on the wire: URL shape, credentials, request body and answer shape.

An `azure/<deployment>` model is served through the OpenAI SDK's `AzureOpenAI` client
(`litellm/llms/azure/azure.py`, client built in `litellm/llms/azure/common_utils.py:737`). With a dated or
preview `api_version` the SDK posts to `{api_base}/openai/deployments/{deployment}/chat/completions` with the
version in the `api-version` query, and the deployment key goes in the `api-key` header. A static
`azure_ad_token` is sent as `Authorization: Bearer <token>` instead. When `api_base` already contains
`/openai/deployments/<name>` it is used as the SDK `base_url` (`common_utils.py:548`), so the deployment
named there wins over the model name.

The body is built by `AzureOpenAIConfig.transform_request` (`litellm/llms/azure/chat/gpt_transformation.py:291`)
or the GPT-5 subclass: the deployment name is the body `model`, `tool_choice` is dropped when no tools are
sent, and GPT-5 deployments get `max_completion_tokens` instead of `max_tokens`. Azure answers with OpenAI's
chat shape plus content filter annotations: `prompt_filter_results` at the top level is copied onto the
response (`convert_dict_to_response.py:955`) and per-choice `content_filter_results` lands in
`choices[i].provider_specific_fields` (`convert_dict_to_response.py:904`).

All of this runs on the Python path today. A Rust front door or a Rust Azure client has to keep the exact
URL, query, header and body bytes, since Azure rejects unknown fields and wrong credentials outright.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    API_KEY,
    CONTENT_FILTER_RESULT,
    DATED_API_VERSION,
    DEPLOYMENT,
    ENTRA_TOKEN,
    GPT5_DEPLOYMENT,
    WEATHER_TOOL,
    azure,
    body_of,
    chat_json,
    chat_reply,
    json_reply,
    marker,
    path_of,
    query_of,
)


def _ok(_: Request) -> Reply:
    return chat_reply("pong")


def test_a_chat_call_posts_to_the_deployment_path_with_the_configured_api_version(gateway: Gateway) -> None:
    """An `azure/gpt-4o-dec1` deployment with `api_version: 2024-10-21` serves one chat call.

    Azure must see exactly one `POST /openai/deployments/gpt-4o-dec1/chat/completions` whose query is exactly
    `{"api-version": "2024-10-21"}`, and the caller gets HTTP 200. The deployment segment is the Azure
    deployment name, not the public model group name the caller used.
    """
    with azure(gateway, _ok) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert path_of(request) == f"/openai/deployments/{DEPLOYMENT}/chat/completions"
        assert query_of(request) == {"api-version": DATED_API_VERSION}


def test_the_deployment_key_is_sent_in_the_api_key_header(gateway: Gateway) -> None:
    """A keyed Azure deployment serves one chat call.

    The upstream request must carry `api-key: synthetic-azure-key`.

    Pinned current (run 44176682961): the request also carries `Authorization: Bearer synthetic-azure-key`.
    The call is sent by the `AsyncAzureOpenAI` client of the locked openai SDK (uv.lock pins openai 2.33.0,
    user agent `AsyncAzureOpenAI/Python 2.33.0`), which adds the base client's bearer next to `api-key`; newer
    SDKs (2.54 `_auth_headers`) send only `api-key`. Whether a real Azure resource accepts the extra bearer was
    not checked here, so the test only requires that no other credential appears in `Authorization`.
    """
    with azure(gateway, _ok) as rig:
        assert rig.chat().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("api-key") == API_KEY
        assert headers.get("authorization", f"Bearer {API_KEY}") == f"Bearer {API_KEY}", headers


def test_the_callers_virtual_key_never_reaches_azure(gateway: Gateway) -> None:
    """The caller authenticates to the proxy with its virtual key in `Authorization: Bearer`.

    No header Azure receives may contain that virtual key; only the deployment's own credential is sent.
    """
    with azure(gateway, _ok) as rig:
        key: Final = rig.scenario.key()
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": rig.model, "messages": [{"role": "user", "content": marker()}]},
            key=key,
        )
        assert response.status_code == 200, response.text
        headers: Final = rig.one_post().headers
        assert all(key not in value for value in headers.values()), headers


def test_the_body_model_is_the_deployment_name_not_the_public_model_group(gateway: Gateway) -> None:
    """The proxy model group is `dec1-azure-<hex>` and the deployment is `azure/gpt-4o-dec1`.

    The JSON body Azure receives must have `model: "gpt-4o-dec1"` and the caller's single user message
    unchanged, plus `stream: false`, which the OpenAI SDK path always sets on a non streaming call.
    """
    prompt: Final = marker()
    with azure(gateway, _ok) as rig:
        assert rig.chat(prompt).status_code == 200
        body: Final = body_of(rig.one_post())
        assert body["model"] == DEPLOYMENT
        assert body["messages"] == [{"role": "user", "content": prompt}]
        assert body.get("stream") is False, body


def test_an_azure_ad_token_without_a_key_is_sent_as_a_bearer(gateway: Gateway) -> None:
    """The deployment has no `api_key` and a static `azure_ad_token: synthetic-entra-access-token`.

    Azure must receive `Authorization: Bearer synthetic-entra-access-token` and no `api-key` header.
    `initialize_azure_sdk_client` passes the token to `AzureOpenAI(azure_ad_token=...)` (`common_utils.py:950`).
    """
    with azure(gateway, _ok, api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        headers: Final = rig.one_post().headers
        assert headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in headers, headers


def test_an_azure_ad_token_wins_over_a_key_when_both_are_configured(gateway: Gateway) -> None:
    """The deployment sets both `api_key` and `azure_ad_token`.

    The AzureOpenAI SDK prefers the token: Azure must receive `Authorization: Bearer <token>` and no `api-key`.
    UNVERIFIED: read from the OpenAI SDK's `AzureOpenAI._prepare_options` (token first, key only when no
    token); confirm in `openai/lib/azure.py` of the pinned SDK version.
    """
    with azure(gateway, _ok, azure_ad_token=ENTRA_TOKEN) as rig:
        assert rig.chat().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "api-key" not in headers, headers


def test_a_deployment_path_inside_api_base_is_used_verbatim(gateway: Gateway) -> None:
    """`api_base` is `<peer>/openai/deployments/pinned-deploy` and the model is `azure/other-name`.

    `select_azure_base_url_or_endpoint` treats such a base as the SDK `base_url` (`common_utils.py:548`), so
    Azure must see `POST /openai/deployments/pinned-deploy/chat/completions` with `api-version: 2024-10-21`,
    while the body `model` is still `other-name`.
    """
    with azure(gateway, _ok, model="azure/other-name", base_path="/openai/deployments/pinned-deploy") as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert path_of(request) == "/openai/deployments/pinned-deploy/chat/completions"
        assert query_of(request) == {"api-version": DATED_API_VERSION}
        assert body_of(request)["model"] == "other-name"


@pytest.mark.parametrize("suffix", ["/", "//"], ids=["one_slash", "two_slashes"])
def test_a_trailing_slash_on_api_base_never_doubles_the_path_separator(gateway: Gateway, suffix: str) -> None:
    """`api_base` ends with `suffix` (one or two slashes).

    Azure must still see exactly `/openai/deployments/gpt-4o-dec1/chat/completions`, never `//openai/...`.
    UNVERIFIED for two slashes: the SDK strips trailing slashes from `azure_endpoint` with `rstrip("/")`.
    """
    with azure(gateway, _ok, base_path=suffix) as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.one_post()) == f"/openai/deployments/{DEPLOYMENT}/chat/completions"


def test_tools_and_tool_choice_reach_azure_unchanged(gateway: Gateway) -> None:
    """The caller sends one function tool and `tool_choice: {"type": "function", "function": {"name": "get_weather"}}`.

    On a dated api-version that supports tools (`2024-10-21`) both must reach Azure exactly as sent.
    """
    choice: Final = {"type": "function", "function": {"name": "get_weather"}}
    with azure(gateway, _ok) as rig:
        assert rig.chat(tools=[WEATHER_TOOL], tool_choice=choice).status_code == 200
        body: Final = body_of(rig.one_post())
        assert body["tools"] == [WEATHER_TOOL]
        assert body["tool_choice"] == choice


def test_tool_choice_without_tools_is_not_sent_to_azure(gateway: Gateway) -> None:
    """The caller sends `tool_choice: "auto"` and no `tools`.

    Azure rejects a `tool_choice` with no tools, so the body must not contain `tool_choice`
    (`gpt_transformation.py:308` keeps it only when `tools` or `functions` is present).
    """
    with azure(gateway, _ok) as rig:
        assert rig.chat(tool_choice="auto").status_code == 200
        assert "tool_choice" not in body_of(rig.one_post())


def test_max_tokens_becomes_max_completion_tokens_on_a_gpt5_deployment(gateway: Gateway) -> None:
    """The deployment is `azure/gpt-5.4-nano` and the caller sends `max_tokens: 64`.

    Azure must receive `max_completion_tokens: 64` and no `max_tokens`, because GPT-5 deployments reject the
    legacy key (`requires_max_completion_tokens`, `gpt_transformation.py:166`).
    """
    with azure(gateway, _ok, model=f"azure/{GPT5_DEPLOYMENT}") as rig:
        assert rig.chat(max_tokens=64).status_code == 200
        body: Final = body_of(rig.one_post())
        assert body.get("max_completion_tokens") == 64, body
        assert "max_tokens" not in body, body


def test_max_tokens_is_kept_on_a_gpt4o_deployment(gateway: Gateway) -> None:
    """The deployment is `azure/gpt-4o-dec1` and the caller sends `max_tokens: 64`.

    The rename is only for reasoning model names, so Azure must receive `max_tokens: 64` unchanged and no
    `max_completion_tokens`.
    """
    with azure(gateway, _ok) as rig:
        assert rig.chat(max_tokens=64).status_code == 200
        body: Final = body_of(rig.one_post())
        assert body.get("max_tokens") == 64, body
        assert "max_completion_tokens" not in body, body


def test_a_developer_message_reaches_azure_as_a_system_message(gateway: Gateway) -> None:
    """The caller sends `[{"role": "developer", "content": "be terse"}, {"role": "user", ...}]` to a gpt-4o deployment.

    `AzureOpenAIConfig.translate_developer_role_to_system_role` (`gpt_transformation.py:111`) only renames the
    role, so Azure must receive the same two messages with the first one's role `system` and its position kept.
    UNVERIFIED: whether the shared proxy calls the translation for a non reasoning deployment; read
    `litellm/utils.py` where `translate_developer_role_to_system_role` is invoked.
    """
    prompt: Final = marker()
    with azure(gateway, _ok) as rig:
        response: Final = rig.chat_body(
            {"messages": [{"role": "developer", "content": "be terse"}, {"role": "user", "content": prompt}]}
        )
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": prompt},
        ]


def test_the_user_field_is_forwarded(gateway: Gateway) -> None:
    """The caller sends `user: "end-user-dec1"`.

    `user` is a supported Azure chat param (`gpt_transformation.py:145`), so the body Azure receives must
    contain `user: "end-user-dec1"`.
    """
    with azure(gateway, _ok) as rig:
        assert rig.chat(user="end-user-dec1").status_code == 200
        assert body_of(rig.one_post()).get("user") == "end-user-dec1"


def test_the_answer_is_returned_in_openai_chat_shape_with_the_azure_usage(gateway: Gateway) -> None:
    """Azure answers `content: "pong"`, `finish_reason: stop` and usage 11 prompt, 7 completion, 18 total.

    The caller must get `object: chat.completion`, one choice with that content and finish reason, and the same
    three usage numbers.
    """
    with azure(gateway, _ok) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        body: Final = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "pong"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert (body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"], body["usage"]["total_tokens"]) == (
            11,
            7,
            18,
        )


def test_a_content_filter_finish_reason_reaches_the_caller(gateway: Gateway) -> None:
    """Azure completes with `finish_reason: content_filter` and `content: null` (the output was filtered).

    The caller must see `finish_reason: "content_filter"` and a null content, not a rewritten `stop`, so it
    can tell a filtered answer from an empty one.
    """
    with azure(gateway, lambda _: chat_reply(None, finish_reason="content_filter")) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["finish_reason"] == "content_filter"
        assert choice["message"]["content"] is None


def test_choice_content_filter_results_are_kept_in_provider_specific_fields(gateway: Gateway) -> None:
    """Azure attaches `content_filter_results` (hate filtered, the rest safe) to its only choice.

    Unknown choice keys are kept in `provider_specific_fields` (`convert_dict_to_response.py:904`), so the
    caller must find `choices[0].provider_specific_fields.content_filter_results` equal to what Azure sent.
    UNVERIFIED: that the proxy serializes `provider_specific_fields` on a non streaming choice.
    """
    with azure(gateway, lambda _: chat_reply(choice_extra={"content_filter_results": CONTENT_FILTER_RESULT})) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        fields: Final = response.json()["choices"][0]["provider_specific_fields"]
        assert fields["content_filter_results"] == CONTENT_FILTER_RESULT


def test_prompt_filter_results_are_copied_onto_the_response(gateway: Gateway) -> None:
    """Azure adds a top-level `prompt_filter_results: [{"prompt_index": 0, "content_filter_results": ...}]`.

    Unknown top-level keys are copied onto the response (`convert_dict_to_response.py:955`), so the caller must
    get the same `prompt_filter_results` list back.
    """
    filters: Final = [{"prompt_index": 0, "content_filter_results": CONTENT_FILTER_RESULT}]
    with azure(gateway, lambda _: chat_reply(top_extra={"prompt_filter_results": filters})) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["prompt_filter_results"] == filters


def test_a_tool_call_answer_reaches_the_caller_with_finish_reason_tool_calls(gateway: Gateway) -> None:
    """Azure answers with one `get_weather` tool call (`arguments: {"city": "Paris"}`) and `finish_reason: tool_calls`.

    The caller must get that tool call id, name and arguments string unchanged, null content and
    `finish_reason: tool_calls`.
    """
    call: Final = {
        "id": "call_dec1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    reply: Final = json_reply(chat_json(None, finish_reason="tool_calls", message_extra={"tool_calls": [call]}))
    with azure(gateway, lambda _: reply) as rig:
        response: Final = rig.chat(tools=[WEATHER_TOOL])
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["id"] == "call_dec1"
        assert choice["message"]["tool_calls"][0]["function"] == call["function"]


def test_json_schema_on_a_gpt35_deployment_is_sent_as_a_forced_tool(gateway: Gateway) -> None:
    """The deployment is `azure/gpt-35-turbo` and the caller asks for `response_format: {"type": "json_schema", ...}`.

    gpt-35 does not support `response_format` (`gpt_transformation.py:175`), so Azure must receive no
    `response_format`, one tool named `json_tool_call` whose `parameters` is the schema, and
    `tool_choice: {"type": "function", "function": {"name": "json_tool_call"}}`.
    """
    schema: Final = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    response_format: Final = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    with azure(gateway, _ok, model="azure/gpt-35-turbo") as rig:
        assert rig.chat(response_format=response_format).status_code == 200
        body: Final = body_of(rig.one_post())
        assert "response_format" not in body, body
        assert body["tools"] == [{"type": "function", "function": {"name": "json_tool_call", "parameters": schema}}]
        assert body["tool_choice"] == {"type": "function", "function": {"name": "json_tool_call"}}


def test_a_json_tool_call_answer_comes_back_as_plain_content(gateway: Gateway) -> None:
    """Same gpt-35 deployment; Azure answers with one `json_tool_call` tool call whose arguments are `{"answer": "42"}`.

    In json mode the tool call is turned back into content (`convert_tool_call_to_json_mode`,
    `convert_dict_to_response.py:149`): the caller must get `content == '{"answer": "42"}'`, no tool calls and
    `finish_reason: stop`.
    """
    schema: Final = {"type": "object", "properties": {"answer": {"type": "string"}}}
    call: Final = {
        "id": "call_json",
        "type": "function",
        "function": {"name": "json_tool_call", "arguments": '{"answer": "42"}'},
    }
    reply: Final = json_reply(chat_json(None, finish_reason="tool_calls", message_extra={"tool_calls": [call]}))
    with azure(gateway, lambda _: reply, model="azure/gpt-35-turbo") as rig:
        response: Final = rig.chat(
            response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}}
        )
        assert response.status_code == 200, response.text
        choice: Final = response.json()["choices"][0]
        assert choice["message"]["content"] == '{"answer": "42"}'
        assert not choice["message"].get("tool_calls")
        assert choice["finish_reason"] == "stop"


def test_json_schema_on_a_supported_deployment_is_sent_as_response_format(gateway: Gateway) -> None:
    """A gpt-4o deployment on `2024-10-21` (after the 2024-08 cutoff in `litellm/types/llms/azure.py:3`).

    The caller's `response_format` must reach Azure unchanged and no `json_tool_call` tool may be added.
    """
    schema: Final = {"type": "object", "properties": {"answer": {"type": "string"}}}
    response_format: Final = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    with azure(gateway, _ok) as rig:
        assert rig.chat(response_format=response_format).status_code == 200
        body: Final = body_of(rig.one_post())
        assert body["response_format"] == response_format
        assert "tools" not in body, body
