"""Azure AI Foundry (`azure_ai/...`) through the proxy: chat URL and auth, DeepSeek, Mistral and Kimi rules, Claude.

Non Claude Foundry models use `AzureAIStudioConfig` (`litellm/llms/azure_ai/chat/transformation.py`), an
OpenAI compatible config over litellm's own HTTP client. The URL is `{api_base}/chat/completions`, or
`/models/chat/completions` on a `*.services.ai.azure.com` host, with an optional `api-version` query from the
deployment (`transformation.py:430`); an existing path suffix is never doubled (`add_path_to_api_base`). A
key is sent as `Authorization: Bearer` except on Azure OpenAI v1 hosts, which take `api-key`
(`api_key_header_for_base`); with no key the Entra ID token is used as a bearer. Messages are cleaned for
Foundry backends that set `additionalProperties: false`: `thinking_blocks`, `provider_specific_fields`,
`cache_control` and (except for `FW-` and `kimi-` models, since #51747 on 2026-11-13) `reasoning_content` are
removed, and text only content lists collapse to a string.

`reasoning_effort` on Foundry changed twice this week (#51747 on 2026-11-13, #51775 on 2026-11-16): a model
whose catalog row declares `reasoning_effort_levels` (DeepSeek V4 Flash and Pro: `none`, `high`, `max`;
FW-Kimi-K3: seven levels) gets a declared level as sent, `default` and `auto` are left to the model (removed),
and an undeclared level is a 400 `UnsupportedParamsError` or dropped under `drop_params`. The deployment's
`base_model` row is checked first. Usage that reports `reasoning_tokens` outside `completion_tokens` is
folded into `completion_tokens` (`XAIChatConfig.fold_reasoning_tokens_into_completion`).

Foundry also answers 422 for parameters a deployment does not pass through; the config retries once or twice
without them (`should_retry_llm_api_inside_llm_translation_on_http_error`, `transformation.py:665`).

Claude on Foundry (`azure_ai/claude-*`) posts the Anthropic Messages API at `/anthropic/v1/messages`
(`azure_anthropic_messages_url`, `litellm/llms/azure_ai/anthropic/transformation.py:14`) from both
`/v1/chat/completions` and `/v1/messages`. `/v1/messages` for Foundry Claude is now served by Rust only
(#51721, 2026-11-14), which ported key and Entra token resolution; the msgprov suite
(`messages_endpoint/test_dec1_msgprov_azure.py`) covers the body and headers, so this file adds the auth and
join cases it leaves out.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_azure_support import (
    ENTRA_TOKEN,
    FOUNDRY_KEY,
    WEATHER_TOOL,
    always,
    anthropic_message_json,
    azure_error,
    body_of,
    chat_reply,
    error_message,
    failing_then,
    foundry,
    json_reply,
    marker,
    path_of,
    query_of,
)

_MISTRAL: Final = "azure_ai/mistral-large-3"
_DEEPSEEK_FLASH: Final = "azure_ai/deepseek-v4-flash"
_KIMI_K3: Final = "azure_ai/FW-Kimi-K3"
_CLAUDE: Final = "azure_ai/claude-sonnet-4-5"


def _ok(_: Request) -> Reply:
    return chat_reply("pong", model="mistral-large-3")


def _anthropic(_: Request) -> Reply:
    return json_reply(anthropic_message_json("pong"))


def test_a_foundry_chat_call_posts_to_chat_completions_under_api_base(gateway: Gateway) -> None:
    """An `azure_ai/mistral-large-3` deployment whose `api_base` is `<peer>/models` (not a services.ai.azure.com host).

    Foundry must see exactly one `POST /models/chat/completions` with no query, and the body `model` must be
    `mistral-large-3` without the `azure_ai/` prefix.
    """
    with foundry(gateway, _ok, model=_MISTRAL, base_path="/models") as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/models/chat/completions", {})
        assert body_of(request)["model"] == "mistral-large-3"


def test_a_foundry_api_version_is_added_as_a_query(gateway: Gateway) -> None:
    """The Foundry deployment sets `api_version: 2024-05-01-preview`.

    Foundry must see `POST /chat/completions?api-version=2024-05-01-preview` (`transformation.py:463`).
    """
    with foundry(gateway, _ok, model=_MISTRAL, api_version="2024-05-01-preview") as rig:
        assert rig.chat().status_code == 200
        request: Final = rig.one_post()
        assert (path_of(request), query_of(request)) == ("/chat/completions", {"api-version": "2024-05-01-preview"})


def test_an_api_base_that_already_ends_in_chat_completions_is_not_doubled(gateway: Gateway) -> None:
    """`api_base` is `<peer>/v1/chat/completions`.

    `add_path_to_api_base` merges overlapping segments (`litellm/utils.py:11900`), so Foundry must see
    `POST /v1/chat/completions`, never `/v1/chat/completions/chat/completions`.
    """
    with foundry(gateway, _ok, model=_MISTRAL, base_path="/v1/chat/completions") as rig:
        assert rig.chat().status_code == 200
        assert path_of(rig.one_post()) == "/v1/chat/completions"


def test_an_api_version_already_in_api_base_is_kept_over_the_deployment_one(gateway: Gateway) -> None:
    """`api_base` is `<peer>?api-version=2024-05-01-preview` and the deployment sets `api_version: 2025-01-01`.

    The base query wins (`"api-version" not in query_params`, `transformation.py:466`): Foundry must see
    `api-version=2024-05-01-preview` exactly once.
    """
    with foundry(
        gateway, _ok, model=_MISTRAL, base_path="?api-version=2024-05-01-preview", api_version="2025-01-01"
    ) as rig:
        assert rig.chat().status_code == 200
        assert query_of(rig.one_post()) == {"api-version": "2024-05-01-preview"}


def test_a_foundry_key_is_sent_as_a_bearer_on_a_non_azure_openai_host(gateway: Gateway) -> None:
    """A keyed Foundry deployment on a loopback host.

    `api_key_header_for_base` returns `Authorization` for any host that is not an Azure OpenAI v1 host, so
    Foundry must receive `Authorization: Bearer synthetic-foundry-key` and no `api-key` header.
    """
    with foundry(gateway, _ok, model=_MISTRAL) as rig:
        assert rig.chat().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("authorization") == f"Bearer {FOUNDRY_KEY}"
        assert "api-key" not in headers, headers


def test_a_foundry_deployment_with_only_an_azure_ad_token_sends_it_as_a_bearer(gateway: Gateway) -> None:
    """A Foundry deployment with no key and `azure_ad_token: synthetic-entra-access-token`.

    With no key `validate_environment` falls back to `base_validate_azure_environment` (`transformation.py:417`),
    so Foundry must receive `Authorization: Bearer synthetic-entra-access-token`.
    """
    with foundry(gateway, _ok, model=_MISTRAL, api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        assert rig.chat().status_code == 200
        assert rig.one_post().headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"


def test_text_only_content_lists_are_collapsed_to_a_string(gateway: Gateway) -> None:
    """The caller sends user content `[{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]`.

    Many Foundry backends refuse list content, so the message Foundry receives must have `content` as the
    single string `a b` (`convert_content_list_to_str`, `transformation.py:556`). UNVERIFIED: the join has no
    separator, so the text is the plain concatenation of the parts.
    """
    with foundry(gateway, _ok, model=_MISTRAL) as rig:
        content: Final = [{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]
        assert rig.chat_body({"messages": [{"role": "user", "content": content}]}).status_code == 200
        assert body_of(rig.one_post())["messages"] == [{"role": "user", "content": "a b"}]


def test_image_content_lists_are_sent_as_lists(gateway: Gateway) -> None:
    """The caller sends user content with a text part and an `image_url` part.

    A list holding an image is user intended and kept (`_has_list_only_content_part`, `transformation.py:90`),
    so Foundry must receive the same two part list unchanged.
    """
    content: Final = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]
    with foundry(gateway, _ok, model=_MISTRAL) as rig:
        assert rig.chat_body({"messages": [{"role": "user", "content": content}]}).status_code == 200
        assert body_of(rig.one_post())["messages"][0]["content"] == content  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


_HISTORY_WITH_REASONING: Final = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "42", "reasoning_content": "thought about it"},
    {"role": "user", "content": "second"},
]


def test_replayed_reasoning_content_is_stripped_for_a_mistral_deployment(gateway: Gateway) -> None:
    """The caller replays an assistant turn carrying `reasoning_content` to `azure_ai/mistral-large-3`.

    Foundry's Mistral rejects unknown message fields, so the assistant message it receives must be exactly
    `{"role": "assistant", "content": "42"}`.
    """
    with foundry(gateway, _ok, model=_MISTRAL) as rig:
        assert rig.chat_body({"messages": _HISTORY_WITH_REASONING}).status_code == 200
        assert body_of(rig.one_post())["messages"][1] == {"role": "assistant", "content": "42"}  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


@pytest.mark.parametrize("model", [_KIMI_K3, "azure_ai/kimi-k2.6"], ids=["fireworks_hosted", "moonshot_format"])
def test_replayed_reasoning_content_is_kept_for_kimi_deployments(gateway: Gateway, model: str) -> None:
    """The same history to an `FW-` (Fireworks hosted) or `kimi-` Foundry deployment.

    Kimi reads its earlier reasoning back across tool calls, so since #51747 the assistant message must keep
    `reasoning_content: "thought about it"` (`_message_fields_to_strip`, `transformation.py:163`).
    """
    with foundry(gateway, _ok, model=model) as rig:
        assert rig.chat_body({"messages": _HISTORY_WITH_REASONING}).status_code == 200
        assert body_of(rig.one_post())["messages"][1]["reasoning_content"] == "thought about it"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


@pytest.mark.parametrize("level", ["none", "high", "max"])
def test_a_declared_deepseek_reasoning_effort_is_sent_as_is(gateway: Gateway, level: str) -> None:
    """`azure_ai/deepseek-v4-flash` declares `reasoning_effort_levels: [none, high, max]` in the cost map.

    A caller `reasoning_effort` of one of those levels must reach Foundry unchanged (#51775).
    """
    with foundry(gateway, _ok, model=_DEEPSEEK_FLASH) as rig:
        assert rig.chat(reasoning_effort=level).status_code == 200
        assert body_of(rig.one_post())["reasoning_effort"] == level


def test_an_undeclared_deepseek_reasoning_effort_is_a_400_before_foundry(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: "low"` to `azure_ai/deepseek-v4-flash` without drop_params.

    The caller must get HTTP 400 whose message contains `azure_ai cannot send reasoning_effort='low'` and
    `the model catalog lists only ['none', 'high', 'max']` (`transformation.py:311`), and Foundry must
    receive no request.
    """
    with foundry(gateway, _ok, model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort="low")
        assert response.status_code == 400, response.text
        message: Final = error_message(response)
        assert "azure_ai cannot send reasoning_effort='low'" in message
        assert "the model catalog lists only ['none', 'high', 'max']" in message
        assert rig.posts() == ()


def test_an_undeclared_deepseek_reasoning_effort_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """Same `reasoning_effort: "low"` with `drop_params: true` on the request.

    The call must succeed and the body Foundry receives must have no `reasoning_effort`.
    """
    with foundry(gateway, _ok, model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort="low", drop_params=True)
        assert response.status_code == 200, response.text
        assert "reasoning_effort" not in body_of(rig.one_post())


@pytest.mark.parametrize("level", ["default", "auto"])
def test_default_and_auto_reasoning_effort_are_left_to_the_model(gateway: Gateway, level: str) -> None:
    """The caller sends `reasoning_effort` `default` or `auto` to `azure_ai/deepseek-v4-flash`.

    Both mean "let the model decide" (`_LEVELS_LEFT_TO_THE_MODEL`), so the call must succeed and the body
    must carry no `reasoning_effort`; Foundry refuses `default` on DeepSeek V4.
    """
    with foundry(gateway, _ok, model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort=level)
        assert response.status_code == 200, response.text
        assert "reasoning_effort" not in body_of(rig.one_post())


def test_kimi_k3_accepts_its_xhigh_reasoning_level(gateway: Gateway) -> None:
    """`azure_ai/FW-Kimi-K3` declares seven levels including `xhigh`.

    `reasoning_effort: "xhigh"` must reach Foundry unchanged (#51747).
    """
    with foundry(gateway, _ok, model=_KIMI_K3) as rig:
        assert rig.chat(reasoning_effort="xhigh").status_code == 200
        assert body_of(rig.one_post())["reasoning_effort"] == "xhigh"


def test_an_effort_object_sends_only_its_level(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: {"effort": "high", "summary": "auto"}` to `azure_ai/deepseek-v4-flash`.

    Foundry takes only a level name (`_SendReasoningEffort`, `transformation.py:293`), so the body must carry
    `reasoning_effort: "high"` and the other field is dropped with a warning.
    """
    with foundry(gateway, _ok, model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort={"effort": "high", "summary": "auto"})
        assert response.status_code == 200, response.text
        assert body_of(rig.one_post())["reasoning_effort"] == "high"


def test_the_base_model_row_decides_the_reasoning_levels_of_a_custom_deployment_name(gateway: Gateway) -> None:
    """The deployment is `azure_ai/my-deepseek` with `base_model: azure_ai/deepseek-v4-flash`; the caller sends `low`.

    Levels are read from the `base_model` row first (`_catalog_reasoning_efforts`, `transformation.py:217`), so
    the caller must get HTTP 400 naming `['none', 'high', 'max']` even though `my-deepseek` has no row.
    UNVERIFIED: that the router passes `base_model` from `litellm_params` into `map_openai_params`.
    """
    with foundry(gateway, _ok, model="azure_ai/my-deepseek", base_model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat(reasoning_effort="low")
        assert response.status_code == 400, response.text
        assert "['none', 'high', 'max']" in error_message(response)


def test_reasoning_tokens_outside_completion_tokens_are_folded_in(gateway: Gateway) -> None:
    """Foundry reports `prompt_tokens: 10`, `completion_tokens: 5`, `total_tokens: 25` and `reasoning_tokens: 10`.

    The reasoning was counted outside `completion_tokens`, so the caller must get `completion_tokens: 15` and
    `total_tokens: 25`, keeping total = prompt + completion (`fold_reasoning_tokens_into_completion`).
    """
    usage: Final = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    with foundry(gateway, always(chat_reply("pong", usage=usage)), model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        got: Final = response.json()["usage"]
        assert (got["prompt_tokens"], got["completion_tokens"], got["total_tokens"]) == (10, 15, 25)


def test_reasoning_tokens_already_inside_completion_tokens_are_not_double_counted(gateway: Gateway) -> None:
    """Foundry reports `prompt_tokens: 10`, `completion_tokens: 15`, `total_tokens: 25` and `reasoning_tokens: 10`.

    The total already adds up, so the caller must get `completion_tokens: 15` unchanged.
    """
    usage: Final = {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    with foundry(gateway, always(chat_reply("pong", usage=usage)), model=_DEEPSEEK_FLASH) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["completion_tokens"] == 15


def test_a_422_naming_extra_parameters_is_retried_without_them(gateway: Gateway) -> None:
    """The caller sends `seed: 7`; Foundry answers 422 `Extra parameters ['seed'] are not allowed when extra-parameters is not set or set to be 'error'. Set extra-parameters to 'pass-through'...` and then 200.

    The config retries without the named parameters (`_drop_extra_params_from_request_data`): Foundry must see
    two POSTs, the second without `seed`, and the caller gets HTTP 200.
    """
    rejection: Final = azure_error(
        422,
        "extra_parameters_not_allowed",
        "Extra parameters ['seed'] are not allowed when extra-parameters is not set or set to be 'error'. "
        "Set extra-parameters to 'pass-through' to pass to the model.",
    )
    with foundry(gateway, failing_then([rejection], chat_reply("pong")), model=_MISTRAL) as rig:
        response: Final = rig.chat(seed=7, num_retries=0)
        assert response.status_code == 200, response.text
        posts: Final = rig.posts()
        assert len(posts) == 2
        assert body_of(posts[0])["seed"] == 7
        assert "seed" not in body_of(posts[1])


def test_a_422_on_a_tool_level_extra_field_drops_only_that_field(gateway: Gateway) -> None:
    """The caller sends a tool with `strict: true`; Foundry answers 422 `Extra inputs are not permitted` at `tools[0].strict`, then 200.

    The retry drops only `strict` from each tool (`_drop_tool_level_extra_fields`, `transformation.py:707`), even
    without drop_params: the second POST must keep the tool name and parameters and have no `strict`.
    """
    strict_tool: Final = {"type": "function", "function": WEATHER_TOOL["function"], "strict": True}
    rejection: Final = azure_error(422, "invalid_payload", "tools[0].strict: Extra inputs are not permitted")
    with foundry(gateway, failing_then([rejection], chat_reply("pong")), model=_MISTRAL) as rig:
        response: Final = rig.chat(tools=[strict_tool], num_retries=0)
        assert response.status_code == 200, response.text
        posts: Final = rig.posts()
        assert len(posts) == 2
        retried_tool: Final = body_of(posts[1])["tools"][0]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]
        assert "strict" not in retried_tool
        assert retried_tool["function"]["name"] == "get_weather"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


def test_foundry_claude_on_chat_completions_posts_anthropic_messages(gateway: Gateway) -> None:
    """An `azure_ai/claude-sonnet-4-5` deployment at a resource root serves `/v1/chat/completions`.

    The chat call is translated to the Anthropic Messages API: Foundry must see `POST /anthropic/v1/messages`
    whose body has `model: claude-sonnet-4-5` and the user turn as an Anthropic message, and the caller gets a
    chat completion whose content is `pong`.
    """
    prompt: Final = marker()
    with foundry(gateway, _anthropic, model=_CLAUDE) as rig:
        response: Final = rig.chat(prompt, max_tokens=64)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["message"]["content"] == "pong"
        request: Final = rig.one_post()
        assert path_of(request) == "/anthropic/v1/messages"
        body: Final = body_of(request)
        assert body["model"] == "claude-sonnet-4-5"
        assert body["messages"][0]["role"] == "user"  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportCallIssue, reportArgumentType]


def test_foundry_claude_on_chat_completions_drops_stream_options(gateway: Gateway) -> None:
    """The caller streams with `stream_options: {"include_usage": true}` to Foundry Claude via chat completions.

    Azure's Anthropic endpoint rejects `stream_options`, `extra_body` and `max_retries`
    (`AzureAnthropicConfig.transform_request`, `anthropic/transformation.py:150`), so the body Foundry receives
    must contain none of them. The answer here is not streamed; only the request body is checked.
    """
    with foundry(gateway, _anthropic, model=_CLAUDE) as rig:
        rig.chat(max_tokens=64, stream_options={"include_usage": True})
        body: Final = body_of(rig.one_post())
        assert "stream_options" not in body, body
        assert "max_retries" not in body, body


def test_foundry_claude_on_chat_completions_sends_the_key_as_api_key(gateway: Gateway) -> None:
    """A keyed Foundry Claude deployment serves `/v1/chat/completions`.

    `AzureAnthropicConfig.validate_environment` uses Azure auth (`api-key` from `base_validate_azure_environment`),
    so Foundry must receive `api-key: synthetic-foundry-key`. UNVERIFIED: whether `x-api-key` is also sent with
    the same value, since the Anthropic headers are merged underneath (`anthropic/transformation.py:116`).
    """
    with foundry(gateway, _anthropic, model=_CLAUDE) as rig:
        assert rig.chat(max_tokens=64).status_code == 200
        assert rig.one_post().headers.get("api-key") == FOUNDRY_KEY


def test_foundry_claude_on_messages_with_only_an_entra_token_sends_a_bearer(gateway: Gateway) -> None:
    """A Foundry Claude deployment with no key and `azure_ad_token: synthetic-entra-access-token` serves `/v1/messages`.

    Rust now resolves Entra tokens the way Python does (#51721): Foundry must receive
    `Authorization: Bearer synthetic-entra-access-token` on `POST /anthropic/v1/messages` and no `x-api-key`.
    """
    with foundry(gateway, _anthropic, model=_CLAUDE, api_key=None, azure_ad_token=ENTRA_TOKEN) as rig:
        response: Final = rig.messages()
        assert response.status_code == 200, response.text
        request: Final = rig.one_post()
        assert path_of(request) == "/anthropic/v1/messages"
        assert request.headers.get("authorization") == f"Bearer {ENTRA_TOKEN}"
        assert "x-api-key" not in request.headers, request.headers


def test_foundry_claude_on_messages_sends_one_credential_only(gateway: Gateway) -> None:
    """A keyed Foundry Claude deployment serves `/v1/messages`.

    Foundry must receive `x-api-key: synthetic-foundry-key` and neither `api-key` nor `Authorization`, so the
    key is never sent twice and the caller's proxy key never leaks.
    """
    with foundry(gateway, _anthropic, model=_CLAUDE) as rig:
        assert rig.messages().status_code == 200
        headers: Final = rig.one_post().headers
        assert headers.get("x-api-key") == FOUNDRY_KEY
        assert "api-key" not in headers, headers
        assert "authorization" not in headers, headers


def test_a_caller_api_key_header_replaces_the_foundry_operator_key_on_messages(gateway: Gateway) -> None:
    """The caller authenticates with its proxy bearer and also sends `api-key: synthetic-caller-foundry-key` on `/v1/messages`.

    Rust replaces the operator key with the caller's key and sends it as `x-api-key` (#51721, "let a caller
    api-key replace the azure_ai operator key"): Foundry must receive `x-api-key: synthetic-caller-foundry-key`
    and the operator key nowhere. UNVERIFIED: that the proxy forwards a caller `api-key` header to `azure_ai`
    on `/v1/messages` without extra forwarding config.
    """
    with foundry(gateway, _anthropic, model=_CLAUDE) as rig:
        response: Final = rig.messages(headers={"api-key": "synthetic-caller-foundry-key"})
        assert response.status_code == 200, response.text
        headers: Final = rig.one_post().headers
        assert headers.get("x-api-key") == "synthetic-caller-foundry-key"
        assert all(FOUNDRY_KEY != value for value in headers.values()), headers


@pytest.mark.parametrize(
    ("base_path", "expected"),
    [
        ("/gateway/anthropic", "/gateway/anthropic/v1/messages"),
        ("/gateway", "/gateway/anthropic/v1/messages"),
        ("/anthropic/v1", "/anthropic/v1/messages"),
        ("/gateway/anthropic/v1/messages/", "/gateway/anthropic/v1/messages"),
    ],
    ids=["prefixed_anthropic_root", "prefix_without_anthropic", "anthropic_v1", "full_url_trailing_slash"],
)
def test_foundry_claude_api_base_join_keeps_a_path_prefix(gateway: Gateway, base_path: str, expected: str) -> None:
    """Foundry Claude `api_base` values with a gateway path prefix in front of the resource.

    `azure_anthropic_messages_url` cuts at the first `/anthropic` in the path and appends
    `anthropic/v1/messages` (`anthropic/transformation.py:14`), so `/v1/messages` must post to `expected`:
    a prefix is kept, `/anthropic/v1` is not doubled and a full URL with a trailing slash is used as is.
    The msgprov suite covers the four bases without a prefix.
    """
    with foundry(gateway, _anthropic, model=_CLAUDE, base_path=base_path) as rig:
        response: Final = rig.messages()
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == expected


def test_a_non_claude_foundry_model_on_messages_is_bridged_to_chat_completions(gateway: Gateway) -> None:
    """`/v1/messages` on `azure_ai/mistral-large-3`.

    A non Claude Foundry model has no native Messages config, so the call goes through the chat bridge: Foundry
    must see `POST /chat/completions` with an OpenAI chat body, and the caller gets an Anthropic message whose
    first text block is `pong`. UNVERIFIED: that `azure_ai` non Claude models take the chat bridge
    (`litellm/messages/dispatch.py`) rather than refusing.
    """
    with foundry(gateway, _ok, model=_MISTRAL) as rig:
        response: Final = rig.messages()
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == "/chat/completions"
        assert response.json()["content"][0]["text"] == "pong"
