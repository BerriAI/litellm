"""`POST /v1/messages` served by a Vertex AI Anthropic partner model (`vertex_ai/claude-*`).

`VertexAIPartnerModelsAnthropicMessagesConfig`
(`litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/messages/transformation.py`) sends the
Anthropic body to `POST {api_base}/v1/projects/{project}/locations/{location}/publishers/anthropic/
models/{model}:rawPredict` (`:streamRawPredict?alt=sse` when streaming) with
`Authorization: Bearer <access token>` from the deployment's service account. The body drops
`model`, keeps `stream`, gains `anthropic_version` (`vertex-2023-10-16` for the caller versions
"", 2023-01-01 and 2023-06-01, the caller's own value for 2026-10-21, and an UnsupportedParamsError for
anything else unless `drop_params` is set), rebuilds `system` as blocks and loses
`cache_control.scope`. The `anthropic-version` header itself is removed in `sign_request`, and
`anthropic-beta` is filtered through the `vertex_ai` row of `anthropic_beta_headers_config.json`.

Access tokens come from the shared scripted upstream's `/_oauth/token`, so the owned peer sees only
model calls. Vertex count tokens has its own wire suite (`providers/test_vertex_partner_count_tokens_wire.py`)
and is not repeated here. Vertex is on Python today; the Rust `inference-messages` crate already
carries a Vertex config, so these tests are the parity contract for that port.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT, Rig, anthropic_error, sse_events
from integration.messages_endpoint.dec1_msgprov_support import (
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    TOOL,
    VERTEX_BACKEND,
    VERTEX_LOCATION,
    VERTEX_PROJECT,
    VERTEX_TOKEN,
    VERTEX_VERSION,
    anthropic_error_reply,
    anthropic_json_reply,
    at,
    message_id,
    only_spend_row,
    path_of,
    provider_deployment,
    query_of,
    spend_of,
    unique_text,
    vertex_credentials,
    vertex_echo,
    vertex_path,
)
from pydantic import JsonValue

_RAW_PREDICT: Final = vertex_path() + ":rawPredict"
_STREAM_RAW_PREDICT: Final = vertex_path() + ":streamRawPredict"
_SUPPORTED_BETA: Final = "context-management-2025-06-27"
_UNKNOWN_BETA: Final = "made-up-future-beta-2099-01-01"
_BILLING_BLOCK: Final[dict[str, JsonValue]] = {
    "type": "text",
    "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli",
}


def _vertex(
    gateway: Gateway, respond: Callable[[Request], Reply] = vertex_echo, **params: JsonValue
) -> AbstractContextManager[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "vertex_project": VERTEX_PROJECT,
        "vertex_location": VERTEX_LOCATION,
        "vertex_credentials": vertex_credentials(gateway),
        **params,
    }
    return provider_deployment(gateway, respond, model=f"vertex_ai/{VERTEX_BACKEND}", api_key=None, **settings)


# ---------------------------------------------------------------- URL, method and auth


def test_a_vertex_claude_deployment_posts_to_raw_predict_with_the_access_token(gateway: Gateway) -> None:
    """The deployment receives one `POST /v1/projects/<project>/locations/us-east5/publishers/anthropic/
    models/claude-sonnet-4-5@20250929:rawPredict` with `Authorization: Bearer scripted-token`, the
    token minted from the service account, and no `x-api-key`."""
    with _vertex(gateway) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        call: Final = rig.one_call()
        assert (call.method, path_of(call)) == ("POST", _RAW_PREDICT)
        assert call.headers["authorization"] == f"Bearer {VERTEX_TOKEN}"
        assert "x-api-key" not in call.headers


def test_a_streaming_request_posts_to_stream_raw_predict_with_alt_sse(gateway: Gateway) -> None:
    """With `stream: true` the path ends in `:streamRawPredict` and the query is exactly `alt=sse`,
    without which Vertex answers a JSON array instead of SSE."""
    with _vertex(gateway) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        call: Final = rig.one_call()
        assert (path_of(call), query_of(call)) == (_STREAM_RAW_PREDICT, "alt=sse")


def test_the_global_location_is_used_in_the_path(gateway: Gateway) -> None:
    """`vertex_location: global` puts `locations/global` in the path; Claude models on Vertex are
    served from the global endpoint as well as from regions."""
    with _vertex(gateway, vertex_location="global") as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path(location="global") + ":rawPredict"


def test_a_missing_location_falls_back_to_us_central1(gateway: Gateway) -> None:
    """Without `vertex_location` (and no VERTEXAI_LOCATION env on the shared proxy) the path uses
    `locations/us-central1` (`get_vertex_region`). UNVERIFIED: that the shared proxy has no
    Vertex location env set."""
    with _vertex(gateway, vertex_location=None) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path(location="us-central1") + ":rawPredict"


def test_the_project_comes_from_the_service_account_when_vertex_project_is_unset(gateway: Gateway) -> None:
    """Without `vertex_project`, the project in the path is the `project_id` of the service account
    JSON in `vertex_credentials`."""
    other_project: Final = "synthetic-credentials-project"
    with _vertex(gateway, vertex_project=None, vertex_credentials=vertex_credentials(gateway, other_project)) as rig:
        assert rig.post(rig.body()).status_code == 200
        assert path_of(rig.one_call()) == vertex_path(project=other_project) + ":rawPredict"


# ---------------------------------------------------------------- body translation


@pytest.mark.parametrize("stream", [False, True])
def test_the_vertex_body_has_vertex_anthropic_version_and_no_model(gateway: Gateway, stream: bool) -> None:
    """A plain request reaches Vertex as exactly `{"max_tokens": 64, "messages": [...], "stream": <flag>,
    "anthropic_version": "vertex-2023-10-16"}`. Vertex takes the model from the path and rejects it in
    the body."""
    with _vertex(gateway) as rig:
        body: Final = rig.body(stream=stream)
        if stream:
            rig.stream(body)
        else:
            assert rig.post(body).status_code == 200
        assert rig.one_body() == {
            "max_tokens": 64,
            "messages": body["messages"],
            "stream": stream,
            "anthropic_version": VERTEX_VERSION,
        }


@pytest.mark.parametrize("caller_version", ["2023-01-01", "2023-06-01"])
def test_classic_anthropic_versions_are_sent_as_the_vertex_default(gateway: Gateway, caller_version: str) -> None:
    """A caller `anthropic-version` of 2023-01-01 or 2023-06-01 is served as
    `anthropic_version: vertex-2023-10-16` in the body, and no `anthropic-version` header reaches
    Vertex."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": caller_version}).status_code == 200
        call: Final = rig.one_call()
        assert JSON_OBJECT.validate_json(call.body)["anthropic_version"] == VERTEX_VERSION
        assert "anthropic-version" not in call.headers


def test_the_2026_10_21_version_is_passed_through_in_the_body(gateway: Gateway) -> None:
    """`anthropic-version: 2026-10-21`, which Vertex accepts in the body, is sent as
    `anthropic_version: "2026-10-21"` rather than rewritten to the Vertex default."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2026-10-21"}).status_code == 200
        assert rig.one_body()["anthropic_version"] == "2026-10-21"


def test_an_unsupported_anthropic_version_is_refused_without_a_vertex_call(gateway: Gateway) -> None:
    """`anthropic-version: 2024-02-02` has no Vertex equivalent, so the caller gets HTTP 400
    `invalid_request_error` naming `anthropic-version=2024-02-02` and Vertex is never called."""
    with _vertex(gateway) as rig:
        response: Final = rig.post(rig.body(), headers={"anthropic-version": "2024-02-02"})
        assert response.status_code == 400, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == "invalid_request_error"
        assert "anthropic-version=2024-02-02" in str(error["message"])
        assert rig.calls() == ()


def test_drop_params_serves_an_unsupported_version_as_the_vertex_default(gateway: Gateway) -> None:
    """With `drop_params: true` on the deployment, the same unsupported version is replaced by
    `vertex-2023-10-16` and the call succeeds."""
    with _vertex(gateway, drop_params=True) as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2024-02-02"}).status_code == 200
        assert rig.one_body()["anthropic_version"] == VERTEX_VERSION


def test_unknown_betas_are_dropped_and_supported_ones_forwarded(gateway: Gateway) -> None:
    """`anthropic-beta: context-management-2025-06-27,made-up-future-beta-2099-01-01` reaches Vertex as
    `anthropic-beta: context-management-2025-06-27`; a beta not in the `vertex_ai` row is removed
    because Vertex refuses unknown betas with a 400."""
    with _vertex(gateway) as rig:
        headers: Final = {"anthropic-beta": f"{_SUPPORTED_BETA},{_UNKNOWN_BETA}"}
        assert rig.post(rig.body(), headers=headers).status_code == 200
        assert rig.one_call().headers["anthropic-beta"] == _SUPPORTED_BETA


@pytest.mark.parametrize(
    ("caller_beta", "sent_beta"),
    [
        ("advanced-tool-use-2025-11-20", "tool-search-tool-2025-10-19"),
        ("files-api-2025-04-14", None),
    ],
    ids=["renamed", "dropped"],
)
def test_the_vertex_beta_row_renames_or_drops_a_beta(gateway: Gateway, caller_beta: str, sent_beta: str | None) -> None:
    """The `vertex_ai` row of `anthropic_beta_headers_config.json` maps `advanced-tool-use-2025-11-20` to
    Vertex's `tool-search-tool-2025-10-19` and `files-api-2025-04-14` to null, so the first is renamed
    and the second leaves no `anthropic-beta` header at all."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(), headers={"anthropic-beta": caller_beta}).status_code == 200
        assert rig.one_call().headers.get("anthropic-beta") == sent_beta


def test_a_web_search_tool_adds_the_web_search_beta(gateway: Gateway) -> None:
    """A `web_search_20250305` server tool makes the proxy add `web-search-2025-03-05` to
    `anthropic-beta` and keep the tool in the body. UNVERIFIED: that the `vertex_ai` beta row keeps
    `web-search-2025-03-05` after filtering."""
    tool: Final[dict[str, JsonValue]] = {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(tools=[tool])).status_code == 200
        call: Final = rig.one_call()
        assert "web-search-2025-03-05" in call.headers.get("anthropic-beta", "").split(","), dict(call.headers)
        assert JSON_OBJECT.validate_json(call.body)["tools"] == [tool]


def test_a_string_system_prompt_becomes_one_text_block(gateway: Gateway) -> None:
    """`system: "<text>"` reaches Vertex as `[{"type": "text", "text": "<text>"}]`."""
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(system="You are terse.")).status_code == 200
        assert rig.one_body()["system"] == [{"type": "text", "text": "You are terse."}]


def test_the_billing_header_block_is_stripped_from_system(gateway: Gateway) -> None:
    """A Claude Code `x-anthropic-billing-header:` system block is removed
    (`should_strip_billing_metadata` is true for Vertex), leaving the caller's other blocks in order."""
    keep: Final[dict[str, JsonValue]] = {"type": "text", "text": "Keep this instruction."}
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(system=[_BILLING_BLOCK, keep])).status_code == 200
        assert rig.one_body()["system"] == [keep]


def test_cache_control_scope_is_removed_and_the_rest_kept(gateway: Gateway) -> None:
    """`cache_control: {"type": "ephemeral", "scope": "global"}` reaches Vertex as
    `{"type": "ephemeral"}`; Vertex rejects the `scope` member."""
    prompt: Final = unique_text()
    content: Final[list[JsonValue]] = [
        {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "scope": "global"}}
    ]
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(messages=[{"role": "user", "content": content}])).status_code == 200
        assert at(rig.one_body(), "messages", 0, "content", 0, "cache_control") == {"type": "ephemeral"}


def test_tools_tool_choice_stop_sequences_and_metadata_are_forwarded(gateway: Gateway) -> None:
    """Native Anthropic fields need no translation on Vertex, so `tools`, `tool_choice`,
    `stop_sequences` and `metadata` arrive unchanged."""
    fields: Final[dict[str, JsonValue]] = {
        "tools": [TOOL],
        "tool_choice": {"type": "any"},
        "stop_sequences": ["###"],
        "metadata": {"user_id": "synthetic-user-01"},
    }
    with _vertex(gateway) as rig:
        assert rig.post(rig.body(**fields)).status_code == 200
        body: Final = rig.one_body()
        assert {key: body[key] for key in fields} == fields


# ---------------------------------------------------------------- response translation


def test_the_vertex_message_is_returned_with_the_public_model_name(gateway: Gateway) -> None:
    """Vertex's Anthropic message comes back unchanged except `model`, which is the proxy model name
    the caller asked for."""
    identity: Final = message_id()
    reply: Final = anthropic_json_reply(identity, model="claude-sonnet-4-5-20250929")
    with _vertex(gateway, lambda _r: reply) as rig:
        payload: Final = JSON_OBJECT.validate_json(rig.post(rig.body()).content)
        assert (payload["id"], payload["model"]) == (identity, rig.model)
        assert payload["content"] == [{"type": "text", "text": "pong"}]
        assert object_value(payload["usage"])["input_tokens"] == 11


def test_the_vertex_sse_stream_is_relayed_in_order(gateway: Gateway) -> None:
    """Vertex's six SSE events reach the caller in the same order, with `message_start.message.model`
    restamped to the proxy model name and the text delta unchanged."""
    with _vertex(gateway) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert [name for name, _data in events] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ], text
        assert object_value(events[0][1]["message"])["model"] == rig.model
        assert object_value(events[2][1]["delta"])["text"] == "hello stream"


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "anthropic_type"),
    [
        (400, "invalid_request_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
    ],
)
def test_a_vertex_error_is_answered_with_its_status_and_type(gateway: Gateway, status: int, anthropic_type: str) -> None:
    """Vertex answers errors in Anthropic's envelope. `_map_vertex_exception` keeps the status for 400,
    403, 404 and 429 and the caller gets the matching Anthropic type with the Vertex text."""
    message: Final = f"synthetic vertex {status}"
    with _vertex(gateway, lambda _r: anthropic_error_reply(status, anthropic_type, message)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == status, response.text
        error: Final = anthropic_error(response)
        assert error["type"] == anthropic_type
        assert message in str(error["message"])


def test_a_vertex_529_overload_is_answered_as_overloaded_error(gateway: Gateway) -> None:
    """Claude on Vertex sends 529 `overloaded_error` under load. `_map_vertex_exception` has no 529
    branch, so the generic status mapping decides. UNVERIFIED: the caller is expected to get 529
    `overloaded_error`, the same as the first-party route."""
    reply: Final = anthropic_error_reply(529, "overloaded_error", "Overloaded")
    with _vertex(gateway, lambda _r: reply) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 529, response.text
        assert anthropic_error(response)["type"] == "overloaded_error"


# ---------------------------------------------------------------- spend


def test_the_vertex_spend_row_names_vertex_ai_and_prices_the_tokens(gateway: Gateway) -> None:
    """One call writes one `anthropic_messages` row with `custom_llm_provider = "vertex_ai"`, 11 prompt
    and 7 completion tokens, and `spend = 11 * 0.001 + 7 * 0.002 = 0.025`."""
    with _vertex(gateway, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "vertex_ai")
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert spend_of(row) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)


def test_a_streamed_vertex_call_is_billed_once(gateway: Gateway) -> None:
    """A streamed call writes exactly one row with the same 0.025 spend as the non-streamed call."""
    with _vertex(gateway, **PRICES) as rig:
        rig.stream(rig.body(stream=True))
        assert spend_of(only_spend_row(rig)) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
