"""Contract: Anthropic's newer request and response features pass through the Rust-native `/v1/messages` route.

Since #51119 `POST /v1/messages` on an `anthropic/` deployment runs only on Rust
(`litellm/rust/crates/inference-messages`, `litellm/rust/crates/llms/src/anthropic/messages/transformation.rs`).
The route is a near passthrough: the Python host hands Rust the body fields named in `_BODY_FIELDS`
(`litellm/rust_bridge/host/routes/messages.py:131`, which includes `mcp_servers`, `container`,
`context_management`, `service_tier` and `inference_geo`), and Rust serializes them back as sent through
`MessagesOptionalParams` (`llms-types/src/formats/messages/request.rs:404`). Rust derives only a small set of
betas from the body (`feature_betas`, `transformation.rs:503`): context management, compaction, structured
outputs, fast mode, threads, thinking display, tool search and the advisor tool. Every other beta, including
`mcp-client`, `files-api`, `skills`, `context-1m`, `token-efficient-tools` and `extended-cache-ttl`, is the
caller's job and must be forwarded as sent. On the way back the message body is relayed with its newer fields:
`container`, `stop_reason` values such as `pause_turn`, `refusal` and `model_context_window_exceeded`, and usage
counters such as `service_tier` and `server_tool_use.web_fetch_requests`.

The peer is the round 1 Anthropic peer (`providers/dec1_anthropic_support.py`), so the recorded upstream request
is read back for every request-side test.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply
from integration.providers.dec1_anthropic2_support import (
    CITED_DOCUMENT,
    CLEAR_TOOL_USES,
    COMPACT_BETA,
    CONTAINER_ID,
    CONTEXT_1M_BETA,
    CONTEXT_MANAGEMENT_BETA,
    EXTENDED_CACHE_TTL_BETA,
    FILES_BETA,
    MCP_BETA,
    MCP_TOKEN,
    MEMORY_TOOL,
    SEARCH_RESULT,
    SKILL_ID,
    SKILLS_BETA,
    TOKEN_EFFICIENT_BETA,
    WEB_FETCH_BETA,
    WEB_FETCH_TOOL,
    body_of,
    mcp_server,
    messages_body,
    native,
    native_stream,
    replying,
    sse_events,
)
from integration.providers.dec1_anthropic_support import (
    MESSAGE_STOP,
    Call,
    betas,
    block_delta,
    block_start,
    block_stop,
    default_respond,
    deployment,
    marker,
    message_delta,
    message_start,
    peer,
    stream,
    streaming,
    usage_block,
)
from pydantic import JsonValue


def _user(*blocks: JsonValue) -> list[JsonValue]:
    return [{"role": "user", "content": [{"type": "text", "text": marker()}, *blocks]}]


def test_mcp_servers_are_forwarded_with_the_authorization_token(gateway: Gateway) -> None:
    """The caller sends `mcp_servers: [{"type": "url", "url": <url>, "name": "dec1-docs", "authorization_token":
    "dec1-synthetic-mcp-token", "tool_configuration": {"allowed_tools": ["search"]}}]`.

    The upstream body must carry the identical list. The token must reach Anthropic even though Rust redacts it in
    its debug log (`without_mcp_tokens`, `inference-messages/src/handler.rs:1233`); a rewrite that redacts the
    wire body instead would break every authenticated MCP server.
    """
    server: Final = mcp_server(authorization_token=MCP_TOKEN, tool_configuration={"allowed_tools": ["search"]})
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), mcp_servers=[server])))
        assert provider.only().body.get("mcp_servers") == [server]


def test_mcp_servers_add_no_beta_on_their_own(gateway: Gateway) -> None:
    """The caller sends `mcp_servers` with no `anthropic-beta` header.

    Neither the Rust `feature_betas` nor Python's native `_update_headers_with_anthropic_beta` derives
    `mcp-client-2025-04-04` from `mcp_servers` on this route (unlike chat), so the upstream request must carry no
    `anthropic-beta` header. Pinned so a port does not start adding a beta the Python route never sent.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), mcp_servers=[mcp_server()])))
        assert "anthropic-beta" not in provider.only().headers


def test_a_caller_mcp_client_beta_is_forwarded_with_the_servers(gateway: Gateway) -> None:
    """The Anthropic SDK sends `anthropic-beta: mcp-client-2025-04-04` next to `mcp_servers`.

    The upstream request must carry that beta and the servers, both unchanged.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(native(gateway, messages_body(model, mcp_servers=[mcp_server()]), headers={"anthropic-beta": MCP_BETA}))
        call: Final = provider.only()
        assert betas(call) == {MCP_BETA}
        assert call.body.get("mcp_servers") == [mcp_server()]


def test_a_container_with_skills_is_forwarded_as_sent(gateway: Gateway) -> None:
    """The caller sends `container: {"id": "container_dec1reuse", "skills": [{"type": "custom", "skill_id":
    "skill_dec1report", "version": "latest"}]}` with `anthropic-beta: skills-2025-10-02`.

    The upstream body must carry the identical `container` and the header exactly the caller's skills beta. Rust
    keeps `container` as an opaque JSON value (`request.rs:434`), so nested skill fields must not be reshaped.
    """
    container: Final[dict[str, JsonValue]] = {
        "id": CONTAINER_ID,
        "skills": [{"type": "custom", "skill_id": SKILL_ID, "version": "latest"}],
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(native(gateway, messages_body(model, container=container), headers={"anthropic-beta": SKILLS_BETA}))
        call: Final = provider.only()
        assert call.body.get("container") == container
        assert betas(call) == {SKILLS_BETA}


def test_a_container_id_string_is_forwarded_as_sent(gateway: Gateway) -> None:
    """The caller reuses a container by sending `container: "container_dec1reuse"`. The upstream body must carry
    that exact string.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), container=CONTAINER_ID)))
        assert provider.only().body.get("container") == CONTAINER_ID


def test_a_container_upload_block_is_forwarded_without_adding_a_tool(gateway: Gateway) -> None:
    """The user turn holds a text block and `{"type": "container_upload", "file_id": "file_dec1csv"}` with no
    `tools`.

    Unlike chat, this route adds no code execution tool: the upstream body must have no `tools` key and the
    user turn must hold the `container_upload` block unchanged.
    """
    upload: Final[dict[str, JsonValue]] = {"type": "container_upload", "file_id": "file_dec1csv"}
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), messages=_user(upload))))
        call: Final = provider.only()
        assert "tools" not in call.body
        assert upload in list_value(object_value(list_value(call.body["messages"])[0])["content"])


def test_a_file_source_document_is_forwarded_with_the_caller_files_beta(gateway: Gateway) -> None:
    """The user turn holds `{"type": "document", "source": {"type": "file", "file_id": "file_dec1pdf"},
    "citations": {"enabled": true}}` and the caller sends `anthropic-beta: files-api-2025-04-14`.

    The block must arrive unchanged and the header must hold exactly the files beta: Rust does not derive it from
    a file source, so it must not be dropped or duplicated either.
    """
    document: Final[dict[str, JsonValue]] = {
        "type": "document",
        "source": {"type": "file", "file_id": "file_dec1pdf"},
        "citations": {"enabled": True},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(native(gateway, messages_body(model, messages=_user(document)), headers={"anthropic-beta": FILES_BETA}))
        call: Final = provider.only()
        assert document in list_value(object_value(list_value(call.body["messages"])[0])["content"])
        assert betas(call) == {FILES_BETA}


def test_a_document_with_citations_title_and_context_is_forwarded(gateway: Gateway) -> None:
    """The user turn holds a text-source `document` with `title`, `context` and `citations: {"enabled": true}`.

    Rust parses content blocks into typed `ContentBlock`s; the block must still arrive with every key and value
    as sent, since a dropped `citations` silently turns citations off.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), messages=_user(dict(CITED_DOCUMENT)))))
        sent: Final = list_value(object_value(list_value(provider.only().body["messages"])[0])["content"])
        assert dict(CITED_DOCUMENT) in sent


def test_a_search_result_block_in_the_user_turn_is_forwarded(gateway: Gateway) -> None:
    """The user turn holds a `search_result` block with `source`, `title`, text `content` and citations enabled.

    It must arrive unchanged. This is the native twin of the chat case that drops the block today.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), messages=_user(dict(SEARCH_RESULT)))))
        sent: Final = list_value(object_value(list_value(provider.only().body["messages"])[0])["content"])
        assert dict(SEARCH_RESULT) in sent


def test_a_search_result_inside_a_tool_result_is_forwarded(gateway: Gateway) -> None:
    """The history holds an assistant `tool_use` of `kb_lookup` and a user `tool_result` whose content is one
    `search_result` block, with the `kb_lookup` tool declared.

    The upstream `tool_result.content` must be `[<search_result>]` unchanged.
    """
    tool_result: Final[dict[str, JsonValue]] = {
        "type": "tool_result",
        "tool_use_id": "toolu_dec1kb",
        "content": [dict(SEARCH_RESULT)],
    }
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_dec1kb", "name": "kb_lookup", "input": {}}],
        },
        {"role": "user", "content": [tool_result]},
    ]
    kb_tool: Final[dict[str, JsonValue]] = {
        "name": "kb_lookup",
        "description": "Search the synthetic knowledge base",
        "input_schema": {"type": "object", "properties": {}},
    }
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), messages=history, tools=[kb_tool])))
        sent: Final = list_value(object_value(list_value(provider.only().body["messages"])[2])["content"])
        assert object_value(sent[0]).get("content") == [dict(SEARCH_RESULT)]


def test_the_memory_tool_is_forwarded_without_a_derived_beta(gateway: Gateway) -> None:
    """The caller sends `tools: [{"type": "memory_20250818", "name": "memory"}]` and no beta header.

    The tool must arrive unchanged, and no `anthropic-beta` header may be sent: on this route only
    `context_management` derives `context-management-2025-06-27`, not the memory tool (chat does add it).
    UNVERIFIED: whether Rust's typed `MessagesTool` keeps an unknown hosted tool type byte for byte; the
    `Recognized` wrapper is meant to.
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), tools=[dict(MEMORY_TOOL)])))
        call: Final = provider.only()
        assert call.body.get("tools") == [dict(MEMORY_TOOL)]
        assert "anthropic-beta" not in call.headers


def test_the_web_fetch_tool_is_forwarded_with_the_caller_beta(gateway: Gateway) -> None:
    """The caller sends the web fetch tool with `max_uses: 2` and `anthropic-beta: web-fetch-2025-09-10`.

    Both must arrive unchanged, and no second web fetch beta may be appended.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(
            native(
                gateway, messages_body(model, tools=[dict(WEB_FETCH_TOOL)]), headers={"anthropic-beta": WEB_FETCH_BETA}
            )
        )
        call: Final = provider.only()
        assert call.body.get("tools") == [dict(WEB_FETCH_TOOL)]
        assert call.headers.get("anthropic-beta", "").count(WEB_FETCH_BETA) == 1


def test_compact_and_clear_edits_together_add_both_betas(gateway: Gateway) -> None:
    """The caller sends `context_management.edits` holding a `compact_20260112` edit and a
    `clear_tool_uses_20250919` edit.

    `context_management_betas` (`transformation.rs:539`) must add both `compact-2026-01-12` and
    `context-management-2025-06-27`, and the edits must arrive in their original order.
    """
    edits: Final[list[JsonValue]] = [
        {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}},
        dict(CLEAR_TOOL_USES),
    ]
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), context_management={"edits": edits})))
        call: Final = provider.only()
        assert call.body.get("context_management") == {"edits": edits}
        assert {COMPACT_BETA, CONTEXT_MANAGEMENT_BETA} <= betas(call)


def test_openai_list_context_management_is_mapped_to_edits_on_this_route_too(gateway: Gateway) -> None:
    """The caller sends OpenAI's list form `context_management: [{"type": "compaction", "compact_threshold":
    150000}]` to `/v1/messages`.

    Rust ports `map_openai_context_management_to_anthropic` (`transformation.rs:761`), so the upstream body must
    carry `{"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]}` with
    the `compact-2026-01-12` beta, the same as Python's native transform.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(
            native(
                gateway, messages_body(model, context_management=[{"type": "compaction", "compact_threshold": 150000}])
            )
        )
        call: Final = provider.only()
        assert call.body.get("context_management") == {
            "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]
        }
        assert COMPACT_BETA in betas(call)


@pytest.mark.parametrize("tier", ["auto", "standard_only", "priority"])
def test_service_tier_is_forwarded_without_mapping(gateway: Gateway, tier: str) -> None:
    """The caller sends `service_tier` on `/v1/messages`.

    Anthropic is in `MESSAGES_SERVICE_TIER_PROVIDERS`, so the upstream body must carry the caller's value as sent:
    unlike chat, this route neither maps OpenAI names nor refuses `priority`
    (`inference-messages/src/handler.rs:564`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider), service_tier=tier)))
        assert provider.only().body.get("service_tier") == tier


def test_newer_feature_betas_from_the_caller_are_forwarded_together(gateway: Gateway) -> None:
    """Claude Code style callers send `anthropic-beta: context-1m-2025-08-07,token-efficient-tools-2025-02-19,
    extended-cache-ttl-2025-04-11`.

    The upstream header must hold exactly those three values, each once, even though the proxy derives none of
    them. Losing `context-1m` caps the conversation at 200k tokens.
    """
    caller: Final = f"{CONTEXT_1M_BETA},{TOKEN_EFFICIENT_BETA},{EXTENDED_CACHE_TTL_BETA}"
    with peer() as provider, gateway.scenario() as scenario:
        body_of(native(gateway, messages_body(deployment(scenario, provider)), headers={"anthropic-beta": caller}))
        call: Final = provider.only()
        assert betas(call) == {CONTEXT_1M_BETA, TOKEN_EFFICIENT_BETA, EXTENDED_CACHE_TTL_BETA}
        assert len([value for value in call.headers["anthropic-beta"].split(",") if value.strip()]) == 3


def test_caller_betas_and_a_derived_beta_share_one_header(gateway: Gateway) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07` and a `clear_tool_uses_20250919` edit.

    The upstream header must hold both `context-1m-2025-08-07` and the derived `context-management-2025-06-27`
    (`merge_beta_headers`), so a derived beta never replaces the caller's.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(
            native(
                gateway,
                messages_body(model, context_management={"edits": [dict(CLEAR_TOOL_USES)]}),
                headers={"anthropic-beta": CONTEXT_1M_BETA},
            )
        )
        assert betas(provider.only()) == {CONTEXT_1M_BETA, CONTEXT_MANAGEMENT_BETA}


def test_one_hour_cache_control_on_a_tool_and_a_block_is_forwarded(gateway: Gateway) -> None:
    """The caller marks a tool and a user text block with `cache_control: {"type": "ephemeral", "ttl": "1h"}`.

    Both must arrive with `ttl: "1h"`: on the direct Anthropic leg the proxy keeps the TTL (it is only reduced to
    its type for strict non-Anthropic hosts, `common_utils.py:2870`).
    """
    cached_tool: Final[dict[str, JsonValue]] = {
        "name": "kb_lookup",
        "input_schema": {"type": "object", "properties": {}},
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    block: Final[dict[str, JsonValue]] = {
        "type": "text",
        "text": marker(),
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(
            native(gateway, messages_body(model, tools=[cached_tool], messages=[{"role": "user", "content": [block]}]))
        )
        call: Final = provider.only()
        assert object_value(list_value(call.body["tools"])[0]).get("cache_control") == {
            "type": "ephemeral",
            "ttl": "1h",
        }
        assert list_value(object_value(list_value(call.body["messages"])[0])["content"]) == [block]


def test_the_response_container_is_relayed(gateway: Gateway) -> None:
    """The provider answers with a top-level `container: {"id": "container_dec1reuse", "expires_at":
    "2026-11-17T09:30:00Z"}`.

    The caller's body must carry the same `container` (`MessagesResponse.container`,
    `llms-types/src/formats/messages/response.rs:25`).
    """
    container: Final[dict[str, JsonValue]] = {"id": CONTAINER_ID, "expires_at": "2026-11-17T09:30:00Z"}
    with peer(replying(container=container)) as provider, gateway.scenario() as scenario:
        body: Final = body_of(native(gateway, messages_body(deployment(scenario, provider))))
        assert body.get("container") == container


def test_newer_usage_counters_are_relayed_unchanged(gateway: Gateway) -> None:
    """The provider reports `usage` with `service_tier: "priority"`, `server_tool_use: {"web_search_requests": 1,
    "web_fetch_requests": 2}` and a `cache_creation` breakdown `{"ephemeral_5m_input_tokens": 0,
    "ephemeral_1h_input_tokens": 4}`.

    The caller's `usage` must hold all three exactly as sent; unlike chat, this route never drops a counter it does
    not price.
    """
    usage: Final = usage_block(
        service_tier="priority",
        server_tool_use={"web_search_requests": 1, "web_fetch_requests": 2},
        cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4},
        cache_creation_input_tokens=4,
    )
    with peer(replying(usage=usage)) as provider, gateway.scenario() as scenario:
        caller_usage: Final = object_value(
            body_of(native(gateway, messages_body(deployment(scenario, provider))))["usage"]
        )
        assert caller_usage.get("service_tier") == "priority"
        assert caller_usage.get("server_tool_use") == {"web_search_requests": 1, "web_fetch_requests": 2}
        assert caller_usage.get("cache_creation") == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4}


@pytest.mark.parametrize("stop_reason", ["pause_turn", "refusal", "model_context_window_exceeded"])
def test_newer_stop_reasons_are_relayed_unchanged(gateway: Gateway, stop_reason: str) -> None:
    """The provider ends the message with each newer `stop_reason`.

    The caller's body must carry the same `stop_reason` and `stop_sequence: null`. `pause_turn` tells the client
    to resend the turn so a long server tool run can continue; mapping it to anything else ends the run early.
    """
    with peer(replying(stop_reason=stop_reason)) as provider, gateway.scenario() as scenario:
        body: Final = body_of(native(gateway, messages_body(deployment(scenario, provider))))
        assert body.get("stop_reason") == stop_reason
        assert body.get("stop_sequence") is None


def test_mcp_tool_use_and_result_blocks_are_relayed(gateway: Gateway) -> None:
    """The provider answers an `mcp_tool_use` block (`id: mcptoolu_dec1`, `name: search`, `server_name:
    dec1-docs`), its `mcp_tool_result` and a text block.

    The caller's `content` must be those three blocks in the same order and shape.
    """
    content: Final[list[JsonValue]] = [
        {
            "type": "mcp_tool_use",
            "id": "mcptoolu_dec1",
            "name": "search",
            "server_name": "dec1-docs",
            "input": {"q": "blue"},
        },
        {
            "type": "mcp_tool_result",
            "tool_use_id": "mcptoolu_dec1",
            "is_error": False,
            "content": [{"type": "text", "text": "widget: blue"}],
        },
        {"type": "text", "text": "It is blue."},
    ]
    with peer(replying(content)) as provider, gateway.scenario() as scenario:
        body: Final = body_of(
            native(gateway, messages_body(deployment(scenario, provider), mcp_servers=[mcp_server()]))
        )
        assert body.get("content") == content


def test_a_web_fetch_result_block_is_relayed(gateway: Gateway) -> None:
    """The provider answers a `server_tool_use` web_fetch call, its `web_fetch_tool_result` holding a text
    document, and a text answer.

    The caller's `content` must be those three blocks unchanged; Rust must not flatten the fetched document the way
    it flattens replayed unencrypted web search results in requests (`common_utils.rs:808`).
    """
    content: Final[list[JsonValue]] = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_dec1fetch",
            "name": "web_fetch",
            "input": {"url": "https://kb.dec1.invalid/7"},
        },
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_dec1fetch",
            "content": {
                "type": "web_fetch_result",
                "url": "https://kb.dec1.invalid/7",
                "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "blue"}},
            },
        },
        {"type": "text", "text": "Blue."},
    ]
    with peer(replying(content)) as provider, gateway.scenario() as scenario:
        body: Final = body_of(native(gateway, messages_body(deployment(scenario, provider))))
        assert body.get("content") == content


def test_a_streamed_pause_turn_and_server_tool_usage_are_relayed(gateway: Gateway) -> None:
    """The caller streams; the provider sends `message_start`, a `server_tool_use` web_search block, then
    `message_delta` with `stop_reason: pause_turn` and `usage: {"output_tokens": 7, "server_tool_use":
    {"web_search_requests": 1}}`, then `message_stop`.

    The caller's `message_delta` event must carry `delta.stop_reason: "pause_turn"` and the same
    `usage.server_tool_use`, and the stream must end with `message_stop`.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "server_tool_use", "id": "srvtoolu_dec1s", "name": "web_search", "input": {}}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "dec1"}'}),
        block_stop(0),
        message_delta("pause_turn", usage={"output_tokens": 7, "server_tool_use": {"web_search_requests": 1}}),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, text = native_stream(gateway, messages_body(deployment(scenario, provider)))
        events: Final = sse_events(text)
        deltas: Final = [data for name, data in events if name == "message_delta"]
        assert len(deltas) == 1
        assert object_value(deltas[0]["delta"]).get("stop_reason") == "pause_turn"
        assert object_value(deltas[0]["usage"]).get("server_tool_use") == {"web_search_requests": 1}
        assert events[-1][0] == "message_stop"


def test_a_streamed_message_relays_the_rate_limit_and_request_id_headers(gateway: Gateway) -> None:
    """The caller streams; the provider's event stream carries `request-id: req_dec1native` and
    `anthropic-ratelimit-tokens-remaining: 79000`.

    The caller's stream response must carry `llm_provider-request-id: req_dec1native` and
    `llm_provider-anthropic-ratelimit-tokens-remaining: 79000`, as the Python route did through the stream
    wrapper's `_hidden_params`. UNVERIFIED on the Rust route: the non-streamed twin is a known skipped bug in
    `test_dec1_messages_response.py`; this pins the streamed path the existing index assumes works.
    """

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        reply: Final = stream(
            message_start(call.body),
            block_start(0, {"type": "text", "text": ""}),
            block_delta(0, {"type": "text_delta", "text": "ok"}),
            block_stop(0),
            message_delta(),
            MESSAGE_STOP,
        )
        return Reply(
            content_type=reply.content_type,
            chunks=reply.chunks,
            headers={"request-id": "req_dec1native", "anthropic-ratelimit-tokens-remaining": "79000"},
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        headers, _ = native_stream(gateway, messages_body(deployment(scenario, provider)))
        assert headers.get("llm_provider-request-id") == "req_dec1native"
        assert headers.get("llm_provider-anthropic-ratelimit-tokens-remaining") == "79000"
