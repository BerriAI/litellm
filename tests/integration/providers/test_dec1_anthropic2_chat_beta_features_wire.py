"""Contract: newer Anthropic features requested through `POST /v1/chat/completions` reach Anthropic intact.

The chat route stays in Python (`AnthropicConfig`, `litellm/llms/anthropic/chat/transformation.py`, plus
`AnthropicModelInfo.validate_environment` in `litellm/llms/anthropic/common_utils.py`). A caller who speaks the
OpenAI shape can still use Anthropic's newer features: an OpenAI `mcp` tool becomes an Anthropic `mcp_servers`
entry with the `mcp-client-2025-04-04` beta, a `container` with skills adds `skills-2025-10-02`, file ids become
Files API sources with `files-api-2025-04-14`, `context_management` and the memory tool add
`context-management-2025-06-27`, the web fetch tool adds `web-fetch-2025-09-10`, `service_tier` is mapped to
Anthropic's tier names, and caller betas such as the 1M context window, token-efficient tools and extended cache
TTL are merged into one `anthropic-beta` header. Any rewrite of the chat translator must keep each of these, since
Anthropic answers 400 when a beta-gated field arrives without its beta.

Every test reads the request the owned peer recorded; each deployment is created for one test.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic2_support import (
    CITED_DOCUMENT,
    CLEAR_TOOL_USES,
    CODE_EXECUTION_BETA,
    CODE_EXECUTION_TOOL,
    COMPACT_BETA,
    CONTAINER_ID,
    CONTEXT_1M_BETA,
    CONTEXT_MANAGEMENT_BETA,
    EXTENDED_CACHE_TTL_BETA,
    FILES_BETA,
    MCP_BETA,
    MCP_NAME,
    MCP_TOKEN,
    MCP_URL,
    MEMORY_TOOL,
    SEARCH_RESULT,
    SKILL_ID,
    SKILLS_BETA,
    TOKEN_EFFICIENT_BETA,
    WEB_FETCH_BETA,
    WEB_FETCH_TOOL,
    mcp_server,
)
from integration.providers.dec1_anthropic_support import (
    betas,
    chat,
    content_blocks,
    deployment,
    error_message,
    marker,
    peer,
    sent_message,
    served,
    tool,
)
from pydantic import JsonValue

_WEATHER: Final = tool("get_weather", "Weather for a city", city={"type": "string"})


def _openai_mcp_tool(**extra: JsonValue) -> dict[str, JsonValue]:
    return {"type": "mcp", "server_label": MCP_NAME, "server_url": MCP_URL, "require_approval": "never", **extra}


def _user_parts(*parts: JsonValue) -> list[JsonValue]:
    return [{"role": "user", "content": [{"type": "text", "text": marker()}, *parts]}]


def test_an_openai_mcp_tool_becomes_an_anthropic_mcp_servers_entry(gateway: Gateway) -> None:
    """The caller sends one OpenAI tool `{"type": "mcp", "server_label": "dec1-docs", "server_url": <url>}`.

    The upstream body must carry `mcp_servers: [{"type": "url", "url": <url>, "name": "dec1-docs"}]`
    (`_map_openai_mcp_server_tool`), so `server_label` becomes `name` and `server_url` becomes `url`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_openai_mcp_tool()]))
        assert provider.only().body.get("mcp_servers") == [{"type": "url", "url": MCP_URL, "name": MCP_NAME}]


def test_an_mcp_tool_is_not_sent_as_a_function_tool(gateway: Gateway) -> None:
    """The caller sends a `get_weather` function tool and an OpenAI `mcp` tool.

    Upstream `tools` must hold only the `get_weather` custom tool and `mcp_servers` the one server
    (`_map_tools` splits them). An MCP entry left in `tools` is a 400 at Anthropic (`Unsupported tool type`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_WEATHER, _openai_mcp_tool()]))
        body: Final = provider.only().body
        assert [object_value(item)["name"] for item in list_value(body["tools"])] == ["get_weather"]
        assert len(list_value(body["mcp_servers"])) == 1


def test_an_mcp_tool_adds_the_mcp_client_beta(gateway: Gateway) -> None:
    """An OpenAI `mcp` tool must add `mcp-client-2025-04-04` to the upstream `anthropic-beta` header
    (`is_mcp_server_used` in `validate_environment`, `common_utils.py:1624`). Without it Anthropic rejects
    `mcp_servers` as an unknown field.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[_openai_mcp_tool()]))
        assert MCP_BETA in betas(provider.only())


def test_an_mcp_bearer_header_becomes_the_authorization_token(gateway: Gateway) -> None:
    """The OpenAI `mcp` tool carries `headers: {"Authorization": "Bearer dec1-synthetic-mcp-token"}`.

    The upstream server entry must carry `authorization_token: "dec1-synthetic-mcp-token"` with the `Bearer `
    prefix removed, and no `headers` key, since Anthropic's MCP connector takes only the bare token.
    """
    with peer() as provider, gateway.scenario() as scenario:
        mcp: Final = _openai_mcp_tool(headers={"Authorization": f"Bearer {MCP_TOKEN}"})
        served(chat(gateway, deployment(scenario, provider), tools=[mcp]))
        server: Final = object_value(list_value(provider.only().body["mcp_servers"])[0])
        assert server.get("authorization_token") == MCP_TOKEN
        assert "headers" not in server


def test_mcp_allowed_tools_become_the_tool_configuration(gateway: Gateway) -> None:
    """The OpenAI `mcp` tool carries `allowed_tools: ["search", "fetch"]`.

    The upstream server entry must carry `tool_configuration: {"allowed_tools": ["search", "fetch"]}` and no
    top-level `allowed_tools` (`AnthropicMcpServerToolConfiguration`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(gateway, deployment(scenario, provider), tools=[_openai_mcp_tool(allowed_tools=["search", "fetch"])])
        )
        server: Final = object_value(list_value(provider.only().body["mcp_servers"])[0])
        assert server.get("tool_configuration") == {"allowed_tools": ["search", "fetch"]}
        assert "allowed_tools" not in server


def test_an_anthropic_url_tool_in_tools_is_moved_to_mcp_servers(gateway: Gateway) -> None:
    """The caller puts an Anthropic-shaped server `{"type": "url", "url": <url>, "name": "dec1-docs",
    "authorization_token": <token>}` directly in `tools`.

    The `url` tool branch of `_map_tool_helper` builds an `AnthropicMcpServerTool` from it, so it must arrive in
    `mcp_servers` exactly as sent and never in `tools`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        server: Final = mcp_server(authorization_token=MCP_TOKEN)
        served(chat(gateway, deployment(scenario, provider), tools=[server]))
        body: Final = provider.only().body
        assert body.get("mcp_servers") == [server]
        assert "tools" not in body


def test_a_top_level_mcp_servers_list_is_forwarded_with_its_beta(gateway: Gateway) -> None:
    """The caller sends Anthropic's own top-level `mcp_servers: [<server>]` in the chat body, which litellm
    passes through as a provider-specific param.

    It must reach Anthropic unchanged and still add `mcp-client-2025-04-04`, because `validate_environment`
    reads `optional_params["mcp_servers"]` whatever produced it. UNVERIFIED: that the proxy passes this unknown
    OpenAI key into `optional_params` (round 1 confirms it for `top_k`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        server: Final = mcp_server()
        served(chat(gateway, deployment(scenario, provider), mcp_servers=[server]))
        call: Final = provider.only()
        assert call.body.get("mcp_servers") == [server]
        assert MCP_BETA in betas(call)


def test_a_container_with_skills_is_forwarded_and_adds_the_skills_beta(gateway: Gateway) -> None:
    """The caller sends `container: {"skills": [{"type": "custom", "skill_id": "skill_dec1report", "version":
    "latest"}]}` and the hosted code execution tool.

    The upstream body must carry the same `container` and the `anthropic-beta` header must hold both
    `skills-2025-10-02` (`is_container_with_skills_used`) and `code-execution-2025-08-25`
    (`is_code_execution_tool_used`, `common_utils.py:1532`).
    """
    container: Final[dict[str, JsonValue]] = {"skills": [{"type": "custom", "skill_id": SKILL_ID, "version": "latest"}]}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), container=container, tools=[dict(CODE_EXECUTION_TOOL)]))
        call: Final = provider.only()
        assert call.body.get("container") == container
        assert {SKILLS_BETA, CODE_EXECUTION_BETA} <= betas(call)


def test_a_container_without_skills_adds_no_skills_beta(gateway: Gateway) -> None:
    """The caller reuses an existing container by id with `container: {"id": "container_dec1reuse"}`.

    The container must be forwarded unchanged, but `skills-2025-10-02` must not be sent: the skills check only
    fires for a non-empty `skills` list (`common_utils.py:1360`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), container={"id": CONTAINER_ID}))
        call: Final = provider.only()
        assert call.body.get("container") == {"id": CONTAINER_ID}
        assert SKILLS_BETA not in betas(call)


def test_a_container_id_string_is_forwarded_as_sent(gateway: Gateway) -> None:
    """Anthropic also accepts the container as a bare id string. The caller sends `container:
    "container_dec1reuse"`; the upstream body must carry exactly that string, not an object.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), container=CONTAINER_ID))
        assert provider.only().body.get("container") == CONTAINER_ID


def test_a_container_upload_with_a_caller_code_execution_tool_does_not_add_a_second_one(gateway: Gateway) -> None:
    """The user turn has a `file` part with `file_id: file_dec1csv` and `format: text/csv` (a `container_upload`
    block) and the caller already sends `code_execution_20250825`.

    The upstream `tools` must hold exactly that one code execution tool: `add_code_execution_tool` returns early
    when any tool type starts with `code_execution` (`transformation.py:2385`).
    """
    upload: Final = {"type": "file", "file": {"file_id": "file_dec1csv", "format": "text/csv"}}
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway, deployment(scenario, provider), messages=_user_parts(upload), tools=[dict(CODE_EXECUTION_TOOL)]
            )
        )
        call: Final = provider.only()
        tools: Final = [object_value(item)["type"] for item in list_value(call.body["tools"])]
        assert tools == ["code_execution_20250825"]
        assert {"type": "container_upload", "file_id": "file_dec1csv"} in content_blocks(sent_message(call, 0))


def test_an_image_file_id_becomes_an_image_with_a_file_source(gateway: Gateway) -> None:
    """The user turn has a `file` part `{"file_id": "file_dec1png", "format": "image/png"}`.

    It must arrive as `{"type": "image", "source": {"type": "file", "file_id": "file_dec1png"}}`
    (`select_anthropic_content_block_type_for_file`), and the request must carry `files-api-2025-04-14` because
    a file source is in the messages (`is_file_id_used`).
    """
    part: Final = {"type": "file", "file": {"file_id": "file_dec1png", "format": "image/png"}}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=_user_parts(part)))
        call: Final = provider.only()
        assert {"type": "image", "source": {"type": "file", "file_id": "file_dec1png"}} in content_blocks(
            sent_message(call, 0)
        )
        assert FILES_BETA in betas(call)


def test_a_plain_text_file_id_becomes_a_document_with_a_file_source(gateway: Gateway) -> None:
    """The user turn has a `file` part `{"file_id": "file_dec1txt", "format": "text/plain"}`. Plain text is a
    document type, so it must arrive as `{"type": "document", "source": {"type": "file", "file_id":
    "file_dec1txt"}}` rather than a `container_upload`.
    """
    part: Final = {"type": "file", "file": {"file_id": "file_dec1txt", "format": "text/plain"}}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=_user_parts(part)))
        assert {"type": "document", "source": {"type": "file", "file_id": "file_dec1txt"}} in content_blocks(
            sent_message(provider.only(), 0)
        )


def test_a_document_part_with_citations_is_forwarded_verbatim(gateway: Gateway) -> None:
    """The user turn carries an Anthropic `document` part with a text source, a `title`, a `context` and
    `citations: {"enabled": True}`.

    The user content loop keeps `document` parts as sent (`factory.py:3139`), so the upstream user turn must hold
    the identical block; dropping `citations` silently turns citations off.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=_user_parts(dict(CITED_DOCUMENT))))
        assert dict(CITED_DOCUMENT) in content_blocks(sent_message(provider.only(), 0))


def test_anthropic_context_management_is_forwarded_with_its_beta(gateway: Gateway) -> None:
    """The caller sends `context_management: {"edits": [<clear_tool_uses_20250919 edit>]}` in Anthropic form.

    The upstream body must carry it unchanged (`map_openai_context_management_to_anthropic` passes a dict with
    `edits` through) and the header must hold `context-management-2025-06-27` but not `compact-2026-01-12`.
    """
    spec: Final[dict[str, JsonValue]] = {"edits": [dict(CLEAR_TOOL_USES)]}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), context_management=spec))
        call: Final = provider.only()
        assert call.body.get("context_management") == spec
        assert CONTEXT_MANAGEMENT_BETA in betas(call)
        assert COMPACT_BETA not in betas(call)


def test_openai_compaction_context_management_becomes_a_compact_edit(gateway: Gateway) -> None:
    """The caller sends OpenAI's list form `context_management: [{"type": "compaction", "compact_threshold":
    150000}]`.

    It must arrive as `{"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value":
    150000}}]}` with the `compact-2026-01-12` beta and without `context-management-2025-06-27`
    (`transformation.py:2024`, `_ensure_context_management_beta_header`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                context_management=[{"type": "compaction", "compact_threshold": 150000}],
            )
        )
        call: Final = provider.only()
        assert call.body.get("context_management") == {
            "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150000}}]
        }
        assert COMPACT_BETA in betas(call)
        assert CONTEXT_MANAGEMENT_BETA not in betas(call)


def test_an_openai_context_management_list_without_compaction_is_not_sent(gateway: Gateway) -> None:
    """OpenAI's list form with only an unknown entry type, `[{"type": "dec1_unknown"}]`, maps to no edits, so
    `map_openai_context_management_to_anthropic` returns None and `context_management` must be absent upstream,
    with no context management beta.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), context_management=[{"type": "dec1_unknown"}]))
        call: Final = provider.only()
        assert "context_management" not in call.body
        assert not ({CONTEXT_MANAGEMENT_BETA, COMPACT_BETA} & betas(call))


def test_the_memory_tool_is_forwarded_and_adds_the_context_management_beta(gateway: Gateway) -> None:
    """The caller sends Anthropic's memory tool `{"type": "memory_20250818", "name": "memory"}` in `tools`.

    It is a hosted tool type (`ANTHROPIC_HOSTED_TOOLS.MEMORY`), so it must arrive exactly as sent, and
    `update_headers_with_optional_anthropic_beta` must add `context-management-2025-06-27` for it
    (`transformation.py:2471`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[dict(MEMORY_TOOL)]))
        call: Final = provider.only()
        assert call.body.get("tools") == [dict(MEMORY_TOOL)]
        assert CONTEXT_MANAGEMENT_BETA in betas(call)


def test_the_web_fetch_tool_is_forwarded_with_its_options_and_beta(gateway: Gateway) -> None:
    """The caller sends `{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2}`.

    The hosted tool branch copies every key other than `type` and `name`, so `max_uses: 2` must arrive, and
    `web-fetch-2025-09-10` must be in the header (`transformation.py:2468`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[dict(WEB_FETCH_TOOL)]))
        call: Final = provider.only()
        assert call.body.get("tools") == [dict(WEB_FETCH_TOOL)]
        assert WEB_FETCH_BETA in betas(call)


@pytest.mark.parametrize(
    "beta",
    [TOKEN_EFFICIENT_BETA, CONTEXT_1M_BETA, EXTENDED_CACHE_TTL_BETA],
    ids=["token_efficient_tools", "context_1m", "extended_cache_ttl"],
)
def test_a_caller_beta_for_a_newer_feature_is_forwarded(gateway: Gateway, beta: str) -> None:
    """The caller asks for a beta the proxy derives from nothing in the body (token-efficient tools, the 1M context
    window, or extended cache TTL) through `extra_headers: {"anthropic-beta": <beta>}`.

    The upstream `anthropic-beta` must contain exactly that value (`user_anthropic_beta_headers` merge in
    `get_anthropic_headers`). Dropping it silently caps the context at 200k tokens or loses the 1h cache.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), extra_headers={"anthropic-beta": beta}))
        assert betas(provider.only()) == {beta}


def test_caller_betas_merge_with_derived_betas_in_one_header(gateway: Gateway) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07,token-efficient-tools-2025-02-19` and the web
    fetch tool, which derives `web-fetch-2025-09-10`.

    The upstream request must carry a single `anthropic-beta` header holding all three values, none dropped and
    none duplicated (`_ensure_beta_header` merges into the existing header).
    """
    caller: Final = f"{CONTEXT_1M_BETA},{TOKEN_EFFICIENT_BETA}"
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                tools=[dict(WEB_FETCH_TOOL)],
                extra_headers={"anthropic-beta": caller},
            )
        )
        call: Final = provider.only()
        assert betas(call) == {CONTEXT_1M_BETA, TOKEN_EFFICIENT_BETA, WEB_FETCH_BETA}
        values: Final = [value.strip() for value in call.headers["anthropic-beta"].split(",")]
        assert len(values) == len(set(values))


def test_a_one_hour_cache_control_on_a_tool_is_kept(gateway: Gateway) -> None:
    """The caller's function tool carries `cache_control: {"type": "ephemeral", "ttl": "1h"}`.

    The upstream tool must carry the same `cache_control` (`_map_tool_helper` copies it), and the proxy must not
    invent the old `extended-cache-ttl-2025-04-11` beta, which Anthropic no longer requires for 1h TTL.
    """
    cached: Final = {**_WEATHER, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), tools=[cached]))
        call: Final = provider.only()
        sent_tool: Final = object_value(list_value(call.body["tools"])[0])
        assert sent_tool.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}
        assert EXTENDED_CACHE_TTL_BETA not in betas(call)


@pytest.mark.parametrize(
    ("requested", "sent"),
    [("auto", "auto"), ("default", "standard_only"), ("standard_only", "standard_only")],
)
def test_service_tier_is_mapped_to_the_anthropic_tier_name(gateway: Gateway, requested: str, sent: str) -> None:
    """The caller sends OpenAI's `service_tier`. Anthropic accepts `auto` and `standard_only`, and OpenAI's
    `default` means "no priority capacity", so the upstream body must carry `service_tier` mapped by
    `OPENAI_TO_ANTHROPIC_SERVICE_TIER` (`transformation.py:603`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), service_tier=requested))
        assert provider.only().body.get("service_tier") == sent


def test_an_unmappable_service_tier_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """The caller sends `service_tier: "priority"`, which Anthropic's API does not accept as a request value.

    Without `drop_params` the proxy must answer 400 with a message containing `does not support
    service_tier='priority'` (`_map_service_tier` raises `UnsupportedParamsError`), and the peer must see no POST.
    """
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider), service_tier="priority")
        assert response.status_code == 400, response.text
        assert "does not support service_tier='priority'" in error_message(response)
        assert provider.posts() == ()


def test_an_unmappable_service_tier_is_dropped_under_drop_params(gateway: Gateway) -> None:
    """The deployment sets `drop_params: true` and the caller sends `service_tier: "flex"`.

    The call must succeed and the upstream body must carry no `service_tier` (`_map_service_tier` returns None
    and logs `DROP_UNSUPPORTED_SERVICE_TIER_WARNING`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider, drop_params=True), service_tier="flex"))
        assert "service_tier" not in provider.only().body


def test_a_search_result_in_a_tool_message_reaches_the_tool_result(gateway: Gateway) -> None:
    """The history holds a `get_weather` tool call and a tool message whose content list is one Anthropic
    `search_result` block (source, title, text content, citations enabled), the RAG pattern Anthropic documents.

    The upstream `tool_result` content must hold that `search_result` block unchanged. BUG: the tool result
    converter (`convert_to_anthropic_tool_result`, `factory.py:1887`) only keeps `text`, `image_url`,
    `tool_reference` and `file` parts and silently drops every other type, so the tool result arrives empty.
    """
    pytest.skip("BUG: a search_result block in a chat tool message is dropped before it reaches Anthropic")
    history: Final[list[JsonValue]] = [
        {"role": "user", "content": marker()},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "toolu_dec1a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": [dict(SEARCH_RESULT)]},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=history, tools=[_WEATHER]))
        tool_result: Final = content_blocks(sent_message(provider.only(), 2))[0]
        assert tool_result.get("content") == [dict(SEARCH_RESULT)]


def test_a_search_result_user_part_reaches_the_user_turn(gateway: Gateway) -> None:
    """The user turn carries a text part and an Anthropic `search_result` part.

    The upstream user turn must hold the `search_result` block unchanged. BUG: the user content loop in
    `anthropic_messages_pt` (`factory.py:3094`) only handles `image_url`, `text`, `document` and `file`, so the
    block is dropped without an error and the model never sees the source.
    """
    pytest.skip("BUG: a search_result user content part on chat is dropped before it reaches Anthropic")
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=_user_parts(dict(SEARCH_RESULT))))
        assert dict(SEARCH_RESULT) in content_blocks(sent_message(provider.only(), 0))
