"""`anthropic-beta` filtering on `/v1/chat/completions` for Claude on Anthropic, Vertex AI and Azure AI Foundry.

A caller's `anthropic-beta` header travels to a Claude deployment, but only the values the provider's row in
`litellm/anthropic_beta_headers_config.json` maps to a non null name survive
(`filter_and_transform_beta_headers`, `litellm/anthropic_beta_headers_manager.py:279`): an unknown value is
dropped, a value mapped to `null` is dropped, a value mapped to another name is renamed (Vertex calls
`advanced-tool-use-2025-11-20` `tool-search-tool-2025-10-19`), the survivors are de-duplicated and sorted, and
the header is removed when none survive (`update_headers_with_filtered_beta`, `:419`). The chat handler applies
this per provider (`litellm/llms/anthropic/chat/handler.py:521`, `update_request_with_filtered_beta`).

The `/v1/messages` side is pinned by the messages suites and Bedrock by `test_bedrock_anthropic_beta_header_wire.py`;
the chat path on these three providers was not (`index/test_dec1_anthropic_.md`). A port that forwards the
header raw gets 400 `invalid beta` from Vertex and Foundry. Assumes the bundled beta config is in force (the
shared proxy runs with the local model cost map; a remote config fetch could add rows).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_bigprov2_support import marker
from integration.providers.dec1_vertex_support import claude_json, claude_sse

_CONTEXT_1M: Final = "context-1m-2025-08-07"
_TOKEN_EFFICIENT: Final = "token-efficient-tools-2025-02-19"
_UNKNOWN: Final = "dec1-not-a-beta-2099-01-01"
_NULL_EVERYWHERE: Final = "bash_20241022"
_FILES_API: Final = "files-api-2025-04-14"
_TOOL_USE: Final = "advanced-tool-use-2025-11-20"
_VERTEX_TOOL_SEARCH: Final = "tool-search-tool-2025-10-19"


def _anthropic_peer(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith("/_oauth/token"):
        return Reply(body=b'{"access_token": "dec1-beta-token", "token_type": "Bearer", "expires_in": 3600}')
    if (
        b'"stream": true' in request.body
        or b'"stream":true' in request.body
        or b"streamRawPredict" in request.target.encode()
    ):
        return claude_sse()
    return Reply(body=json.dumps(claude_json()).encode())


@contextmanager
def _deployment(gateway: Gateway, provider: str) -> Iterator[tuple[Wire, str]]:
    with wire_server(_anthropic_peer) as wire, gateway.scenario() as scenario:
        base: Final = wire.url.rstrip("/")
        if provider == "anthropic":
            model = scenario.model(model="anthropic/claude-sonnet-4-5-20250929", api_base=base, api_key="dec1-key")
        elif provider == "vertex_ai":
            model = scenario.model(
                model="vertex_ai/claude-sonnet-4-5@20250929",
                api_key=None,
                api_base=base,
                vertex_project="dec1-beta-project",
                vertex_location="us-east5",
                vertex_credentials=service_account_json("dec1-beta-project", base),
            )
        else:
            model = scenario.model(model="azure_ai/claude-sonnet-4-5", api_base=base, api_key="dec1-foundry-key")
        yield wire, model


def _sent_beta(wire: Wire) -> str | None:
    posts: Final = [
        request for request in wire.drain() if request.method == "POST" and not request.target.startswith("/_oauth")
    ]
    assert len(posts) == 1, [request.target for request in posts]
    return posts[0].headers.get("anthropic-beta")


def _chat(gateway: Gateway, model: str, beta: str, *, stream: bool = False) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": stream, "num_retries": 0},
        headers={"anthropic-beta": beta},
    )
    assert response.status_code == 200, response.text


_PROVIDERS: Final = ("anthropic", "vertex_ai", "azure_ai")
# On `/v1/chat/completions` the proxy hands the caller's `anthropic-*` headers to `azure_ai` (and `databricks`)
# only for native `/v1/messages` calls (`_NATIVE_MESSAGES_ANTHROPIC_API_HEADER_PROVIDERS` with
# `native_anthropic_messages_only=True`, litellm/proxy/litellm_pre_call_utils.py:4815-4817 and :4849-4853 on
# main), so a Foundry Claude chat call never carries a caller beta and the filter rows apply to these two only.
_FORWARDING_PROVIDERS: Final = ("anthropic", "vertex_ai")


@pytest.mark.parametrize("provider", _FORWARDING_PROVIDERS)
def test_an_unknown_beta_is_dropped_and_a_supported_one_forwarded(gateway: Gateway, provider: str) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07,dec1-not-a-beta-2099-01-01`. Every provider row
    lists `context-1m-2025-08-07`, none lists the second value, so the provider must receive exactly
    `anthropic-beta: context-1m-2025-08-07`."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, f"{_CONTEXT_1M},{_UNKNOWN}")
        assert _sent_beta(wire) == _CONTEXT_1M


@pytest.mark.parametrize("provider", _FORWARDING_PROVIDERS)
def test_a_beta_mapped_to_null_is_dropped(gateway: Gateway, provider: str) -> None:
    """`bash_20241022` is in every row with a `null` value (a tool version, not a beta). Sent next to
    `context-1m-2025-08-07`, only `context-1m-2025-08-07` reaches the provider."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, f"{_NULL_EVERYWHERE}, {_CONTEXT_1M}")
        assert _sent_beta(wire) == _CONTEXT_1M


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_the_header_is_removed_when_no_beta_survives(gateway: Gateway, provider: str) -> None:
    """The caller sends only `dec1-not-a-beta-2099-01-01,bash_20241022`. No value survives, so the provider must
    receive no `anthropic-beta` header at all rather than an empty one."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, f"{_UNKNOWN},{_NULL_EVERYWHERE}")
        assert _sent_beta(wire) is None


@pytest.mark.parametrize("provider", _FORWARDING_PROVIDERS)
def test_surviving_betas_are_sorted_and_deduplicated(gateway: Gateway, provider: str) -> None:
    """The caller sends `token-efficient-tools-2025-02-19, context-1m-2025-08-07, context-1m-2025-08-07`. The
    provider receives `context-1m-2025-08-07,token-efficient-tools-2025-02-19`: one copy each, sorted, joined
    with a bare comma. UNVERIFIED: no auto added beta joins the list for a plain request."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, f"{_TOKEN_EFFICIENT}, {_CONTEXT_1M}, {_CONTEXT_1M}")
        assert _sent_beta(wire) == f"{_CONTEXT_1M},{_TOKEN_EFFICIENT}"


def test_vertex_renames_the_advanced_tool_use_beta(gateway: Gateway) -> None:
    """On a Vertex Claude deployment, `advanced-tool-use-2025-11-20` is sent to Vertex as
    `tool-search-tool-2025-10-19`, the name Vertex's row maps it to."""
    with _deployment(gateway, "vertex_ai") as (wire, model):
        _chat(gateway, model, _TOOL_USE)
        assert _sent_beta(wire) == _VERTEX_TOOL_SEARCH


def test_anthropic_keeps_the_advanced_tool_use_beta_name(gateway: Gateway) -> None:
    """On the Anthropic API the same `advanced-tool-use-2025-11-20` is forwarded under its own name."""
    with _deployment(gateway, "anthropic") as (wire, model):
        _chat(gateway, model, _TOOL_USE)
        assert _sent_beta(wire) == _TOOL_USE


@pytest.mark.parametrize(("provider", "forwarded"), [("anthropic", True), ("azure_ai", False), ("vertex_ai", False)])
def test_the_files_api_beta_follows_each_provider_row(gateway: Gateway, provider: str, forwarded: bool) -> None:
    """`files-api-2025-04-14` is supported on the Anthropic API and mapped to `null` for Vertex. Sent alone, it
    reaches Anthropic unchanged and Vertex receives no beta header. Foundry's row lists it too, but a chat call
    to Foundry never carries the caller's beta (see `_FORWARDING_PROVIDERS`), so Foundry receives none."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, _FILES_API)
        assert _sent_beta(wire) == (_FILES_API if forwarded else None)


@pytest.mark.parametrize("provider", _FORWARDING_PROVIDERS)
def test_streamed_chat_filters_the_beta_the_same_way(gateway: Gateway, provider: str) -> None:
    """The unknown plus supported pair is sent on a streamed chat. The streaming request takes the same filter,
    so the provider receives exactly `anthropic-beta: context-1m-2025-08-07`."""
    with _deployment(gateway, provider) as (wire, model):
        _chat(gateway, model, f"{_CONTEXT_1M},{_UNKNOWN}", stream=True)
        assert _sent_beta(wire) == _CONTEXT_1M


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "streamed"])
def test_a_foundry_claude_chat_call_does_not_carry_the_callers_beta(gateway: Gateway, stream: bool) -> None:
    """The caller sends `anthropic-beta: context-1m-2025-08-07,dec1-not-a-beta-2099-01-01` on a chat call to an
    `azure_ai/claude-...` deployment. The caller's `anthropic-*` headers are scoped to native `/v1/messages`
    calls for Foundry (litellm/proxy/litellm_pre_call_utils.py:4849-4853 on main), so Foundry receives no
    `anthropic-beta`, not even the supported value."""
    with _deployment(gateway, "azure_ai") as (wire, model):
        _chat(gateway, model, f"{_CONTEXT_1M},{_UNKNOWN}", stream=stream)
        assert _sent_beta(wire) is None
