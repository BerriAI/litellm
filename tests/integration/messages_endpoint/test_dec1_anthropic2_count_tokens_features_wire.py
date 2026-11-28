"""Contract: `POST /v1/messages/count_tokens` on an Anthropic deployment counts the newer features correctly.

Claude Code counts tokens before every turn with the same body it will send to `/v1/messages`. For an
`anthropic/` deployment the proxy calls Anthropic's own count endpoint through `AnthropicCountTokensConfig`
(`litellm/llms/anthropic/count_tokens/transformation.py`, handler in `count_tokens/handler.py`). The count body
keeps `model`, `messages` (as sent, after a leading system run is lifted), `system`, `tools`,
`context_management` when it has edits, and the options in `COUNT_TOKEN_OPTION_NAMES` (`thinking`, `tool_choice`,
`output_config`); every other generation field is left out. The headers carry `token-counting-2024-11-01` plus
the betas the counted body needs: context management or compaction betas for edits and the thinking betas
(`get_count_tokens_headers`, `transformation.py:578`). The caller's own `anthropic-beta` is not merged in.

A rewrite of the count route must keep this, or Claude Code's context meter drifts from what inference bills.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic2_support import (
    CITED_DOCUMENT,
    CLEAR_TOOL_USES,
    COMPACT_BETA,
    CONTAINER_ID,
    CONTEXT_MANAGEMENT_BETA,
    FILES_BETA,
    MEMORY_TOOL,
    SEARCH_RESULT,
    TOKEN_COUNTING_BETA,
    WEB_FETCH_TOOL,
    body_of,
    count,
    mcp_server,
    messages_body,
)
from integration.providers.dec1_anthropic_support import (
    COUNTED_TOKENS,
    betas,
    deployment,
    marker,
    peer,
)
from pydantic import JsonValue


def _counted(
    gateway: Gateway, **fields: JsonValue
) -> tuple[dict[str, JsonValue], frozenset[str], dict[str, JsonValue]]:
    """Count a request with `fields` on a fresh deployment; return the count body sent, its betas, and the answer."""
    with peer() as provider, gateway.scenario() as scenario:
        answer: Final = body_of(count(gateway, messages_body(deployment(scenario, provider), **fields)))
        call: Final = provider.only()
        assert call.path == "/v1/messages/count_tokens"
        return dict(call.body), betas(call), answer


def test_a_count_with_clear_tool_uses_edits_sends_them_with_the_context_management_beta(gateway: Gateway) -> None:
    """The caller counts a body with `context_management: {"edits": [<clear_tool_uses_20250919 edit>]}`.

    The count body must carry the same `context_management` and the headers both `token-counting-2024-11-01` and
    `context-management-2025-06-27` (`context_management_betas`), so Anthropic counts the cleared history, and the
    answer must be the provider's `input_tokens: 42`.
    """
    spec: Final[dict[str, JsonValue]] = {"edits": [dict(CLEAR_TOOL_USES)]}
    sent, sent_betas, answer = _counted(gateway, context_management=spec)
    assert sent.get("context_management") == spec
    assert {TOKEN_COUNTING_BETA, CONTEXT_MANAGEMENT_BETA} <= sent_betas
    assert answer.get("input_tokens") == COUNTED_TOKENS


def test_a_count_with_a_compact_edit_sends_the_compact_beta(gateway: Gateway) -> None:
    """The caller counts with `context_management: {"edits": [{"type": "compact_20260112"}]}`.

    The count headers must hold `compact-2026-01-12` next to the token counting beta, and not
    `context-management-2025-06-27`, mirroring what inference sends for the same body.
    """
    sent, sent_betas, _ = _counted(gateway, context_management={"edits": [{"type": "compact_20260112"}]})
    assert sent.get("context_management") == {"edits": [{"type": "compact_20260112"}]}
    assert COMPACT_BETA in sent_betas
    assert CONTEXT_MANAGEMENT_BETA not in sent_betas


def test_a_count_with_empty_edits_sends_no_context_management(gateway: Gateway) -> None:
    """The caller counts with `context_management: {"edits": []}`.

    `context_management_with_edits` returns None for a spec with no edits, so the count body must have no
    `context_management` key and the only beta must be `token-counting-2024-11-01`.
    """
    sent, sent_betas, _ = _counted(gateway, context_management={"edits": []})
    assert "context_management" not in sent
    assert sent_betas == {TOKEN_COUNTING_BETA}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mcp_servers", [mcp_server()]),
        ("container", CONTAINER_ID),
        ("service_tier", "priority"),
        ("inference_geo", "us"),
    ],
)
def test_inference_only_fields_are_left_out_of_the_count(gateway: Gateway, field: str, value: JsonValue) -> None:
    """The caller counts a body that also carries `mcp_servers`, `container`, `service_tier` or `inference_geo`.

    None of them is in the count body the transform builds (`transform_request_to_count_tokens` keeps only model,
    messages, system, tools, context_management and `COUNT_TOKEN_OPTION_NAMES`), so the count request must not
    carry the field, and the count must still succeed with the provider's number.
    """
    sent, _, answer = _counted(gateway, **{field: value})
    assert field not in sent
    assert answer.get("input_tokens") == COUNTED_TOKENS


def test_a_count_keeps_a_search_result_block_in_the_messages(gateway: Gateway) -> None:
    """The counted user turn holds a text block and a `search_result` block.

    The count body must carry the user turn with both blocks unchanged (`_count_block` only rewrites `image_url`
    parts and images inside tool results), so the cited source is counted.
    """
    text: Final = {"type": "text", "text": marker()}
    sent, _, _ = _counted(gateway, messages=[{"role": "user", "content": [text, dict(SEARCH_RESULT)]}])
    assert list_value(object_value(list_value(sent["messages"])[0])["content"]) == [text, dict(SEARCH_RESULT)]


def test_a_count_keeps_a_cited_document_in_the_messages(gateway: Gateway) -> None:
    """The counted user turn holds a text-source `document` with `citations: {"enabled": true}`.

    It must be in the count body unchanged; dropping the document would undercount a large attachment.
    """
    text: Final = {"type": "text", "text": marker()}
    sent, _, _ = _counted(gateway, messages=[{"role": "user", "content": [text, dict(CITED_DOCUMENT)]}])
    assert dict(CITED_DOCUMENT) in list_value(object_value(list_value(sent["messages"])[0])["content"])


def test_a_count_keeps_hosted_tools_as_sent(gateway: Gateway) -> None:
    """The caller counts with `tools: [<memory_20250818>, <web_fetch_20250910 with max_uses 2>]`.

    The count body `tools` must be the same two tools in order: hosted tool definitions count toward the prompt,
    and `tools_without_custom_field` only hoists `defer_loading` or drops `custom`.
    """
    tools: Final[list[JsonValue]] = [dict(MEMORY_TOOL), dict(WEB_FETCH_TOOL)]
    sent, _, _ = _counted(gateway, tools=tools)
    assert sent.get("tools") == tools


def test_a_count_with_display_updates_thinking_sends_the_thinking_and_its_beta(gateway: Gateway) -> None:
    """The caller counts with `thinking: {"type": "adaptive", "display": "updates"}`.

    `thinking` is a count option (`COUNT_TOKEN_OPTION_NAMES`), so it must be in the count body, and the headers
    must carry `thinking-display-updates-2026-08-18` from `AnthropicModelInfo.thinking_betas`
    (`get_count_tokens_headers`, #51079). UNVERIFIED: whether `messages_count_options` rewrites `adaptive` for the
    pre-adaptive backend `claude-sonnet-4-5-20250929` the way inference does; only presence is pinned. In CI run
    44176682961 only `token-counting-2024-11-01` was sent although `thinking_betas` adds the display beta
    (litellm/llms/anthropic/count_tokens/transformation.py:460-471 on main); the failure now prints the `thinking`
    the count body carried, to tell a rewrite that dropped `display` from a lost beta.
    """
    sent, sent_betas, _ = _counted(gateway, thinking={"type": "adaptive", "display": "updates"})
    assert "thinking" in sent
    assert "thinking-display-updates-2026-08-18" in sent_betas, (sent.get("thinking"), sorted(sent_betas))


def test_a_count_forwards_tool_choice(gateway: Gateway) -> None:
    """The caller counts with one tool and `tool_choice: {"type": "any"}`.

    `tool_choice` changes the system prompt Anthropic injects, so it is a count option since #51133 and must be in
    the count body unchanged.
    """
    lookup: Final[dict[str, JsonValue]] = {"name": "kb_lookup", "input_schema": {"type": "object", "properties": {}}}
    sent, _, _ = _counted(gateway, tools=[lookup], tool_choice={"type": "any"})
    assert sent.get("tool_choice") == {"type": "any"}


def test_a_count_does_not_merge_the_caller_beta_header(gateway: Gateway) -> None:
    """The caller counts with `anthropic-beta: context-1m-2025-08-07`.

    The count headers are built from the auth header and the body alone (`get_count_tokens_headers`), so the
    upstream beta set must be exactly `{token-counting-2024-11-01}`. Pinned as today's behaviour; see the next
    test for the case where that is wrong.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(count(gateway, messages_body(model), headers={"anthropic-beta": "context-1m-2025-08-07"}))
        assert betas(provider.only()) == {TOKEN_COUNTING_BETA}


def test_a_count_with_a_file_source_document_carries_the_files_beta(gateway: Gateway) -> None:
    """The caller counts a user turn holding `{"type": "document", "source": {"type": "file", "file_id":
    "file_dec1pdf"}}` and sends `anthropic-beta: files-api-2025-04-14`, the same request inference accepts.

    The count request must carry `files-api-2025-04-14`, or Anthropic rejects the file source. BUG (UNVERIFIED
    against the live API): the count headers neither merge the caller's beta nor derive the files beta from a file
    source (`get_count_tokens_headers` only adds compaction, context management and thinking betas), so the count
    of any Files API request fails while inference succeeds.
    """
    pytest.skip("BUG: count_tokens drops the files-api beta for a file-source document")
    document: Final[dict[str, JsonValue]] = {"type": "document", "source": {"type": "file", "file_id": "file_dec1pdf"}}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        body_of(
            count(
                gateway,
                messages_body(
                    model, messages=[{"role": "user", "content": [{"type": "text", "text": marker()}, document]}]
                ),
                headers={"anthropic-beta": FILES_BETA},
            )
        )
        assert FILES_BETA in betas(provider.only())
