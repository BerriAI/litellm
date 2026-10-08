"""
Tests for POST /policy/templates/test endpoint logic.

Tests _test_guardrail_definitions and _compute_overall_action directly
without needing a running proxy.
"""

import pytest
from fastapi import Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.policy_endpoints.endpoints import (
    GuardrailTestResultEntry,
    _compute_overall_action,
    _test_guardrail_definitions,
    list_policies,
)
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.types.proxy.policy_engine import (
    PolicyGuardrailsResponse,
    PolicyListResponse,
    PolicyScopeResponse,
    PolicySummaryItem,
)


@pytest.mark.asyncio
async def test_pattern_based_guardrail_masks_pii():
    """A pattern-based guardrail should mask matching PII."""
    guardrail_defs = [
        {
            "guardrail_name": "test-ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks US SSNs"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="My SSN is 123-45-6789",
    )

    assert len(results) == 1
    assert results[0]["guardrail_name"] == "test-ssn-masker"
    assert results[0]["action"] == "masked"
    assert "123-45-6789" not in results[0]["output_text"]
    assert "REDACTED" in results[0]["output_text"]


@pytest.mark.asyncio
async def test_blocked_words_guardrail_blocks():
    """A blocked_words guardrail should block matching text."""
    guardrail_defs = [
        {
            "guardrail_name": "test-word-blocker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "blocked_words": [
                    {
                        "keyword": "forbidden_word",
                        "action": "BLOCK",
                        "description": "test block",
                    }
                ],
            },
            "guardrail_info": {"description": "Blocks forbidden words"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="This contains forbidden_word in it",
    )

    assert len(results) == 1
    assert results[0]["guardrail_name"] == "test-word-blocker"
    assert results[0]["action"] == "blocked"


@pytest.mark.asyncio
async def test_clean_text_passes():
    """Clean text should pass all guardrails."""
    guardrail_defs = [
        {
            "guardrail_name": "test-ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
            },
            "guardrail_info": {"description": "Masks US SSNs"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="Hello, this is a perfectly clean message.",
    )

    assert len(results) == 1
    assert results[0]["action"] == "passed"
    assert results[0]["output_text"] == "Hello, this is a perfectly clean message."


@pytest.mark.asyncio
async def test_unsupported_guardrail_type():
    """Non-litellm_content_filter types should return unsupported."""
    guardrail_defs = [
        {
            "guardrail_name": "test-mcp",
            "litellm_params": {
                "guardrail": "mcp_security",
                "mode": "pre_call",
            },
            "guardrail_info": {"description": "MCP guardrail"},
        }
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="Any text",
    )

    assert len(results) == 1
    assert results[0]["action"] == "unsupported"
    assert "mcp_security" in results[0]["details"]


@pytest.mark.asyncio
async def test_multiple_guardrails_mixed_results():
    """Multiple guardrails with different outcomes."""
    guardrail_defs = [
        {
            "guardrail_name": "ssn-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "us_ssn",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks SSNs"},
        },
        {
            "guardrail_name": "email-masker",
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "patterns": [
                    {
                        "pattern_type": "prebuilt",
                        "pattern_name": "email",
                        "action": "MASK",
                    }
                ],
                "pattern_redaction_format": "[{pattern_name}_REDACTED]",
            },
            "guardrail_info": {"description": "Masks emails"},
        },
    ]

    results = await _test_guardrail_definitions(
        guardrail_definitions=guardrail_defs,
        text="My SSN is 123-45-6789 but no email here",
    )

    assert len(results) == 2
    ssn_result = next(r for r in results if r["guardrail_name"] == "ssn-masker")
    email_result = next(r for r in results if r["guardrail_name"] == "email-masker")
    assert ssn_result["action"] == "masked"
    assert email_result["action"] == "passed"


def test_compute_overall_action_blocked_wins():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(
            guardrail_name="a", action="passed", output_text="", details=""
        ),
        GuardrailTestResultEntry(
            guardrail_name="b", action="blocked", output_text="", details=""
        ),
        GuardrailTestResultEntry(
            guardrail_name="c", action="masked", output_text="", details=""
        ),
    ]
    assert _compute_overall_action(results) == "blocked"


def test_compute_overall_action_masked_wins_over_passed():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(
            guardrail_name="a", action="passed", output_text="", details=""
        ),
        GuardrailTestResultEntry(
            guardrail_name="b", action="masked", output_text="", details=""
        ),
    ]
    assert _compute_overall_action(results) == "masked"


def test_compute_overall_action_all_passed():
    results: list[GuardrailTestResultEntry] = [
        GuardrailTestResultEntry(
            guardrail_name="a", action="passed", output_text="", details=""
        ),
        GuardrailTestResultEntry(
            guardrail_name="b", action="passed", output_text="", details=""
        ),
    ]
    assert _compute_overall_action(results) == "passed"


def test_compute_overall_action_empty():
    assert _compute_overall_action([]) == "passed"


class TestEnrichPolicyTemplateStreamKeepalive:
    async def _collect_endpoint_body(self, monkeypatch, interval, delay=0.3) -> tuple[list[bytes], dict]:
        import asyncio
        from unittest.mock import MagicMock

        import litellm
        import litellm.proxy.management_endpoints.policy_endpoints.endpoints as policy_endpoints
        import litellm.proxy.proxy_server as proxy_server
        from fastapi.responses import StreamingResponse
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.management_endpoints.policy_endpoints.endpoints import (
            EnrichTemplateRequest,
            enrich_policy_template_stream,
        )

        monkeypatch.setattr(litellm, "sse_keepalive_ping_interval_seconds", interval)

        async def _name_chunks():
            await asyncio.sleep(delay)
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Rival Air\n"
            yield chunk

        class SlowRouter:
            async def acompletion(self, **kwargs):
                return _name_chunks()

        async def _no_variations(competitors, model):
            return {}

        monkeypatch.setattr(proxy_server, "llm_router", SlowRouter())
        monkeypatch.setattr(policy_endpoints, "_generate_competitor_variations", _no_variations)

        response = await enrich_policy_template_stream(
            data=EnrichTemplateRequest(
                template_id="competitor-mention-detection",
                parameters={"brand_name": "Acme"},
                model="gpt-5.4-mini",
            ),
            request=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN),
        )
        assert isinstance(response, StreamingResponse)
        chunks = [chunk if isinstance(chunk, bytes) else chunk.encode() async for chunk in response.body_iterator]
        return chunks, dict(response.headers)

    @pytest.mark.asyncio
    async def test_endpoint_pings_while_competitor_discovery_is_still_running(self, monkeypatch):
        chunks, headers = await self._collect_endpoint_body(monkeypatch, interval=0.05)

        assert headers["content-type"].startswith("text/event-stream")
        assert headers["cache-control"] == "no-cache"
        assert headers["x-accel-buffering"] == "no"
        assert chunks[0] == b": ping\n\n"
        assert chunks.count(b": ping\n\n") >= 3
        assert b'data: {"type": "competitor", "name": "Rival Air"}\n\n' in chunks
        assert chunks[-1].startswith(b'data: {"type": "done"')

    @pytest.mark.asyncio
    async def test_endpoint_stream_is_untouched_while_keepalives_are_unconfigured(self, monkeypatch):
        chunks, _ = await self._collect_endpoint_body(monkeypatch, interval=None, delay=0.15)

        assert b": ping\n\n" not in chunks
        assert chunks[0] == b'data: {"type": "competitor", "name": "Rival Air"}\n\n'
        assert chunks[-1].startswith(b'data: {"type": "done"')


def _policy_list_request() -> Request:
    return Request({"type": "http", "method": "GET", "path": "/policy/list", "headers": []})


def _summary_item(inherit, resolved_guardrails, inheritance_chain) -> PolicySummaryItem:
    return PolicySummaryItem(
        inherit=inherit,
        scope=PolicyScopeResponse(),
        guardrails=PolicyGuardrailsResponse(),
        resolved_guardrails=resolved_guardrails,
        inheritance_chain=inheritance_chain,
    )


@pytest.fixture
def policy_registry():
    registry = get_policy_registry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.mark.asyncio
async def test_list_policies_is_empty_before_any_policy_is_loaded(policy_registry):
    response = await list_policies(request=_policy_list_request(), user_api_key_dict=UserAPIKeyAuth())

    assert response == PolicyListResponse(policies={}, total_count=0)


@pytest.mark.parametrize(
    ("policies_config", "expected_policies"),
    [
        ({}, {}),
        (
            {"solo": {"description": "standalone", "guardrails": {"add": ["pii"]}}},
            {"solo": _summary_item(None, ["pii"], ["solo"])},
        ),
        (
            {
                "base": {"guardrails": {"add": ["pii"]}},
                "child": {"inherit": "base", "guardrails": {"add": ["audit"], "remove": ["pii"]}},
                "conditional": {"guardrails": {"add": ["toxicity"]}, "condition": {"model": "gpt-4.*"}},
                "empty": {"guardrails": {"remove": ["pii"]}},
            },
            {
                "base": _summary_item(None, ["pii"], ["base"]),
                "child": _summary_item("base", ["audit"], ["base", "child"]),
                "conditional": _summary_item(None, ["toxicity"], ["conditional"]),
                "empty": _summary_item(None, [], ["empty"]),
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_list_policies_reports_each_loaded_policy_with_its_resolved_guardrails(
    policy_registry, policies_config, expected_policies
):
    policy_registry.load_policies(policies_config)

    response = await list_policies(request=_policy_list_request(), user_api_key_dict=UserAPIKeyAuth())

    assert response == PolicyListResponse(policies=expected_policies, total_count=0)
    assert list(response.policies) == list(expected_policies)
