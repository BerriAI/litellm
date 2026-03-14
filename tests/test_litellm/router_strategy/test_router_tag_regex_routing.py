"""
Unit tests for tag_regex routing.

Tests _is_valid_deployment_tag_regex() and get_deployments_for_tag() with tag_regex
patterns, verifying that regex-based header matching works correctly alongside
existing tag-based routing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import MagicMock

from litellm.router_strategy import tag_based_routing
from litellm.router_strategy.tag_based_routing import get_deployments_for_tag

_is_valid_deployment_tag_regex = tag_based_routing._is_valid_deployment_tag_regex


# ---------------------------------------------------------------------------
# _is_valid_deployment_tag_regex unit tests
# ---------------------------------------------------------------------------


def test_regex_matches_claude_code_user_agent():
    """^User-Agent: claude-code/ matches a claude-code UA string."""
    result = _is_valid_deployment_tag_regex(
        tag_regexes=[r"^User-Agent: claude-code\/"],
        header_strings=["User-Agent: claude-code/1.2.3"],
    )
    assert result == r"^User-Agent: claude-code\/"


def test_regex_no_match_for_other_ua():
    """Pattern does not match a non-claude-code User-Agent."""
    result = _is_valid_deployment_tag_regex(
        tag_regexes=[r"^User-Agent: claude-code\/"],
        header_strings=["User-Agent: Mozilla/5.0 (browser)"],
    )
    assert result is None


def test_regex_returns_first_matching_pattern():
    """When multiple patterns are provided, returns the first match."""
    result = _is_valid_deployment_tag_regex(
        tag_regexes=[r"^User-Agent: cursor\/", r"^User-Agent: claude-code\/"],
        header_strings=["User-Agent: claude-code/2.0.0"],
    )
    assert result == r"^User-Agent: claude-code\/"


def test_regex_empty_inputs_return_none():
    """Empty lists return None without errors."""
    assert _is_valid_deployment_tag_regex([], ["User-Agent: claude-code/1.0"]) is None
    assert _is_valid_deployment_tag_regex([r"^User-Agent: claude-code\/"], []) is None


def test_invalid_regex_skipped_does_not_raise():
    """An invalid regex pattern is skipped (warning logged) — no exception raised."""
    result = _is_valid_deployment_tag_regex(
        tag_regexes=["[invalid(regex"],
        header_strings=["User-Agent: claude-code/1.0"],
    )
    assert result is None


def test_regex_matches_version_range():
    """Semver-aware pattern matches multiple versions."""
    pattern = r"^User-Agent: claude-code\/\d"
    for ua in ["claude-code/1.0", "claude-code/2.0.0-beta.1", "claude-code/99.0"]:
        result = _is_valid_deployment_tag_regex(
            tag_regexes=[pattern],
            header_strings=[f"User-Agent: {ua}"],
        )
        assert result == pattern, f"Expected match for UA: {ua}"


# ---------------------------------------------------------------------------
# get_deployments_for_tag integration tests
# ---------------------------------------------------------------------------

CLAUDE_CODE_DEPLOYMENT = {
    "model_name": "claude-sonnet",
    "litellm_params": {
        "model": "openai/claude-code-deployment",
        "api_key": "fake",
        "mock_response": "cc",
        "tag_regex": [r"^User-Agent: claude-code\/"],
    },
    "model_info": {"id": "claude-code-deployment"},
}

REGULAR_DEPLOYMENT = {
    "model_name": "claude-sonnet",
    "litellm_params": {
        "model": "openai/regular-deployment",
        "api_key": "fake",
        "mock_response": "regular",
        "tags": ["default"],
    },
    "model_info": {"id": "regular-deployment"},
}

ALL_DEPLOYMENTS = [CLAUDE_CODE_DEPLOYMENT, REGULAR_DEPLOYMENT]


def _make_router_mock(enable_tag_filtering=True, match_any=True):
    mock = MagicMock()
    mock.enable_tag_filtering = enable_tag_filtering
    mock.tag_filtering_match_any = match_any
    return mock


@pytest.mark.asyncio
async def test_claude_code_ua_routes_to_cc_deployment():
    """claude-code/x.y.z UA → claude-code-deployment via tag_regex."""
    router = _make_router_mock()
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=ALL_DEPLOYMENTS,
        request_kwargs={"metadata": {"user_agent": "claude-code/1.2.3"}},
    )
    assert len(result) == 1
    assert result[0]["model_info"]["id"] == "claude-code-deployment"


@pytest.mark.asyncio
async def test_regular_ua_routes_to_default_deployment():
    """Mozilla UA → regular-deployment via default tag fallback."""
    router = _make_router_mock()
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=ALL_DEPLOYMENTS,
        request_kwargs={"metadata": {"user_agent": "Mozilla/5.0 (browser)"}},
    )
    assert len(result) == 1
    assert result[0]["model_info"]["id"] == "regular-deployment"


@pytest.mark.asyncio
async def test_no_ua_routes_to_default_deployment():
    """No User-Agent → default deployment."""
    router = _make_router_mock()
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=ALL_DEPLOYMENTS,
        request_kwargs={"metadata": {}},
    )
    assert len(result) == 1
    assert result[0]["model_info"]["id"] == "regular-deployment"


@pytest.mark.asyncio
async def test_tag_routing_metadata_written_for_regex_match():
    """tag_routing metadata block is populated when regex matches."""
    router = _make_router_mock()
    metadata: dict = {"user_agent": "claude-code/2.0.0-beta.1"}
    await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=ALL_DEPLOYMENTS,
        request_kwargs={"metadata": metadata},
    )
    assert "tag_routing" in metadata
    tr = metadata["tag_routing"]
    assert tr["matched_via"] == "tag_regex"
    assert tr["matched_value"] == r"^User-Agent: claude-code\/"
    assert tr["user_agent"] == "claude-code/2.0.0-beta.1"


@pytest.mark.asyncio
async def test_tag_filtering_disabled_returns_all_deployments():
    """When enable_tag_filtering is False, all deployments returned regardless of UA."""
    router = _make_router_mock(enable_tag_filtering=False)
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=ALL_DEPLOYMENTS,
        request_kwargs={"metadata": {"user_agent": "claude-code/1.0"}},
    )
    assert result == ALL_DEPLOYMENTS


@pytest.mark.asyncio
async def test_explicit_tag_match_takes_precedence_over_regex():
    """A deployment with both tags and tag_regex: exact tag match fires first."""
    deployment_with_both = {
        "model_name": "claude-sonnet",
        "litellm_params": {
            "model": "openai/both-deployment",
            "api_key": "fake",
            "tags": ["premium"],
            "tag_regex": [r"^User-Agent: claude-code\/"],
        },
        "model_info": {"id": "both-deployment"},
    }
    router = _make_router_mock()
    metadata: dict = {
        "tags": ["premium"],
        "user_agent": "claude-code/1.0",
    }
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=[deployment_with_both],
        request_kwargs={"metadata": metadata},
    )
    assert len(result) == 1
    tr = metadata.get("tag_routing", {})
    assert tr.get("matched_via") == "tags"


@pytest.mark.asyncio
async def test_user_agent_present_no_tag_regex_deployments_does_not_raise():
    """
    Backwards-compat: a request that carries a User-Agent but targets plain-tag
    deployments (no tag_regex) must NOT raise ValueError — it should fall
    through to the default/all-deployments path just like before.
    """
    plain_tag_only_deployments = [
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/premium-deployment",
                "api_key": "fake",
                "tags": ["premium"],
            },
            "model_info": {"id": "premium-deployment"},
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "openai/free-deployment",
                "api_key": "fake",
                "tags": ["free"],
            },
            "model_info": {"id": "free-deployment"},
        },
    ]
    router = _make_router_mock()
    # The request has a User-Agent (as all proxy requests do) but NO tags and
    # neither deployment has tag_regex — must not raise, must return all.
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="gpt-4",
        healthy_deployments=plain_tag_only_deployments,
        request_kwargs={"metadata": {"user_agent": "Mozilla/5.0 (any-client)"}},
    )
    # Falls through to "return healthy_deployments" path unchanged
    assert result == plain_tag_only_deployments


@pytest.mark.asyncio
async def test_tag_routing_metadata_not_overwritten_for_multiple_matches():
    """
    When multiple deployments match, tag_routing records only the first match
    so the provenance reflects what the load balancer likely selected.
    """
    deployment_a = {
        "model_name": "claude-sonnet",
        "litellm_params": {
            "model": "openai/cc-deployment-a",
            "api_key": "fake",
            "tag_regex": [r"^User-Agent: claude-code\/"],
        },
        "model_info": {"id": "cc-deployment-a"},
    }
    deployment_b = {
        "model_name": "claude-sonnet",
        "litellm_params": {
            "model": "openai/cc-deployment-b",
            "api_key": "fake",
            "tag_regex": [r"^User-Agent: claude-code\/"],
        },
        "model_info": {"id": "cc-deployment-b"},
    }
    router = _make_router_mock()
    metadata: dict = {"user_agent": "claude-code/1.0"}
    result = await get_deployments_for_tag(
        llm_router_instance=router,
        model="claude-sonnet",
        healthy_deployments=[deployment_a, deployment_b],
        request_kwargs={"metadata": metadata},
    )
    assert len(result) == 2
    # tag_routing recorded once and reflects the first match
    tr = metadata.get("tag_routing", {})
    assert tr.get("matched_deployment") == "claude-sonnet"
    assert tr.get("matched_via") == "tag_regex"
