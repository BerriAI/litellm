import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Dict, Final, List, Optional, Union
from unittest.mock import Mock

import pytest

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.constants import ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.proxy._types import UserAPIKeyAuth
from litellm.litellm_core_utils.routing_common_utils import (
    _deployment_supports_web_search,
    add_model_file_id_mappings,
    filter_team_based_models,
    filter_web_search_deployments,
    pick_preferring_web_search,
    prefer_web_search_deployments,
    format_content_policy_no_fallback_note,
    format_fallback_outcome_message,
    provider_for_generic_call,
    resolve_model_group_alias,
    resolve_served_model,
    truncate_fallback_error_detail,
    PROVIDER_SCOPED_CREDENTIAL_PARAMS,
    warn_on_provider_credential_mismatch,
)
from litellm.router_utils.pre_call_checks.model_rate_limit_check import ModelRateLimitingCheck
from litellm.types.router import Deployment, FallbackSkipReason, LiteLLM_Params, ModelInfo, SkippedFallbackTarget
from litellm.types.utils import ProviderSpecificModelInfo


class TestFilterTeamBasedModels:
    """Test cases for filter_team_based_models function"""

    @pytest.fixture
    def sample_deployments_with_teams(self) -> List[Dict]:
        """Sample deployments where some have team_id and some don't"""
        return [
            {"model_info": {"id": "deployment-1", "team_id": "team-a"}},
            {"model_info": {"id": "deployment-2", "team_id": "team-b"}},
            {
                "model_info": {
                    "id": "deployment-3"
                    # No team_id - should always be included
                }
            },
            {"model_info": {"id": "deployment-4", "team_id": "team-a"}},
        ]

    @pytest.fixture
    def sample_deployments_no_teams(self) -> List[Dict]:
        """Sample deployments with no team_id restrictions"""
        return [
            {"model_info": {"id": "deployment-1"}},
            {"model_info": {"id": "deployment-2"}},
        ]

    def test_filter_team_based_models_none_request_kwargs(
        self, sample_deployments_with_teams
    ):
        """Test that when request_kwargs is None, all deployments are returned unchanged"""
        result = filter_team_based_models(sample_deployments_with_teams, None)
        assert result == sample_deployments_with_teams

    def test_filter_team_based_models_empty_request_kwargs(
        self, sample_deployments_with_teams
    ):
        """Test with empty request_kwargs"""
        result = filter_team_based_models(sample_deployments_with_teams, {})
        # Should include all deployments since no team_id in request
        assert len(result) == 1

    def test_filter_team_based_models_no_metadata(self, sample_deployments_with_teams):
        """Test with request_kwargs that has no metadata"""
        request_kwargs = {"some_other_key": "value"}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)
        # Should include only non-team based deployments
        assert len(result) == 1

    def test_filter_team_based_models_team_match_metadata(
        self, sample_deployments_with_teams
    ):
        """Test filtering when team_id is in metadata"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should include:
        # - deployment-1 (team-a matches)
        # - deployment-3 (no team_id restriction)
        # - deployment-4 (team-a matches)
        # Should exclude:
        # - deployment-2 (team-b doesn't match)
        expected_ids = ["deployment-1", "deployment-3", "deployment-4"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_team_match_litellm_metadata(
        self, sample_deployments_with_teams
    ):
        """Test filtering when team_id is in litellm_metadata"""
        request_kwargs = {"litellm_metadata": {"user_api_key_team_id": "team-b"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should include:
        # - deployment-2 (team-b matches)
        # - deployment-3 (no team_id restriction)
        # Should exclude:
        # - deployment-1 (team-a doesn't match)
        # - deployment-4 (team-a doesn't match)
        expected_ids = ["deployment-2", "deployment-3"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_priority_metadata_over_litellm(
        self, sample_deployments_with_teams
    ):
        """Test that metadata.user_api_key_team_id takes priority over litellm_metadata.user_api_key_team_id"""
        request_kwargs = {
            "metadata": {
                "user_api_key_team_id": "team-a",  # This should take priority
                "litellm_metadata": {"user_api_key_team_id": "team-b"},
            }
        }
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should filter based on team-a (from metadata, not litellm_metadata)
        expected_ids = ["deployment-1", "deployment-3", "deployment-4"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)

    def test_filter_team_based_models_no_matching_team(
        self, sample_deployments_with_teams
    ):
        """Test when request team doesn't match any deployment teams"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "team-nonexistent"}}
        result = filter_team_based_models(sample_deployments_with_teams, request_kwargs)

        # Should only include deployment-3 (no team_id restriction)
        expected_ids = ["deployment-3"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert result_ids == expected_ids

    def test_filter_team_based_models_no_team_restrictions(
        self, sample_deployments_no_teams
    ):
        """Test with deployments that have no team restrictions"""
        request_kwargs = {"metadata": {"user_api_key_team_id": "any-team"}}
        result = filter_team_based_models(sample_deployments_no_teams, request_kwargs)

        # Should include all deployments since none have team_id restrictions
        assert result == sample_deployments_no_teams

    def test_filter_team_based_models_missing_model_info(self):
        """Test with deployments missing model_info"""
        deployments = [
            {"model_info": {"id": "deployment-1", "team_id": "team-a"}},
            {
                # Missing model_info entirely
            },
            {
                "model_info": {
                    # Missing id
                    "team_id": "team-b"
                }
            },
        ]

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(deployments, request_kwargs)

        # Should handle missing model_info gracefully
        # deployment-1 should be included (team matches)
        # others should be included since they don't have proper team_id setup
        assert len(result) >= 1  # At least deployment-1 should be included

    def test_filter_team_based_models_dict_input(self):
        """Test with Dict input instead of List[Dict]"""
        # Note: Based on the function signature, it accepts Union[List[Dict], Dict]
        # But the implementation seems to expect List[Dict] for the filtering logic
        # This test documents the current behavior
        deployments_dict = {"key1": "value1", "key2": "value2"}

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}

        # This should not crash, though the filtering logic won't apply to Dict input
        result = filter_team_based_models(deployments_dict, request_kwargs)
        # The function will likely return the dict unchanged or handle it differently
        assert result is not None

    def test_filter_team_based_models_empty_deployments(self):
        """Test with empty deployments list"""
        result = filter_team_based_models(
            [], {"metadata": {"user_api_key_team_id": "team-a"}}
        )
        assert result == []

    def test_filter_team_based_models_none_team_id_in_deployment(self):
        """Test with explicit None team_id in deployment"""
        deployments = [
            {"model_info": {"id": "deployment-1", "team_id": None}},
            {"model_info": {"id": "deployment-2", "team_id": "team-a"}},
        ]

        request_kwargs = {"metadata": {"user_api_key_team_id": "team-a"}}
        result = filter_team_based_models(deployments, request_kwargs)

        # Both should be included:
        # - deployment-1 (None team_id is treated as no restriction)
        # - deployment-2 (team matches)
        expected_ids = ["deployment-1", "deployment-2"]
        result_ids = [d.get("model_info", {}).get("id") for d in result]
        assert sorted(result_ids) == sorted(expected_ids)


class TestDeploymentSupportsWebSearch:
    """Test cases for _deployment_supports_web_search helper function"""

    def test_model_info_true(self):
        """model_info.supports_web_search=True returns True"""
        deployment = {"model_info": {"supports_web_search": True}}
        assert _deployment_supports_web_search(deployment) is True

    def test_model_info_false(self):
        """model_info.supports_web_search=False returns False"""
        deployment = {"model_info": {"supports_web_search": False}}
        assert _deployment_supports_web_search(deployment) is False

    def test_no_config_defaults_to_true(self):
        """When no supports_web_search in config, default to True"""
        deployment = {"litellm_params": {"model": "gpt-4"}, "model_info": {"id": "123"}}
        assert _deployment_supports_web_search(deployment) is True

    def test_empty_deployment_defaults_to_true(self):
        """Empty deployment defaults to True"""
        assert _deployment_supports_web_search({}) is True

    def test_missing_model_info_defaults_to_true(self):
        """When model_info missing, default to True"""
        deployment = {"litellm_params": {"model": "gpt-4"}}
        assert _deployment_supports_web_search(deployment) is True

    @pytest.mark.parametrize("supports_web_search", [True, False])
    def test_model_info_object_setting_is_read(self, supports_web_search: bool) -> None:
        deployment: Final = {"model_info": ModelInfo(id="1", supports_web_search=supports_web_search)}
        assert _deployment_supports_web_search(deployment) is supports_web_search


class TestFilterWebSearchDeployments:
    """Test cases for filter_web_search_deployments function"""

    @pytest.fixture
    def sample_deployments(self) -> List[Dict]:
        """Sample deployments with varying web search support"""
        return [
            {"model_info": {"id": "deployment-1"}},  # default True
            {"model_info": {"id": "deployment-2", "supports_web_search": True}},
            {"model_info": {"id": "deployment-3", "supports_web_search": False}},
        ]

    def test_no_request_kwargs_returns_all(self, sample_deployments):
        """When request_kwargs is None, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, None)
        assert result == sample_deployments

    def test_no_tools_returns_all(self, sample_deployments):
        """When no tools in request, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, {"other": "value"})
        assert result == sample_deployments

    def test_empty_tools_returns_all(self, sample_deployments):
        """When tools list is empty, return all deployments"""
        result = filter_web_search_deployments(sample_deployments, {"tools": []})
        assert result == sample_deployments

    def test_none_tools_returns_all(self, sample_deployments):
        """When tools is explicitly None, return all deployments (regression test for #17672)"""
        result = filter_web_search_deployments(sample_deployments, {"tools": None})
        assert result == sample_deployments

    def test_non_web_search_tools_returns_all(self, sample_deployments):
        """When tools don't include web_search, return all deployments"""
        request_kwargs = {"tools": [{"type": "function", "function": {}}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert result == sample_deployments

    def test_web_search_filters_unsupported(self, sample_deployments):
        """When web_search tool present, filter out deployments that don't support it"""
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        # Should exclude deployment-3 (supports_web_search=False)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-1" in result_ids
        assert "deployment-2" in result_ids
        assert "deployment-3" not in result_ids

    def test_web_search_in_a_tuple_filters_unsupported(self, sample_deployments: list[dict[str, object]]) -> None:
        result: Final = filter_web_search_deployments(sample_deployments, {"tools": ({"type": "web_search"},)})
        assert [d["model_info"]["id"] for d in result] == ["deployment-1", "deployment-2"]

    @pytest.mark.parametrize("tool_type", [{}, [], None, 5], ids=["object", "list", "null", "int"])
    def test_a_non_string_tool_type_is_no_web_search_request(
        self, sample_deployments: list[dict[str, object]], tool_type: object
    ) -> None:
        result: Final = filter_web_search_deployments(
            sample_deployments, {"tools": [{"type": tool_type}, {"type": "function"}]}
        )
        assert result == sample_deployments

    def test_web_search_preview_filters_unsupported(self, sample_deployments):
        """web_search_preview type should also trigger filtering"""
        request_kwargs = {"tools": [{"type": "web_search_preview"}]}
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-3" not in result_ids

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_search_20260209", "name": "web_search"},
            {"type": "web_search_preview_2025_03_11"},
            {"googleSearch": {}},
            {"google_search": {}},
            {"googleSearchRetrieval": {}},
            {"google_search_retrieval": {}},
            {"enterpriseWebSearch": {}},
            {"enterprise_web_search": {}},
            {"type": "google_search"},
        ],
    )
    def test_native_web_search_tools_filter_unsupported(
        self, sample_deployments: list[dict[str, object]], tool: dict[str, object]
    ) -> None:
        result: Final = prefer_web_search_deployments(sample_deployments, {"tools": [tool]})
        assert result == sample_deployments[:2]

    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "name": "googleSearch"},
            {"urlContext": {}},
            {"googleSearch": None},
            {"google_search": None, "urlContext": {}},
        ],
    )
    def test_non_web_search_tools_preserve_deployments(
        self, sample_deployments: list[dict[str, object]], tool: dict[str, object]
    ) -> None:
        result: Final = prefer_web_search_deployments(sample_deployments, {"tools": [tool]})
        assert result == sample_deployments

    def test_web_search_with_other_tools(self, sample_deployments):
        """Web search filtering works when mixed with other tools"""
        request_kwargs = {
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "web_search"},
            ]
        }
        result = filter_web_search_deployments(sample_deployments, request_kwargs)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "deployment-3" not in result_ids

    def test_all_deployments_support_web_search(self):
        """When all deployments support web search, none are filtered"""
        deployments = [
            {"model_info": {"id": "d1", "supports_web_search": True}},
            {"model_info": {"id": "d2", "supports_web_search": True}},
        ]
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployments, request_kwargs)
        assert len(result) == 2

    @pytest.mark.parametrize("tool_type", ["web_search", "web_search_preview"])
    def test_no_deployment_supporting_openai_web_search_leaves_none(self, tool_type: str) -> None:
        deployments: Final = [
            {"model_info": {"id": "d1", "supports_web_search": False}},
            {"model_info": {"id": "d2", "supports_web_search": False}},
        ]
        assert filter_web_search_deployments(deployments, {"tools": [{"type": tool_type}]}) == []

    @pytest.mark.parametrize(
        "request_kwargs",
        [
            {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
            {"tools": [{"type": "web_search_2025_08_26"}]},
            {"tools": [{"googleSearch": {}}]},
            {"tools": [{"type": "web_search"}]},
            {"web_search_options": {}},
        ],
    )
    def test_no_deployment_supporting_web_search_keeps_every_deployment(self, request_kwargs: dict[str, object]) -> None:
        deployments: Final[list[dict[str, object]]] = [
            {"model_info": {"id": "d1", "supports_web_search": False}},
            {"model_info": {"id": "d2", "supports_web_search": False}},
        ]
        assert prefer_web_search_deployments(deployments, request_kwargs) == deployments

    def test_a_tool_that_is_not_an_object_is_no_web_search_request(
        self, sample_deployments: list[dict[str, object]]
    ) -> None:
        result: Final = prefer_web_search_deployments(sample_deployments, {"tools": ["web_search", None]})
        assert result == sample_deployments

    def test_web_search_options_filter_unsupported(self, sample_deployments: list[dict[str, object]]) -> None:
        result: Final = prefer_web_search_deployments(sample_deployments, {"web_search_options": {}})
        assert result == sample_deployments[:2]

    def test_null_web_search_options_preserve_deployments(self, sample_deployments: list[dict[str, object]]) -> None:
        result: Final = prefer_web_search_deployments(sample_deployments, {"web_search_options": None})
        assert result == sample_deployments

    def test_missing_config_defaults_to_supported(self):
        """Deployments without supports_web_search config default to True"""
        deployments = [
            {"model_info": {"id": "d1"}},  # No supports_web_search - defaults to True
            {"model_info": {"id": "d2"}},  # No supports_web_search - defaults to True
            {
                "model_info": {"id": "d3", "supports_web_search": False}
            },  # Explicit False
        ]
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployments, request_kwargs)
        # d1 and d2 should be included (default True), d3 excluded (explicit False)
        assert len(result) == 2
        result_ids = [d["model_info"]["id"] for d in result]
        assert "d1" in result_ids
        assert "d2" in result_ids
        assert "d3" not in result_ids

    def test_empty_deployments_list(self):
        """Empty deployments list returns empty list"""
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments([], request_kwargs)
        assert result == []

    def test_dict_deployment_passthrough(self):
        """When deployment is a dict (single deployment), pass through unchanged"""
        deployment = {"model_info": {"id": "d1", "supports_web_search": False}}
        request_kwargs = {"tools": [{"type": "web_search"}]}
        result = filter_web_search_deployments(deployment, request_kwargs)
        # Should return the dict unchanged, not filter it
        assert result == deployment


def _router_with_a_deployment_that_disables_web_search(routing_strategy: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "search-model",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key"},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
            {
                "model_name": "search-model",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key"},
                "model_info": {"id": "search"},
            },
        ],
        routing_strategy=routing_strategy,
    )


_ANTHROPIC_WEB_SEARCH_REQUEST: Final = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}


@pytest.mark.parametrize("routing_strategy", ["simple-shuffle", "usage-based-routing"])
def test_sync_pick_skips_deployments_without_web_search(routing_strategy: str) -> None:
    router: Final = _router_with_a_deployment_that_disables_web_search(routing_strategy)
    picks: Final = {
        router.get_available_deployment(model="search-model", request_kwargs=dict(_ANTHROPIC_WEB_SEARCH_REQUEST))[
            "model_info"
        ]["id"]
        for _ in range(20)
    }
    assert picks == {"search"}


def _router_with_a_group_that_disables_web_search(routing_strategy: str = "simple-shuffle") -> Router:
    return Router(
        model_list=[
            {
                "model_name": "flagged-model",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "mock_response": "flagged"},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
            {
                "model_name": "search-model",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "searched"},
                "model_info": {"id": "search"},
            },
        ],
        fallbacks=[{"flagged-model": ["search-model"]}],
        routing_strategy=routing_strategy,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "expected_content"),
    [
        ([{"type": "web_search"}], "searched"),
        ([{"type": "web_search_20250305", "name": "web_search"}], "flagged"),
        ([{"googleSearch": dict[str, object]()}], "flagged"),
    ],
)
async def test_web_search_to_a_group_without_web_search_routes_like_main(
    tools: list[dict[str, object]], expected_content: str
) -> None:
    response: Final = await _router_with_a_group_that_disables_web_search().acompletion(
        model="flagged-model", messages=[{"role": "user", "content": "hi"}], tools=tools
    )
    assert response.choices[0].message.content == expected_content


@pytest.mark.parametrize("routing_strategy", ["simple-shuffle", "usage-based-routing"])
def test_sync_pick_keeps_a_group_without_web_search_for_openai_web_search(routing_strategy: str) -> None:
    router: Final = _router_with_a_group_that_disables_web_search(routing_strategy)
    deployment: Final = router.get_available_deployment(
        model="flagged-model", request_kwargs={"tools": [{"type": "web_search"}]}
    )
    assert deployment["model_info"]["id"] == "no-search"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "routing_strategy", ["usage-based-routing", "usage-based-routing-v2", "least-busy", "latency-based-routing"]
)
async def test_async_strategy_pick_skips_deployments_without_web_search(routing_strategy: str) -> None:
    router: Final = _router_with_a_deployment_that_disables_web_search(routing_strategy)
    picks: Final = {
        (
            await router.async_get_available_deployment(
                model="search-model", request_kwargs=dict(_ANTHROPIC_WEB_SEARCH_REQUEST)
            )
        )["model_info"]["id"]
        for _ in range(20)
    }
    assert picks == {"search"}


def _order_param(order: int | None) -> dict[str, int]:
    return {} if order is None else {"order": order}


def _router_with_a_mixed_web_search_group(
    enable_tag_filtering: bool, orders: tuple[int | None, int | None] = (None, None)
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "tags": ["cheap"],
                    **_order_param(orders[0]),
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "tags": ["premium"],
                    **_order_param(orders[1]),
                },
                "model_info": {"id": "search"},
            },
        ],
        enable_tag_filtering=enable_tag_filtering,
    )


def _cool_down(router: Router, model_ids: tuple[str, ...]) -> None:
    for model_id in model_ids:
        router.cooldown_cache.add_deployment_to_cooldown(
            model_id=model_id, original_exception=Exception("rate limited"), exception_status=429, cooldown_time=300
        )


_LATER_FILTER_CASES: Final = pytest.mark.parametrize(
    ("orders", "cooled_down", "request_kwargs", "expected_id"),
    [
        ((None, None), (), _ANTHROPIC_WEB_SEARCH_REQUEST, "search"),
        ((None, None), (), {"web_search_options": {}}, "search"),
        ((1, 2), (), _ANTHROPIC_WEB_SEARCH_REQUEST, "no-search"),
        ((1, 2), (), {"web_search_options": {}}, "no-search"),
        ((None, None), ("search",), _ANTHROPIC_WEB_SEARCH_REQUEST, "no-search"),
        ((None, None), ("search",), {"web_search_options": {}}, "no-search"),
        ((None, None), (), {**_ANTHROPIC_WEB_SEARCH_REQUEST, "_excluded_deployment_ids": ["search"]}, "no-search"),
        ((None, None), (), {**_ANTHROPIC_WEB_SEARCH_REQUEST, "_retry_skipped_deployment_ids": ("search",)}, "no-search"),
    ],
    ids=[
        "same-tier",
        "same-tier-options",
        "lower-order-tier",
        "lower-order-tier-options",
        "cooldown",
        "cooldown-options",
        "weighted-failover",
        "retry-skip",
    ],
)


@pytest.mark.asyncio
@_LATER_FILTER_CASES
async def test_async_pick_prefers_web_search_until_later_filters_leave_none_that_can(
    orders: tuple[int | None, int | None],
    cooled_down: tuple[str, ...],
    request_kwargs: dict[str, object],
    expected_id: str,
) -> None:
    router: Final = _router_with_a_mixed_web_search_group(False, orders)
    _cool_down(router, cooled_down)
    deployment: Final = await router.async_get_available_deployment(
        model="mixed", messages=[{"role": "user", "content": "hi"}], request_kwargs=dict(request_kwargs)
    )
    assert deployment["model_info"]["id"] == expected_id


@_LATER_FILTER_CASES
def test_sync_pick_prefers_web_search_until_later_filters_leave_none_that_can(
    orders: tuple[int | None, int | None],
    cooled_down: tuple[str, ...],
    request_kwargs: dict[str, object],
    expected_id: str,
) -> None:
    router: Final = _router_with_a_mixed_web_search_group(False, orders)
    _cool_down(router, cooled_down)
    deployment: Final = router.get_available_deployment(
        model="mixed", messages=[{"role": "user", "content": "hi"}], request_kwargs=dict(request_kwargs)
    )
    assert deployment["model_info"]["id"] == expected_id


_WEB_SEARCH_REQUESTS: Final = pytest.mark.parametrize(
    "web_search_request",
    [_ANTHROPIC_WEB_SEARCH_REQUEST, {"web_search_options": {}}],
    ids=["web-search-tool", "web-search-options"],
)


@pytest.mark.asyncio
@_WEB_SEARCH_REQUESTS
async def test_order_fallback_answers_a_web_search_request_from_the_next_tier_when_the_capable_one_fails(
    web_search_request: dict[str, object],
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "ordered",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": Exception("connection error"),
                    "order": 1,
                },
                "model_info": {"id": "search"},
            },
            {
                "model_name": "ordered",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "hi", "order": 2},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        num_retries=0,
    )
    response: Final = await router.acompletion(
        model="ordered", messages=[{"role": "user", "content": "hi"}], **web_search_request
    )
    assert response._hidden_params["model_id"] == "no-search"


@pytest.mark.asyncio
@_WEB_SEARCH_REQUESTS
async def test_web_search_request_prefers_the_capable_deployment_inside_its_order_tier(
    web_search_request: dict[str, object],
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "ordered",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "order": 1},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
            {
                "model_name": "ordered",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "order": 1},
                "model_info": {"id": "search"},
            },
            {
                "model_name": "ordered",
                "litellm_params": {"model": "openai/gpt-z", "api_key": "fake-key", "order": 2},
                "model_info": {"id": "later-no-search", "supports_web_search": False},
            },
        ],
    )
    picks: Final = {
        (
            await router.async_get_available_deployment(
                model="ordered", messages=[{"role": "user", "content": "hi"}], request_kwargs=dict(web_search_request)
            )
        )["model_info"]["id"]
        for _ in range(20)
    }
    assert picks == {"search"}


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
async def test_web_search_retry_goes_to_an_untried_deployment_once_the_capable_deployment_failed_this_request(
    metadata_field: str,
) -> None:
    router: Final = _router_with_a_mixed_web_search_group(False)
    picks: Final = {
        (
            await router.async_get_available_deployment(
                model="mixed",
                messages=[{"role": "user", "content": "hi"}],
                request_kwargs={
                    **_ANTHROPIC_WEB_SEARCH_REQUEST,
                    metadata_field: {"previous_models": ({"deployment_id": "search", "exception_type": "Timeout"},)},
                },
            )
        )["model_info"]["id"]
        for _ in range(40)
    }
    assert picks == {"no-search"}



@pytest.mark.parametrize(
    ("deployments", "failed", "expected"),
    [
        (
            (
                {"id": "search"},
                {"id": "flag-a", "supports_web_search": False},
                {"id": "flag-b", "supports_web_search": False},
            ),
            ("search", "flag-a"),
            ("flag-b",),
        ),
        (
            ({"id": "search"}, {"id": "flag-a", "supports_web_search": False}),
            ("search", "flag-a"),
            ("search", "flag-a"),
        ),
        (
            ({"id": "flag-a", "supports_web_search": False}, {"id": "flag-b", "supports_web_search": False}),
            ("flag-a",),
            ("flag-a", "flag-b"),
        ),
    ],
    ids=["skips-every-tried-one", "keeps-all-once-all-were-tried", "no-deployment-can-search"],
)
def test_web_search_retry_prefers_untried_deployments_only_after_one_that_can_search_was_tried(
    deployments: tuple[dict[str, object], ...], failed: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    request_kwargs: Final = {
        **_ANTHROPIC_WEB_SEARCH_REQUEST,
        "metadata": {"previous_models": tuple({"deployment_id": d, "exception_type": "Timeout"} for d in failed)},
    }
    kept: Final = prefer_web_search_deployments([{"model_info": info} for info in deployments], request_kwargs)
    assert tuple(d["model_info"]["id"] for d in kept) == expected


@pytest.mark.parametrize(
    ("failed", "at_limit", "expected"),
    [((), ("search",), ("search", "flag")), (("search",), (), ("flag",))],
    ids=["at-its-limit-keeps-all", "failed-keeps-the-untried-one"],
)
def test_a_web_search_deployment_only_at_its_limit_leaves_every_deployment_to_the_weights_and_fallbacks(
    failed: tuple[str, ...], at_limit: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    request_kwargs: Final = {
        **_ANTHROPIC_WEB_SEARCH_REQUEST,
        "metadata": {"previous_models": tuple({"deployment_id": d, "exception_type": "Timeout"} for d in failed)},
    }
    infos: Final = ({"id": "search"}, {"id": "flag", "supports_web_search": False})
    kept: Final = prefer_web_search_deployments(
        [{"model_info": info} for info in infos], request_kwargs, frozenset(at_limit)
    )
    assert tuple(d["model_info"]["id"] for d in kept) == expected

@pytest.mark.asyncio
@_WEB_SEARCH_REQUESTS
@pytest.mark.parametrize(
    "caller_credentials", [{}, {"api_key": "caller-key"}], ids=["configured-credentials", "caller-credentials"]
)
@pytest.mark.parametrize(
    "operator_model_info", [{}, {"original_model_id": "set-by-operator"}], ids=["plain", "operator-original-model-id"]
)
async def test_retry_after_a_retryable_failure_on_the_capable_deployment_reaches_the_other_one(
    web_search_request: dict[str, object],
    caller_credentials: dict[str, str],
    operator_model_info: dict[str, str],
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": Exception("connection error"),
                    "weight": 1,
                },
                "model_info": {"id": "search", **operator_model_info},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "from no-search",
                    "weight": 10**6,
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        num_retries=1,
        disable_cooldowns=True,
    )
    response: Final = await router.acompletion(
        model="mixed", messages=[{"role": "user", "content": "hi"}], **web_search_request, **caller_credentials
    )
    assert response.choices[0].message.content == "from no-search"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_field", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize(("request_retry_count", "flagged_reachable"), [(1, False), (2, True)], ids=["kept", "trimmed"])
async def test_web_search_preference_steps_aside_only_once_the_retry_history_was_trimmed(
    metadata_field: str, request_retry_count: int, flagged_reachable: bool
) -> None:
    router: Final = Router(
        model_list=[
            *(
                {
                    "model_name": "mixed",
                    "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "mock_response": "hi"},
                    "model_info": {"id": deployment_id},
                }
                for deployment_id in ("search-a", "search-b")
            ),
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "weight": 10**6,
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
    )
    picks: Final = {
        (
            await router.async_get_available_deployment(
                model="mixed",
                messages=[{"role": "user", "content": "hi"}],
                request_kwargs={
                    **_ANTHROPIC_WEB_SEARCH_REQUEST,
                    metadata_field: {
                        "previous_models": ({"deployment_id": "search-a", "exception_type": "Timeout"},),
                        "request_retry_count": request_retry_count,
                    },
                },
            )
        )["model_info"]["id"]
        for _ in range(20)
    }
    assert ("no-search" in picks) is flagged_reachable


@pytest.mark.asyncio
async def test_retries_past_the_kept_retry_history_reach_the_deployment_that_cannot_search() -> None:
    capable_deployments: Final = [
        {
            "model_name": "mixed",
            "litellm_params": {
                "model": "openai/gpt-x",
                "api_key": "fake-key",
                "mock_response": Exception("connection error"),
            },
            "model_info": {"id": f"search-{index}"},
        }
        for index in range(5)
    ]
    router: Final = Router(
        model_list=[
            *capable_deployments,
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "from no-search",
                    "weight": 10**6,
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        num_retries=5,
        disable_cooldowns=True,
    )
    response: Final = await router.acompletion(
        model="mixed", messages=[{"role": "user", "content": "hi"}], **_ANTHROPIC_WEB_SEARCH_REQUEST
    )
    assert response.choices[0].message.content == "from no-search"


@pytest.mark.asyncio
async def test_retry_after_a_capable_deployment_fails_goes_to_another_capable_one_before_any_other() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": Exception("connection error"),
                    "weight": 10**6,
                },
                "model_info": {"id": "dead-search"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "mock_response": "from live-search"},
                "model_info": {"id": "live-search"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "from no-search"},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        num_retries=1,
        disable_cooldowns=True,
    )
    replies: Final = tuple(
        [
            (
                await router.acompletion(
                    model="mixed", messages=[{"role": "user", "content": "hi"}], **_ANTHROPIC_WEB_SEARCH_REQUEST
                )
            )
            .choices[0]
            .message.content
            for _ in range(5)
        ]
    )
    assert replies == ("from live-search",) * 5


@pytest.mark.asyncio
@pytest.mark.parametrize("web_search_options", [False, "", 0, []], ids=["false", "empty-string", "zero", "list"])
async def test_web_search_options_that_is_not_an_object_routes_like_a_plain_request(
    web_search_options: object,
) -> None:
    router: Final = _router_with_a_mixed_web_search_group(False)
    picks: Final = {
        (
            await router.async_get_available_deployment(
                model="mixed",
                messages=[{"role": "user", "content": "hi"}],
                request_kwargs={"web_search_options": web_search_options},
            )
        )["model_info"]["id"]
        for _ in range(40)
    }
    assert picks == {"search", "no-search"}


_SEARCH_RPM_OF_ONE: Final = (
    pytest.param({"litellm_params": {"rpm": 1}}, id="int-rpm-in-litellm-params"),
    pytest.param({"model_info": {"rpm": 1.0}}, id="float-rpm-in-model-info"),
    pytest.param({"model_info": {"rpm": True}}, id="bool-rpm-in-model-info"),
)


_ONE_MINUTE: Final = datetime(2026, 10, 28, 12, 0, 30, tzinfo=timezone.utc)


def _web_search_group_with_model_rate_limits(
    search_rpm: Mapping[str, Mapping[str, object]] = MappingProxyType({"litellm_params": {"rpm": 1}}),
) -> Router:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-x",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    **search_rpm.get("litellm_params", {}),
                },
                "model_info": {"id": "search", **search_rpm.get("model_info", {})},
            },
            {
                "model_name": "mixed",
                "litellm_params": {
                    "model": "openai/gpt-y",
                    "api_key": "fake-key",
                    "mock_response": "hi",
                    "rpm": 10**6,
                },
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
    )
    check: Final = ModelRateLimitingCheck(dual_cache=DualCache(), clock=lambda: _ONE_MINUTE)
    router.optional_callbacks = [check]
    litellm.logging_callback_manager.add_litellm_callback(check)
    return router


@pytest.mark.asyncio
@pytest.mark.parametrize("search_rpm", _SEARCH_RPM_OF_ONE)
async def test_web_search_request_goes_to_the_other_deployment_once_the_capable_one_hits_its_enforced_rpm(
    search_rpm: Mapping[str, Mapping[str, object]],
) -> None:
    router: Final = _web_search_group_with_model_rate_limits(search_rpm)
    served_by: Final = tuple(
        [
            (
                await router.acompletion(
                    model="mixed", messages=[{"role": "user", "content": "hi"}], **_ANTHROPIC_WEB_SEARCH_REQUEST
                )
            )._hidden_params["model_id"]
            for _ in range(3)
        ]
    )
    assert served_by == ("search", "no-search", "no-search")


@pytest.mark.parametrize("search_rpm", _SEARCH_RPM_OF_ONE)
def test_sync_web_search_request_goes_to_the_other_deployment_once_the_capable_one_hits_its_enforced_rpm(
    search_rpm: Mapping[str, Mapping[str, object]],
) -> None:
    router: Final = _web_search_group_with_model_rate_limits(search_rpm)
    served_by: Final = tuple(
        router.completion(
            model="mixed", messages=[{"role": "user", "content": "hi"}], num_retries=0, **_ANTHROPIC_WEB_SEARCH_REQUEST
        )._hidden_params["model_id"]
        for _ in range(3)
    )
    assert served_by == ("search", "no-search", "no-search")


@pytest.mark.asyncio
@pytest.mark.parametrize("zero_limit", ["rpm", "itpm", "otpm"])
async def test_web_search_request_never_goes_to_a_capable_deployment_whose_enforced_limit_is_zero(
    zero_limit: str,
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "mock_response": "hi", zero_limit: 0},
                "model_info": {"id": "search"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "hi", "rpm": 1000},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        optional_pre_call_checks=["enforce_model_rate_limits"],
    )
    served_by: Final = tuple(
        [
            (
                await router.acompletion(
                    model="mixed",
                    messages=[{"role": "user", "content": "hi"}],
                    num_retries=0,
                    **_ANTHROPIC_WEB_SEARCH_REQUEST,
                )
            )._hidden_params["model_id"]
            for _ in range(5)
        ]
    )
    assert served_by == ("no-search",) * 5


def _rate_limited_web_search_group(routing_strategy: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key", "mock_response": "hi", "tpm": 1},
                "model_info": {"id": "search"},
            },
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key", "mock_response": "hi", "rpm": 1000},
                "model_info": {"id": "no-search", "supports_web_search": False},
            },
        ],
        routing_strategy=routing_strategy,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["usage-based-routing-v2", "cost-based-routing"])
async def test_web_search_request_goes_to_the_other_deployment_when_the_capable_one_has_no_tpm_room(
    routing_strategy: str,
) -> None:
    router: Final = _rate_limited_web_search_group(routing_strategy)
    response: Final = await router.acompletion(
        model="mixed", messages=[{"role": "user", "content": "hi"}], **_ANTHROPIC_WEB_SEARCH_REQUEST
    )
    assert response._hidden_params["model_id"] == "no-search"


def test_sync_web_search_request_goes_to_the_other_deployment_when_the_capable_one_has_no_tpm_room() -> None:
    router: Final = _rate_limited_web_search_group("usage-based-routing-v2")
    response: Final = router.completion(
        model="mixed", messages=[{"role": "user", "content": "hi"}], **_ANTHROPIC_WEB_SEARCH_REQUEST
    )
    assert response._hidden_params["model_id"] == "no-search"


def test_sync_pick_falls_back_to_every_deployment_when_the_strategy_finds_none_that_can_search() -> None:
    deployments: Final = [
        {"model_info": {"id": "search"}},
        {"model_info": {"id": "no-search", "supports_web_search": False}},
    ]
    picked: Final = pick_preferring_web_search(
        lambda candidates: next((d for d in candidates if d["model_info"]["id"] == "no-search"), None),
        deployments,
        _ANTHROPIC_WEB_SEARCH_REQUEST,
    )
    assert picked == deployments[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("search_order", "no_search_order"),
    [([1], [1]), ({"tier": 1}, None)],
    ids=["list-order", "dict-order"],
)
async def test_web_search_request_routes_on_a_group_whose_order_is_not_a_number(
    search_order: object, no_search_order: object
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-x", "api_key": "fake-key"},
                "model_info": {"id": "search", "order": search_order},
            },
            {
                "model_name": "mixed",
                "litellm_params": {"model": "openai/gpt-y", "api_key": "fake-key"},
                "model_info": {
                    "id": "no-search",
                    "supports_web_search": False,
                    **({} if no_search_order is None else {"order": no_search_order}),
                },
            },
        ],
    )
    deployment: Final = await router.async_get_available_deployment(
        model="mixed", messages=[{"role": "user", "content": "hi"}], request_kwargs=dict(_ANTHROPIC_WEB_SEARCH_REQUEST)
    )
    assert deployment["model_info"]["id"] == "search"


@pytest.mark.asyncio
async def test_tag_routing_to_a_deployment_without_web_search_keeps_it() -> None:
    deployment: Final = await _router_with_a_mixed_web_search_group(True).async_get_available_deployment(
        model="mixed",
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs={"web_search_options": {}, "metadata": {"tags": ["cheap"]}},
    )
    assert deployment["model_info"]["id"] == "no-search"


@pytest.mark.asyncio
async def test_openai_web_search_still_finds_no_deployment_while_the_only_one_that_can_search_cools_down() -> None:
    router: Final = _router_with_a_mixed_web_search_group(False)
    _cool_down(router, ("search",))
    with pytest.raises(litellm.RouterRateLimitError):
        await router.async_get_available_deployment(
            model="mixed",
            messages=[{"role": "user", "content": "hi"}],
            request_kwargs={"tools": [{"type": "web_search"}]},
        )


def test_invalidate_model_group_info_cache():
    """Test that _invalidate_model_group_info_cache clears the LRU cache."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ]
    )
    # Populate the cache
    router._cached_get_model_group_info("gpt-4")
    assert router._cached_get_model_group_info.cache_info().currsize > 0

    # Invalidate and verify cache is cleared
    router._invalidate_model_group_info_cache()
    assert router._cached_get_model_group_info.cache_info().currsize == 0


def test_filter_deployments_by_model_access_groups_access_group_only_key():
    """
    Access-group-only keys should only route to deployments in allowed groups,
    even when multiple deployments share the same public model name.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {"model": "openai/gpt-5.1", "api_key": "key-1"},
                "model_info": {"access_groups": ["AG1"]},
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "key-2"},
                "model_info": {"access_groups": ["AG2"]},
            },
        ]
    )

    scoped_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team-2",
        models=["AG2"],
        team_models=["AG2"],
    )

    filtered = router._filter_deployments_by_model_access_groups(
        model="gpt-5",
        healthy_deployments=router.get_all_deployments(model_name="gpt-5"),
        request_kwargs={
            "metadata": {
                "user_api_key_team_id": "team-2",
                "user_api_key_auth": scoped_key,
            }
        },
        request_team_id="team-2",
    )

    assert len(filtered) == 1
    assert filtered[0].get("model_info", {}).get("access_groups") == ["AG2"]


class TestAddModelFileIdMappings:
    """Test cases for add_model_file_id_mappings.

    The router may pass either a list of deployment dicts (multiple matched
    deployments) or a single deployment dict (when a specific deployment was
    resolved, e.g. because the requested model matched a `model_info.id`).
    Both shapes must produce a `{model_id: file_id}` mapping by extracting
    `model_info.id` from each deployment.
    """

    @staticmethod
    def _make_response(file_id: str):
        response = Mock()
        response.id = file_id
        return response

    def test_should_map_each_deployment_id_when_given_list(self):
        deployments = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "deployment-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "deployment-2"},
            },
        ]
        responses = [self._make_response("file-1"), self._make_response("file-2")]

        result = add_model_file_id_mappings(deployments, responses)

        assert result == {"deployment-1": "file-1", "deployment-2": "file-2"}

    def test_should_extract_model_info_id_when_given_single_deployment_dict(self):
        """Regression test: when `_common_checks_available_deployment` resolves
        a specific deployment (returned as a dict, not a list), the function
        must still extract `model_info.id` rather than iterate over the
        deployment's own keys (`model_name`, `litellm_params`, `model_info`).
        """
        deployment = {
            "model_name": "gpt-4",
            "litellm_params": {"model": "gpt-4", "api_key": "sk-test"},
            "model_info": {"id": "deployment-1", "mode": "chat"},
        }
        responses = [self._make_response("file-1")]

        result = add_model_file_id_mappings(deployment, responses)

        assert result == {"deployment-1": "file-1"}
        assert all(isinstance(v, str) for v in result.values())

    def test_should_handle_batch_model_when_id_matches_model_name(self):
        """Regression test for the batch-model case: when `model_info.id` is
        intentionally set equal to `model_name`, the router resolves a single
        deployment via `has_model_id` and returns it as a dict. The mapping
        must contain only `{id: file_id}` with string values so the resulting
        `LiteLLM_ManagedFileTable` Pydantic validation passes.
        """
        deployment = {
            "model_name": "openai/openai/gpt-5.5-batch",
            "litellm_params": {
                "model": "openai/gpt-5.5",
                "api_key": "sk-test",
                "tpm": 40000000,
                "rpm": 15000,
            },
            "model_info": {
                "id": "openai/openai/gpt-5.5-batch",
                "mode": "batch",
                "base_model": "gpt-5.5",
                "access_groups": ["default-models"],
            },
        }
        responses = [self._make_response("file-batch-1")]

        result = add_model_file_id_mappings(deployment, responses)

        # Bug case would have produced keys ["model_name", "litellm_params",
        # "model_info"] with non-string values.
        assert result == {"openai/openai/gpt-5.5-batch": "file-batch-1"}
        assert "litellm_params" not in result
        assert "model_info" not in result

    def test_should_skip_deployment_when_model_info_id_missing(self):
        deployments = [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"id": "deployment-2"},
            },
        ]
        responses = [self._make_response("file-1"), self._make_response("file-2")]

        result = add_model_file_id_mappings(deployments, responses)

        assert result == {"deployment-2": "file-2"}

    def test_should_return_empty_mapping_when_given_empty_list(self):
        result = add_model_file_id_mappings([], [])
        assert result == {}


class TestResolveModelGroupAlias:
    """``model_group_alias`` maps reach this helper from validated config and
    from key/team rows, so both entry shapes must resolve and malformed entries
    must not raise mid-request."""

    @pytest.mark.parametrize(
        "alias_map, expected",
        [
            ({"group-a": "group-b"}, "group-b"),
            ({"group-a": {"model": "group-b", "hidden": True}}, "group-b"),
            ({"group-a": {"model": "group-b"}}, "group-b"),
            ({"other": "group-b"}, None),
            ({}, None),
            (None, None),
            ("not-a-map", None),
            ({"group-a": {"hidden": True}}, None),
            ({"group-a": {"model": 5}}, None),
            ({"group-a": 5}, None),
            ({"group-a": None}, None),
            ({"group-a": ""}, None),
        ],
    )
    def test_resolves_both_entry_shapes_and_tolerates_malformed_entries(self, alias_map, expected):
        assert resolve_model_group_alias(alias_map, "group-a") == expected

    def test_router_alias_resolution_uses_the_shared_helper(self):
        router = Router(
            model_list=[
                {
                    "model_name": "group-b",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
                }
            ],
            model_group_alias={"group-a": "group-b", "group-item": {"model": "group-b", "hidden": True}},
        )

        assert router.get_model_from_alias("group-a") == "group-b"
        assert router.get_model_from_alias("group-item") == "group-b"
        assert router.get_model_from_alias("group-b") is None


@pytest.mark.parametrize(
    "model, model_alias_map, key_aliases, model_group_alias, expected",
    [
        pytest.param("a", {"a": "b"}, {"b": "c"}, {"c": "d"}, "d", id="global-key-router-order"),
        pytest.param("a", {}, ["a", "b"], {}, "a", id="non-mapping-key-aliases-ignored"),
        pytest.param("a", {}, {}, {"a": {"model": "x"}}, "x", id="router-item-entry"),
        pytest.param("a", {}, {}, {}, "a", id="no-alias"),
    ],
)
def test_resolves_served_model_aliases_in_order(
    model: str,
    model_alias_map: object,
    key_aliases: object,
    model_group_alias: object,
    expected: str,
) -> None:
    assert resolve_served_model(model, model_alias_map, key_aliases, model_group_alias) == expected


class TestTruncateFallbackErrorDetail:
    def test_short_detail_is_returned_unchanged(self):
        assert truncate_fallback_error_detail("boom") == "boom"

    def test_detail_at_the_limit_is_returned_unchanged(self):
        detail = "x" * ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS
        assert truncate_fallback_error_detail(detail) == detail

    def test_long_detail_is_bounded_and_reports_what_was_dropped(self):
        detail = "x" * (ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS + 500)

        truncated = truncate_fallback_error_detail(detail)

        assert truncated.startswith("x" * ROUTER_FALLBACK_ERROR_DETAIL_MAX_CHARS)
        assert truncated.endswith("... [truncated 500 characters]")
        assert len(truncated) < len(detail)

    def test_a_megabyte_of_detail_comes_back_small(self):
        """The detail is what a fallback level records about the level below it, so it has
        to stay small enough that a walk over many model groups cannot compound it into an
        output volume that starves the process."""
        assert len(truncate_fallback_error_detail("x" * 1_000_000)) < 3_000


def test_format_fallback_outcome_reports_unauthorized_skips_without_failure_detail() -> None:
    message: Final = format_fallback_outcome_message(
        "primary",
        ("restricted-model",),
        "",
        skipped=(SkippedFallbackTarget("restricted-model", FallbackSkipReason.NOT_AUTHORIZED),),
    )

    assert "Fallback to restricted-model was skipped because the caller is not authorized to call it." in message
    assert "also failed" not in message


def test_format_fallback_outcome_separates_attempted_failures_from_skipped_targets() -> None:
    message: Final = format_fallback_outcome_message(
        "primary",
        ("restricted-model", "dead-allowed"),
        "boom",
        skipped=(SkippedFallbackTarget("restricted-model", FallbackSkipReason.NOT_AUTHORIZED),),
    )
    failed_sentence: Final = message.split("Fallback to dead-allowed also failed:", 1)[1].split(
        " Fallback to restricted-model was skipped", 1
    )[0]

    assert "Fallback to dead-allowed also failed: boom" in message
    assert "Fallback to restricted-model was skipped because the caller is not authorized" in message
    assert "restricted-model" not in failed_sentence


def test_format_fallback_outcome_uses_plural_for_multiple_over_budget_skips() -> None:
    message: Final = format_fallback_outcome_message(
        "primary",
        ("paid-one", "paid-two"),
        "",
        skipped=(
            SkippedFallbackTarget("paid-one", FallbackSkipReason.OVER_BUDGET),
            SkippedFallbackTarget("paid-two", FallbackSkipReason.OVER_BUDGET),
        ),
    )

    assert "Fallback to paid-one, paid-two was skipped because the caller is over budget for them." in message


def test_format_fallback_outcome_preserves_attempted_failure_wording_without_skips() -> None:
    assert (
        format_fallback_outcome_message("primary", ("fallback",), "detail")
        == "\n\nLiteLLM: model group 'primary' failed with the error above. "
        "Fallback to fallback also failed: detail"
    )


def test_content_policy_note_explains_the_failure_when_fallbacks_are_a_plain_model_list() -> None:
    note: Final = format_content_policy_no_fallback_note("gpt-bg", ("gpt-bg",), None, ["backup"], False)

    assert "content_policy_fallback=None" in note
    assert "LiteLLM: model group 'gpt-bg' failed with the error above. No fallback was attempted." in note
    assert "Fallbacks are configured for" not in note


class TestWarnOnProviderCredentialMismatch:
    """A deployment that carries one provider's credentials while resolving to
    another is silently broken: litellm ignores the credentials and sends the
    request to the resolved provider, which 401s. The classic shape is a bedrock
    model group where one entry lost its route prefix, which fails only on the
    requests the router happens to send to that entry."""

    def test_warns_when_aws_params_sit_on_an_anthropic_model(self):
        warning = warn_on_provider_credential_mismatch(
            model_name="claude-sonnet-5",
            litellm_params={"model": "claude-sonnet-5", "aws_region_name": "eu-central-1"},
        )

        assert warning is not None
        assert "aws_region_name" in warning
        assert "anthropic" in warning
        assert "bedrock/claude-sonnet-5" in warning

    def test_silent_when_the_prefix_is_present(self):
        assert (
            warn_on_provider_credential_mismatch(
                model_name="claude-sonnet-5",
                litellm_params={
                    "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "aws_region_name": "eu-central-1",
                },
            )
            is None
        )

    def test_silent_when_custom_llm_provider_supplies_the_route(self):
        """An operator may name the provider explicitly instead of prefixing the
        model; that is consistent and must not warn."""
        assert (
            warn_on_provider_credential_mismatch(
                model_name="claude-sonnet-5",
                litellm_params={
                    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "custom_llm_provider": "bedrock",
                    "aws_region_name": "eu-central-1",
                },
            )
            is None
        )

    def test_silent_when_no_provider_scoped_credentials_are_set(self):
        assert (
            warn_on_provider_credential_mismatch(
                model_name="gpt-5.5", litellm_params={"model": "gpt-5.5"}
            )
            is None
        )

    def test_vertex_params_name_vertex_not_bedrock(self):
        """The hint must follow the params that were actually set, otherwise it
        sends the operator to the wrong prefix."""
        warning = warn_on_provider_credential_mismatch(
            model_name="claude-on-vertex",
            litellm_params={"model": "claude-sonnet-5", "vertex_project": "my-project"},
        )

        assert warning is not None
        assert "vertex_ai/claude-sonnet-5" in warning
        assert "bedrock" not in warning

    def test_silent_for_a_model_litellm_cannot_classify(self):
        """An unresolvable model must not warn and must not raise: this runs on
        the router startup path, so a wrong guess would spam every boot."""
        assert (
            warn_on_provider_credential_mismatch(
                model_name="mystery",
                litellm_params={"model": "not-a-real-provider-model-xyz", "aws_region_name": "us-east-1"},
            )
            is None
        )

    def test_router_warns_for_a_config_shaped_model_list(self, caplog):
        """The whole point is that this fires where operators declare models, so
        drive Router rather than the helper."""
        with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
            Router(
                model_list=[
                    {
                        "model_name": "claude-sonnet-5",
                        "litellm_params": {
                            "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                            "aws_region_name": "us-east-1",
                        },
                    },
                    {
                        "model_name": "claude-sonnet-5",
                        "litellm_params": {
                            "model": "claude-sonnet-5",
                            "aws_region_name": "us-east-1",
                        },
                    },
                ]
            )

        mismatch_warnings = [r for r in caplog.records if "resolves to provider" in r.getMessage()]
        assert len(mismatch_warnings) == 1, (
            "exactly the prefix-less deployment should warn; "
            f"got {[r.getMessage() for r in mismatch_warnings]}"
        )
        assert "aws_region_name" in mismatch_warnings[0].getMessage()

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/mantle/anthropic.claude-sonnet-4-5-20250929-v1:0",
            "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "sagemaker/my-endpoint",
        ],
    )
    def test_silent_for_every_aws_family_route(self, model):
        """The AWS family is wider than 'bedrock': mantle, sagemaker and the
        sagemaker variants all read aws_* legitimately. Warning on any of them
        would tell an operator to 'fix' a working deployment, so the provider
        set is derived from LlmProviders rather than hand-listed."""
        assert (
            warn_on_provider_credential_mismatch(
                model_name="aws-deployment",
                litellm_params={"model": model, "aws_region_name": "us-east-1"},
            )
            is None
        )

    def test_every_aws_family_provider_is_covered(self):
        """Pins the derivation itself: a newly added bedrock_*/sagemaker_* provider
        must join the set automatically, or it starts drawing false warnings."""
        from litellm.types.utils import LlmProviders

        aws_family = {p.value for p in LlmProviders if p.value.startswith(("bedrock", "sagemaker"))}
        assert aws_family <= PROVIDER_SCOPED_CREDENTIAL_PARAMS["aws_region_name"]
        assert {"bedrock", "bedrock_mantle", "sagemaker", "sagemaker_chat", "sagemaker_nova"} <= aws_family

    def test_silent_when_credentials_come_from_a_named_credential(self):
        """Named credentials resolve after registration, so the params are absent
        here. Warning on that absence would fire on every such deployment."""
        assert (
            warn_on_provider_credential_mismatch(
                model_name="claude-sonnet-5",
                litellm_params={
                    "model": "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "litellm_credential_name": "my-aws-creds",
                },
            )
            is None
        )

    @pytest.mark.parametrize("provider", ["bedrock_mantle", "sagemaker_nova"])
    def test_silent_for_aws_providers_named_explicitly(self, provider):
        """The false-positive shape: an operator names a less common AWS provider
        directly, so the model string carries no route prefix to key off. A
        hand-listed provider set misses these and tells them to 'fix' a working
        deployment by prefixing it with bedrock/."""
        assert (
            warn_on_provider_credential_mismatch(
                model_name="aws-deployment",
                litellm_params={
                    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                    "custom_llm_provider": provider,
                    "aws_region_name": "us-east-1",
                },
            )
            is None
        )


@pytest.mark.parametrize(
    ("litellm_params", "expected"),
    [
        ({"model": "azure_ai/gpt-5.4-mini", "custom_llm_provider": "azure"}, "azure"),
        ({"model": "azure_ai/gpt-5.4-mini", "api_base": "https://my-resource.openai.azure.com"}, "azure_ai"),
        ({"model": "cohere/command-r"}, "cohere"),
        ({"model": "gpt-5.4-mini"}, "openai"),
        ({"model": "no-provider-knows-this-model"}, None),
        ({"api_base": "https://my-resource.openai.azure.com"}, None),
    ],
    ids=["declared_wins", "prefix_beats_host_flip", "prefix_beats_cohere_chat_flip", "unprefixed_inferred", "unknown", "no_model"],
)
def test_provider_for_generic_call(litellm_params, expected, monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://unrelated.openai.azure.com")
    assert provider_for_generic_call(litellm_params) == expected


def test_router_warns_only_for_a_capability_flag_set_under_litellm_params(caplog):
    bedrock_model: Final = "bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        Router(
            model_list=[
                {
                    "model_name": "flag-in-litellm-params",
                    "litellm_params": {"model": bedrock_model, "supports_max_reasoning_effort": False},
                },
                {
                    "model_name": "flag-in-model-info",
                    "litellm_params": {"model": bedrock_model},
                    "model_info": {"supports_max_reasoning_effort": False},
                },
                {
                    "model_name": "declared-litellm-param",
                    "litellm_params": {"model": bedrock_model, "supports_system_message": False},
                },
            ]
        )

    capability_warnings: Final = [r.getMessage() for r in caplog.records if "model_info" in r.getMessage()]
    assert len(capability_warnings) == 1, capability_warnings
    assert "flag-in-litellm-params" in capability_warnings[0]
    assert "supports_max_reasoning_effort" in capability_warnings[0]


def test_router_serves_deployments_without_model_capability_flags():
    capability_flags: Final = frozenset(ProviderSpecificModelInfo.__annotations__) | frozenset(
        key for key in chain.from_iterable(GetModelCostMap.loaded_model_cost_map().values()) if key.startswith("supports_")
    )
    declared_params: Final = {"supports_system_message": False, "supports_system_prompt": True}
    router: Final = Router(
        model_list=[
            {
                "model_name": "every-flag",
                "litellm_params": {
                    "model": "replicate/meta/meta-llama-3-8b-instruct",
                    **dict.fromkeys(capability_flags - declared_params.keys(), True),
                    **declared_params,
                },
            }
        ]
    )

    served_params: Final = router.get_model_list()[0]["litellm_params"]
    assert sorted(capability_flags & served_params.keys() - declared_params.keys()) == []
    assert {key: served_params.get(key) for key in declared_params} == declared_params


def test_router_keeps_a_provider_param_that_another_deployments_model_info_names():
    Router(
        model_list=[
            {
                "model_name": "registers-the-name",
                "litellm_params": {"model": "openai/litellm-qa-registered-capability"},
                "model_info": {"supports_system_prompt": True},
            }
        ]
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "replicate-system-prompt",
                "litellm_params": {"model": "replicate/meta/meta-llama-3-8b-instruct", "supports_system_prompt": True},
            }
        ]
    )

    assert router.get_model_list()[0]["litellm_params"].get("supports_system_prompt") is True


def test_router_warns_once_and_serves_no_flag_for_a_deployment_synced_from_the_db(caplog):
    router: Final = Router(model_list=[])
    db_deployment: Final = Deployment(
        model_name="db-model",
        litellm_params=LiteLLM_Params(
            model="bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            supports_max_reasoning_effort=False,
        ),
        model_info=ModelInfo(id="db-model-id"),
    )
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        first_sync: Final = router.upsert_deployment(db_deployment)
        second_sync: Final = router.upsert_deployment(db_deployment)

    capability_warnings: Final = [r.getMessage() for r in caplog.records if "model_info" in r.getMessage()]
    assert first_sync is not None
    assert second_sync is None
    assert len(capability_warnings) == 1, capability_warnings
    assert "db-model" in capability_warnings[0]
    assert "supports_max_reasoning_effort" not in router.get_model_list()[0]["litellm_params"]
