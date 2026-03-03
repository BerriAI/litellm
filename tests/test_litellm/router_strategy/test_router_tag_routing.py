#### What this tests ####
# This tests litellm router

import asyncio
import os
import sys
import time
import traceback

import openai
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from dotenv import load_dotenv

import litellm
from litellm import Router
from litellm._logging import verbose_logger


@pytest.mark.asyncio()
async def test_router_free_paid_tier():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["free"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-expensive-model"


@pytest.mark.asyncio()
async def test_router_free_paid_tier_embeddings():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                    "mock_response": ["1", "2", "3"],
                },
                "model_info": {"id": "default-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aembedding(
            model="gpt-4",
            input="Tell me a joke.",
            metadata={"tags": ["free"]},
            mock_response=[1, 2, 3],
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-expensive-model
        response = await router.aembedding(
            model="gpt-4",
            input="Tell me a joke.",
            metadata={"tags": ["paid"]},
            mock_response=[1, 2, 3],
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-expensive-model"


@pytest.mark.asyncio()
async def test_default_tagged_deployments():
    """
    - only use default deployment for untagged requests
    - if a request has tag "default", use default deployment
    """

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default"],
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # Untagged request, this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests tagged with "default", this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["default"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests with invalid tags, this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["invalid-tag"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "default-model"



@pytest.mark.asyncio()
async def test_error_from_tag_routing():
    """
    Tests the correct error raised when no deployments found for tag
    """
    import logging

    from litellm._logging import verbose_logger

    verbose_logger.setLevel(logging.DEBUG)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "default-model-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    try:
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        pytest.fail("this should have failed - expected it to fail")
    except Exception as e:
        from litellm.types.router import RouterErrors

        assert RouterErrors.no_deployments_with_tag_routing.value in str(e)
        print("got expected exception = ", e)
        pass


def test_tag_routing_with_list_of_tags():
    """
    Test that the router can handle a list of tags with match_any behavior
    """
    from litellm.router_strategy.tag_based_routing import is_valid_deployment_tag

    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA"])
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamB"])
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamC"])
    assert is_valid_deployment_tag(["teamA"], ["teamA", "teamB"])
    assert not is_valid_deployment_tag(["teamA", "teamB"], ["teamC"])
    assert not is_valid_deployment_tag(["teamA", "teamB"], [])
    assert not is_valid_deployment_tag(["default"], ["teamA"])

def test_tag_routing_with_list_of_tags_match_all():
    """
    Test that the router can handle a list of tags with match_all behavior
    """
    from litellm.router_strategy.tag_based_routing import is_valid_deployment_tag

    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA"], match_any=False)
    assert is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamB"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB", "teamC"], ["teamA", "teamD"], match_any=False)
    assert not is_valid_deployment_tag(["teamA"], ["teamA", "teamB"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB"], ["teamA", "teamC"], match_any=False)
    assert not is_valid_deployment_tag(["teamA", "teamB"], [], match_any=False)
    assert not is_valid_deployment_tag(["default"], ["teamA"], match_any=False)

@pytest.mark.asyncio()
async def test_router_free_paid_tier_with_responses_api():
    """
    Pass list of orgs in 1 model definition,
    expect a unique deployment for each to be created
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "very-cheap-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "very-expensive-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aresponses(
            model="gpt-4",
            input="Tell me a joke.",
            litellm_metadata={"tags": ["free"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aresponses(
            model="gpt-4",
            input="Tell me a joke.",
            litellm_metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        print("Response: ", response)

        response_extra_info = response._hidden_params
        print("response_extra_info: ", response_extra_info)

        assert response_extra_info["model_id"] == "very-expensive-model"

def test_get_tags_from_request_kwargs_none():
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    # None request kwargs should safely return empty list
    assert _get_tags_from_request_kwargs(None) == []


def test_get_tags_from_request_kwargs_various_inputs():
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    # Direct "metadata" path
    assert _get_tags_from_request_kwargs({"metadata": {"tags": ["free"]}}) == ["free"]
    assert _get_tags_from_request_kwargs({"metadata": {"tags": []}}) == []
    assert _get_tags_from_request_kwargs({"metadata": {"tags": None}}) == []
    assert _get_tags_from_request_kwargs({"metadata": {}}) == []
    assert _get_tags_from_request_kwargs({"metadata": None}) == []

    # Indirect via "litellm_params" - metadata inside
    assert (
        _get_tags_from_request_kwargs(
            {"litellm_params": {"metadata": {"tags": ["paid"]}}}
        )
        == ["paid"]
    )
    assert _get_tags_from_request_kwargs({"litellm_params": {"metadata": None}}) == []
    assert _get_tags_from_request_kwargs({"litellm_params": {}}) == []

    # Alternate metadata variable name: "litellm_metadata"
    assert (
        _get_tags_from_request_kwargs(
            {"litellm_metadata": {"tags": ["alt"]}},
            metadata_variable_name="litellm_metadata",
        )
        == ["alt"]
    )
    assert (
        _get_tags_from_request_kwargs(
            {"litellm_params": {"litellm_metadata": {"tags": ["nested-alt"]}}},
            metadata_variable_name="litellm_metadata",
        )
        == ["nested-alt"]
    )

    # No relevant keys present
    assert _get_tags_from_request_kwargs({"foo": "bar"}) == []


@pytest.mark.asyncio()
async def test_tag_filtering_without_global_flag():
    """
    Test that tag filtering works automatically when the request carries tags,
    even without enable_tag_filtering=True on the router.

    This is the Tag Management workflow: create tag → select models → set tag
    on team → routing just works. No global flag needed.

    Regression test for: tag filtering required enable_tag_filtering global flag
    even when tags were present in the request.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["team-a"],
                },
                "model_info": {"id": "model-team-a"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["team-b"],
                },
                "model_info": {"id": "model-team-b"},
            },
        ],
        # NOTE: enable_tag_filtering is NOT set (defaults to None)
    )

    # Request with team-a tag should route to model-team-a
    for _ in range(3):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"tags": ["team-a"]},
            mock_response="Hello from team A",
        )
        assert response._hidden_params["model_id"] == "model-team-a"

    # Request with team-b tag should route to model-team-b
    for _ in range(3):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"tags": ["team-b"]},
            mock_response="Hello from team B",
        )
        assert response._hidden_params["model_id"] == "model-team-b"

    # Untagged request should load-balance across both (no default-deployment filtering)
    model_ids_seen = set()
    for _ in range(20):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hello",
        )
        model_ids_seen.add(response._hidden_params["model_id"])
    # With 20 requests and 2 deployments, we should see both
    assert "model-team-a" in model_ids_seen and "model-team-b" in model_ids_seen


@pytest.mark.asyncio()
async def test_tag_filtering_graceful_fallthrough_without_global_flag():
    """
    Test that when request has tags but no deployment matches, the router
    falls through gracefully (load-balances) instead of erroring — when
    enable_tag_filtering is NOT explicitly set.

    This prevents breaking users who have tags on teams/keys for non-routing
    purposes (e.g., budget tracking) and whose deployments have unrelated tags.

    With enable_tag_filtering=True, the error IS raised (tested in
    test_error_from_tag_routing).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["team-a"],
                },
                "model_info": {"id": "model-team-a"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "model-no-tags"},
            },
        ],
        # NOTE: enable_tag_filtering is NOT set
    )

    # Request with a non-matching tag should fall through and load-balance,
    # NOT raise an error
    model_ids_seen = set()
    for _ in range(20):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"tags": ["nonexistent-tag"]},
            mock_response="Hello",
        )
        model_ids_seen.add(response._hidden_params["model_id"])

    # Should see both deployments (graceful fallthrough = load-balanced)
    assert "model-team-a" in model_ids_seen and "model-no-tags" in model_ids_seen


@pytest.mark.asyncio
async def test_tag_filtering_explicit_false_disables():
    """
    When enable_tag_filtering is explicitly set to False, tag filtering is
    completely disabled — even if the request carries tags, all deployments
    should be used (no filtering applied).

    This tests the opt-out path: users who have tags on teams/keys for
    non-routing purposes can set enable_tag_filtering=False to ensure
    tag-based routing never activates.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["team-a"],
                },
                "model_info": {"id": "model-team-a"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["team-b"],
                },
                "model_info": {"id": "model-team-b"},
            },
        ],
        enable_tag_filtering=False,  # Explicitly disabled
    )

    # Even though request carries tags that match a specific deployment,
    # filtering should NOT be applied — both deployments should be used
    model_ids_seen = set()
    for _ in range(20):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"tags": ["team-a"]},
            mock_response="Hello",
        )
        model_ids_seen.add(response._hidden_params["model_id"])

    # Should see BOTH deployments (filtering disabled)
    assert "model-team-a" in model_ids_seen and "model-team-b" in model_ids_seen