#### What this tests ####
# This tests litellm router

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
import logging
import os


import litellm
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

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

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

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-expensive-model
        response = await router.aembedding(
            model="gpt-4",
            input="Tell me a joke.",
            metadata={"tags": ["paid"]},
            mock_response=[1, 2, 3],
        )

        response_extra_info = response._hidden_params

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

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests tagged with "default", this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["default"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"

    for _ in range(5):
        # requests with invalid tags, this should pick model with id == "default-model"
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["invalid-tag"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_error_from_tag_routing():
    """
    Tests the correct error raised when no deployments found for tag
    """
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
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        pytest.fail("this should have failed - expected it to fail")
    except Exception as e:
        from litellm.types.router import RouterErrors

        assert RouterErrors.no_deployments_with_tag_routing.value in str(e)
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


def test_strict_tag_routing_without_request_tags_blocks_header_regex_fallback():
    """
    When tag_filtering_match_any=False, deployments with plain tags must require
    those request tags before header regex can match. A spoofed User-Agent must
    not route to a tagged deployment when the request has no tags.
    """
    from litellm.router_strategy.tag_based_routing import _match_deployment

    deployment = {
        "model_name": "restricted-model",
        "litellm_params": {
            "model": "gpt-4o",
            "tags": ["internal"],
            "tag_regex": ["^User-Agent: internal-tool"],
        },
    }

    assert (
        _match_deployment(
            deployment=deployment,
            request_tags=None,
            header_strings=["User-Agent: internal-tool"],
            match_any=False,
        )
        is None
    )


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

        response_extra_info = response._hidden_params

        assert response_extra_info["model_id"] == "very-cheap-model"

    for _ in range(5):
        # this should pick model with id == very-cheap-model
        response = await router.aresponses(
            model="gpt-4",
            input="Tell me a joke.",
            litellm_metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )

        response_extra_info = response._hidden_params

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
    assert _get_tags_from_request_kwargs({"litellm_params": {"metadata": {"tags": ["paid"]}}}) == ["paid"]
    assert _get_tags_from_request_kwargs({"litellm_params": {"metadata": None}}) == []
    assert _get_tags_from_request_kwargs({"litellm_params": {}}) == []

    # Alternate metadata variable name: "litellm_metadata"
    assert _get_tags_from_request_kwargs(
        {"litellm_metadata": {"tags": ["alt"]}},
        metadata_variable_name="litellm_metadata",
    ) == ["alt"]
    assert _get_tags_from_request_kwargs(
        {"litellm_params": {"litellm_metadata": {"tags": ["nested-alt"]}}},
        metadata_variable_name="litellm_metadata",
    ) == ["nested-alt"]

    # No relevant keys present
    assert _get_tags_from_request_kwargs({"foo": "bar"}) == []


# --- _split_tags unit tests ---


def test_split_tags_positive_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["paid", "teamA"])
    assert positive == ["paid", "teamA"]
    assert excluded == []


def test_split_tags_negation_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["!provider:anthropic"])
    assert positive == []
    assert excluded == ["provider:anthropic"]


def test_split_tags_mixed():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags(["paid", "!provider:anthropic", "!inference:cerebras"])
    assert positive == ["paid"]
    assert len(excluded) == 2


def test_split_tags_bare_bang_skipped():
    from litellm.router_strategy.tag_based_routing import _split_tags

    # A bare "!" with nothing after it is not a valid negation tag; skip it
    positive, excluded = _split_tags(["paid", "!"])
    assert positive == ["paid"]
    assert excluded == []


def test_split_tags_empty():
    from litellm.router_strategy.tag_based_routing import _split_tags

    positive, excluded = _split_tags([])
    assert positive == []
    assert excluded == []


# --- get_deployments_for_tag negation integration tests ---


@pytest.mark.asyncio()
async def test_negation_excludes_matching_deployments():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "model:claude-sonnet-4-6"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai", "model:gpt-4o"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-model"


@pytest.mark.asyncio()
async def test_negation_multiple_tags_exclude_multiple_providers():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:vertex"],
                },
                "model_info": {"id": "vertex-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic", "!provider:openai"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "vertex-model"


@pytest.mark.asyncio()
async def test_negation_with_positive_tag():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid", "provider:anthropic"],
                },
                "model_info": {"id": "anthropic-paid"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid", "provider:openai"],
                },
                "model_info": {"id": "openai-paid"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free", "provider:openai"],
                },
                "model_info": {"id": "openai-free"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["paid", "!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-paid"


@pytest.mark.asyncio()
async def test_negation_all_excluded_raises():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_negation_ban_only_cannot_escape_default_pool():
    # A ban-only request must not route to tagged deployments outside the default pool.
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
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # Sending only "!default" must NOT route to the paid deployment.
    # The base pool for ban-only is the default pool; banning the only
    # default deployment should raise rather than falling through to paid.
    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!default"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_negation_ban_only_respects_default_pool():
    # A ban-only request stays within the default pool; non-default deployments
    # remain unreachable even when the negation tag is unrelated to the default.
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
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # "!paid" bans the paid deployment, but the base pool for ban-only is
    # already restricted to defaults; default-model must still be returned.
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!paid"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_negation_untagged_deployment_kept():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                },
                "model_info": {"id": "untagged-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "untagged-model"


@pytest.mark.asyncio()
async def test_negation_literal_only_no_partial_match():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic-haiku"],
                },
                "model_info": {"id": "anthropic-haiku-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # "!provider:anthropic" should NOT match "provider:anthropic-haiku" — exact tag match only
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] in (
            "anthropic-haiku-model",
            "openai-model",
        )


@pytest.mark.asyncio()
async def test_negation_regex_pattern_treated_as_literal():
    # "!provider:(anthropic|openai)" looks like a regex but is treated as a literal string.
    # It does NOT exclude deployments tagged "provider:anthropic" or "provider:openai".
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    # The regex-like string matches no deployment tag literally, so all
    # candidates survive and both model IDs are reachable.
    seen_ids = set()
    for _ in range(10):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:(anthropic|openai)"]},
            mock_response="hi",
        )
        seen_ids.add(response._hidden_params["model_id"])

    assert seen_ids == {"anthropic-model", "openai-model"}


@pytest.mark.asyncio()
async def test_positive_tags_unchanged_by_negation():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["free"],
                },
                "model_info": {"id": "free-model"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["paid"],
                },
                "model_info": {"id": "paid-model"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["free"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "free-model"


@pytest.mark.asyncio()
async def test_negation_skips_banned_group_and_uses_fallback():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-primary"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-fallback"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["!provider:anthropic"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "openai-fallback"


@pytest.mark.asyncio()
async def test_negation_exhausts_entire_fallback_chain():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-primary"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-fallback"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception) as exc_info:
        await router.acompletion(
            model="primary",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_tag_regex_survives_when_negation_removes_other_deployment():
    # Negation removes a plain-tagged deployment; the surviving tag_regex deployment
    # is still matched by User-Agent and selected.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                },
                "model_info": {"id": "claude-code-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-deployment"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "claude-code-deployment"


@pytest.mark.asyncio()
async def test_negation_removes_tag_regex_deployment_falls_to_ban_only():
    # When a negation tag removes the only tag_regex deployment, no regex deployments
    # remain in the candidate pool. has_tag_filter becomes False, ban_only fires,
    # and the remaining plain-tagged deployment is returned via the ban-only path.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                    "tags": ["group:claude"],
                },
                "model_info": {"id": "claude-code-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai"],
                },
                "model_info": {"id": "openai-deployment"},
            },
        ],
        enable_tag_filtering=True,
        tag_filtering_match_any=True,
    )

    # !group:claude removes the tag_regex deployment from candidates, so no regex
    # deployments remain. The ban-only path fires and returns the openai deployment.
    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!group:claude"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-deployment"


@pytest.mark.asyncio()
async def test_request_level_enable_tag_filtering_applies_when_global_off():
    """
    A request carrying enable_tag_filtering=True (set by the proxy from key/team
    router_settings) must activate tag filtering even when the router-level flag
    is off. Without this, a team's "Enable Tag Filtering" toggle saved in the UI
    is silently ignored at request time.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment"},
            },
        ],
        enable_tag_filtering=False,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            enable_tag_filtering=True,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-a-deployment"

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamB"]},
            enable_tag_filtering=True,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-b-deployment"


@pytest.mark.asyncio()
async def test_request_level_enable_tag_filtering_false_cannot_disable_global():
    """
    A request-level enable_tag_filtering=False must not bypass a router-level
    True: tag filtering can be an operator-level restriction on which
    deployments a caller may reach, so per-request settings may only scope
    down, never escape the global policy.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            enable_tag_filtering=False,
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-a-deployment"
