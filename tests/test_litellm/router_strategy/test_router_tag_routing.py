#### What this tests ####
# This tests litellm router


import pytest

import logging


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

    from litellm.types.router import RouterErrors

    with pytest.raises(
        Exception, match=RouterErrors.no_deployments_with_tag_routing.value
    ):
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Tell me a joke."}],
            metadata={"tags": ["paid"]},
            mock_response="Tell me a joke.",
        )


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


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"metadata": "not-a-dict"},
        {"litellm_metadata": "not-a-dict"},
        {"litellm_metadata": ["not", "a", "dict"]},
        {"litellm_params": "not-a-dict"},
        {"litellm_params": {"metadata": "not-a-dict"}},
        {"metadata": {"tags": "free"}},
        {"metadata": {"tags": {"free": "paid"}}},
    ],
)
def test_get_tags_from_request_kwargs_reads_no_tags_from_a_non_dict_shape(request_kwargs):
    """Metadata and `tags` are request-controlled, so a client can send either as a
    string, a list or null. Every shape that cannot hold string tags reads as untagged
    instead of raising, because callers run on the hot request path."""
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    assert _get_tags_from_request_kwargs(request_kwargs) == []


def test_get_tags_from_request_kwargs_keeps_only_string_tags():
    from litellm.router_strategy.tag_based_routing import _get_tags_from_request_kwargs

    assert _get_tags_from_request_kwargs({"metadata": {"tags": ["free", 7, None, "paid"]}}) == ["free", "paid"]


# --- _split_tags unit tests ---


def test_split_tags_positive_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    required, positive, excluded = _split_tags(["paid", "teamA"])
    assert required == ()
    assert positive == ["paid", "teamA"]
    assert excluded == ()


def test_split_tags_negation_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    required, positive, excluded = _split_tags(["!provider:anthropic"])
    assert required == ()
    assert positive == []
    assert excluded == ("provider:anthropic",)


def test_split_tags_required_only():
    from litellm.router_strategy.tag_based_routing import _split_tags

    required, positive, excluded = _split_tags(["&reasoning_type:high", "&provider:anthropic"])
    assert required == ("reasoning_type:high", "provider:anthropic")
    assert positive == []
    assert excluded == ()


def test_split_tags_mixed():
    from litellm.router_strategy.tag_based_routing import _split_tags

    required, positive, excluded = _split_tags(
        ["paid", "!provider:anthropic", "!inference:cerebras", "&reasoning_type:high"]
    )
    assert required == ("reasoning_type:high",)
    assert positive == ["paid"]
    assert len(excluded) == 2


def test_split_tags_bare_bang_and_amp_skipped():
    from litellm.router_strategy.tag_based_routing import _split_tags

    # A bare "!" or "&" with nothing after it is not a valid tag; skip it
    required, positive, excluded = _split_tags(["paid", "!", "&"])
    assert required == ()
    assert positive == ["paid"]
    assert excluded == ()


def test_split_tags_empty():
    from litellm.router_strategy.tag_based_routing import _split_tags

    required, positive, excluded = _split_tags([])
    assert required == ()
    assert positive == []
    assert excluded == ()


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

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
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
    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
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

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
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


# --- model_info.enable_tag_filtering per-chain override ---


class _FakeRouterForChainOverride:
    def __init__(self, all_deployments):
        self._all_deployments = all_deployments

    def _get_all_deployments(self, model_name):
        return self._all_deployments


def test_chain_tag_filtering_override_reads_any_member():
    from litellm.router_strategy.tag_based_routing import _chain_tag_filtering_override

    deployments = [
        {"model_info": {}},
        {"model_info": {"enable_tag_filtering": False}},
    ]
    router = _FakeRouterForChainOverride(deployments)
    assert _chain_tag_filtering_override(router, "gpt-4", deployments) is False


def test_chain_tag_filtering_override_none_when_unset_anywhere():
    from litellm.router_strategy.tag_based_routing import _chain_tag_filtering_override

    deployments = [{"model_info": {}}, {}]
    router = _FakeRouterForChainOverride(deployments)
    assert _chain_tag_filtering_override(router, "gpt-4", deployments) is None


def test_chain_tag_filtering_override_survives_the_overriding_member_going_unhealthy():
    # Regression: the per-group override must be resolved from every deployment
    # configured for the model, not just the ones that survived cooldown/health
    # filtering. async_get_healthy_deployments filters cooldowns before calling
    # into get_deployments_for_tag, so healthy_deployments alone can be missing
    # the one deployment that carries the group's only explicit override.
    from litellm.router_strategy.tag_based_routing import _chain_tag_filtering_override

    all_deployments = [
        {"model_info": {"enable_tag_filtering": True}},
        {"model_info": {}},
    ]
    router = _FakeRouterForChainOverride(all_deployments)
    # The overriding deployment (index 0) is cooled down and absent from
    # healthy_deployments -- the override must still be found via the full-group
    # lookup, not silently lost.
    healthy_deployments = [all_deployments[1]]
    assert _chain_tag_filtering_override(router, "gpt-4", healthy_deployments) is True


def test_chain_tag_filtering_override_falls_back_to_healthy_deployments_on_lookup_error():
    from litellm.router_strategy.tag_based_routing import _chain_tag_filtering_override

    class _BrokenRouter:
        def _get_all_deployments(self, model_name):
            raise RuntimeError("model group not found")

    healthy_deployments = [{"model_info": {"enable_tag_filtering": False}}]
    assert _chain_tag_filtering_override(_BrokenRouter(), "gpt-4", healthy_deployments) is False


@pytest.mark.asyncio()
async def test_chain_enable_tag_filtering_true_overrides_router_level_false():
    # Router-wide tag filtering is off; this model group opts in on its own via
    # model_info.enable_tag_filtering, so tags still apply to requests for it.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment", "enable_tag_filtering": True},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment", "enable_tag_filtering": True},
            },
        ],
        enable_tag_filtering=False,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "team-a-deployment"


@pytest.mark.asyncio()
async def test_chain_enable_tag_filtering_false_overrides_router_level_true():
    # Router-wide tag filtering is on, but this model group opts itself out via
    # model_info.enable_tag_filtering: tags are ignored for requests to this group,
    # so an untagged-style request just gets ordinary load-balanced routing.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment", "enable_tag_filtering": False},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment", "enable_tag_filtering": False},
            },
        ],
        enable_tag_filtering=True,
    )

    seen_ids = set()
    for _ in range(10):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["teamA"]},
            mock_response="hi",
        )
        seen_ids.add(response._hidden_params["model_id"])

    assert seen_ids == {"team-a-deployment", "team-b-deployment"}


@pytest.mark.asyncio()
async def test_request_level_enable_tag_filtering_still_wins_over_chain_level_false():
    # A key/team's own request-level enable_tag_filtering=True must still win over
    # a chain that opted itself out, exactly as it already wins over the router
    # default: request-level escalation is the highest-precedence layer.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamA"],
                },
                "model_info": {"id": "team-a-deployment", "enable_tag_filtering": False},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["teamB"],
                },
                "model_info": {"id": "team-b-deployment", "enable_tag_filtering": False},
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


# --- _require_all_tags / _chain_allows_fail_open unit tests ---


def test_require_all_tags_empty_required_set_is_noop():
    from litellm.router_strategy.tag_based_routing import _require_all_tags

    deployments = [{"litellm_params": {"tags": ["a"]}}, {"litellm_params": {"tags": []}}]
    assert _require_all_tags(deployments, frozenset()) == tuple(deployments)


def test_require_all_tags_keeps_only_deployments_with_every_required_tag():
    from litellm.router_strategy.tag_based_routing import _require_all_tags

    has_both = {"litellm_params": {"tags": ["reasoning_type:high", "provider:anthropic"]}}
    has_one = {"litellm_params": {"tags": ["reasoning_type:high"]}}
    has_neither = {"litellm_params": {"tags": ["provider:openai"]}}

    result = _require_all_tags(
        [has_both, has_one, has_neither], frozenset({"reasoning_type:high", "provider:anthropic"})
    )
    assert result == (has_both,)


def test_chain_allows_fail_open_true_when_any_member_sets_flag():
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [
        {"model_info": {}, "litellm_params": {"tags": ["provider:anthropic"]}},
        {"model_info": {"allow_fail_open": True}, "litellm_params": {"tags": ["provider:openai"]}},
    ]
    assert _chain_allows_fail_open(deployments, frozenset(), frozenset({"provider:anthropic"}), frozenset()) is True


def test_chain_allows_fail_open_false_by_default():
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [{"model_info": {}}, {}]
    assert _chain_allows_fail_open(deployments, frozenset(), frozenset(), frozenset()) is False


def test_chain_allows_fail_open_true_when_no_required_tag_is_known_at_all():
    # An entirely-invented required tag with nothing else known to compare against
    # has no narrower answer to hide; a single-deployment catch-all fallback is a
    # legitimate use of allow_fail_open, not something to deny.
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [
        {"model_info": {"allow_fail_open": True}, "litellm_params": {"tags": ["default", "reasoning_type:low"]}},
    ]
    assert _chain_allows_fail_open(deployments, frozenset(), frozenset({"reasoning_type:high"}), frozenset()) is True


def test_unknown_required_tag_hides_an_answer_denies_fail_open():
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [
        {
            "model_info": {},
            "litellm_params": {"tags": ["provider:anthropic", "region:us-east"]},
        },
        {
            "model_info": {"allow_fail_open": True},
            "litellm_params": {"tags": ["default", "provider:openai"]},
        },
    ]
    # region:us-east is real and satisfiable on the first deployment; the invented tag
    # alone forces emptiness. Dropping it reveals a specific, non-default answer, so
    # fail-open must be denied even though the flag is set on the group.
    assert (
        _chain_allows_fail_open(
            deployments, frozenset(), frozenset({"region:us-east", "totally-invented-tag-nobody-has"}), frozenset()
        )
        is False
    )


def test_unknown_required_tag_allows_fail_open_when_no_answer_is_hidden():
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [
        {
            "model_info": {"allow_fail_open": True},
            "litellm_params": {"tags": ["provider:anthropic", "region:us-east"]},
        },
        {
            "model_info": {"allow_fail_open": True},
            "litellm_params": {"tags": ["provider:eu", "region:eu"]},
        },
        {
            "model_info": {"allow_fail_open": True},
            "litellm_params": {"tags": ["default", "provider:openai"]},
        },
    ]
    # region:us-east and region:eu are both real, known tags; no single deployment
    # carries both, so this is a genuinely unsatisfiable combination, not an invented
    # tag masking a narrower answer. Fail-open must proceed normally.
    assert (
        _chain_allows_fail_open(deployments, frozenset(), frozenset({"region:us-east", "region:eu"}), frozenset())
        is True
    )


# --- _strip_routing_prefix / _bare_tag_value unit tests ---


def test_strip_routing_prefix_empty_prefix_is_noop():
    from litellm.router_strategy.tag_based_routing import _strip_routing_prefix

    tags = ["provider:anthropic", "&region:eu", "!region:us"]
    rewritten, confirmed = _strip_routing_prefix(tags, "")
    assert rewritten == tuple(tags)
    assert confirmed == frozenset()


def test_strip_routing_prefix_splits_routed_from_other():
    from litellm.router_strategy.tag_based_routing import _strip_routing_prefix

    rewritten, confirmed = _strip_routing_prefix(["feature:demo", "route:!provider:openai"], "route:")
    assert rewritten == ("feature:demo", "!provider:openai")
    assert confirmed == frozenset({"provider:openai"})


def test_strip_routing_prefix_confirmed_matches_bare_required_and_excluded_values():
    # Regression: confirmed must carry the same bare (marker-stripped) form that
    # _split_tags produces for required_set/excluded_set downstream. A prior bug
    # left the "&"/"!" marker in `confirmed`, so `required_set & routing_confirmed`
    # never intersected for any prefixed "&"/"!" tag -- the entire "trusted,
    # caller-declared required/excluded tag" mechanism silently no-opped.
    from litellm.router_strategy.tag_based_routing import _strip_routing_prefix

    _, confirmed = _strip_routing_prefix(["route:&provider:anthropic", "route:!region:eu"], "route:")
    assert confirmed == frozenset({"provider:anthropic", "region:eu"})


def test_strip_routing_prefix_lone_marker_confirms_nothing():
    from litellm.router_strategy.tag_based_routing import _strip_routing_prefix

    # A lone "&"/"!" with nothing after it parses to nothing in required_set,
    # excluded_set, or positive_tags (see test_split_tags_bare_bang_and_amp_skipped);
    # confirmed must not invent a value for it either.
    _, confirmed = _strip_routing_prefix(["route:&", "route:!"], "route:")
    assert confirmed == frozenset()


def test_chain_allows_fail_open_true_when_prefixed_unknown_required_tag_is_confirmed():
    # Regression for the same bug: a required tag no deployment carries is normally
    # treated as invented noise that can hide a narrower answer (see
    # test_unknown_required_tag_hides_an_answer_denies_fail_open) -- but once the
    # caller has explicitly marked it via the routing prefix, it counts as a known,
    # honest ask, and fail-open must proceed rather than get denied.
    from litellm.router_strategy.tag_based_routing import _chain_allows_fail_open

    deployments = [
        {
            "model_info": {"allow_fail_open": True},
            "litellm_params": {"tags": ["default", "provider:anthropic"]},
        },
    ]
    required_set = frozenset({"provider:anthropic", "typo-tag"})
    assert _chain_allows_fail_open(deployments, frozenset(), required_set, frozenset()) is False
    assert _chain_allows_fail_open(deployments, frozenset(), required_set, required_set) is True


# --- get_deployments_for_tag required-AND ("&") integration tests ---


@pytest.mark.asyncio()
async def test_required_and_matches_deployment_with_all_tags():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:openai"],
                },
                "model_info": {"id": "high-reasoning-openai"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "&provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning-anthropic"


@pytest.mark.asyncio()
async def test_required_and_excludes_deployment_missing_one_tag():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low", "provider:anthropic"],
                },
                "model_info": {"id": "low-reasoning-anthropic"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "&provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning-anthropic"


@pytest.mark.asyncio()
async def test_required_and_composes_with_negation():
    # &reasoning_type:high requires the tag; !provider:anthropic bans that provider.
    # Negation applies first, so the anthropic deployment is excluded even though
    # it satisfies the required tag.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:openai"],
                },
                "model_info": {"id": "high-reasoning-openai"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "!provider:anthropic"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning-openai"


@pytest.mark.asyncio()
async def test_required_and_combines_with_positive_or_preference():
    # &reasoning_type:high is a hard requirement; provider:anthropic/provider:openai
    # is a preference (OR) applied on top of the survivors.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:vertex"],
                },
                "model_info": {"id": "high-reasoning-vertex"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low", "provider:anthropic"],
                },
                "model_info": {"id": "low-reasoning-anthropic"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "provider:anthropic", "provider:openai"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning-anthropic"


@pytest.mark.asyncio()
async def test_required_and_single_tag_matches_trivially():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                },
                "model_info": {"id": "high-reasoning"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low"],
                },
                "model_info": {"id": "low-reasoning"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning"


@pytest.mark.asyncio()
async def test_required_and_unmatched_raises_by_default():
    # allow_fail_open unset -> unmatched required-AND raises, same as today's "!" behavior.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low"],
                },
                "model_info": {"id": "low-reasoning"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_required_and_combined_with_positive_unmatched_raises_by_default():
    # &A eliminates every candidate before the positive-tag preference even runs;
    # this must be gated by allow_fail_open too, not just the required-AND-only path.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low", "provider:anthropic"],
                },
                "model_info": {"id": "low-reasoning-anthropic"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "provider:anthropic"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


# --- get_deployments_for_tag allow_fail_open integration tests ---


@pytest.mark.asyncio()
async def test_allow_fail_open_required_and_unmatched_falls_back_to_default_pool():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "reasoning_type:low"],
                },
                "model_info": {"id": "default-model", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "default-model"


@pytest.mark.asyncio()
async def test_allow_fail_open_negation_eliminates_everything_includes_banned_deployment():
    # The core backwards-compatibility risk: once allow_fail_open opts a chain in,
    # a "!" ban that eliminates every deployment falls back to the full default
    # pool, INCLUDING the deployment the request tried to ban. This must never
    # silently disappear (still raise) nor silently reappear on chains without
    # the flag set (see test_negation_all_excluded_raises).
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model", "allow_fail_open": True},
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
        assert response._hidden_params["model_id"] == "anthropic-model"


@pytest.mark.asyncio()
async def test_allow_fail_open_prefers_default_tagged_deployment_on_fallback():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic"],
                },
                "model_info": {"id": "anthropic-model", "allow_fail_open": True},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "default"],
                },
                "model_info": {"id": "anthropic-default-model", "allow_fail_open": True},
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
        assert response._hidden_params["model_id"] == "anthropic-default-model"


@pytest.mark.asyncio()
async def test_allow_fail_open_per_hop_across_fallback_chain():
    # required-AND fail-open must be re-evaluated fresh on every hop, the same
    # per-hop guarantee the negation feature already established.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low"],
                },
                "model_info": {"id": "primary-low-reasoning"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "reasoning_type:low"],
                },
                "model_info": {"id": "fallback-model", "allow_fail_open": True},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["&reasoning_type:high"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "fallback-model"


@pytest.mark.asyncio()
async def test_allow_fail_open_resolves_locally_without_triggering_external_fallback():
    # allow_fail_open on the primary group's own default deployment absorbs the
    # exhaustion internally (_resolve_or_fail_open returns a non-empty pool, so
    # get_deployments_for_tag never raises); router.async_function_with_fallbacks
    # only invokes the configured "fallbacks" chain on an exception, so a
    # separate, unrelated fallback group must never be touched even though one is
    # configured. A fallback deployment that would trivially satisfy the request
    # tag if it were ever consulted makes this a meaningful negative assertion,
    # not a vacuous one.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                },
                "model_info": {"id": "primary-high-reasoning"},
            },
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "reasoning_type:low"],
                },
                "model_info": {"id": "primary-default", "allow_fail_open": True},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["region:eu"],
                },
                "model_info": {"id": "fallback-should-never-be-used"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["&region:eu"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "primary-default"


# --- allow_fail_open must also gate "!" exhaustion combined with a plain positive tag ---


@pytest.mark.asyncio()
async def test_negation_combined_with_positive_unmatched_raises_by_default():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "paid"],
                },
                "model_info": {"id": "anthropic-paid"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic", "paid"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_negation_combined_with_positive_unmatched_falls_open_when_allowed():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "paid", "default"],
                },
                "model_info": {"id": "anthropic-paid", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["!provider:anthropic", "paid"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "anthropic-paid"


# --- a required-AND-only request must not be diluted by incidental regex/header preference ---


@pytest.mark.asyncio()
async def test_required_and_only_returns_every_matching_deployment_despite_regex_header():
    # Deployment A satisfies &reasoning_type:high and also happens to carry a tag_regex
    # that matches the caller's User-Agent. Deployment B also satisfies the required tag
    # but has no tag_regex at all. A required-AND-only request (no plain positive tags)
    # must be free to route to either survivor, not be narrowed down to only the one
    # that happens to match the incidental regex/header preference.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                },
                "model_info": {"id": "high-reasoning-with-regex"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                },
                "model_info": {"id": "high-reasoning-no-regex"},
            },
        ],
        enable_tag_filtering=True,
    )

    seen_ids = set()
    for _ in range(30):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        seen_ids.add(response._hidden_params["model_id"])

    assert seen_ids == {"high-reasoning-with-regex", "high-reasoning-no-regex"}


@pytest.mark.asyncio()
async def test_required_and_only_excludes_regex_deployment_missing_the_required_tag():
    # The tag_regex deployment matches the caller's User-Agent but does NOT carry the
    # required tag; a required-AND-only request must not let it through on the strength
    # of the regex/header match alone.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low"],
                    "tag_regex": ["^User-Agent: claude-code\\/"],
                },
                "model_info": {"id": "low-reasoning-with-regex"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                },
                "model_info": {"id": "high-reasoning-no-regex"},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high"], "user_agent": "claude-code/1.2.3"},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "high-reasoning-no-regex"


# --- allow_fail_open must also gate exhaustion after a non-empty required-AND survivor
# set fails to match a plain preference tag, not just full !/& exhaustion ---


@pytest.mark.asyncio()
async def test_mixed_constraint_survivor_unmatched_by_positive_tag_raises_by_default():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "reasoning_type:low"],
                },
                "model_info": {"id": "default-fallback"},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "provider:openai"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_mixed_constraint_survivor_unmatched_by_positive_tag_falls_open_when_allowed():
    # &reasoning_type:high survives to a non-empty candidate set (the anthropic
    # deployment), but the plain preference tag provider:openai matches none of the
    # survivors, and the surviving deployment itself is not "default"-tagged (so the
    # pre-existing in-loop default-collection escape hatch can't mask the fix). Greptile
    # flagged this exact path as bypassing allow_fail_open by raising unconditionally;
    # it must instead fall back to the group's actual default-tagged deployment, which
    # is a different deployment than the one &reasoning_type:high matched.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high", "provider:anthropic"],
                },
                "model_info": {"id": "high-reasoning-anthropic", "allow_fail_open": True},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "reasoning_type:low"],
                },
                "model_info": {"id": "default-fallback", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&reasoning_type:high", "provider:openai"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "default-fallback"


# --- allow_fail_open must not be triggerable by an invented tag the chain has never
# carried; a caller-supplied garbage tag must not be able to force an otherwise-
# satisfiable constraint (e.g. one inherited from the key/team) to be discarded ---


@pytest.mark.asyncio()
async def test_allow_fail_open_denied_when_request_includes_unknown_tag():
    # region:us-east is a real, satisfiable constraint on anthropic-deployment. Adding
    # a single invented tag no deployment in this group has ever carried empties the
    # required-AND set regardless of region:us-east's own satisfiability. allow_fail_open
    # is set on the default deployment, but must not fire here: none of the *other*
    # deployments carry the invented tag either, so it is unknown to the chain, and
    # falling back would silently discard the still-satisfiable region:us-east ask.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "region:us-east"],
                },
                "model_info": {"id": "anthropic-deployment"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "provider:openai"],
                },
                "model_info": {"id": "openai-default", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&region:us-east", "&totally-invented-tag-nobody-has"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_allow_fail_open_still_fires_when_every_requested_tag_is_known():
    # region:us-east and region:eu are both real tags this chain uses; no single
    # deployment carries both, so the combination is genuinely unsatisfiable, not
    # invented. allow_fail_open must still fall back normally in this case.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "region:us-east"],
                },
                "model_info": {"id": "anthropic-deployment", "allow_fail_open": True},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:eu", "region:eu"],
                },
                "model_info": {"id": "eu-deployment", "allow_fail_open": True},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "provider:openai"],
                },
                "model_info": {"id": "openai-default", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&region:us-east", "&region:eu"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "openai-default"


# --- required-AND, allow_fail_open, and the unknown-tag denial across fallback
# chains spanning multiple model groups ---


@pytest.mark.asyncio()
async def test_required_and_exhausts_primary_group_falls_through_to_fallback_group():
    # &reasoning_type:high matches nothing on "primary" (raises internally, same as
    # negation's own fallback-chain behavior), so the router advances to "fallback"
    # where the tag is satisfiable. No allow_fail_open involved; this is the plain
    # fallback-chain mechanics already established for "!" extended to "&".
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:low"],
                },
                "model_info": {"id": "primary-low-reasoning"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["reasoning_type:high"],
                },
                "model_info": {"id": "fallback-high-reasoning"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["&reasoning_type:high"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "fallback-high-reasoning"


@pytest.mark.asyncio()
async def test_required_and_negation_and_allow_fail_open_combine_across_three_model_groups():
    # A single request routes through three independent model groups via two
    # fallback hops, exercising "!", "&", and allow_fail_open together at each hop:
    # - "primary" is banned outright by "!provider:anthropic" -> raises, advances.
    # - "secondary" satisfies the negation but not &reasoning_type:high, and has no
    #   allow_fail_open -> raises exactly as today, advances.
    # - "tertiary" has reasoning_type:high, but only on the deployment the same
    #   "!provider:anthropic" also bans; the tag is known to the chain but its only
    #   carrier is legitimately excluded, not hidden behind an invented tag, so the
    #   opted-in allow_fail_open falls back to the group's own default deployment.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "reasoning_type:high"],
                },
                "model_info": {"id": "primary-anthropic"},
            },
            {
                "model_name": "secondary",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:openai", "reasoning_type:low"],
                },
                "model_info": {"id": "secondary-openai"},
            },
            {
                "model_name": "tertiary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "reasoning_type:high", "region:eu"],
                },
                "model_info": {"id": "tertiary-anthropic-high-reasoning"},
            },
            {
                "model_name": "tertiary",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "provider:openai", "reasoning_type:low"],
                },
                "model_info": {"id": "tertiary-default", "allow_fail_open": True},
            },
        ],
        fallbacks=[{"primary": ["secondary"]}, {"secondary": ["tertiary"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["!provider:anthropic", "&reasoning_type:high"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "tertiary-default"


@pytest.mark.asyncio()
async def test_unknown_tag_denial_is_scoped_per_hop_not_leaked_across_fallback_groups():
    # On "primary": region:us-east is real and satisfiable there, but the invented
    # tag masks it -> denies fail-open -> raises -> advances to "fallback".
    # On "fallback": neither region:us-east nor the invented tag is known to this
    # entirely different, unrelated group at all, so there's no answer for the
    # invented tag to hide -> falls open normally. Each hop must independently
    # discover what its own group knows; a deny decision from a prior hop's group
    # must not leak forward and block a later hop that has no relevant knowledge.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["region:us-east"],
                },
                "model_info": {"id": "primary-us-east", "allow_fail_open": True},
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "provider:openai"],
                },
                "model_info": {"id": "fallback-default", "allow_fail_open": True},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        enable_tag_filtering=True,
    )

    response = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["&region:us-east", "&totally-invented-tag-nobody-has"]},
        mock_response="hi",
    )
    assert response._hidden_params["model_id"] == "fallback-default"


@pytest.mark.asyncio()
async def test_required_and_only_finds_compliant_non_default_deployment_over_noncompliant_default():
    # A required-AND-only request must be checked against every deployment in the
    # group, not just the one tagged "default". A compliant, healthy deployment that
    # simply isn't the operator's default must win over routing to a noncompliant
    # default just because allow_fail_open happened to be set.
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["provider:anthropic", "region:us-east"],
                },
                "model_info": {"id": "anthropic-us-east"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "provider:openai"],
                },
                "model_info": {"id": "openai-default", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["&region:us-east"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] == "anthropic-us-east"


# --- plain positive-tag exhaustion must not be masked by a universally-applied
# "default" tag; allow_fail_open must still be consulted (or hard-fail without it) ---


def _quality_high_cost_low_router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "quality:high"],
                },
                "model_info": {"id": "quality-high-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "quality:high"],
                },
                "model_info": {"id": "quality-high-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "cost:low"],
                },
                "model_info": {"id": "cost-low-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["default", "cost:low"],
                },
                "model_info": {"id": "cost-low-2"},
            },
        ],
        enable_tag_filtering=True,
    )


@pytest.mark.asyncio()
async def test_plain_tag_exhaustion_with_universal_default_tag_raises_by_default():
    # Every deployment in the group is tagged "default" (a legitimate cross-cutting
    # safety-net pattern), so default_deployments is never empty on its own. With
    # the quality:high deployments unhealthy, a request asking for quality:high
    # must still hard-fail, not silently get served by a cost:low deployment just
    # because it happens to also carry "default".
    from unittest.mock import AsyncMock, patch

    router = _quality_high_cost_low_router()

    with patch(
        "litellm.router._async_get_cooldown_deployments",
        new=AsyncMock(return_value=["quality-high-1", "quality-high-2"]),
    ):
        with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
            await router.acompletion(
                model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"tags": ["quality:high"]},
                mock_response="hi",
            )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_plain_tag_exhaustion_with_universal_default_tag_falls_open_when_allowed():
    router = _quality_high_cost_low_router()
    for deployment in router.model_list:
        deployment["model_info"]["allow_fail_open"] = True

    from unittest.mock import AsyncMock, patch

    with patch(
        "litellm.router._async_get_cooldown_deployments",
        new=AsyncMock(return_value=["quality-high-1", "quality-high-2"]),
    ):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["quality:high"]},
            mock_response="hi",
        )

    assert response._hidden_params["model_id"] in ("cost-low-1", "cost-low-2")


@pytest.mark.asyncio()
async def test_plain_tag_unknown_to_group_still_falls_back_silently_unconditionally():
    # A tag that no deployment in this group has ever carried (foreign to this
    # group entirely, e.g. an attribution tag meant for an unrelated mechanism
    # sharing the same request-tags list) must keep falling back to the
    # "default"-tagged pool unconditionally, exactly like today, regardless of
    # allow_fail_open. Only a tag that IS part of this group's real vocabulary
    # triggers the new hard-fail/fail-open gate.
    router = _quality_high_cost_low_router()

    for _ in range(5):
        response = await router.acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["llm-preference-include:some-unrelated-mechanism"]},
            mock_response="hi",
        )
        assert response._hidden_params["model_id"] in (
            "quality-high-1",
            "quality-high-2",
            "cost-low-1",
            "cost-low-2",
        )


def test_tag_known_to_group_true_for_real_tag():
    from litellm.router_strategy.tag_based_routing import _tag_known_to_group

    router = _quality_high_cost_low_router()
    assert _tag_known_to_group(router, "gpt-4", ["quality:high"], frozenset()) is True


def test_tag_known_to_group_false_for_foreign_tag():
    from litellm.router_strategy.tag_based_routing import _tag_known_to_group

    router = _quality_high_cost_low_router()
    assert _tag_known_to_group(router, "gpt-4", ["llm-preference-include:unrelated"], frozenset()) is False


def test_inherited_constraint_sets_none_when_inherited_tags_absent():
    from litellm.router_strategy.tag_based_routing import _inherited_constraint_sets

    assert _inherited_constraint_sets(None, "") == (None, None)


def test_inherited_constraint_sets_splits_required_and_excluded():
    from litellm.router_strategy.tag_based_routing import _inherited_constraint_sets

    inherited_required_set, inherited_excluded_set = _inherited_constraint_sets(
        ["&region:eu", "!region:us", "plain"], ""
    )
    assert inherited_required_set == frozenset({"region:eu"})
    assert inherited_excluded_set == frozenset({"region:us"})


def test_inherited_constraint_sets_none_for_non_sequence_value():
    from litellm.router_strategy.tag_based_routing import _inherited_constraint_sets

    # A malformed/unexpected inherited_tags value (anything but a list/tuple) must
    # be treated the same as "no origin information", never as "nothing is
    # inherited" -- the two are not interchangeable, see _trusted_only_pool.
    assert _inherited_constraint_sets("not-a-sequence", "") == (None, None)


def test_trusted_only_pool_discards_everything_when_inherited_sets_are_none():
    from litellm.router_strategy.tag_based_routing import _trusted_only_pool

    deployments = ({"litellm_params": {"tags": ["region:us"]}},)
    # No origin info at all -> reproduce the pre-provenance unconditional
    # fall-open: the trusted-only pool ignores excluded_set/required_set entirely.
    assert _trusted_only_pool(deployments, frozenset({"region:eu"}), frozenset({"region:apac"}), None, None) == deployments


def test_trusted_only_pool_keeps_constraint_backed_by_inherited_tags():
    from litellm.router_strategy.tag_based_routing import _trusted_only_pool

    eu = {"litellm_params": {"tags": ["region:eu"]}}
    us = {"litellm_params": {"tags": ["region:us"]}}
    # required_set={"region:eu"} IS in inherited_required_set -> protected, kept.
    result = _trusted_only_pool(
        (eu, us), frozenset(), frozenset({"region:eu"}), frozenset(), frozenset({"region:eu"})
    )
    assert result == (eu,)


def test_trusted_only_pool_discards_a_value_with_no_inherited_backing_even_if_the_caller_also_sent_it():
    # Regression for the value-collision bypass Greptile and veria-ai both
    # flagged: a value with zero inherited backing is discardable even when it
    # happens to be the exact value the caller submitted -- there is nothing here
    # to distinguish "caller-only" from "caller happened to guess a real policy
    # value" at this function's level, which is exactly why protection must be
    # keyed off presence in inherited_required_set, never absence from a
    # caller-supplied set (see the router-level regression below for the full
    # bypass this replaces).
    from litellm.router_strategy.tag_based_routing import _trusted_only_pool

    eu = {"litellm_params": {"tags": ["region:eu"]}}
    us = {"litellm_params": {"tags": ["region:us"]}}
    result = _trusted_only_pool((eu, us), frozenset(), frozenset({"region:eu"}), frozenset(), frozenset())
    assert result == (eu, us)


def _eu_region_router():
    # eu-1 deliberately carries no "default" tag, and us-default is the only
    # "default"-tagged deployment -- this keeps _default_tagged_pool's outcome a
    # single, deterministic deployment id in every scenario below, regardless of
    # which of the two candidate pools (trusted-only vs fully-unconstrained) a
    # given code path resolves to.
    return litellm.Router(
        model_list=[
            {
                "model_name": "chat",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["region:eu"],
                },
                "model_info": {"id": "eu-1", "allow_fail_open": True},
            },
            {
                "model_name": "chat",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_base": "https://exampleopenaiendpoint-production.up.railway.app/",
                    "tags": ["region:us", "default"],
                },
                "model_info": {"id": "us-default", "allow_fail_open": True},
            },
        ],
        enable_tag_filtering=True,
    )


@pytest.mark.asyncio()
async def test_allow_fail_open_preserves_inherited_constraint_when_caller_tag_causes_exhaustion():
    # &region:eu simulates a key/team-inherited hard requirement, captured in
    # inherited_tags (a snapshot taken before the caller's own tags are merged
    # in); !region:eu simulates the caller's own tag. Combined they exhaust the
    # pool (nothing can both carry and not carry region:eu), but allow_fail_open
    # must fall back to what still satisfies the inherited requirement, not the
    # fully-unconstrained default pool (us-default), and not raise either.
    router = _eu_region_router()

    response = await router.acompletion(
        model="chat",
        messages=[{"role": "user", "content": "hi"}],
        metadata={
            "tags": ["&region:eu", "!region:eu"],
            "inherited_tags": ["&region:eu"],
            "caller_tags": ["!region:eu"],
        },
        mock_response="hi",
    )

    assert response._hidden_params["model_id"] == "eu-1"


@pytest.mark.asyncio()
async def test_allow_fail_open_stays_protected_when_caller_duplicates_the_inherited_tag():
    # Regression for the value-collision bypass Greptile and veria-ai both
    # flagged: a caller who resubmits the exact value of an inherited "&" tag
    # (here alongside a conflicting "!" for the same value) must not be able to
    # strip that value's protection just because it now also appears in
    # caller_tags. Protection is keyed off presence in inherited_tags, not
    # absence from caller_tags -- if it were the latter, subtracting
    # caller_required_set={"region:eu"} from required_set would zero out the
    # inherited requirement entirely and this would incorrectly resolve to
    # us-default instead of eu-1.
    router = _eu_region_router()

    response = await router.acompletion(
        model="chat",
        messages=[{"role": "user", "content": "hi"}],
        metadata={
            "tags": ["&region:eu", "!region:eu"],
            "inherited_tags": ["&region:eu"],
            "caller_tags": ["&region:eu", "!region:eu"],
        },
        mock_response="hi",
    )

    assert response._hidden_params["model_id"] == "eu-1"


@pytest.mark.asyncio()
async def test_allow_fail_open_raises_when_inherited_constraint_alone_is_unsatisfiable():
    # Both region:eu and region:us are known to the group (so the unknown-tag
    # masking guard does not apply), but no single deployment carries both, and
    # inherited_tags confirms the entire required-AND set traces back to policy.
    # allow_fail_open must not paper over an inherited requirement that is
    # unsatisfiable on its own; it should raise exactly as it would with
    # allow_fail_open unset.
    router = _eu_region_router()

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="chat",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "tags": ["&region:eu", "&region:us"],
                "inherited_tags": ["&region:eu", "&region:us"],
                "caller_tags": [],
            },
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_allow_fail_open_unconditional_discard_when_inherited_tags_key_absent():
    # No "inherited_tags" key at all (e.g. a direct SDK Router call that never
    # went through the proxy's litellm_pre_call_utils.py) must reproduce the exact
    # pre-provenance behavior: unconditional fall-open to the default pool, even
    # though region:eu here would otherwise look like an inherited requirement.
    router = _eu_region_router()

    response = await router.acompletion(
        model="chat",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["&region:eu", "!region:eu"]},
        mock_response="hi",
    )

    assert response._hidden_params["model_id"] == "us-default"


# --- tag_routing_prefix must be configurable through every settings-update
# path the router already supports for its sibling enable_tag_filtering, not
# just the config.yaml constructor argument ---


def test_router_update_settings_applies_tag_routing_prefix():
    # Regression: tag_routing_prefix was missing from Router.update_settings's
    # _allowed_settings, so an operator configuring it via the DB-backed
    # router_settings path (proxy_server.py's _add_router_settings_from_db_config,
    # which calls update_settings directly) had the value silently ignored.
    router = litellm.Router(model_list=[{"model_name": "x", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
    assert router.tag_routing_prefix == ""

    router.update_settings(tag_routing_prefix="route:")

    assert router.tag_routing_prefix == "route:"


def test_router_get_settings_includes_tag_routing_prefix():
    router = litellm.Router(model_list=[{"model_name": "x", "litellm_params": {"model": "openai/gpt-4o-mini"}}])
    router.update_settings(tag_routing_prefix="route:")

    assert router.get_settings()["tag_routing_prefix"] == "route:"


def test_update_router_config_schema_includes_tag_routing_prefix():
    # The Admin UI's POST /config/update path validates through
    # UpdateRouterConfig before calling update_settings; a field missing here
    # causes model_dump(exclude_none=True) to silently drop it before
    # update_settings is ever called -- the same bug shape LIT-3152 fixed for
    # retry_policy (see tests/test_litellm/test_router_retry_policy_update.py).
    from litellm.types.router import UpdateRouterConfig

    config = UpdateRouterConfig(tag_routing_prefix="route:")
    assert config.model_dump(exclude_none=True)["tag_routing_prefix"] == "route:"


# --- issue #36621: the request tags that selected a tagged pre-routing strategy
# (e.g. an auto_router marker) are consumed by that selection and must not
# re-apply to the routed tier's model group; key/team-inherited constraints
# must keep applying there ---


class _RewriteToTierStrategy:
    def __init__(self, rewrite_to: str):
        self.rewrite_to = rewrite_to

    async def async_pre_routing_hook(
        self, model, request_kwargs, messages=None, input=None, specific_deployment=False
    ):
        from litellm.types.router import PreRoutingHookResponse

        return PreRoutingHookResponse(model=self.rewrite_to, messages=messages)


def _tagged_marker_router(tier_tags=None):
    from litellm.types.router import TaggedPreRoutingStrategy

    tier_params = {"model": "gemini/gemini-3.6-flash"}
    if tier_tags is not None:
        tier_params["tags"] = tier_tags
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "plain-gpt4o"},
            },
            {
                "model_name": "gemini-flash",
                "litellm_params": tier_params,
                "model_info": {"id": "tier-gemini-flash"},
            },
        ],
        enable_tag_filtering=True,
    )
    router.auto_routers = {
        "gpt4o": [TaggedPreRoutingStrategy(tags=("route",), strategy=_RewriteToTierStrategy("gemini-flash"))]
    }
    return router


@pytest.mark.asyncio()
async def test_router_selecting_tag_is_not_reapplied_to_the_routed_tier():
    # The exact request the auto-router exists to serve: tags=["route"] selects
    # the tagged marker, the strategy rewrites to gemini-flash, and the untagged
    # tier deployment must serve it instead of 401ing on the already-spent tag.
    router = _tagged_marker_router()

    response = await router.acompletion(
        model="gpt4o",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        metadata={"tags": ["route"], "inherited_tags": []},
        mock_response="Paris",
    )

    assert response._hidden_params["model_id"] == "tier-gemini-flash"


@pytest.mark.asyncio()
async def test_router_selecting_tag_is_consumed_on_litellm_metadata_shaped_requests():
    # /v1/messages (and other litellm_metadata endpoints) store proxy metadata,
    # including x-litellm-tags header tags, under "litellm_metadata"; consumption
    # must read and stamp that same bucket instead of only "metadata".
    router = _tagged_marker_router()

    deployment = await router.async_get_available_deployment(
        model="gpt4o",
        request_kwargs={"litellm_metadata": {"tags": ["route"], "inherited_tags": []}},
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    assert deployment["model_info"]["id"] == "tier-gemini-flash"


def test_consumed_request_tags_stamp_names_the_routed_group_and_spent_tags_only_on_a_tag_match():
    from litellm.types.router import ConsumedRequestTagsStamp, PreRoutingHookResponse

    router = _tagged_marker_router()
    strategy = router.auto_routers["gpt4o"][0]
    rewrite = PreRoutingHookResponse(model="gemini-flash", messages=None)

    consumed = router._consumed_request_tags_stamp(
        selected_strategy=strategy, pre_routing_hook_response=rewrite, request_tags=["route"]
    )
    unmatched = router._consumed_request_tags_stamp(
        selected_strategy=strategy, pre_routing_hook_response=rewrite, request_tags=["other"]
    )

    assert consumed == ConsumedRequestTagsStamp(model_group="gemini-flash", tags=("route",))
    assert unmatched is None


@pytest.mark.asyncio()
async def test_tagged_request_direct_to_plain_group_still_rejected():
    # Sent straight to the tier, no router selection consumed the tag, so strict
    # tag filtering must reject exactly as before.
    router = _tagged_marker_router()

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gemini-flash",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["route"], "inherited_tags": []},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_caller_forged_consumption_stamp_is_neutralized_by_the_hook():
    # A caller pre-loading the stamp in metadata must not unlock a plain group:
    # the pre-routing hook writes-or-clears the stamp on every attempt, and this
    # group has no registered strategy, so the forged value is cleared before
    # tag filtering runs.
    router = _tagged_marker_router()

    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await router.acompletion(
            model="gemini-flash",
            messages=[{"role": "user", "content": "hi"}],
            metadata={
                "tags": ["route"],
                "inherited_tags": [],
                "_consumed_request_tags": {"model_group": "gemini-flash", "tags": ["route"]},
            },
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)


@pytest.mark.asyncio()
async def test_inherited_constraint_still_applies_to_the_routed_tier():
    # &region:eu comes from key/team policy (present in inherited_tags):
    # consuming the router-selecting "route" tag must not also discard the
    # inherited requirement, so a tier without the tag still raises...
    with pytest.raises(Exception, match='Not allowed to access model due to tags configuration\\.') as exc_info:
        await _tagged_marker_router().acompletion(
            model="gpt4o",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tags": ["route", "&region:eu"], "inherited_tags": ["&region:eu"]},
            mock_response="hi",
        )

    from litellm.types.router import RouterErrors

    assert RouterErrors.no_deployments_with_tag_routing.value in str(exc_info.value)

    # ...and a tier carrying it serves the request even though it lacks "route".
    response = await _tagged_marker_router(tier_tags=["region:eu"]).acompletion(
        model="gpt4o",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["route", "&region:eu"], "inherited_tags": ["&region:eu"]},
        mock_response="hi",
    )

    assert response._hidden_params["model_id"] == "tier-gemini-flash"


def test_request_tags_after_router_consumption_scopes_to_the_stamped_group():
    from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY
    from litellm.router_strategy.tag_based_routing import _request_tags_after_router_consumption
    from litellm.types.router import ConsumedRequestTagsStamp

    metadata = {
        "tags": ["route", "&region:eu"],
        "inherited_tags": ["&region:eu"],
        CONSUMED_REQUEST_TAGS_METADATA_KEY: ConsumedRequestTagsStamp(model_group="gemini-flash", tags=("route",)),
    }
    assert _request_tags_after_router_consumption(metadata, "gemini-flash") == ("&region:eu",)
    assert _request_tags_after_router_consumption(metadata, "other-group") == ["route", "&region:eu"]


def test_request_tags_after_router_consumption_drops_only_the_consumed_tags():
    from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY
    from litellm.router_strategy.tag_based_routing import _request_tags_after_router_consumption
    from litellm.types.router import ConsumedRequestTagsStamp

    fully_consumed = {
        "tags": ["route"],
        CONSUMED_REQUEST_TAGS_METADATA_KEY: ConsumedRequestTagsStamp(model_group="gemini-flash", tags=("route",)),
    }
    assert _request_tags_after_router_consumption(fully_consumed, "gemini-flash") is None

    partially_consumed = {
        "tags": ["route", "deploy:us"],
        "inherited_tags": [],
        CONSUMED_REQUEST_TAGS_METADATA_KEY: ConsumedRequestTagsStamp(model_group="gemini-flash", tags=("route",)),
    }
    assert _request_tags_after_router_consumption(partially_consumed, "gemini-flash") == ("deploy:us",)


@pytest.mark.asyncio()
async def test_non_router_tags_still_pick_the_matching_tier_deployment():
    # tags=["route", "deploy:us"]: "route" picks the router and is spent there,
    # but "deploy:us" must keep constraining deployment choice inside the routed
    # group instead of being dropped with it.
    from litellm.types.router import TaggedPreRoutingStrategy

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "plain-gpt4o"},
            },
            {
                "model_name": "gemini-flash",
                "litellm_params": {"model": "gemini/gemini-3.6-flash", "tags": ["deploy:us"]},
                "model_info": {"id": "tier-gemini-flash-us"},
            },
            {
                "model_name": "gemini-flash",
                "litellm_params": {"model": "gemini/gemini-3.6-flash", "tags": ["deploy:eu"]},
                "model_info": {"id": "tier-gemini-flash-eu"},
            },
        ],
        enable_tag_filtering=True,
    )
    router.auto_routers = {
        "gpt4o": [TaggedPreRoutingStrategy(tags=("route",), strategy=_RewriteToTierStrategy("gemini-flash"))]
    }

    response = await router.acompletion(
        model="gpt4o",
        messages=[{"role": "user", "content": "hi"}],
        metadata={"tags": ["route", "deploy:us"], "inherited_tags": []},
        mock_response="hi",
    )

    assert response._hidden_params["model_id"] == "tier-gemini-flash-us"
