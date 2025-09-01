"""
This tests the pattern matching router

Pattern matching router is used to match patterns like openai/*, vertex_ai/*, anthropic/* etc. (wildcard matching)
"""

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, AsyncMock

load_dotenv()

from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


def test_pattern_match_router_initialization():
    router = PatternMatchRouter()
    assert router.patterns == {}


def test_add_pattern():
    """
    Tests that openai/* is added to the patterns

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment.to_json(exclude_none=True))
    assert len(router.patterns) == 1
    assert list(router.patterns.keys())[0] == "openai/(.*)"

    # try getting the pattern
    assert router.route(request="openai/gpt-15") == [
        deployment.to_json(exclude_none=True)
    ]


def test_add_pattern_vertex_ai():
    """
    Tests that vertex_ai/* is added to the patterns

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="this-can-be-anything",
        litellm_params=LiteLLM_Params(model="vertex_ai/gemini-1.5-flash-latest"),
        model_info=ModelInfo(),
    )
    router.add_pattern("vertex_ai/*", deployment.to_json(exclude_none=True))
    assert len(router.patterns) == 1
    assert list(router.patterns.keys())[0] == "vertex_ai/(.*)"

    # try getting the pattern
    assert router.route(request="vertex_ai/gemini-1.5-flash-latest") == [
        deployment.to_json(exclude_none=True)
    ]


def test_add_multiple_deployments():
    """
    Tests adding multiple deployments for the same pattern

    when we try to get the pattern, it should return the deployment
    """
    router = PatternMatchRouter()
    deployment1 = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    deployment2 = Deployment(
        model_name="openai-2",
        litellm_params=LiteLLM_Params(model="gpt-4"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment1.to_json(exclude_none=True))
    router.add_pattern("openai/*", deployment2.to_json(exclude_none=True))
    assert len(router.route("openai/gpt-4o")) == 2


def test_pattern_to_regex():
    """
    Tests that the pattern is converted to a regex
    """
    router = PatternMatchRouter()
    assert router._pattern_to_regex("openai/*") == "openai/(.*)"
    assert (
        router._pattern_to_regex("openai/fo::*::static::*")
        == "openai/fo::(.*)::static::(.*)"
    )


def test_route_with_none():
    """
    Tests that the router returns None when the request is None
    """
    router = PatternMatchRouter()
    assert router.route(None) is None


def test_route_with_multiple_matching_patterns():
    """
    Tests that the router returns the first matching pattern when there are multiple matching patterns
    """
    router = PatternMatchRouter()
    deployment1 = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    deployment2 = Deployment(
        model_name="openai-2",
        litellm_params=LiteLLM_Params(model="gpt-4"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment1.to_json(exclude_none=True))
    router.add_pattern("openai/gpt-*", deployment2.to_json(exclude_none=True))
    assert router.route("openai/gpt-3.5-turbo") == [
        deployment2.to_json(exclude_none=True)
    ]


# Add this test to check for exception handling
def test_route_with_exception():
    """
    Tests that the router returns None when there is an exception calling router.route()
    """
    router = PatternMatchRouter()
    deployment = Deployment(
        model_name="openai-1",
        litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
        model_info=ModelInfo(),
    )
    router.add_pattern("openai/*", deployment.to_json(exclude_none=True))

    router.patterns = (
        []
    )  # this will cause router.route to raise an exception, since router.patterns should be a dict

    result = router.route("openai/gpt-3.5-turbo")
    assert result is None


@pytest.mark.asyncio
async def test_route_with_no_matching_pattern():
    """
    Tests that the router returns None when there is no matching pattern
    """
    from litellm.types.router import RouterErrors

    router = Router(
        model_list=[
            {
                "model_name": "*meta.llama3*",
                "litellm_params": {"model": "bedrock/meta.llama3*"},
            }
        ]
    )

    ## WORKS
    result = await router.acompletion(
        model="bedrock/meta.llama3-70b",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Works",
    )
    assert result.choices[0].message.content == "Works"

    ## WORKS
    result = await router.acompletion(
        model="meta.llama3-70b-instruct-v1:0",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="Works",
    )
    assert result.choices[0].message.content == "Works"

    ## FAILS
    with pytest.raises(litellm.BadRequestError) as e:
        await router.acompletion(
            model="my-fake-model",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response="Works",
        )

    assert RouterErrors.no_deployments_available.value not in str(e.value)

    with pytest.raises(litellm.BadRequestError):
        await router.aembedding(
            model="my-fake-model",
            input="Hello, world!",
        )


def test_router_pattern_match_e2e():
    """
    Tests the end to end flow of the router
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    router = Router(
        model_list=[
            {
                "model_name": "llmengine/*",
                "litellm_params": {"model": "anthropic/*", "api_key": "test"},
            }
        ]
    )

    with patch.object(client, "post", new=MagicMock()) as mock_post:

        router.completion(
            model="llmengine/my-custom-model",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            client=client,
            api_key="test",
        )
        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs["data"])
        mock_post.call_args.kwargs["data"] == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        }


def test_pattern_matching_router_with_default_wildcard():
    """
    Tests that the router returns the default wildcard model when the pattern is not found

    Make sure generic '*' allows all models to be passed through.
    """
    router = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {"model": "*"},
                "model_info": {"access_groups": ["default"]},
            },
            {
                "model_name": "anthropic-claude",
                "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
            },
        ]
    )

    assert len(router.pattern_router.patterns) > 0

    router.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
    )


def test_pattern_matching_router_with_default_wildcard_and_model_wildcard():
    """
    Match to more specific pattern first.
    """
    router = Router(
        model_list=[
            {
                "model_name": "*",
                "litellm_params": {"model": "*"},
                "model_info": {"access_groups": ["default"]},
            },
            {
                "model_name": "llmengine/*",
                "litellm_params": {"model": "openai/*"},
            },
        ]
    )

    assert len(router.pattern_router.patterns) > 0

    pattern_router = router.pattern_router
    deployments = pattern_router.route("llmengine/gpt-3.5-turbo")
    assert len(deployments) == 1
    assert deployments[0]["model_name"] == "llmengine/*"


def test_sorted_patterns():
    """
    Tests that the pattern specificity is calculated correctly
    """
    from litellm.router_utils.pattern_match_deployments import PatternUtils

    sorted_patterns = PatternUtils.sorted_patterns(
        {
            "llmengine/*": [{"model_name": "anthropic/claude-3-5-sonnet"}],
            "*": [{"model_name": "openai/*"}],
        },
    )
    assert sorted_patterns[0][0] == "llmengine/*"


def test_calculate_pattern_specificity():
    from litellm.router_utils.pattern_match_deployments import PatternUtils

    assert PatternUtils.calculate_pattern_specificity("llmengine/*") == (11, 1)
    assert PatternUtils.calculate_pattern_specificity("*") == (1, 1)
