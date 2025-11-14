import copy
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import litellm
from litellm.router_utils.fallback_event_handlers import run_async_fallback


def test_update_kwargs_does_not_mutate_defaults_and_merges_metadata():
    # initialize a real Router (env‑vars can be empty)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            }
        ],
    )

    # override to known defaults for the test
    router.default_litellm_params = {
        "foo": "bar",
        "metadata": {"baz": 123},
    }
    original = copy.deepcopy(router.default_litellm_params)
    kwargs: dict = {}

    # invoke the helper
    router._update_kwargs_with_default_litellm_params(
        kwargs=kwargs,
        metadata_variable_name="litellm_metadata",
    )

    # 1) router.defaults must be unchanged
    assert router.default_litellm_params == original

    # 2) non‑metadata keys get merged
    assert kwargs["foo"] == "bar"

    # 3) metadata lands under "metadata"
    assert kwargs["litellm_metadata"] == {"baz": 123}


def test_router_with_model_info_and_model_group():
    """
    Test edge case where user specifies model_group in model_info
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
                "model_info": {
                    "tpm": 1000,
                    "rpm": 1000,
                    "model_group": "gpt-3.5-turbo",
                },
            }
        ],
    )

    router._set_model_group_info(
        model_group="gpt-3.5-turbo",
        user_facing_model_group_name="gpt-3.5-turbo",
    )


@pytest.mark.asyncio
async def test_arouter_with_tags_and_fallbacks():
    """
    If fallback model missing tag, raise error
    """
    from litellm import Router

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "Hello, world!",
                    "tags": ["test"],
                },
            },
            {
                "model_name": "anthropic-claude-3-5-sonnet",
                "litellm_params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "mock_response": "Hello, world 2!",
                },
            },
        ],
        fallbacks=[
            {"gpt-3.5-turbo": ["anthropic-claude-3-5-sonnet"]},
        ],
        enable_tag_filtering=True,
    )

    with pytest.raises(Exception):
        response = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_testing_fallbacks=True,
            metadata={"tags": ["test"]},
        )


@pytest.mark.asyncio
async def test_async_router_acreate_file():
    """
    Write to all deployments of a model
    """
    from unittest.mock import MagicMock, call, patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            {"model_name": "gpt-3.5-turbo", "litellm_params": {"model": "gpt-4o-mini"}},
        ],
    )

    with patch("litellm.acreate_file", return_value=MagicMock()) as mock_acreate_file:
        mock_acreate_file.return_value = MagicMock()
        response = await router.acreate_file(
            model="gpt-3.5-turbo",
            purpose="test",
            file=MagicMock(),
        )

        # assert that the mock_acreate_file was called twice
        assert mock_acreate_file.call_count == 2


@pytest.mark.asyncio
async def test_async_router_acreate_file_with_jsonl():
    """
    Test router.acreate_file with both JSONL and non-JSONL files
    """
    import json
    from io import BytesIO
    from unittest.mock import MagicMock, patch

    # Create test JSONL content
    jsonl_data = [
        {
            "body": {
                "model": "gpt-3.5-turbo-router",
                "messages": [{"role": "user", "content": "test"}],
            }
        },
        {
            "body": {
                "model": "gpt-3.5-turbo-router",
                "messages": [{"role": "user", "content": "test2"}],
            }
        },
    ]
    jsonl_content = "\n".join(json.dumps(item) for item in jsonl_data)
    jsonl_file = BytesIO(jsonl_content.encode("utf-8"))
    jsonl_file.name = "test.jsonl"

    # Create test non-JSONL content
    non_jsonl_content = "This is not a JSONL file"
    non_jsonl_file = BytesIO(non_jsonl_content.encode("utf-8"))
    non_jsonl_file.name = "test.txt"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo-router",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
            {
                "model_name": "gpt-3.5-turbo-router",
                "litellm_params": {"model": "gpt-4o-mini"},
            },
        ],
    )

    with patch("litellm.acreate_file", return_value=MagicMock()) as mock_acreate_file:
        # Test with JSONL file
        response = await router.acreate_file(
            model="gpt-3.5-turbo-router",
            purpose="batch",
            file=jsonl_file,
        )

        # Verify mock was called twice (once for each deployment)
        print(f"mock_acreate_file.call_count: {mock_acreate_file.call_count}")
        print(f"mock_acreate_file.call_args_list: {mock_acreate_file.call_args_list}")
        assert mock_acreate_file.call_count == 2

        # Get the file content passed to the first call
        first_call_file = mock_acreate_file.call_args_list[0][1]["file"]
        first_call_content = first_call_file.read().decode("utf-8")

        # Verify the model name was replaced in the JSONL content
        first_line = json.loads(first_call_content.split("\n")[0])
        assert first_line["body"]["model"] == "gpt-3.5-turbo"

        # Reset mock for next test
        mock_acreate_file.reset_mock()

        # Test with non-JSONL file
        response = await router.acreate_file(
            model="gpt-3.5-turbo-router",
            purpose="user_data",
            file=non_jsonl_file,
        )

        # Verify mock was called twice
        assert mock_acreate_file.call_count == 2

        # Get the file content passed to the first call
        first_call_file = mock_acreate_file.call_args_list[0][1]["file"]
        first_call_content = first_call_file.read().decode("utf-8")

        # Verify the non-JSONL content was not modified
        assert first_call_content == non_jsonl_content


@pytest.mark.asyncio
async def test_arouter_async_get_healthy_deployments():
    """
    Test that afile_content returns the correct file content
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            },
        ],
    )

    result = await router.async_get_healthy_deployments(
        model="gpt-3.5-turbo",
        request_kwargs={},
        messages=None,
        input=None,
        specific_deployment=False,
        parent_otel_span=None,
    )

    assert len(result) == 1
    assert result[0]["model_name"] == "gpt-3.5-turbo"
    assert result[0]["litellm_params"]["model"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
@patch("litellm.amoderation")
async def test_arouter_amoderation_with_credential_name(mock_amoderation):
    """
    Test that router.amoderation passes litellm_credential_name to the underlying litellm.amoderation call
    """
    mock_amoderation.return_value = AsyncMock()

    router = litellm.Router(
        model_list=[
            {
                "model_name": "text-moderation-stable",
                "litellm_params": {
                    "model": "text-moderation-stable",
                    "litellm_credential_name": "my-custom-auth",
                },
            },
        ],
    )

    await router.amoderation(input="I love everyone!", model="text-moderation-stable")

    mock_amoderation.assert_called_once()
    call_kwargs = mock_amoderation.call_args[1]  # Get the kwargs of the call
    print(
        "call kwargs for router.amoderation=",
        json.dumps(call_kwargs, indent=4, default=str),
    )
    assert call_kwargs["litellm_credential_name"] == "my-custom-auth"
    assert call_kwargs["model"] == "text-moderation-stable"


def test_arouter_test_team_model():
    """
    Test that router.test_team_model returns the correct model
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "team_id": "test-team",
                    "team_public_model_name": "test-model",
                },
            },
        ],
    )

    result = router.map_team_model(team_model_name="test-model", team_id="test-team")
    assert result is not None


def test_arouter_ignore_invalid_deployments():
    """
    Test that router.ignore_invalid_deployments is set to True
    """
    from litellm.types.router import Deployment

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "my-bad-model"},
            },
        ],
        ignore_invalid_deployments=True,
    )

    assert router.ignore_invalid_deployments is True
    assert router.get_model_list() == []

    ## check upsert deployment
    router.upsert_deployment(
        Deployment(
            model_name="gpt-3.5-turbo",
            litellm_params={"model": "my-bad-model"},  # type: ignore
            model_info={"tpm": 1000, "rpm": 1000},
        )
    )

    assert router.get_model_list() == []


@pytest.mark.asyncio
async def test_arouter_aretrieve_batch():
    """
    Test that router.aretrieve_batch returns the correct response
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "custom_llm_provider": "azure",
                    "api_key": "my-custom-key",
                    "api_base": "my-custom-base",
                },
            }
        ],
    )

    with patch.object(
        litellm, "aretrieve_batch", return_value=AsyncMock()
    ) as mock_aretrieve_batch:
        try:
            response = await router.aretrieve_batch(
                model="gpt-3.5-turbo",
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_aretrieve_batch.assert_called_once()

        print(mock_aretrieve_batch.call_args.kwargs)
        assert mock_aretrieve_batch.call_args.kwargs["api_key"] == "my-custom-key"
        assert mock_aretrieve_batch.call_args.kwargs["api_base"] == "my-custom-base"


@pytest.mark.asyncio
async def test_arouter_aretrieve_file_content():
    """
    Test that router.acreate_file with JSONL file returns the correct response
    """

    with patch.object(
        litellm, "afile_content", return_value=AsyncMock()
    ) as mock_afile_content:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "custom_llm_provider": "azure",
                        "api_key": "my-custom-key",
                        "api_base": "my-custom-base",
                    },
                }
            ],
        )
        try:
            response = await router.afile_content(
                **{
                    "model": "gpt-3.5-turbo",
                    "file_id": "my-unique-file-id",
                }
            )  # type: ignore
        except Exception as e:
            print(f"Error: {e}")

        mock_afile_content.assert_called_once()

        print(mock_afile_content.call_args.kwargs)
        assert mock_afile_content.call_args.kwargs["api_key"] == "my-custom-key"
        assert mock_afile_content.call_args.kwargs["api_base"] == "my-custom-base"


@pytest.mark.asyncio
async def test_arouter_filter_team_based_models():
    """
    Test that router.filter_team_based_models filters out models that are not in the team
    """
    from litellm.types.router import Deployment

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "team_id": "test-team",
                },
            },
        ],
    )

    # WORKS
    result = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        metadata={"user_api_key_team_id": "test-team"},
        mock_response="Hello, world!",
    )

    assert result is not None

    # FAILS
    with pytest.raises(Exception) as e:
        result = await router.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
            metadata={"user_api_key_team_id": "test-team-2"},
            mock_response="Hello, world!",
        )
    assert "No deployments available" in str(e.value)

    ## ADD A MODEL THAT IS NOT IN THE TEAM
    router.add_deployment(
        Deployment(
            model_name="gpt-3.5-turbo",
            litellm_params={"model": "gpt-3.5-turbo"},  # type: ignore
            model_info={"tpm": 1000, "rpm": 1000},
        )
    )

    result = await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        metadata={"user_api_key_team_id": "test-team-2"},
        mock_response="Hello, world!",
    )

    assert result is not None


def test_arouter_should_include_deployment():
    """
    Test the should_include_deployment method with various scenarios

    The method logic:
    1. Returns True if: team_id matches AND model_name matches team_public_model_name
    2. Returns True if: model_name matches AND deployment has no team_id
    3. Otherwise returns False
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "team_id": "test-team",
                },
            },
        ],
    )

    # Test deployment structures
    deployment_with_team_and_public_name = {
        "model_name": "gpt-3.5-turbo",
        "model_info": {
            "team_id": "test-team",
            "team_public_model_name": "team-gpt-model",
        },
    }

    deployment_with_team_no_public_name = {
        "model_name": "gpt-3.5-turbo",
        "model_info": {
            "team_id": "test-team",
        },
    }

    deployment_without_team = {
        "model_name": "gpt-4",
        "model_info": {},
    }

    deployment_different_team = {
        "model_name": "claude-3",
        "model_info": {
            "team_id": "other-team",
            "team_public_model_name": "team-claude-model",
        },
    }

    # Test Case 1: Team-specific deployment - team_id and team_public_model_name match
    result = router.should_include_deployment(
        model_name="team-gpt-model",
        model=deployment_with_team_and_public_name,
        team_id="test-team",
    )
    assert (
        result is True
    ), "Should return True when team_id and team_public_model_name match"

    # Test Case 2: Team-specific deployment - team_id matches but model_name doesn't match team_public_model_name
    result = router.should_include_deployment(
        model_name="different-model",
        model=deployment_with_team_and_public_name,
        team_id="test-team",
    )
    assert (
        result is False
    ), "Should return False when team_id matches but model_name doesn't match team_public_model_name"

    # Test Case 3: Team-specific deployment - team_id doesn't match
    result = router.should_include_deployment(
        model_name="team-gpt-model",
        model=deployment_with_team_and_public_name,
        team_id="different-team",
    )
    assert result is False, "Should return False when team_id doesn't match"

    # Test Case 4: Team-specific deployment with no team_public_model_name - should fail
    result = router.should_include_deployment(
        model_name="gpt-3.5-turbo",
        model=deployment_with_team_no_public_name,
        team_id="test-team",
    )
    assert (
        result is True
    ), "Should return True when team deployment has no team_public_model_name to match"

    # Test Case 5: Non-team deployment - model_name matches and no team_id
    result = router.should_include_deployment(
        model_name="gpt-4", model=deployment_without_team, team_id=None
    )
    assert (
        result is True
    ), "Should return True when model_name matches and deployment has no team_id"

    # Test Case 6: Non-team deployment - model_name matches but team_id provided (should still work)
    result = router.should_include_deployment(
        model_name="gpt-4", model=deployment_without_team, team_id="any-team"
    )
    assert (
        result is True
    ), "Should return True when model_name matches non-team deployment, regardless of team_id param"

    # Test Case 7: Non-team deployment - model_name doesn't match
    result = router.should_include_deployment(
        model_name="different-model", model=deployment_without_team, team_id=None
    )
    assert result is False, "Should return False when model_name doesn't match"

    # Test Case 8: Team deployment accessed without matching team_id
    result = router.should_include_deployment(
        model_name="gpt-3.5-turbo",
        model=deployment_with_team_and_public_name,
        team_id=None,
    )
    assert (
        result is True
    ), "Should return True when matching model with exact model_name"


def test_arouter_responses_api_bridge():
    """
    Test that router.responses_api_bridge returns the correct response
    """
    from unittest.mock import MagicMock, patch

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    router = litellm.Router(
        model_list=[
            {
                "model_name": "[IP-approved] o3-pro",
                "litellm_params": {
                    "model": "azure/responses/o_series/webinterface-o3-pro",
                    "api_base": "https://webhook.site/fba79dae-220a-4bb7-9a3a-8caa49604e55",
                    "api_key": "sk-1234567890",
                    "api_version": "preview",
                    "stream": True,
                },
                "model_info": {
                    "input_cost_per_token": 0.00002,
                    "output_cost_per_token": 0.00008,
                },
            }
        ],
    )

    ## CONFIRM BRIDGE IS CALLED
    with patch.object(litellm, "responses", return_value=AsyncMock()) as mock_responses:
        result = router.completion(
            model="[IP-approved] o3-pro",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
        assert mock_responses.call_count == 1

    ## CONFIRM MODEL NAME IS STRIPPED
    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            result = router.completion(
                model="[IP-approved] o3-pro",
                messages=[{"role": "user", "content": "Hello, world!"}],
                client=client,
                num_retries=0,
            )
        except Exception as e:
            print(f"Error: {e}")

        assert mock_post.call_count == 1
        assert (
            mock_post.call_args.kwargs["url"]
            == "https://webhook.site/fba79dae-220a-4bb7-9a3a-8caa49604e55/openai/v1/responses?api-version=preview"
        )
        assert mock_post.call_args.kwargs["json"]["model"] == "webinterface-o3-pro"


@pytest.mark.asyncio
async def test_router_v1_messages_fallbacks():
    """
    Test that router.v1_messages_fallbacks returns the correct response
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-5-20250929",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5-20250929",
                    "mock_response": "litellm.InternalServerError",
                },
            },
            {
                "model_name": "bedrock-claude",
                "litellm_params": {
                    "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "mock_response": "Hello, world I am a fallback!",
                },
            },
        ],
        fallbacks=[
            {"claude-sonnet-4-5-20250929": ["bedrock-claude"]},
        ],
    )

    result = await router.aanthropic_messages(
        model="claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=256,
    )
    assert result is not None

    print(result)
    assert result["content"][0]["text"] == "Hello, world I am a fallback!"


def test_add_invalid_provider_to_router():
    """
    Test that router.add_deployment raises an error if the provider is invalid
    """
    from litellm.types.router import Deployment

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
    )

    with pytest.raises(Exception) as e:
        router.add_deployment(
            Deployment(
                model_name="vertex_ai/*",
                litellm_params={
                    "model": "vertex_ai/*",
                    "custom_llm_provider": "vertex_ai_eu",
                },
            )
        )

    assert router.pattern_router.patterns == {}


@pytest.mark.asyncio
async def test_router_ageneric_api_call_with_fallbacks_helper():
    """
    Test the _ageneric_api_call_with_fallbacks_helper method with various scenarios
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "https://api.openai.com/v1",
                },
                "model_info": {
                    "tpm": 1000,
                    "rpm": 1000,
                },
            },
        ],
    )

    # Test 1: Successful call
    async def mock_generic_function(**kwargs):
        return {"result": "success", "model": kwargs.get("model")}

    with patch.object(router, "async_get_available_deployment") as mock_get_deployment:
        mock_get_deployment.return_value = {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "api_base": "https://api.openai.com/v1",
            },
        }

        with patch.object(
            router, "_update_kwargs_with_deployment"
        ) as mock_update_kwargs:
            with patch.object(
                router, "async_routing_strategy_pre_call_checks"
            ) as mock_pre_call_checks:
                with patch.object(
                    router, "_get_client", return_value=None
                ) as mock_get_client:
                    result = await router._ageneric_api_call_with_fallbacks_helper(
                        model="gpt-3.5-turbo",
                        original_generic_function=mock_generic_function,
                        messages=[{"role": "user", "content": "test"}],
                    )

                    assert result is not None
                    assert result["result"] == "success"
                    mock_get_deployment.assert_called_once()
                    mock_update_kwargs.assert_called_once()
                    mock_pre_call_checks.assert_called_once()

    # Test 2: Passthrough on no deployment (success case)
    async def mock_passthrough_function(**kwargs):
        return {"result": "passthrough", "model": kwargs.get("model")}

    with patch.object(router, "async_get_available_deployment") as mock_get_deployment:
        mock_get_deployment.side_effect = Exception("No deployment available")

        result = await router._ageneric_api_call_with_fallbacks_helper(
            model="gpt-3.5-turbo",
            original_generic_function=mock_passthrough_function,
            passthrough_on_no_deployment=True,
            messages=[{"role": "user", "content": "test"}],
        )

        assert result is not None
        assert result["result"] == "passthrough"
        assert result["model"] == "gpt-3.5-turbo"

    # Test 3: No deployment available and passthrough=False (should raise exception)
    with patch.object(router, "async_get_available_deployment") as mock_get_deployment:
        mock_get_deployment.side_effect = Exception("No deployment available")

        with pytest.raises(Exception) as exc_info:
            await router._ageneric_api_call_with_fallbacks_helper(
                model="gpt-3.5-turbo",
                original_generic_function=mock_generic_function,
                passthrough_on_no_deployment=False,
                messages=[{"role": "user", "content": "test"}],
            )

        assert "No deployment available" in str(exc_info.value)

    # Test 4: Test with semaphore (rate limiting)
    import asyncio

    async def mock_semaphore_function(**kwargs):
        return {"result": "semaphore_success", "model": kwargs.get("model")}

    with patch.object(router, "async_get_available_deployment") as mock_get_deployment:
        mock_get_deployment.return_value = {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "api_base": "https://api.openai.com/v1",
            },
        }

        mock_semaphore = asyncio.Semaphore(1)

        with patch.object(
            router, "_update_kwargs_with_deployment"
        ) as mock_update_kwargs:
            with patch.object(
                router, "_get_client", return_value=mock_semaphore
            ) as mock_get_client:
                with patch.object(
                    router, "async_routing_strategy_pre_call_checks"
                ) as mock_pre_call_checks:
                    result = await router._ageneric_api_call_with_fallbacks_helper(
                        model="gpt-3.5-turbo",
                        original_generic_function=mock_semaphore_function,
                        messages=[{"role": "user", "content": "test"}],
                    )

                    assert result is not None
                    assert result["result"] == "semaphore_success"
                    mock_get_client.assert_called_once()
                    mock_pre_call_checks.assert_called_once()

    # Test 5: Test call tracking (success and failure counts)
    initial_success_count = router.success_calls.get("gpt-3.5-turbo", 0)
    initial_fail_count = router.fail_calls.get("gpt-3.5-turbo", 0)

    async def mock_failing_function(**kwargs):
        raise Exception("Mock failure")

    with patch.object(router, "async_get_available_deployment") as mock_get_deployment:
        mock_get_deployment.return_value = {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "test-key",
                "api_base": "https://api.openai.com/v1",
            },
        }

        with patch.object(
            router, "_update_kwargs_with_deployment"
        ) as mock_update_kwargs:
            with patch.object(
                router, "_get_client", return_value=None
            ) as mock_get_client:
                with patch.object(
                    router, "async_routing_strategy_pre_call_checks"
                ) as mock_pre_call_checks:
                    with pytest.raises(Exception) as exc_info:
                        await router._ageneric_api_call_with_fallbacks_helper(
                            model="gpt-3.5-turbo",
                            original_generic_function=mock_failing_function,
                            messages=[{"role": "user", "content": "test"}],
                        )

                    assert "Mock failure" in str(exc_info.value)
                    # Check that fail_calls was incremented
                    assert router.fail_calls["gpt-3.5-turbo"] == initial_fail_count + 1


def test_router_get_model_access_groups_team_only_models():
    """
    Test that Router.get_model_access_groups returns the correct response for team-only models
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model-name",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {
                    "team_id": "team_1",
                    "access_groups": ["default-models"],
                    "team_public_model_name": "gpt-3.5-turbo",
                },
            },
        ]
    )

    access_groups = router.get_model_access_groups(
        model_name="gpt-3.5-turbo", team_id=None
    )
    assert len(access_groups) == 0

    access_groups = router.get_model_access_groups(
        model_name="gpt-3.5-turbo", team_id="team_1"
    )
    assert list(access_groups.keys()) == ["default-models"]


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator():
    """Test _acompletion_streaming_iterator for normal streaming and fallback behavior."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.utils import ModelResponseStream

    # Helper class for creating async iterators
    class AsyncIterator:
        def __init__(self, items, error_after=None):
            self.items = items
            self.index = 0
            self.error_after = error_after

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.error_after is not None and self.index >= self.error_after:
                raise self.error_after
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    # Set up router with fallback configuration
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key-1"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key-2"},
            },
        ],
        fallbacks=[{"gpt-4": ["gpt-3.5-turbo"]}],
        set_verbose=True,
    )

    # Test data
    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "gpt-4", "stream": True, "temperature": 0.7}

    # Test 1: Successful streaming (no errors)
    print("\n=== Test 1: Successful streaming ===")

    # Mock successful streaming response
    mock_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" there"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
    ]

    mock_response = AsyncIterator(mock_chunks)

    setattr(mock_response, "model", "gpt-4")
    setattr(mock_response, "custom_llm_provider", "openai")
    setattr(mock_response, "logging_obj", MagicMock())

    result = await router._acompletion_streaming_iterator(
        model_response=mock_response, messages=messages, initial_kwargs=initial_kwargs
    )

    # Collect streamed chunks
    collected_chunks = []
    async for chunk in result:
        collected_chunks.append(chunk)

    assert len(collected_chunks) == 3
    assert all(chunk in mock_chunks for chunk in collected_chunks)
    print("✓ Successfully streamed all chunks")

    # Test 2: MidStreamFallbackError with fallback
    print("\n=== Test 2: MidStreamFallbackError with fallback ===")

    # Create error that should trigger after first chunk
    error = MidStreamFallbackError(
        message="Connection lost",
        model="gpt-4",
        llm_provider="openai",
        generated_content="Hello",
    )

    class AsyncIteratorWithError:
        def __init__(self, items, error_after_index):
            self.items = items
            self.index = 0
            self.error_after_index = error_after_index
            self.chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            if self.index == self.error_after_index:
                raise error
            item = self.items[self.index]
            self.index += 1
            return item

    mock_error_response = AsyncIteratorWithError(
        mock_chunks, 1
    )  # Error after first chunk

    setattr(mock_error_response, "model", "gpt-4")
    setattr(mock_error_response, "custom_llm_provider", "openai")
    setattr(mock_error_response, "logging_obj", MagicMock())

    # Mock the fallback response
    fallback_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
    ]

    mock_fallback_response = AsyncIterator(fallback_chunks)

    # Mock the fallback function
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=mock_fallback_response,
    ) as mock_fallback_utils:
        collected_chunks = []
        result = await router._acompletion_streaming_iterator(
            model_response=mock_error_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        async for chunk in result:
            collected_chunks.append(chunk)

        # Verify fallback was called
        assert mock_fallback_utils.called
        call_args = mock_fallback_utils.call_args

        # Check that generated content was added to messages
        fallback_kwargs = call_args.kwargs["kwargs"]
        modified_messages = fallback_kwargs["messages"]

        # Should have original message + system message + assistant message with prefix
        assert len(modified_messages) == 3
        assert modified_messages[0] == {"role": "user", "content": "Hello"}
        assert modified_messages[1]["role"] == "system"
        assert "continuation" in modified_messages[1]["content"]
        assert modified_messages[2]["role"] == "assistant"
        assert modified_messages[2]["content"] == "Hello"
        assert modified_messages[2]["prefix"] == True

        # Verify fallback parameters
        assert call_args.kwargs["disable_fallbacks"] == False
        assert call_args.kwargs["model_group"] == "gpt-4"

        # Should get original chunk + fallback chunks
        assert len(collected_chunks) == 3  # 1 original + 2 fallback
        print("✓ Fallback system called correctly with proper message modification")

    print("\n=== All tests passed! ===")


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_edge_cases():
    """Test edge cases for _acompletion_streaming_iterator."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
        set_verbose=True,
    )

    messages = [{"role": "user", "content": "Test"}]
    initial_kwargs = {"model": "gpt-4", "stream": True}

    # Test: Empty generated content
    empty_error = MidStreamFallbackError(
        message="Error",
        model="gpt-4",
        llm_provider="openai",
        generated_content="",  # Empty content
    )

    class AsyncIteratorImmediateError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise empty_error

    mock_response = AsyncIteratorImmediateError()

    # Mock empty fallback response using AsyncIterator
    class EmptyAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    mock_fallback_response = EmptyAsyncIterator()

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=mock_fallback_response,
    ) as mock_fallback_utils:
        collected_chunks = []
        iterator = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        async for chunk in iterator:
            collected_chunks.append(chunk)

        # Should still call fallback even with empty content
        assert mock_fallback_utils.called
        fallback_kwargs = mock_fallback_utils.call_args.kwargs["kwargs"]
        modified_messages = fallback_kwargs["messages"]

        # Should have assistant message with empty content
        assert modified_messages[2]["content"] == ""
        print("✓ Handles empty generated content correctly")

    print("✓ Edge case tests passed!")


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_common_utils():
    """Test the async_function_with_fallbacks_common_utils method"""
    # Create a basic router for testing
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                },
            }
        ],
        max_fallbacks=5,
    )

    # Test case 1: disable_fallbacks=True should raise original exception
    test_exception = Exception("Test error")
    with pytest.raises(Exception, match="Test error"):
        await router.async_function_with_fallbacks_common_utils(
            e=test_exception,
            disable_fallbacks=True,
            fallbacks=None,
            context_window_fallbacks=None,
            content_policy_fallbacks=None,
            model_group="gpt-3.5-turbo",
            args=(),
            kwargs=MagicMock(),
        )

    # Test case 2: original_model_group=None should raise original exception
    with pytest.raises(Exception, match="Test error"):
        await router.async_function_with_fallbacks_common_utils(
            e=test_exception,
            disable_fallbacks=False,
            fallbacks=None,
            context_window_fallbacks=None,
            content_policy_fallbacks=None,
            model_group="gpt-3.5-turbo",
            args=(),
            kwargs={},  # No model key
        )


def test_should_include_deployment():
    """Test that Router.should_include_deployment returns the correct response"""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "model_name_a28a12f9-3e44-4861-bd4f-325f2d309ce8_cd5dc6fb-b046-4e05-ae1d-32ba4d936266",
                "litellm_params": {"model": "openai/*"},
                "model_info": {
                    "team_id": "a28a12f9-3e44-4861-bd4f-325f2d309ce8",
                    "team_public_model_name": "openai/*",
                },
            }
        ],
    )

    model = {
        "model_name": "model_name_a28a12f9-3e44-4861-bd4f-325f2d309ce8_cd5dc6fb-b046-4e05-ae1d-32ba4d936266",
        "litellm_params": {
            "api_key": "sk-proj-1234567890",
            "custom_llm_provider": "openai",
            "use_in_pass_through": False,
            "use_litellm_proxy": False,
            "merge_reasoning_content_in_choices": False,
            "model": "openai/*",
        },
        "model_info": {
            "id": "95f58039-d54a-4d1c-b700-5e32e99a1120",
            "db_model": True,
            "updated_by": "64a2f787-0863-4d76-9516-2dc49c1598e8",
            "created_by": "64a2f787-0863-4d76-9516-2dc49c1598e8",
            "team_id": "a28a12f9-3e44-4861-bd4f-325f2d309ce8",
            "team_public_model_name": "openai/*",
            "mode": "completion",
            "access_groups": ["restricted-models-openai"],
        },
    }
    model_name = "openai/o4-mini-deep-research"
    team_id = "a28a12f9-3e44-4861-bd4f-325f2d309ce8"
    assert router.get_model_list(
        model_name=model_name,
        team_id=team_id,
    )


def test_get_deployment_model_info_base_model_flow():
    """Test that get_deployment_model_info correctly handles the base model flow"""
    from unittest.mock import patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
    )

    # Mock data for the test
    mock_custom_model_info = {
        "base_model": "gpt-3.5-turbo",
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "custom_field": "custom_value",
    }

    mock_base_model_info = {
        "key": "gpt-3.5-turbo",
        "max_tokens": 4096,
        "max_input_tokens": 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0015,  # This should be overridden by custom model info
        "output_cost_per_token": 0.002,
        "litellm_provider": "openai",
        "mode": "chat",
        "supported_openai_params": ["temperature", "max_tokens"],
    }

    mock_litellm_model_name_info = {
        "key": "test-model",
        "max_tokens": 2048,
        "max_input_tokens": 2048,
        "max_output_tokens": 2048,
        "input_cost_per_token": 0.0005,
        "output_cost_per_token": 0.001,
        "litellm_provider": "test_provider",
        "mode": "completion",
        "supported_openai_params": ["temperature"],
    }

    # Test Case 1: Base model flow with custom model info that has base_model
    with patch.object(
        litellm, "model_cost", {"test-custom-model": mock_custom_model_info}
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            # Configure mock returns
            mock_get_model_info.side_effect = lambda model: {
                "gpt-3.5-turbo": mock_base_model_info,
                "test-model": mock_litellm_model_name_info,
            }.get(model)

            result = router.get_deployment_model_info(
                model_id="test-custom-model", model_name="test-model"
            )

            # Verify that get_model_info was called for both base model and model name
            assert mock_get_model_info.call_count == 2
            mock_get_model_info.assert_any_call(
                model="gpt-3.5-turbo"
            )  # base model call
            mock_get_model_info.assert_any_call(model="test-model")  # model name call

            # Verify the result contains merged information
            assert result is not None

            # Test the correct merging behavior after fix:
            # 1. base_model_info provides defaults, custom_model_info overrides (correct priority)
            # 2. The result of step 1 gets merged into litellm_model_name_info (custom+base override litellm)

            # Fields from custom model (should override base model values)
            assert (
                result["input_cost_per_token"] == 0.001
            )  # From custom model (overrides base 0.0015)
            assert (
                result["output_cost_per_token"] == 0.002
            )  # From custom model (same as base)
            assert result["custom_field"] == "custom_value"  # From custom model

            # Fields from base model that weren't overridden by custom
            assert result["max_tokens"] == 4096  # From base model
            assert result["litellm_provider"] == "openai"  # From base model
            assert (
                result["mode"] == "chat"
            )  # From base model (overrides litellm "completion")

            # The key field comes from base model since both base and litellm have it
            # and base model info overrides litellm model name info in final merge
            assert (
                result["key"] == "gpt-3.5-turbo"
            )  # From base model (overrides litellm key)

    # Test Case 2: Custom model info without base_model
    mock_custom_model_info_no_base = {
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
        "custom_field": "custom_value",
    }

    with patch.object(
        litellm,
        "model_cost",
        {"test-custom-model-no-base": mock_custom_model_info_no_base},
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            mock_get_model_info.side_effect = lambda model: {
                "test-model": mock_litellm_model_name_info,
            }.get(model)

            result = router.get_deployment_model_info(
                model_id="test-custom-model-no-base", model_name="test-model"
            )

            # Should only call get_model_info once for model name (no base model)
            assert mock_get_model_info.call_count == 1
            mock_get_model_info.assert_called_with(model="test-model")

            # Verify the result contains merged information
            assert result is not None
            assert result["input_cost_per_token"] == 0.001  # From custom model
            assert result["max_tokens"] == 2048  # From litellm model name info
            assert result["custom_field"] == "custom_value"  # From custom model
            assert result["mode"] == "completion"  # From litellm model name info

    # Test Case 3: No custom model info, only litellm model name info
    with patch.object(litellm, "model_cost", {}):  # Empty model cost
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            mock_get_model_info.side_effect = lambda model: {
                "test-model": mock_litellm_model_name_info,
            }.get(model)

            result = router.get_deployment_model_info(
                model_id="non-existent-model", model_name="test-model"
            )

            # Should only call get_model_info once for model name
            assert mock_get_model_info.call_count == 1
            mock_get_model_info.assert_called_with(model="test-model")

            # Result should be just the litellm model name info
            assert result is not None
            assert result == mock_litellm_model_name_info

    # Test Case 4: Base model info retrieval fails (exception handling)
    mock_custom_model_info_invalid_base = {
        "base_model": "invalid-base-model",
        "input_cost_per_token": 0.001,
        "output_cost_per_token": 0.002,
    }

    with patch.object(
        litellm,
        "model_cost",
        {"test-custom-model-invalid": mock_custom_model_info_invalid_base},
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            # Mock get_model_info to raise exception for invalid base model
            def mock_get_model_info_side_effect(model):
                if model == "invalid-base-model":
                    raise Exception("Model not found")
                elif model == "test-model":
                    return mock_litellm_model_name_info
                return None

            mock_get_model_info.side_effect = mock_get_model_info_side_effect

            result = router.get_deployment_model_info(
                model_id="test-custom-model-invalid", model_name="test-model"
            )

            # Should handle exception gracefully and still return merged result
            assert result is not None
            assert result["input_cost_per_token"] == 0.001  # From custom model
            assert result["mode"] == "completion"  # From litellm model name info

    # Test Case 5: Both model_cost.get() and get_model_info() return None
    with patch.object(litellm, "model_cost", {}):
        with patch.object(
            litellm, "get_model_info", side_effect=Exception("Not found")
        ):
            result = router.get_deployment_model_info(
                model_id="non-existent", model_name="non-existent"
            )

            # Should return None when no model info is found
            assert result is None

    print("✓ All base model flow test cases passed!")


@patch("litellm.model_cost", {})
def test_get_deployment_model_info_base_model_merge_priority():
    """Test that base model info merging respects the correct priority order"""
    from unittest.mock import patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
    )

    # Test data with overlapping fields to test merge priority
    mock_custom_model_info = {
        "base_model": "gpt-4",
        "input_cost_per_token": 0.01,  # Should override base model value
        "max_tokens": 8000,  # Should override base model value
        "custom_only_field": "custom_value",
    }

    mock_base_model_info = {
        "key": "gpt-4",
        "max_tokens": 4096,  # Should be overridden by custom model
        "input_cost_per_token": 0.03,  # Should be overridden by custom model
        "output_cost_per_token": 0.06,  # Should be preserved (not in custom)
        "litellm_provider": "openai",
        "base_only_field": "base_value",
    }

    mock_litellm_model_name_info = {
        "key": "test-model",
        "max_tokens": 2048,  # Should be overridden by final custom model info
        "input_cost_per_token": 0.005,  # Should be overridden by final custom model info
        "output_cost_per_token": 0.01,  # Should be overridden by final custom model info
        "mode": "completion",
        "litellm_only_field": "litellm_value",
    }

    with patch.object(
        litellm, "model_cost", {"custom-model-id": mock_custom_model_info}
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            mock_get_model_info.side_effect = lambda model: {
                "gpt-4": mock_base_model_info,
                "test-model": mock_litellm_model_name_info,
            }.get(model)

            result = router.get_deployment_model_info(
                model_id="custom-model-id", model_name="test-model"
            )

            assert result is not None

            # Test correct merge priority after fix:
            # 1. base_model_info provides defaults
            # 2. custom_model_info overrides base_model_info
            # 3. Result from steps 1-2 overrides litellm_model_name_info

            # Fields that should come from custom model info (highest priority)
            assert (
                result["input_cost_per_token"] == 0.01
            )  # From custom model (overrides base 0.03)
            assert (
                result["max_tokens"] == 8000
            )  # From custom model (overrides base 4096)
            assert result["custom_only_field"] == "custom_value"  # From custom model

            # Fields that should come from base model (not overridden by custom)
            assert (
                result["output_cost_per_token"] == 0.06
            )  # From base model (not in custom)
            assert (
                result["litellm_provider"] == "openai"
            )  # From base model (not in custom)
            assert (
                result["base_only_field"] == "base_value"
            )  # From base model (not in custom)

            # Fields that should come from litellm model name info (not overridden by custom+base)
            assert (
                result["mode"] == "completion"
            )  # From litellm model name info (not in custom or base)
            assert (
                result["litellm_only_field"] == "litellm_value"
            )  # From litellm model name info (not in custom or base)

            # Key comes from base model since both base and litellm have key fields
            # and the merged custom+base overrides litellm in the final merge
            assert result["key"] == "gpt-4"

    print("✓ Base model merge priority test passed!")


def test_add_deployment_model_to_endpoint_for_llm_passthrough_route():
    """
    Test that _add_deployment_model_to_endpoint_for_llm_passthrough_route correctly strips bedrock provider prefix
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "special-bedrock-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                },
            }
        ],
    )

    # Test Case 1: Bedrock model with provider prefix - should strip "bedrock/" prefix
    kwargs = {
        "endpoint": "/model/special-bedrock-model/invoke",
        "custom_llm_provider": "bedrock",
    }
    result = router._add_deployment_model_to_endpoint_for_llm_passthrough_route(
        kwargs=kwargs,
        model="special-bedrock-model",
        model_name="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    assert (
        result["endpoint"]
        == "/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke"
    ), f"Expected '/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke', got '{result['endpoint']}'"

    # Test Case 2: Bedrock invoke-with-response-stream endpoint
    kwargs = {
        "endpoint": "/model/special-bedrock-model/invoke-with-response-stream",
        "custom_llm_provider": "bedrock",
    }
    result = router._add_deployment_model_to_endpoint_for_llm_passthrough_route(
        kwargs=kwargs,
        model="special-bedrock-model",
        model_name="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    assert (
        result["endpoint"]
        == "/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke-with-response-stream"
    ), f"Expected streaming endpoint with stripped prefix, got '{result['endpoint']}'"

    # Test Case 3: Bedrock converse endpoint
    kwargs = {
        "endpoint": "/model/bedrock-model/converse",
        "custom_llm_provider": "bedrock",
    }
    result = router._add_deployment_model_to_endpoint_for_llm_passthrough_route(
        kwargs=kwargs,
        model="bedrock-model",
        model_name="bedrock/us.meta.llama3-8b-instruct-v1:0",
    )
    assert (
        result["endpoint"] == "/model/us.meta.llama3-8b-instruct-v1:0/converse"
    ), f"Expected '/model/us.meta.llama3-8b-instruct-v1:0/converse', got '{result['endpoint']}'"

    # Test Case 4: Bedrock provider prefix auto-detected from model_name
    kwargs = {
        "endpoint": "/model/router-model/invoke",
    }
    result = router._add_deployment_model_to_endpoint_for_llm_passthrough_route(
        kwargs=kwargs,
        model="router-model",
        model_name="bedrock/us.meta.llama3-8b-instruct-v1:0",
    )
    assert (
        result["endpoint"] == "/model/us.meta.llama3-8b-instruct-v1:0/invoke"
    ), f"Expected '/model/us.meta.llama3-8b-instruct-v1:0/invoke', got '{result['endpoint']}'"


@pytest.mark.asyncio
async def test_router_acompletion_with_unknown_model_and_default_fallback():
    """
    Test that the router successfully uses a default fallback when a completely
    unknown model is requested. It should not raise a BadRequestError.
    This test verifies the fix for issue #15114.
    """
    model_list = [
        {
            "model_name": "gpt-4o",  # This is the fallback model
            "litellm_params": {
                "model": "azure/gpt-4o-real",  # The actual underlying model name
                "api_key": "fake-key",
                "api_base": "https://fake-endpoint.openai.azure.com/",
                "mock_response": "this is the fallback response",  # Mocked response to prevent real API calls
            },
        }
    ]

    # Initialize the router with a default fallback
    router = litellm.Router(model_list=model_list, default_fallbacks=["gpt-4o"])

    messages = [
        {"role": "user", "content": "This call should succeed by falling back."}
    ]

    # Call completion with a model name that is NOT in the model_list
    response = await router.acompletion(
        model="completely-unknown-model", messages=messages
    )

    # Check that the call did not fail and we received a valid response object.
    assert response is not None

    # Check that the content of the response is from the MOCKED fallback model.
    assert response.choices[0].message.content == "this is the fallback response"

    # Check that the response object reports the model that was *actually* called.
    assert response.model == "gpt-4o-real"


@pytest.mark.asyncio
async def test_router_acompletion_with_unknown_model_and_no_fallback():
    """
    Test that the router still raises a BadRequestError for an unknown model
    when no default fallbacks are configured. This ensures we don't break
    the original behavior.
    """
    model_list = [
        {
            "model_name": "gpt-4o",
            "litellm_params": {
                "model": "azure/gpt-4o-real",
                "api_key": "fake-key",
                "mock_response": "this should not be called",
            },
        }
    ]

    # Initialize the router WITHOUT any default fallbacks
    router = litellm.Router(model_list=model_list)

    messages = [{"role": "user", "content": "This call should fail."}]

    # Use pytest.raises to assert that a BadRequestError is thrown.
    with pytest.raises(litellm.BadRequestError) as excinfo:
        await router.acompletion(model="completely-unknown-model", messages=messages)

    # Check that the error message is correct.
    # The router returns 'no healthy deployments' because get_model_list returns [] not None.
    assert "no healthy deployments for this model" in str(excinfo.value)
