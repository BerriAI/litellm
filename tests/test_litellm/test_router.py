import copy
import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import litellm


def test_update_kwargs_does_not_mutate_defaults_and_merges_metadata():
    # initialize a real Router (env‑vars can be empty)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/chatgpt-v-3",
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
                    "model": "claude-3-5-sonnet-latest",
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
                "model_name": "claude-3-5-sonnet-latest",
                "litellm_params": {
                    "model": "anthropic/claude-3-5-sonnet-latest",
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
            {"claude-3-5-sonnet-latest": ["bedrock-claude"]},
        ],
    )

    result = await router.aanthropic_messages(
        model="claude-3-5-sonnet-latest",
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