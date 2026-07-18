import asyncio
import copy
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import litellm
from litellm.exceptions import MidStreamFallbackError


def test_update_kwargs_does_not_mutate_defaults_and_merges_metadata():
    # initialize a real Router (env‑vars can be empty)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": os.getenv("AZURE_AI_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
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


def test_router_model_group_encrypted_content_affinity_callback_registration():
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    model_group_affinity_config = {
        model_group: ["encrypted_content_affinity"],
    }
    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = []
    router = None

    try:
        router = litellm.Router(
            model_list=[
                {
                    "model_name": model_group,
                    "litellm_params": {
                        "model": "openai/gpt-5.1-codex",
                        "api_key": "mock-api-key",
                    },
                }
            ],
            model_group_affinity_config=model_group_affinity_config,
            num_retries=0,
        )
        callbacks = router.optional_callbacks or []
        encrypted_content_callbacks = [
            cb for cb in callbacks if isinstance(cb, EncryptedContentAffinityCheck)
        ]
        deployment_callback = next(
            cb for cb in callbacks if isinstance(cb, DeploymentAffinityCheck)
        )
        assert len(encrypted_content_callbacks) == 1
        assert encrypted_content_callbacks[0].enable_global_affinity is False
        assert (
            encrypted_content_callbacks[0].model_group_affinity_config
            == model_group_affinity_config
        )
        assert callbacks.index(encrypted_content_callbacks[0]) < callbacks.index(
            deployment_callback
        )
        assert litellm.callbacks.index(encrypted_content_callbacks[0]) < (
            litellm.callbacks.index(deployment_callback)
        )

        router._add_encrypted_content_affinity_check(enable_global_affinity=True)

        callbacks = router.optional_callbacks or []
        encrypted_content_callbacks = [
            cb for cb in callbacks if isinstance(cb, EncryptedContentAffinityCheck)
        ]
        assert len(encrypted_content_callbacks) == 1
        assert encrypted_content_callbacks[0].enable_global_affinity is True
        assert encrypted_content_callbacks[0].router is router
    finally:
        if router is not None:
            router.discard()
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_encrypted_content_affinity_model_group_config_is_additive():
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    target_deployment = {
        "model_name": model_group,
        "litellm_params": {"model": "openai/gpt-5.1-codex"},
        "model_info": {"id": "deployment-b"},
    }
    healthy_deployments = [
        {
            "model_name": model_group,
            "litellm_params": {"model": "openai/gpt-5.1-codex"},
            "model_info": {"id": "deployment-a"},
        },
        target_deployment,
    ]
    encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(
        "deployment-b", "rs_test"
    )

    assert EncryptedContentAffinityCheck.has_model_group_affinity_enabled(
        {model_group: ["encrypted_content_affinity"]}
    )
    assert not EncryptedContentAffinityCheck.has_model_group_affinity_enabled(None)

    per_group_check = EncryptedContentAffinityCheck(
        enable_global_affinity=False,
        model_group_affinity_config={
            model_group: ["encrypted_content_affinity"],
        },
    )
    request_kwargs = {
        "input": [{"type": "reasoning", "id": encoded_id}],
        "litellm_metadata": {},
    }
    filtered = await per_group_check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=request_kwargs,
    )

    assert filtered == [target_deployment]
    assert request_kwargs["litellm_metadata"]["encrypted_content_affinity_enabled"]

    disabled_check = EncryptedContentAffinityCheck(
        enable_global_affinity=False,
        model_group_affinity_config={
            "other-model-group": ["encrypted_content_affinity"],
        },
    )
    disabled_request_kwargs = {
        "input": [{"type": "reasoning", "id": encoded_id}],
        "litellm_metadata": {},
    }
    unfiltered = await disabled_check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=disabled_request_kwargs,
    )

    assert unfiltered == healthy_deployments
    assert (
        "encrypted_content_affinity_enabled"
        not in disabled_request_kwargs["litellm_metadata"]
    )

    global_check = EncryptedContentAffinityCheck(
        enable_global_affinity=True,
        model_group_affinity_config={
            model_group: ["deployment_affinity"],
        },
    )
    global_request_kwargs = {
        "input": [{"type": "reasoning", "id": encoded_id}],
        "litellm_metadata": {},
    }
    globally_filtered = await global_check.async_filter_deployments(
        model=model_group,
        healthy_deployments=healthy_deployments,
        messages=None,
        request_kwargs=global_request_kwargs,
    )

    assert globally_filtered == [target_deployment]
    assert global_request_kwargs["litellm_metadata"][
        "encrypted_content_affinity_enabled"
    ]


@pytest.mark.asyncio
async def test_encrypted_content_affinity_takes_priority_over_user_key_affinity():
    from litellm.responses.utils import ResponsesAPIRequestUtils
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )
    from litellm.router_utils.pre_call_checks.encrypted_content_affinity_check import (
        EncryptedContentAffinityCheck,
    )

    model_group = "openai.gpt-5.1-codex"
    user_api_key_hash = "test-user-key"
    deployment_a = {
        "model_name": model_group,
        "litellm_params": {
            "model": "openai/gpt-5.1-codex",
            "api_key": "mock-api-key-a",
        },
        "model_info": {"id": "deployment-a"},
    }
    deployment_b = {
        "model_name": model_group,
        "litellm_params": {
            "model": "openai/gpt-5.1-codex",
            "api_key": "mock-api-key-b",
        },
        "model_info": {"id": "deployment-b"},
    }
    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = []
    router = None

    try:
        router = litellm.Router(
            model_list=[deployment_a, deployment_b],
            model_group_affinity_config={
                model_group: [
                    "deployment_affinity",
                    "encrypted_content_affinity",
                ],
            },
            num_retries=0,
        )
        callbacks = router.optional_callbacks or []
        deployment_callback = next(
            cb for cb in callbacks if isinstance(cb, DeploymentAffinityCheck)
        )
        encrypted_content_callback = next(
            cb for cb in callbacks if isinstance(cb, EncryptedContentAffinityCheck)
        )
        assert callbacks.index(encrypted_content_callback) < callbacks.index(
            deployment_callback
        )
        assert litellm.callbacks.index(encrypted_content_callback) < (
            litellm.callbacks.index(deployment_callback)
        )

        cache_key = DeploymentAffinityCheck.get_affinity_cache_key(
            model_group=model_group,
            user_key=user_api_key_hash,
        )
        await deployment_callback.cache.async_set_cache(
            key=cache_key,
            value={"model_id": "deployment-a"},
            ttl=60,
        )
        encoded_id = ResponsesAPIRequestUtils._build_encrypted_item_id(
            "deployment-b", "rs_test"
        )
        request_kwargs = {
            "input": [{"type": "reasoning", "id": encoded_id}],
            "litellm_metadata": {"user_api_key_hash": user_api_key_hash},
        }

        filtered = await router.async_callback_filter_deployments(
            model=model_group,
            healthy_deployments=[deployment_a, deployment_b],
            messages=None,
            parent_otel_span=None,
            request_kwargs=request_kwargs,
        )

        assert filtered == [deployment_b]
        assert request_kwargs.get("_encrypted_content_affinity_pinned") is True
    finally:
        if router is not None:
            router.discard()
        litellm.callbacks = original_callbacks


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
    from unittest.mock import MagicMock, patch

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
async def test_async_router_acreate_file_uses_deployment_custom_llm_provider():
    """
    Ensure file routing preserves deployment custom_llm_provider instead of
    inferring provider from model string alone.
    """
    from unittest.mock import MagicMock, patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "team-azure-batch",
                "litellm_params": {
                    "model": "gpt-4.1-mini",
                    "custom_llm_provider": "azure",
                    "api_base": "https://example-resource.openai.azure.com",
                },
            },
        ],
    )

    with patch("litellm.acreate_file", return_value=MagicMock()) as mock_acreate_file:
        await router.acreate_file(
            model="team-azure-batch",
            purpose="batch",
            file=MagicMock(),
        )

        assert mock_acreate_file.call_count == 1
        assert mock_acreate_file.call_args.kwargs["custom_llm_provider"] == "azure"


@pytest.mark.asyncio
async def test_async_router_afile_content_uses_deployment_custom_llm_provider():
    """
    Regression test: Ensure afile_content preserves deployment custom_llm_provider
    when model name lacks provider prefix (e.g., "gpt-4.1-mini" instead of "azure/gpt-4.1-mini").

    This prevents "None is not a valid LlmProviders" errors when calling file content operations.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    router = litellm.Router(
        model_list=[
            {
                "model_name": "team-azure-batch",
                "litellm_params": {
                    "model": "gpt-4.1-mini",  # No provider prefix
                    "custom_llm_provider": "azure",
                    "api_base": "https://example-resource.openai.azure.com",
                    "api_key": "test-key",
                },
            },
        ],
    )

    # Mock the Azure file handler's afile_content method
    mock_response = MagicMock(spec=HttpxBinaryResponseContent)
    mock_response.response = MagicMock()

    with patch(
        "litellm.llms.azure.files.handler.AzureOpenAIFilesAPI.afile_content",
        return_value=mock_response,
    ) as mock_afile_content:
        result = await router.afile_content(
            model="team-azure-batch",
            file_id="file-123",
        )

        # Verify the call was made (proves custom_llm_provider was correctly passed)
        assert mock_afile_content.call_count == 1
        assert result == mock_response


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

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "output": [],
    }
    mock_response.text = (
        '{"id": "resp_test", "object": "response", "status": "completed", "output": []}'
    )

    with patch.object(client, "post", return_value=mock_response) as mock_post:
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
                    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
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
    from unittest.mock import patch

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


@pytest.mark.asyncio
async def test_ageneric_api_call_deployment_model_overrides_alias():
    """
    Regression: when a model alias (e.g. "not-gemini-2.5-flash") maps to a deployment
    with model="vertex_ai/gemini-2.5-flash", the underlying litellm function must receive
    the deployment model, not the alias. Before the fix, **kwargs overwrote data["model"].
    """
    from unittest.mock import patch

    captured: dict = {}

    async def capture_model(**kwargs):
        captured["model"] = kwargs.get("model")
        return {"result": "ok"}

    router = litellm.Router(
        model_list=[
            {
                "model_name": "not-gemini-2.5-flash",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-flash",
                    "api_key": "fake-key",
                },
            }
        ]
    )

    def inject_alias_into_kwargs(deployment, kwargs, function_name=None):
        # Simulate the alias leaking into kwargs (as happens when
        # _ageneric_api_call_with_fallbacks sets kwargs["model"] = alias before
        # calling the helper through async_function_with_fallbacks).
        kwargs["model"] = "not-gemini-2.5-flash"

    with (
        patch.object(router, "async_get_available_deployment") as mock_dep,
        patch.object(
            router,
            "_update_kwargs_with_deployment",
            side_effect=inject_alias_into_kwargs,
        ),
        patch.object(router, "async_routing_strategy_pre_call_checks"),
        patch.object(router, "_get_client", return_value=None),
    ):
        mock_dep.return_value = {
            "model_name": "not-gemini-2.5-flash",
            "litellm_params": {
                "model": "vertex_ai/gemini-2.5-flash",
                "api_key": "fake-key",
            },
        }

        await router._ageneric_api_call_with_fallbacks_helper(
            model="not-gemini-2.5-flash",
            original_generic_function=capture_model,
        )

    assert (
        captured["model"] == "vertex_ai/gemini-2.5-flash"
    ), f"Expected deployment model 'vertex_ai/gemini-2.5-flash', got '{captured['model']}'"


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


def test_cached_get_model_group_info():
    """
    Test that _cached_get_model_group_info caches results and
    invalidates on deployment changes.
    """
    from litellm.types.router import Deployment, LiteLLM_Params

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {"tpm": 1000, "rpm": 100},
            },
        ]
    )

    # First call should compute and cache
    result1 = router._cached_get_model_group_info("gpt-4")
    assert result1 is not None
    assert result1.tpm == 1000

    # Second call should hit cache (same object)
    result2 = router._cached_get_model_group_info("gpt-4")
    assert result1 is result2

    # Add a deployment — cache should be invalidated
    router.add_deployment(
        Deployment(
            model_name="gpt-4",
            litellm_params=LiteLLM_Params(model="gpt-4", api_key="fake2"),
            model_info={"tpm": 2000, "rpm": 200},
        )
    )
    result3 = router._cached_get_model_group_info("gpt-4")
    assert result3 is not result2
    assert result3 is not None
    assert result3.tpm == 3000  # 1000 + 2000

    # Delete a deployment — cache should be invalidated
    deployment_id = router.model_list[-1]["model_info"]["id"]
    router.delete_deployment(id=deployment_id)
    result4 = router._cached_get_model_group_info("gpt-4")
    assert result4 is not result3
    assert result4 is not None
    assert result4.tpm == 1000

    # set_model_list — cache should be invalidated
    router.set_model_list(
        [
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {"tpm": 5000},
            },
        ]
    )
    result5 = router._cached_get_model_group_info("gpt-4")
    assert result5 is not result4
    assert result5 is not None
    assert result5.tpm == 5000

    # Verify cache still works after invalidation
    result6 = router._cached_get_model_group_info("gpt-4")
    assert result5 is result6


def test_model_group_info_cost_from_db_model_info():
    """
    When get_deployment_model_info fails (model_info is None fallback),
    input_cost_per_token and output_cost_per_token should be read from db model_info.
    """
    from unittest.mock import patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model",
                "litellm_params": {
                    "model": "openai/my-custom-model",
                    "api_key": "fake",
                    "api_base": "https://my-custom-endpoint.com",
                },
                "model_info": {
                    "input_cost_per_token": 0.0001,
                    "output_cost_per_token": 0.0002,
                },
            },
        ]
    )

    with patch.object(
        router, "get_deployment_model_info", side_effect=Exception("not found")
    ):
        result = router._cached_get_model_group_info("my-custom-model")
        assert result is not None
        assert result.input_cost_per_token == 0.0001
        assert result.output_cost_per_token == 0.0002


def test_model_group_info_cost_none_when_db_model_info_has_no_cost():
    """
    When get_deployment_model_info fails and db model_info has no cost fields,
    input/output_cost_per_token should be None.
    """
    from unittest.mock import patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model-no-cost",
                "litellm_params": {
                    "model": "openai/my-custom-model-no-cost",
                    "api_key": "fake",
                    "api_base": "https://my-custom-endpoint.com",
                },
                "model_info": {},
            },
        ]
    )

    with patch.object(
        router, "get_deployment_model_info", side_effect=Exception("not found")
    ):
        result = router._cached_get_model_group_info("my-custom-model-no-cost")
        assert result is not None
        assert result.input_cost_per_token is None
        assert result.output_cost_per_token is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1e-05", 1e-05),
        ("0.00001", 1e-05),
        (1e-05, 1e-05),
        (5, 5.0),
        (None, None),
        ("not-a-number", None),
    ],
)
def test_cost_value_as_float(value, expected):
    from litellm.router import _cost_value_as_float

    assert _cost_value_as_float(value) == expected


def test_model_group_info_with_stringified_cost_values():
    """
    YAML 1.2 parsers emit '1e-05' (integer mantissa) as a string, so cost
    values in deployment model_info can arrive as str. Aggregating the model
    group must not raise TypeError('>' between str and float) and must return
    float costs.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model",
                "litellm_params": {
                    "model": "openai/my-custom-backend-1",
                    "api_key": "fake",
                },
                "model_info": {
                    "input_cost_per_token": "1e-05",
                    "output_cost_per_token": "1e-05",
                },
            },
            {
                "model_name": "my-custom-model",
                "litellm_params": {
                    "model": "openai/my-custom-backend-2",
                    "api_key": "fake",
                },
                "model_info": {
                    "input_cost_per_token": "2e-05",
                    "output_cost_per_token": "2e-05",
                },
            },
        ]
    )

    def _model_info_with_str_costs(model_id: str, model_name: str):
        for model in router.model_list:
            if model["model_info"]["id"] == model_id:
                return {
                    "key": model_name,
                    "input_cost_per_token": model["model_info"]["input_cost_per_token"],
                    "output_cost_per_token": model["model_info"]["output_cost_per_token"],
                    "litellm_provider": "openai",
                    "mode": "chat",
                }
        return None

    with patch.object(
        router, "get_deployment_model_info", side_effect=_model_info_with_str_costs
    ):
        result = router._set_model_group_info(
            model_group="my-custom-model",
            user_facing_model_group_name="my-custom-model",
        )

    assert result is not None
    assert result.input_cost_per_token == 2e-05
    assert result.output_cost_per_token == 2e-05
    assert isinstance(result.input_cost_per_token, float)
    assert isinstance(result.output_cost_per_token, float)


def test_model_group_info_db_fallback_with_stringified_cost_values():
    """
    Fallback path: when get_deployment_model_info returns nothing, costs are
    read straight from the deployment's model_info dict, which can hold
    stringified floats parsed from YAML. They must be coerced to float.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model",
                "litellm_params": {
                    "model": "openai/my-custom-backend-1",
                    "api_key": "fake",
                },
                "model_info": {
                    "input_cost_per_token": "1e-05",
                    "output_cost_per_token": "3e-05",
                },
            },
            {
                "model_name": "my-custom-model",
                "litellm_params": {
                    "model": "openai/my-custom-backend-2",
                    "api_key": "fake",
                },
                "model_info": {
                    "input_cost_per_token": "2e-05",
                    "output_cost_per_token": "2e-05",
                },
            },
        ]
    )

    with patch.object(
        router, "get_deployment_model_info", side_effect=Exception("not found")
    ):
        result = router._set_model_group_info(
            model_group="my-custom-model",
            user_facing_model_group_name="my-custom-model",
        )

    assert result is not None
    assert result.input_cost_per_token == 2e-05
    assert result.output_cost_per_token == 3e-05
    assert isinstance(result.input_cost_per_token, float)
    assert isinstance(result.output_cost_per_token, float)


def test_get_model_access_groups_caching():
    """
    Test that get_model_access_groups caches the no-args result
    and invalidates on deployment changes.
    """
    from litellm.types.router import Deployment, LiteLLM_Params

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"access_groups": ["premium"]},
            },
        ]
    )

    # First call computes and populates cache
    result1 = router.get_model_access_groups()
    assert "premium" in result1

    # All subsequent calls should return the same cached object (including first)
    result2 = router.get_model_access_groups()
    assert result1 is result2

    # Calls with args should bypass cache
    result_with_args = router.get_model_access_groups(model_name="gpt-4")
    assert result_with_args is not result2

    # Add a deployment — cache should be invalidated
    router.add_deployment(
        Deployment(
            model_name="gpt-3.5",
            litellm_params=LiteLLM_Params(model="gpt-3.5-turbo"),
            model_info={"access_groups": ["default"]},
        )
    )
    result3 = router.get_model_access_groups()
    assert result3 is not result2
    assert "premium" in result3
    assert "default" in result3

    # Delete the deployment — cache should be invalidated again
    deployment_id = None
    for m in router.model_list:
        if m.get("model_name") == "gpt-3.5":
            deployment_id = m.get("model_info", {}).get("id")
            break
    assert deployment_id is not None
    router.delete_deployment(id=deployment_id)
    result4 = router.get_model_access_groups()
    assert result4 is not result3
    assert "default" not in result4
    assert "premium" in result4


def test_get_model_access_groups_cache_invalidation_set_model_list():
    """
    Test that set_model_list invalidates the access groups cache.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"access_groups": ["premium"]},
            },
        ]
    )

    # Populate cache
    result1 = router.get_model_access_groups()
    assert "premium" in result1

    # set_model_list should invalidate cache
    router.set_model_list(
        [
            {
                "model_name": "claude-3",
                "litellm_params": {"model": "anthropic/claude-3-opus-20240229"},
                "model_info": {"access_groups": ["research"]},
            },
        ]
    )
    result2 = router.get_model_access_groups()
    assert result2 is not result1
    assert "research" in result2
    assert "premium" not in result2


def test_get_model_access_groups_cache_invalidation_upsert_deployment():
    """
    Test that upsert_deployment invalidates the access groups cache.
    """
    from litellm.types.router import Deployment, LiteLLM_Params

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4"},
                "model_info": {"access_groups": ["premium"]},
            },
        ]
    )

    # Populate cache
    result1 = router.get_model_access_groups()
    assert "premium" in result1

    # Get the existing deployment's ID
    existing_id = router.model_list[0]["model_info"]["id"]

    # Upsert with the same ID but different params — triggers pop + re-add
    router.upsert_deployment(
        Deployment(
            model_name="gpt-4-updated",
            litellm_params=LiteLLM_Params(model="gpt-4-turbo"),
            model_info={"id": existing_id, "access_groups": ["updated-group"]},
        )
    )
    result2 = router.get_model_access_groups()
    assert result2 is not result1
    assert "updated-group" in result2


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator():
    """Test _acompletion_streaming_iterator for normal streaming and fallback behavior."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError

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

        # Empty content → pre-first-chunk path uses original messages
        # (no continuation prompt added)
        assert modified_messages == messages
        print("✓ Handles empty generated content correctly")

    print("✓ Edge case tests passed!")


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_preserves_hidden_params():
    """
    Regression test: FallbackStreamWrapper must copy _hidden_params from the
    original CustomStreamWrapper so that x-litellm-overhead-duration-ms (and
    other hidden params) are present in the proxy response headers for streaming.
    """
    from unittest.mock import MagicMock

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
    )

    # Simulate a CustomStreamWrapper that already has timing metadata set by
    # update_response_metadata (litellm_overhead_time_ms, _response_ms, etc.)
    mock_response = MagicMock()
    mock_response.model = "gpt-4"
    mock_response.custom_llm_provider = "openai"
    mock_response.logging_obj = MagicMock()
    mock_response._hidden_params = {
        "litellm_overhead_time_ms": 12.34,
        "_response_ms": 500.0,
        "litellm_call_id": "test-call-id",
        "api_base": "https://api.openai.com",
        "additional_headers": {},
    }

    # Make the mock iterable (yields nothing — we only care about hidden_params)
    async def _empty():
        return
        yield  # make it an async generator

    mock_response.__aiter__ = lambda self: _empty().__aiter__()

    result = await router._acompletion_streaming_iterator(
        model_response=mock_response,
        messages=[{"role": "user", "content": "hi"}],
        initial_kwargs={"model": "gpt-4", "stream": True},
    )

    # The returned FallbackStreamWrapper must carry the original _hidden_params
    assert hasattr(result, "_hidden_params"), "result must have _hidden_params"
    assert result._hidden_params.get("litellm_overhead_time_ms") == 12.34, (
        "litellm_overhead_time_ms must be preserved — "
        "this is what drives x-litellm-overhead-duration-ms in streaming responses"
    )
    assert result._hidden_params.get("litellm_call_id") == "test-call-id"
    assert result._hidden_params.get("_response_ms") == 500.0


def test_completion_streaming_iterator_fallback_on_429():
    """Sync streaming: MidStreamFallbackError (429 pre-first-chunk) triggers fallback.

    This is the sync counterpart of test_acompletion_streaming_iterator.
    Before this fix, __next__ raised RateLimitError directly and the Router
    never got a chance to fall back.
    """
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
    )

    messages = [{"role": "user", "content": "Test"}]
    initial_kwargs = {"model": "gpt-4", "stream": True}

    rate_limit_error = MidStreamFallbackError(
        message="Resource exhausted",
        model="gpt-4",
        llm_provider="vertex_ai",
        generated_content="",
        is_pre_first_chunk=True,
    )

    class SyncIteratorImmediateError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __iter__(self):
            return self

        def __next__(self):
            raise rate_limit_error

    mock_response = SyncIteratorImmediateError()

    # Fallback returns a simple non-streaming response (fallback may not stream)
    mock_fallback_response = MagicMock()
    mock_fallback_response.__iter__ = MagicMock(return_value=iter([]))

    with patch.object(
        router,
        "function_with_fallbacks",
        return_value=mock_fallback_response,
    ) as mock_fallback:
        result = router._completion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        collected_chunks = list(result)

        assert mock_fallback.called
        call_kwargs = mock_fallback.call_args
        # Pre-first-chunk: should use original messages, no continuation prompt
        assert call_kwargs.kwargs.get("messages") == messages
        # Verify original_function is _completion (sync)
        assert call_kwargs.kwargs.get("original_function") == router._completion


def test_completion_streaming_iterator_preserves_hidden_params():
    """SyncFallbackStreamWrapper must copy _hidden_params from original response."""
    from unittest.mock import MagicMock

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
    )

    mock_response = MagicMock()
    mock_response.model = "gpt-4"
    mock_response.custom_llm_provider = "openai"
    mock_response.logging_obj = MagicMock()
    mock_response._hidden_params = {
        "litellm_overhead_time_ms": 42.0,
        "litellm_call_id": "test-sync-call",
    }
    mock_response.__iter__ = MagicMock(return_value=iter([]))

    result = router._completion_streaming_iterator(
        model_response=mock_response,
        messages=[{"role": "user", "content": "hi"}],
        initial_kwargs={"model": "gpt-4", "stream": True},
    )

    assert hasattr(result, "_hidden_params")
    assert result._hidden_params.get("litellm_overhead_time_ms") == 42.0
    assert result._hidden_params.get("litellm_call_id") == "test-sync-call"


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_pre_first_chunk_skips_continuation():
    """When MidStreamFallbackError has is_pre_first_chunk=True, use original messages."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
    )

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "gpt-4", "stream": True}

    pre_first_chunk_error = MidStreamFallbackError(
        message="429 Resource exhausted",
        model="gpt-4",
        llm_provider="vertex_ai",
        generated_content="",
        is_pre_first_chunk=True,
    )

    class AsyncIteratorPreFirstChunkError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise pre_first_chunk_error

    mock_response = AsyncIteratorPreFirstChunkError()

    class EmptyAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=EmptyAsyncIterator(),
    ) as mock_fallback_utils:
        iterator = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )
        async for _ in iterator:
            pass

        assert mock_fallback_utils.called
        fallback_kwargs = mock_fallback_utils.call_args.kwargs["kwargs"]
        # Pre-first-chunk: should use original messages, no continuation prompt
        assert fallback_kwargs["messages"] == messages


# ---------------------------------------------------------------------------
# Shared helpers for the _aresponses_streaming_iterator test suite.
# ---------------------------------------------------------------------------
def _make_responses_iterator(
    *,
    chunks=(),
    error=None,
    bridge=False,
    model="gpt-4",
    hidden_params=None,
    chat_chunks=None,
):
    """Build a minimal mock Responses-API streaming iterator.

    Bypasses BaseResponsesAPIStreamingIterator.__init__ but mirrors every
    attribute production code reads. Yields *chunks*, then raises *error*
    (or StopAsyncIteration). Set bridge=True to inherit from
    LiteLLMCompletionStreamingIterator so the wrapper's bridge-path
    isinstance check (used by usage extraction) matches.
    """
    from litellm.responses.litellm_completion_transformation.streaming_iterator import (
        LiteLLMCompletionStreamingIterator,
    )
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    base = (
        LiteLLMCompletionStreamingIterator
        if bridge
        else BaseResponsesAPIStreamingIterator
    )

    class _Iter(base):
        def __init__(self):
            self._chunks = list(chunks)
            self._idx = 0
            self._hidden_params = hidden_params or {}
            self.model = model
            self.custom_llm_provider = "anthropic"
            self.logging_obj = MagicMock()
            self.litellm_metadata = None
            self.responses_api_provider_config = None
            self.finished = False
            self.completed_response = None
            self.response = None
            self.start_time = None
            self.request_data = {}
            self.call_type = None
            if chat_chunks is not None:
                self.collected_chat_completion_chunks = chat_chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx < len(self._chunks):
                self._idx += 1
                return self._chunks[self._idx - 1]
            if error is not None:
                raise error
            raise StopAsyncIteration

    return _Iter()


class _AsyncList:
    """Generic async iterator over a list — used as the fallback response."""

    def __init__(self, items=()):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _make_router_with_fallback(primary="gpt-4", secondary="gpt-3.5-turbo"):
    return litellm.Router(
        model_list=[
            {
                "model_name": primary,
                "litellm_params": {"model": primary, "api_key": "k1"},
            },
            {
                "model_name": secondary,
                "litellm_params": {"model": secondary, "api_key": "k2"},
            },
        ],
        fallbacks=[{primary: [secondary]}],
    )


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_fallback():
    """Catches MidStreamFallbackError, re-enters the fallback chain via
    async_function_with_fallbacks_common_utils with the per-attempt helper
    and original_generic_function preserved. Mirrors
    test_acompletion_streaming_iterator for the aresponses path."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    router = _make_router_with_fallback(
        "anthropic/claude-sonnet-4-6", "vertex_ai/claude-sonnet-4-6"
    )
    src = _make_responses_iterator(
        chunks=[MagicMock(type="response.created")],
        error=MidStreamFallbackError(
            message="anthropic socket timeout",
            model="anthropic/claude-sonnet-4-6",
            llm_provider="anthropic",
            is_pre_first_chunk=False,
            generated_content="",
        ),
        model="anthropic/claude-sonnet-4-6",
        hidden_params={"model_id": "src-deployment-1"},
    )
    fallback_chunks = [
        MagicMock(type="response.output_text.delta"),
        MagicMock(type="response.completed"),
    ]

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(fallback_chunks),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "anthropic/claude-sonnet-4-6",
                "stream": True,
                "input": "Hi",
                "original_generic_function": litellm.aresponses,
            },
        )
        assert isinstance(wrapped, BaseResponsesAPIStreamingIterator)
        assert wrapped._hidden_params.get("model_id") == "src-deployment-1"
        collected = [c async for c in wrapped]

    assert len(collected) == 3  # 1 primary chunk + 2 fallback chunks
    call_kwargs = mock_fallback_utils.call_args.kwargs
    fbk = call_kwargs["kwargs"]
    # Bound methods compare equal when they share the same instance + __func__.
    assert fbk["original_function"] == router._ageneric_api_call_with_fallbacks_helper
    assert fbk["original_generic_function"] is litellm.aresponses
    assert call_kwargs["model_group"] == "anthropic/claude-sonnet-4-6"
    assert call_kwargs["disable_fallbacks"] is False


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_writes_litellm_metadata_on_fallback():
    """Regression: model_group must land under "litellm_metadata" (the key
    litellm.aresponses reads), not the default "metadata"."""
    from litellm.exceptions import MidStreamFallbackError

    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        error=MidStreamFallbackError(
            message="boom",
            model="gpt-4",
            llm_provider="anthropic",
            is_pre_first_chunk=True,
            generated_content="",
        )
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Hello",
                "original_generic_function": litellm.aresponses,
            },
        )
        async for _ in wrapped:
            pass

    fbk = mock_fallback_utils.call_args.kwargs["kwargs"]
    assert "litellm_metadata" in fbk, "wrong metadata_variable_name"
    assert fbk["litellm_metadata"]["model_group"] == "gpt-4"
    assert "model_group" not in fbk.get(
        "metadata", {}
    ), "model_group leaked into 'metadata' instead of 'litellm_metadata'"


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_pre_first_chunk_skips_continuation():
    """Pre-first-chunk error: original input is preserved unchanged."""
    from litellm.exceptions import MidStreamFallbackError

    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        error=MidStreamFallbackError(
            message="socket timeout before first chunk",
            model="gpt-4",
            llm_provider="anthropic",
            is_pre_first_chunk=True,
            generated_content="",
        )
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Hello",
                "original_generic_function": litellm.aresponses,
            },
        )
        async for _ in wrapped:
            pass

    fbk = mock_fallback_utils.call_args.kwargs["kwargs"]
    assert fbk["input"] == "Hello"  # original input, no continuation messages


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_partial_content_injects_continuation():
    """Mid-stream error: input is rewritten to include user prompt +
    developer instruction + prior assistant message with partial output."""
    from litellm.exceptions import MidStreamFallbackError

    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[MagicMock(type="response.output_text.delta")],
        error=MidStreamFallbackError(
            message="socket reset mid-stream",
            model="gpt-4",
            llm_provider="anthropic",
            is_pre_first_chunk=False,
            generated_content="The capital of France is",
        ),
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "What's the capital of France?",
                "original_generic_function": litellm.aresponses,
            },
        )
        async for _ in wrapped:
            pass

    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    assert isinstance(new_input, list)
    assert new_input[0]["role"] == "user"
    assert new_input[0]["content"][0]["text"] == "What's the capital of France?"
    assert new_input[1]["role"] == "developer"
    assert "do not repeat" in new_input[1]["content"][0]["text"].lower()
    assert new_input[2]["role"] == "assistant"
    assert new_input[2]["content"][0]["type"] == "output_text"
    assert new_input[2]["content"][0]["text"] == "The capital of France is"


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_combines_partial_usage():
    """Partial usage from the bridge path is normalized to ResponseAPIUsage
    and summed onto the fallback's response.completed event — no token-name
    split, clean ResponseAPIUsage on output."""
    from types import SimpleNamespace

    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.llms.openai import (
        ResponseAPIUsage,
        ResponseCompletedEvent,
        ResponsesAPIResponse,
        ResponsesAPIStreamEvents,
    )

    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        bridge=True,
        chat_chunks=[MagicMock()],
        chunks=[MagicMock(type="response.output_text.delta")],
        error=MidStreamFallbackError(
            message="boom",
            model="gpt-4",
            llm_provider="anthropic",
            is_pre_first_chunk=False,
            generated_content="hello",
        ),
    )

    fallback_response_object = ResponsesAPIResponse(
        id="resp_test", created_at=0, model="gpt-4", object="response", output=[]
    )
    fallback_response_object.usage = ResponseAPIUsage(
        input_tokens=20, output_tokens=15, total_tokens=35
    )
    fallback_event = ResponseCompletedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=fallback_response_object,
    )

    with (
        patch(
            "litellm.main.stream_chunk_builder",
            return_value=SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4)
            ),
        ),
        patch.object(
            router,
            "async_function_with_fallbacks_common_utils",
            return_value=_AsyncList([fallback_event]),
        ),
    ):
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "hi",
                "original_generic_function": litellm.aresponses,
            },
        )
        async for _ in wrapped:
            pass

    merged = fallback_response_object.usage
    assert isinstance(merged, ResponseAPIUsage)
    assert merged.input_tokens == 30  # 10 (translated from prompt_tokens) + 20
    assert merged.output_tokens == 19  # 4 (translated from completion_tokens) + 15
    assert merged.total_tokens == 49


def _midstream_rate_limit_error():
    rate_limit_error = litellm.RateLimitError(
        message="vertex_ai_betaException - Resource exhausted.",
        model="gemini",
        llm_provider="vertex_ai_beta",
    )
    midstream_error = MidStreamFallbackError(
        message=str(rate_limit_error),
        model="gemini",
        llm_provider="vertex_ai_beta",
        original_exception=rate_limit_error,
        is_pre_first_chunk=True,
    )
    return rate_limit_error, midstream_error


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_surfaces_rate_limit_without_fallbacks():
    """Regression for #26015: a mid-stream 429 with no fallbacks configured must
    surface a clean RateLimitError, not leak the internal MidStreamFallbackError
    wrapper to the client, and must terminate instead of hanging."""
    rate_limit_error, midstream_error = _midstream_rate_limit_error()

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=0,
    )

    class _RaisingStream:
        def __init__(self):
            self.chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise midstream_error

    stream = _RaisingStream()
    setattr(stream, "model", "gemini")
    setattr(stream, "custom_llm_provider", "vertex_ai_beta")
    setattr(stream, "logging_obj", MagicMock())

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=midstream_error),
    ):
        result = await router._acompletion_streaming_iterator(
            model_response=stream,
            messages=[{"role": "user", "content": "Hello"}],
            initial_kwargs={"model": "gemini", "stream": True},
        )

        async def _consume():
            async for _ in result:
                pass

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await asyncio.wait_for(_consume(), timeout=10)

    assert not isinstance(exc_info.value, MidStreamFallbackError)
    assert exc_info.value.status_code == 429
    assert exc_info.value is rate_limit_error


def test_completion_streaming_iterator_surfaces_rate_limit_without_fallbacks():
    """Sync counterpart of
    test_acompletion_streaming_iterator_surfaces_rate_limit_without_fallbacks."""
    rate_limit_error, midstream_error = _midstream_rate_limit_error()

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=0,
    )

    class _RaisingSyncStream:
        def __init__(self):
            self.model = "gemini"
            self.custom_llm_provider = "vertex_ai_beta"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __iter__(self):
            return self

        def __next__(self):
            raise midstream_error

    with patch.object(
        router,
        "function_with_fallbacks",
        side_effect=midstream_error,
    ):
        result = router._completion_streaming_iterator(
            model_response=_RaisingSyncStream(),
            messages=[{"role": "user", "content": "Hello"}],
            initial_kwargs={"model": "gemini", "stream": True},
        )

        with pytest.raises(litellm.RateLimitError) as exc_info:
            list(result)

    assert not isinstance(exc_info.value, MidStreamFallbackError)
    assert exc_info.value.status_code == 429
    assert exc_info.value is rate_limit_error


@pytest.mark.asyncio
async def test_aresponses_streaming_iterator_surfaces_rate_limit_without_fallbacks():
    """Responses-API counterpart of
    test_acompletion_streaming_iterator_surfaces_rate_limit_without_fallbacks."""
    rate_limit_error, midstream_error = _midstream_rate_limit_error()

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=0,
    )
    src = _make_responses_iterator(error=midstream_error, model="gemini")

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=midstream_error),
    ):
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gemini",
                "stream": True,
                "input": "Hello",
                "original_generic_function": litellm.aresponses,
            },
        )

        async def _consume():
            async for _ in wrapped:
                pass

        with pytest.raises(litellm.RateLimitError) as exc_info:
            await asyncio.wait_for(_consume(), timeout=10)

    assert not isinstance(exc_info.value, MidStreamFallbackError)
    assert exc_info.value.status_code == 429
    assert exc_info.value is rate_limit_error


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


def test_pre_call_checks_skips_token_count_without_max_input_tokens(monkeypatch):
    """
    tiktoken token counting is the dominant on-loop cost for large prompts. When no
    deployment in the group declares max_input_tokens, the count is never consumed, so
    _pre_call_checks must not run it at all.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(router, "get_router_model_info", lambda **kwargs: {})

    calls = []
    monkeypatch.setattr(
        litellm, "token_counter", lambda *a, **k: calls.append(1) or 1000
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d2"}},
    ]
    result = router._pre_call_checks(
        model="m",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert calls == []
    assert len(result) == 2


def test_pre_call_checks_counts_once_and_filters_on_max_input_tokens(monkeypatch):
    """
    When a deployment declares max_input_tokens the count must still run, be performed
    at most once across the group (memoized), and filter deployments whose limit is
    exceeded.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": 5}
    )

    calls = []
    monkeypatch.setattr(
        litellm, "token_counter", lambda *a, **k: calls.append(1) or 1000
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d2"}},
    ]
    with pytest.raises(litellm.ContextWindowExceededError):
        router._pre_call_checks(
            model="m",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert calls == [1]


def test_pre_call_checks_counts_tokens_from_responses_input_string(monkeypatch):
    """
    Responses API calls pass `input` (str) instead of `messages`. Context-window
    checks must count tokens from `input` and filter deployments over the limit. Uses
    the real token_counter so the transform + counting path is a true regression guard.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": 1}
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    with pytest.raises(litellm.ContextWindowExceededError):
        router._pre_call_checks(
            model="m",
            healthy_deployments=deployments,
            input="a very long prompt that exceeds the tiny context window",
        )


def test_pre_call_checks_counts_tokens_from_responses_input_list(monkeypatch):
    """
    Responses API `input` can be a list of input items. It must be normalized to
    chat messages and counted so oversized requests are filtered out. Uses the real
    token_counter (no mock) so the transform + counting path is a true regression guard.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": 1}
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    with pytest.raises(litellm.ContextWindowExceededError):
        router._pre_call_checks(
            model="m",
            healthy_deployments=deployments,
            input=[
                {"role": "user", "content": "count these tokens against the one token limit please"},
            ],
        )


def test_pre_call_checks_counts_responses_instructions_tokens(monkeypatch):
    """
    Responses API `instructions` become a system message the model receives, so their
    tokens must be counted too. A request whose `input` alone fits under the limit but
    whose `input` + `instructions` exceeds it must be filtered (regression for the
    context-window check under-filtering when instructions were ignored).
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]

    short_input = "hi"
    long_instructions = "you are a helpful assistant. " * 20

    input_only_tokens = router._count_pre_call_check_tokens(messages=None, input=short_input)
    with_instructions_tokens = router._count_pre_call_check_tokens(
        messages=None, input=short_input, instructions=long_instructions
    )
    assert with_instructions_tokens > input_only_tokens

    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": input_only_tokens}
    )
    with pytest.raises(litellm.ContextWindowExceededError):
        router._pre_call_checks(
            model="m",
            healthy_deployments=deployments,
            input=short_input,
            request_kwargs={"instructions": long_instructions},
        )


def test_count_pre_call_check_tokens_across_api_surfaces():
    """
    _count_pre_call_check_tokens must count tokens from chat `messages`, a Responses
    API string `input`, and a Responses API list `input`, and raise when given neither.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
    )

    messages_tokens = router._count_pre_call_check_tokens(
        messages=[{"role": "user", "content": "hello world"}], input=None
    )
    string_input_tokens = router._count_pre_call_check_tokens(messages=None, input="hello world")
    list_input_tokens = router._count_pre_call_check_tokens(
        messages=None, input=[{"role": "user", "content": "hello world"}]
    )

    assert messages_tokens > 0
    assert string_input_tokens > 0
    assert list_input_tokens > 0

    with pytest.raises(ValueError):
        router._count_pre_call_check_tokens(messages=None, input=None)


def test_pre_call_checks_no_messages_or_input_does_not_crash(monkeypatch):
    """
    When neither messages nor input is provided (e.g. endpoints without prompt text),
    token counting is skipped gracefully and all deployments are returned.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": 5}
    )

    counted: list[dict] = []
    original = router._count_pre_call_check_tokens
    monkeypatch.setattr(
        router,
        "_count_pre_call_check_tokens",
        lambda **kwargs: counted.append(kwargs) or original(**kwargs),
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    result = router._pre_call_checks(model="m", healthy_deployments=deployments)
    assert len(result) == 1
    assert counted == []  # token counting skipped entirely, so no misleading error is logged


@pytest.mark.asyncio
async def test_aresponses_enforces_context_window_pre_call_check():
    """
    End-to-end router regression: a Responses API call whose `input` exceeds the
    deployment's max_input_tokens must be filtered by the pre-call check, raising
    ContextWindowExceededError instead of being silently routed. This guards the
    wiring that forwards `input` from the generic-call path into deployment selection
    (the deployment uses mock_response, so the check must trip before any real call).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "small-ctx",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
                "model_info": {"max_input_tokens": 5},
            }
        ],
        enable_pre_call_checks=True,
    )
    with pytest.raises(litellm.ContextWindowExceededError):
        await router.aresponses(
            model="small-ctx",
            input="this responses input is definitely much longer than five tokens for sure",
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

    # Test Case 6: custom_model_info present but litellm_model_name_model_info is None
    # (model has custom pricing in config but is not in built-in model_prices_and_context_window.json)
    mock_custom_pricing_only = {
        "input_cost_per_token": 1.74e-06,
        "output_cost_per_token": 3.48e-06,
        "cache_read_input_token_cost": 1.45e-08,
        "mode": "chat",
    }

    with patch.object(
        litellm,
        "model_cost",
        {"custom-model-id": mock_custom_pricing_only},
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:
            # Model NOT in built-in cost map — raise exception
            mock_get_model_info.side_effect = Exception("Model not in cost map")

            result = router.get_deployment_model_info(
                model_id="custom-model-id", model_name="unknown-model"
            )

            # Should return custom_model_info even when litellm_model_name_model_info is None
            assert result is not None
            assert result["input_cost_per_token"] == 1.74e-06
            assert result["output_cost_per_token"] == 3.48e-06
            assert result["cache_read_input_token_cost"] == 1.45e-08
            assert result["mode"] == "chat"

    # Test Case 7: custom_model_info with base_model but litellm_model_name_model_info None
    mock_custom_with_base = {
        "base_model": "some-base-model",
        "input_cost_per_token": 0.01,
        "output_cost_per_token": 0.02,
    }
    mock_base_info = {
        "key": "some-base-model",
        "max_tokens": 8192,
        "mode": "chat",
        "litellm_provider": "openai",
    }

    with patch.object(
        litellm,
        "model_cost",
        {"custom-with-base": mock_custom_with_base},
    ):
        with patch.object(litellm, "get_model_info") as mock_get_model_info:

            def get_info_side_effect(model):
                if model == "some-base-model":
                    return mock_base_info
                raise Exception("Model not in cost map")

            mock_get_model_info.side_effect = get_info_side_effect

            result = router.get_deployment_model_info(
                model_id="custom-with-base", model_name="unknown-model"
            )

            # Should return custom_model_info merged with base model info
            assert result is not None
            assert (
                result["input_cost_per_token"] == 0.01
            )  # From custom (overrides base)
            assert result["max_tokens"] == 8192  # From base model
            assert result["litellm_provider"] == "openai"  # From base model

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
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
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
        model_name="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    assert (
        result["endpoint"]
        == "/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke"
    ), f"Expected '/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke', got '{result['endpoint']}'"

    # Test Case 2: Bedrock invoke-with-response-stream endpoint
    kwargs = {
        "endpoint": "/model/special-bedrock-model/invoke-with-response-stream",
        "custom_llm_provider": "bedrock",
    }
    result = router._add_deployment_model_to_endpoint_for_llm_passthrough_route(
        kwargs=kwargs,
        model="special-bedrock-model",
        model_name="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    assert (
        result["endpoint"]
        == "/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke-with-response-stream"
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


def test_update_kwargs_with_deployment_uses_pass_through_request_timeout():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-bedrock-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
                },
            }
        ],
    )
    deployment = router.model_list[0]
    kwargs: dict = {}

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"pass_through_request_timeout": 6},
    ):
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs=kwargs,
            function_name="_ageneric_api_call_with_fallbacks",
        )

    assert kwargs["timeout"] == 6.0


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


@pytest.mark.asyncio
async def test_router_unknown_model_error_message_renders_model_name_literally():
    """
    The unknown-model error message renders the caller-supplied model name
    verbatim. A name containing Python format-field syntax must be treated as
    literal text, not re-interpreted as a format template, which would distort
    the message and balloon its length.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "azure/gpt-4o-real", "api_key": "fake-key"},
            }
        ]
    )

    weird_model = "ghost{:>200}model"
    messages = [{"role": "user", "content": "hi"}]

    with pytest.raises(litellm.BadRequestError) as excinfo:
        await router.acompletion(model=weird_model, messages=messages)

    message = str(excinfo.value)
    assert weird_model in message
    assert "          " not in message  # no padding run from an expanded format field


def test_get_deployment_credentials_with_provider_aws_bedrock_runtime_endpoint():
    """
    Test that get_deployment_credentials_with_provider correctly copies
    aws_bedrock_runtime_endpoint from deployment litellm_params to credentials.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-claude-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "aws_access_key_id": "test-access-key",
                    "aws_secret_access_key": "test-secret-key",
                    "aws_region_name": "us-east-1",
                    "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                },
            }
        ],
    )

    credentials = router.get_deployment_credentials_with_provider(
        model_id="bedrock-claude-model"
    )

    assert credentials is not None
    assert (
        credentials["aws_bedrock_runtime_endpoint"]
        == "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    assert credentials["aws_access_key_id"] == "test-access-key"
    assert credentials["aws_secret_access_key"] == "test-secret-key"
    assert credentials["aws_region_name"] == "us-east-1"
    assert credentials["custom_llm_provider"] == "bedrock"


def test_get_deployment_credentials_with_provider_includes_bucket_name():
    """
    Regression: bucket_name must survive the CredentialLiteLLMParams filter so
    managed-files batch retrieval can resolve the GCS/S3 bucket. Previously it was
    dropped, causing "GCS bucket_name is required" when fetching batch output files.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "vertex-gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-3.5-flash",
                    "vertex_project": "my-project",
                    "vertex_location": "global",
                    "gcs_bucket_name": "my-batch-bucket",
                },
            }
        ],
    )

    credentials = router.get_deployment_credentials_with_provider(
        model_id="vertex-gemini"
    )

    assert credentials is not None
    assert credentials["gcs_bucket_name"] == "my-batch-bucket"
    assert credentials["vertex_project"] == "my-project"
    assert credentials["custom_llm_provider"] == "vertex_ai"


def test_get_deployment_credentials_with_provider_resolves_credential_name():
    """
    Test that get_deployment_credentials_with_provider correctly resolves
    litellm_credential_name to actual credential values (for UI-created models).
    """
    from litellm.types.utils import CredentialItem

    # Setup credential list with a test credential
    litellm.credential_list = [
        CredentialItem(
            credential_name="test-azure-cred",
            credential_info={"custom_llm_provider": "azure"},
            credential_values={
                "api_key": "resolved-api-key",
                "api_base": "https://resolved.openai.azure.com",
                "api_version": "2024-02-01",
            },
        )
    ]

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-gpt-4",
                "litellm_params": {
                    "model": "azure/gpt-4",
                    "litellm_credential_name": "test-azure-cred",
                },
            }
        ],
    )

    credentials = router.get_deployment_credentials_with_provider(
        model_id="azure-gpt-4"
    )

    assert credentials is not None
    assert credentials["api_key"] == "resolved-api-key"
    assert credentials["api_base"] == "https://resolved.openai.azure.com"
    assert credentials["api_version"] == "2024-02-01"
    assert credentials["custom_llm_provider"] == "azure"
    # Ensure credential name is removed after resolution
    assert "litellm_credential_name" not in credentials

    # Cleanup
    litellm.credential_list = []


def _team_wildcard_model(api_key: str, model_id: str = "team-wildcard-id") -> dict:
    return {
        "model_name": f"model_name_team-1_{model_id}",
        "litellm_params": {"model": "openai/*", "api_key": api_key},
        "model_info": {
            "id": model_id,
            "team_id": "team-1",
            "team_public_model_name": "openai/*",
        },
    }


def test_get_deployment_credentials_with_provider_team_wildcard_priority():
    """
    Regression: a global wildcard pattern (e.g. "openai/*") must not shadow a
    team's own wildcard entry. When team_id is provided, the team wildcard
    deployment's credentials win; without team_id the global one is used.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "global-key"},
            },
            _team_wildcard_model(api_key="team-key"),
        ],
    )

    team_credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-5.2", team_id="team-1"
    )
    assert team_credentials is not None
    assert team_credentials["api_key"] == "team-key"

    global_credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-5.2"
    )
    assert global_credentials is not None
    assert global_credentials["api_key"] == "global-key"


def test_team_wildcard_credentials_not_usable_after_delete_deployment():
    """
    Regression: team_pattern_routers retained deleted deployments, so a team
    user could keep resolving credentials of a deleted wildcard deployment.
    """
    router = litellm.Router(model_list=[_team_wildcard_model(api_key="old-key")])

    assert (
        router.get_deployment_credentials_with_provider(
            model_id="openai/gpt-5.2", team_id="team-1"
        )
        is not None
    )

    router.delete_deployment(id="team-wildcard-id")

    assert (
        router.get_deployment_credentials_with_provider(
            model_id="openai/gpt-5.2", team_id="team-1"
        )
        is None
    )


def test_pattern_match_router_remove_deployment():
    """
    remove_deployment must drop only the deployment with the given model id and
    delete patterns whose deployment list becomes empty.
    """
    from litellm.router_utils.pattern_match_deployments import PatternMatchRouter

    pattern_router = PatternMatchRouter()
    pattern_router.add_pattern(
        "openai/*",
        {"litellm_params": {"model": "openai/*", "api_key": "key-a"}, "model_info": {"id": "dep-a"}},
    )
    pattern_router.add_pattern(
        "openai/*",
        {"litellm_params": {"model": "openai/*", "api_key": "key-b"}, "model_info": {"id": "dep-b"}},
    )

    pattern_router.remove_deployment(model_id="dep-a")
    matches = pattern_router.route("openai/gpt-5.2")
    assert matches is not None
    assert [m["model_info"]["id"] for m in matches] == ["dep-b"]

    pattern_router.remove_deployment(model_id="dep-b")
    assert pattern_router.patterns == {}
    assert pattern_router.route("openai/gpt-5.2") is None


def test_team_wildcard_credentials_refreshed_on_upsert_and_set_model_list():
    """
    Regression: replacing a team wildcard deployment (upsert or model list
    reload) must serve the new credentials, not the stale cached ones.
    """
    from litellm.types.router import Deployment

    router = litellm.Router(model_list=[_team_wildcard_model(api_key="old-key")])

    router.upsert_deployment(
        deployment=Deployment(**_team_wildcard_model(api_key="new-key"))
    )
    credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-5.2", team_id="team-1"
    )
    assert credentials is not None
    assert credentials["api_key"] == "new-key"

    router.set_model_list(model_list=[])
    assert (
        router.get_deployment_credentials_with_provider(
            model_id="openai/gpt-5.2", team_id="team-1"
        )
        is None
    )


def test_get_available_guardrail_single_deployment():
    """
    Test get_available_guardrail returns the single guardrail when only one exists.
    """
    guardrail_config = {
        "guardrail_name": "content-filter",
        "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
        "id": "guardrail-1",
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        guardrail_list=[guardrail_config],
    )

    result = router.get_available_guardrail(guardrail_name="content-filter")
    assert result == guardrail_config


def test_get_available_guardrail_multiple_deployments():
    """
    Test get_available_guardrail load balances across multiple guardrails.
    """
    guardrail_1 = {
        "guardrail_name": "content-filter",
        "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
        "id": "guardrail-1",
    }
    guardrail_2 = {
        "guardrail_name": "content-filter",
        "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
        "id": "guardrail-2",
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        guardrail_list=[guardrail_1, guardrail_2],
    )

    # Call multiple times to verify load balancing
    results = set()
    for _ in range(20):
        result = router.get_available_guardrail(guardrail_name="content-filter")
        results.add(result["id"])

    # Both guardrails should be selected at least once
    assert "guardrail-1" in results or "guardrail-2" in results


def test_get_available_guardrail_not_found():
    """
    Test get_available_guardrail raises ValueError when guardrail not found.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        guardrail_list=[],
    )

    with pytest.raises(ValueError, match="No guardrail found with name"):
        router.get_available_guardrail(guardrail_name="non-existent")


@pytest.mark.asyncio
async def test_aguardrail_helper():
    """
    Test _aguardrail_helper selects a guardrail and executes the original function.
    """
    guardrail_config = {
        "guardrail_name": "content-filter",
        "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
        "id": "guardrail-1",
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        guardrail_list=[guardrail_config],
    )

    # Mock the original function
    async def mock_original_function(**kwargs):
        return {
            "result": "success",
            "selected_guardrail": kwargs.get("selected_guardrail"),
        }

    result = await router._aguardrail_helper(
        model="content-filter",
        original_generic_function=mock_original_function,
    )

    assert result["result"] == "success"
    assert result["selected_guardrail"] == guardrail_config


@pytest.mark.asyncio
async def test_aguardrail():
    """
    Test aguardrail executes a guardrail with load balancing and fallbacks.
    """
    guardrail_config = {
        "guardrail_name": "content-filter",
        "litellm_params": {"guardrail": "custom", "mode": "pre_call"},
        "id": "guardrail-1",
    }

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        guardrail_list=[guardrail_config],
    )

    # Mock the original function
    async def mock_original_function(**kwargs):
        return {
            "result": "success",
            "selected_guardrail": kwargs.get("selected_guardrail"),
        }

    result = await router.aguardrail(
        guardrail_name="content-filter",
        original_function=mock_original_function,
    )

    assert result["result"] == "success"
    assert result["selected_guardrail"]["id"] == "guardrail-1"


@pytest.mark.asyncio
async def test_anthropic_messages_call_type_is_cached():
    """
    Regression test: Verify that anthropic_messages call type is allowed
    in PromptCachingDeploymentCheck.async_log_success_event.
    """
    import asyncio

    from litellm.caching.dual_cache import DualCache
    from litellm.router_utils.pre_call_checks.prompt_caching_deployment_check import (
        PromptCachingDeploymentCheck,
    )
    from litellm.router_utils.prompt_caching_cache import PromptCachingCache
    from litellm.types.utils import (
        CallTypes,
        StandardLoggingHiddenParams,
        StandardLoggingMetadata,
        StandardLoggingModelInformation,
        StandardLoggingPayload,
    )

    # Create mock standard logging payload inline
    def create_standard_logging_payload() -> StandardLoggingPayload:
        return StandardLoggingPayload(
            id="test_id",
            call_type="completion",
            response_cost=0.1,
            response_cost_failure_debug_info=None,
            status="success",
            total_tokens=30,
            prompt_tokens=20,
            completion_tokens=10,
            startTime=1234567890.0,
            endTime=1234567891.0,
            completionStartTime=1234567890.5,
            model_map_information=StandardLoggingModelInformation(
                model_map_key="gpt-3.5-turbo", model_map_value=None
            ),
            model="gpt-3.5-turbo",
            model_id="model-123",
            model_group="openai-gpt",
            api_base="https://api.openai.com",
            metadata=StandardLoggingMetadata(
                user_api_key_hash="test_hash",
                user_api_key_org_id=None,
                user_api_key_alias="test_alias",
                user_api_key_team_id="test_team",
                user_api_key_user_id="test_user",
                user_api_key_team_alias="test_team_alias",
                spend_logs_metadata=None,
                requester_ip_address="127.0.0.1",
                requester_metadata=None,
            ),
            cache_hit=False,
            cache_key=None,
            saved_cache_cost=0.0,
            request_tags=[],
            end_user=None,
            requester_ip_address="127.0.0.1",
            messages=[{"role": "user", "content": "Hello, world!"}],
            response={"choices": [{"message": {"content": "Hi there!"}}]},
            error_str=None,
            model_parameters={"stream": True},
            hidden_params=StandardLoggingHiddenParams(
                model_id="model-123",
                cache_key=None,
                api_base="https://api.openai.com",
                response_cost="0.1",
                additional_headers=None,
            ),
        )

    cache = DualCache()
    deployment_check = PromptCachingDeploymentCheck(cache=cache)
    prompt_cache = PromptCachingCache(cache=cache)

    # Create messages with enough tokens to pass the caching threshold
    test_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "test long message here" * 1024,
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        }
    ]
    test_model_id = "test-model-id-123"

    # Create a payload with anthropic_messages call type
    payload = create_standard_logging_payload()
    payload["call_type"] = CallTypes.anthropic_messages.value
    payload["messages"] = test_messages
    payload["model"] = "anthropic/claude-3-5-sonnet-20240620"
    payload["model_id"] = test_model_id

    # Log the success event (should cache the model_id)
    await deployment_check.async_log_success_event(
        kwargs={"standard_logging_object": payload},
        response_obj={},
        start_time=1234567890.0,
        end_time=1234567891.0,
    )

    # Small delay to ensure cache write completes
    await asyncio.sleep(0.1)

    # Verify that the model_id was actually cached
    cached_result = await prompt_cache.async_get_model_id(
        messages=test_messages,
        tools=None,
    )

    # This assertion will FAIL if anthropic_messages is filtered out
    assert (
        cached_result is not None
    ), "Model ID should be cached for anthropic_messages call type"
    assert (
        cached_result["model_id"] == test_model_id
    ), f"Expected {test_model_id}, got {cached_result['model_id']}"


def test_update_kwargs_with_deployment_propagates_model_tags():
    """
    Test that deployment-level tags from litellm_params are merged into
    kwargs metadata when _update_kwargs_with_deployment is called.

    This ensures model-level tags defined in config.yaml appear in SpendLogs.
    See: https://github.com/BerriAI/litellm/issues/XXXX
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key",
                    "tags": ["openai-account", "production"],
                },
            },
        ],
    )

    kwargs: dict = {"metadata": {}}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-4o-mini"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    # Deployment tags should be propagated to kwargs metadata
    assert "tags" in kwargs["metadata"]
    assert "openai-account" in kwargs["metadata"]["tags"]
    assert "production" in kwargs["metadata"]["tags"]


def test_update_kwargs_with_deployment_merges_tags_without_duplicates():
    """
    Test that when both request-level and deployment-level tags exist,
    they are merged without duplicates.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key",
                    "tags": ["openai-account", "shared-tag"],
                },
            },
        ],
    )

    # Simulate request that already has tags (from request body or key/team level)
    kwargs: dict = {"metadata": {"tags": ["user-tag", "shared-tag"]}}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-4o-mini"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    # Both sources should be merged, no duplicates
    assert "user-tag" in kwargs["metadata"]["tags"]
    assert "openai-account" in kwargs["metadata"]["tags"]
    assert "shared-tag" in kwargs["metadata"]["tags"]
    assert kwargs["metadata"]["tags"].count("shared-tag") == 1


def test_update_kwargs_with_deployment_no_tags():
    """
    Test that when deployment has no tags, kwargs metadata is not affected.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key",
                },
            },
        ],
    )

    kwargs: dict = {"metadata": {}}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="gpt-4o-mini"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    # No tags key should be added if deployment has no tags
    assert "tags" not in kwargs["metadata"]


def test_update_kwargs_with_deployment_merges_tools():
    """
    Test that when both deployment litellm_params and request have tools,
    they are merged (deployment tools first, then request tools).

    Supports proxy-configured tools (e.g. for o3 deep research) merged with
    client-provided tools.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "o3-deep-research",
                "litellm_params": {
                    "model": "openai/o3-deep-research",
                    "api_key": "fake-key",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                },
            },
        ],
    )

    kwargs: dict = {
        "metadata": {},
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "description": "Get weather"},
            },
        ],
    }
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="o3-deep-research"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    # Tools should be merged: deployment first, then request
    assert "tools" in kwargs
    assert len(kwargs["tools"]) == 2
    assert kwargs["tools"][0] == {"type": "web_search"}
    assert kwargs["tools"][1]["function"]["name"] == "get_weather"
    # tool_choice from request (none) - deployment's should be used
    assert kwargs["tool_choice"] == "auto"


def test_update_kwargs_with_deployment_merge_tools_deployment_only():
    """
    Test that when only deployment has tools, they are applied to kwargs.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "o3-deep-research",
                "litellm_params": {
                    "model": "openai/o3-deep-research",
                    "api_key": "fake-key",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "required",
                },
            },
        ],
    )

    kwargs: dict = {"metadata": {}}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="o3-deep-research"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    assert kwargs["tools"] == [{"type": "web_search"}]
    assert kwargs["tool_choice"] == "required"


def test_update_kwargs_with_deployment_merge_tools_request_overrides_tool_choice():
    """
    Test that when request has tool_choice, it overrides deployment's.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "o3-deep-research",
                "litellm_params": {
                    "model": "openai/o3-deep-research",
                    "api_key": "fake-key",
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                },
            },
        ],
    )

    kwargs: dict = {
        "metadata": {},
        "tool_choice": "none",
    }
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="o3-deep-research"
    )
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    # Request tool_choice should be preserved (merged tools still applied)
    assert kwargs["tool_choice"] == "none"


def test_credential_name_injected_as_tag():
    """
    Test that litellm_credential_name from deployment litellm_params
    is injected as a tag into metadata during _update_kwargs_with_deployment.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "xai-model",
                "litellm_params": {
                    "model": "xai/grok-4-1-fast",
                    "litellm_credential_name": "xAI",
                },
            }
        ],
    )

    kwargs: dict = {"metadata": {"tags": ["A.101"]}}
    deployment = router.get_deployment_by_model_group_name(model_group_name="xai-model")
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    assert "Credential: xAI" in kwargs["metadata"]["tags"]
    assert "A.101" in kwargs["metadata"]["tags"]


def test_credential_name_not_duplicated_in_tags():
    """
    Test that if the credential tag already exists in the tags list,
    it is not duplicated.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "xai-model",
                "litellm_params": {
                    "model": "xai/grok-4-1-fast",
                    "litellm_credential_name": "xAI",
                },
            }
        ],
    )

    kwargs: dict = {"metadata": {"tags": ["Credential: xAI", "A.101"]}}
    deployment = router.get_deployment_by_model_group_name(model_group_name="xai-model")
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    assert kwargs["metadata"]["tags"].count("Credential: xAI") == 1


def test_credential_name_not_injected_when_absent():
    """
    Test that when no litellm_credential_name is set, tags are unchanged.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-model",
                "litellm_params": {
                    "model": "gpt-4o",
                },
            }
        ],
    )

    kwargs: dict = {"metadata": {"tags": ["A.101"]}}
    deployment = router.get_deployment_by_model_group_name(model_group_name="gpt-model")
    router._update_kwargs_with_deployment(deployment=deployment, kwargs=kwargs)

    assert kwargs["metadata"]["tags"] == ["A.101"]


def test_update_kwargs_with_deployment_model_info_in_litellm_metadata():
    """For generic_api_call, model_info with pricing must go to litellm_metadata.

    Routes like /messages and /responses use generic_api_call which stores
    model_info under litellm_metadata. Regression test for #23185.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "api_key": "fake-key",
                },
                "model_info": {
                    "id": "custom-pricing-id",
                    "input_cost_per_token": 0.0003,
                    "output_cost_per_token": 0.0015,
                },
            },
        ],
    )

    kwargs: dict = {}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="claude-sonnet-4"
    )
    router._update_kwargs_with_deployment(
        deployment=deployment, kwargs=kwargs, function_name="generic_api_call"
    )

    assert "litellm_metadata" in kwargs
    model_info = kwargs["litellm_metadata"]["model_info"]
    assert model_info["id"] == "custom-pricing-id"
    assert model_info["input_cost_per_token"] == 0.0003
    assert model_info["output_cost_per_token"] == 0.0015


def test_update_kwargs_with_deployment_model_info_in_metadata():
    """For acompletion (function_name=None), model_info goes to metadata.

    /chat/completions uses acompletion which stores model_info under metadata.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-20250514",
                    "api_key": "fake-key",
                },
                "model_info": {
                    "id": "custom-pricing-id",
                    "input_cost_per_token": 0.0003,
                    "output_cost_per_token": 0.0015,
                },
            },
        ],
    )

    kwargs: dict = {}
    deployment = router.get_deployment_by_model_group_name(
        model_group_name="claude-sonnet-4"
    )
    router._update_kwargs_with_deployment(
        deployment=deployment, kwargs=kwargs, function_name=None
    )

    assert "metadata" in kwargs
    model_info = kwargs["metadata"]["model_info"]
    assert model_info["id"] == "custom-pricing-id"
    assert model_info["input_cost_per_token"] == 0.0003
    assert model_info["output_cost_per_token"] == 0.0015


def test_combine_fallback_usage():
    """Test that _combine_fallback_usage merges partial and fallback usage."""
    from litellm.router import Router
    from litellm.types.utils import Usage

    # Create a stream chunk with usage
    chunk = litellm.ModelResponseStream(
        id="test",
        model="gpt-4o",
        choices=[],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    # Call _combine_fallback_usage with no extra usage
    Router._combine_fallback_usage(chunk, None)
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5
    assert chunk.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_does_not_log_success_on_terminal_failure():
    """A mid-stream failure with no successful fallback raises and is logged as
    a failure, so the router must never dispatch it as a success. Partial-spend
    recovery for the failure row happens in the streaming handler, not here, so
    this guards only against reintroducing a success log for a failed stream.
    """
    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.utils import Delta, StreamingChoices, Usage

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key-1"},
            },
        ],
        set_verbose=True,
    )

    error = MidStreamFallbackError(
        message="Connection lost",
        model="gpt-4",
        llm_provider="openai",
        generated_content="The Roman Empire began when",
    )

    def _make_interrupted_model_response():
        partial_chunk = litellm.ModelResponseStream(
            id="chatcmpl-partial-1",
            created=1742056047,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        content="The Roman Empire began when", role="assistant"
                    ),
                )
            ],
            usage=Usage(prompt_tokens=17, completion_tokens=9, total_tokens=26),
        )

        class _RaisingStream:
            def __init__(self):
                self.index = 0
                self.chunks = [partial_chunk]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index == 0:
                    self.index += 1
                    return partial_chunk
                raise error

        stream = _RaisingStream()
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()
        logging_obj.model_call_details = {}
        setattr(stream, "model", "gpt-4")
        setattr(stream, "custom_llm_provider", "openai")
        setattr(stream, "logging_obj", logging_obj)
        return stream, logging_obj

    messages = [{"role": "user", "content": "Hello"}]
    initial_kwargs = {"model": "gpt-4", "stream": True}

    # Terminal path: no successful fallback -> the error propagates and the
    # router never dispatches a success for the failed stream.
    model_response, logging_obj = _make_interrupted_model_response()
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=error),
    ):
        result = await router._acompletion_streaming_iterator(
            model_response=model_response,
            messages=messages,
            initial_kwargs=dict(initial_kwargs),
        )
        collected = []
        with pytest.raises(MidStreamFallbackError):
            async for chunk in result:
                collected.append(chunk)

    assert len(collected) == 1
    logging_obj.dispatch_success_handlers.assert_not_called()

    # Fallback success: the fallback stream owns success accounting via
    # _combine_fallback_usage, so this iterator must not dispatch its own.
    model_response, logging_obj = _make_interrupted_model_response()

    class _FallbackStream:
        def __init__(self, items):
            self.items = items
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.items):
                raise StopAsyncIteration
            item = self.items[self.index]
            self.index += 1
            return item

    fallback_stream = _FallbackStream(
        [
            litellm.ModelResponseStream(
                id="chatcmpl-fallback-1",
                model="gpt-3.5-turbo",
                object="chat.completion.chunk",
                choices=[
                    StreamingChoices(
                        finish_reason=None,
                        index=0,
                        delta=Delta(content=" continued", role="assistant"),
                    )
                ],
            )
        ]
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ):
        result = await router._acompletion_streaming_iterator(
            model_response=model_response,
            messages=messages,
            initial_kwargs=dict(initial_kwargs),
        )
        collected = []
        async for chunk in result:
            collected.append(chunk)

    assert len(collected) == 2
    logging_obj.dispatch_success_handlers.assert_not_called()


@pytest.mark.asyncio
async def test_team_scoped_model_fallback():
    """
    Test that fallback works correctly for team-scoped models.

    When a team-scoped model fails and the fallback model is also team-scoped,
    the router should find the fallback deployment by matching team_public_model_name.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "team-a-primary-internal",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake"},
                "model_info": {
                    "team_id": "team-a",
                    "team_public_model_name": "primary-model",
                },
            },
            {
                "model_name": "team-a-fallback-internal",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake",
                    "mock_response": "fallback success from team-a",
                },
                "model_info": {
                    "team_id": "team-a",
                    "team_public_model_name": "fallback-model",
                },
            },
        ],
        fallbacks=[{"primary-model": ["fallback-model"]}],
    )

    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={"user_api_key_team_id": "team-a"},
        mock_testing_fallbacks=True,
    )
    assert response is not None
    assert response.choices[0].message.content == "fallback success from team-a"


@pytest.mark.asyncio
async def test_team_scoped_model_fallback_to_global():
    """
    Test that a team-scoped model can fall back to a global (non-team) model.

    Global models (no team_id on deployment) should be accessible as fallback
    targets for team-scoped requests.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "team-a-primary-internal",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake"},
                "model_info": {
                    "team_id": "team-a",
                    "team_public_model_name": "primary-model",
                },
            },
            {
                "model_name": "global-fallback",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake",
                    "mock_response": "global fallback success",
                },
            },
        ],
        fallbacks=[{"primary-model": ["global-fallback"]}],
    )

    response = await router.acompletion(
        model="primary-model",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={"user_api_key_team_id": "team-a"},
        mock_testing_fallbacks=True,
    )
    assert response is not None
    assert response.choices[0].message.content == "global fallback success"


@pytest.mark.asyncio
async def test_team_scoped_model_fallback_cross_team_blocked():
    """
    Test that cross-team fallback is correctly blocked.

    When team-a's model fails and the fallback target is scoped to team-b,
    the router should NOT use it (team isolation).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "team-a-primary-internal",
                "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake"},
                "model_info": {
                    "team_id": "team-a",
                    "team_public_model_name": "primary-model",
                },
            },
            {
                "model_name": "team-b-fallback-internal",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fake",
                    "mock_response": "team-b response - should not reach here",
                },
                "model_info": {
                    "team_id": "team-b",
                    "team_public_model_name": "fallback-model",
                },
            },
        ],
        fallbacks=[{"primary-model": ["fallback-model"]}],
    )

    with pytest.raises(Exception):
        await router.acompletion(
            model="primary-model",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"user_api_key_team_id": "team-a"},
            mock_testing_fallbacks=True,
        )


def test_get_all_deployments_with_team_id():
    """
    Test that _get_all_deployments with team_id can find deployments
    by team_public_model_name when the model_name is not in the index.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "internal-team-deployment",
                "litellm_params": {"model": "gpt-4", "api_key": "fake"},
                "model_info": {
                    "team_id": "team-x",
                    "team_public_model_name": "gpt-4",
                },
            },
        ],
    )

    # Without team_id: "gpt-4" is not in the model_name index (internal name is different)
    deployments = router._get_all_deployments(model_name="gpt-4")
    assert len(deployments) == 0

    # With correct team_id: should find via O(n) scan matching team_public_model_name
    deployments = router._get_all_deployments(model_name="gpt-4", team_id="team-x")
    assert len(deployments) == 1
    assert deployments[0]["model_name"] == "internal-team-deployment"

    # With wrong team_id: should find nothing
    deployments = router._get_all_deployments(model_name="gpt-4", team_id="team-y")
    assert len(deployments) == 0


def test_multiregion_team_deployments_unique_model_names():
    """
    Simulates athenahealth's exact setup: unique model_names per deployment,
    same team_public_model_name, multiple regions.

    Verifies that _get_all_deployments returns ALL regional deployments
    for a team when queried by team_public_model_name.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "metis-claude-us-east-1",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet",
                    "aws_region_name": "us-east-1",
                    "api_key": "fake",
                },
                "model_info": {
                    "team_id": "metis-team",
                    "team_public_model_name": "claude-sonnet",
                },
            },
            {
                "model_name": "metis-claude-us-west-2",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet",
                    "aws_region_name": "us-west-2",
                    "api_key": "fake",
                },
                "model_info": {
                    "team_id": "metis-team",
                    "team_public_model_name": "claude-sonnet",
                },
            },
        ],
    )

    # "claude-sonnet" is NOT in the model_name index
    assert "claude-sonnet" not in router.model_names

    # Without team_id: returns nothing (no model_name="claude-sonnet" in index, no O(n) scan)
    deployments = router._get_all_deployments(model_name="claude-sonnet")
    assert len(deployments) == 0

    # With team_id: O(n) scan finds BOTH regional deployments
    deployments = router._get_all_deployments(
        model_name="claude-sonnet", team_id="metis-team"
    )
    assert len(deployments) == 2
    deployment_names = {d["model_name"] for d in deployments}
    assert deployment_names == {"metis-claude-us-east-1", "metis-claude-us-west-2"}

    # Each deployment has a unique ID (critical for cooldown/retry to work)
    deployment_ids = {d["model_info"]["id"] for d in deployments}
    assert (
        len(deployment_ids) == 2
    ), "Each deployment must have a unique ID for cooldown tracking"

    # Wrong team: returns nothing
    deployments = router._get_all_deployments(
        model_name="claude-sonnet", team_id="other-team"
    )
    assert len(deployments) == 0


@pytest.mark.asyncio
async def test_multiregion_team_failover_between_regions():
    """
    Simulates athenahealth's multiregion failover scenario:
    - Two Bedrock deployments (us-east-1 and us-west-2) with unique model_names
    - Same team_public_model_name ("claude-sonnet")
    - Primary region fails → router should failover to second region

    This is the exact scenario Sean Glover from athenahealth will demonstrate.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "metis-claude-us-east-1",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet",
                    "api_key": "fake",
                    "mock_response": "response from us-east-1",
                },
                "model_info": {
                    "team_id": "metis-team",
                    "team_public_model_name": "claude-sonnet",
                },
            },
            {
                "model_name": "metis-claude-us-west-2",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-sonnet",
                    "api_key": "fake",
                    "mock_response": "response from us-west-2",
                },
                "model_info": {
                    "team_id": "metis-team",
                    "team_public_model_name": "claude-sonnet",
                },
            },
        ],
        num_retries=1,
    )

    # Verify the router finds both deployments for the team
    deployments = router._get_all_deployments(
        model_name="claude-sonnet", team_id="metis-team"
    )
    assert (
        len(deployments) == 2
    ), "Router must find both regional deployments by team_public_model_name"

    # Make a normal request — should succeed from one of the regions
    response = await router.acompletion(
        model="claude-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        metadata={"user_api_key_team_id": "metis-team"},
    )
    assert response is not None
    assert response.choices[0].message.content in [
        "response from us-east-1",
        "response from us-west-2",
    ]


def test_access_group_scoped_key_filters_deployments_with_same_public_model():
    """
    If a key can access a model only via access group membership,
    router candidate deployments for that public model should be constrained
    to deployments in the allowed access group.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "key1",
                    "mock_response": "response-via-AG1",
                },
                "model_info": {"access_groups": ["AG1"]},
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "key2",
                    "mock_response": "response-via-AG2",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
        ]
    )

    scoped_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team2",
        models=["AG2"],
        team_models=["AG2"],
    )

    _model, deployments = router._common_checks_available_deployment(
        model="gpt-5",
        request_kwargs={
            "metadata": {
                "user_api_key_team_id": "team2",
                "user_api_key_auth": scoped_key,
            }
        },
    )

    assert len(deployments) == 1
    assert deployments[0].get("model_info", {}).get("access_groups") == ["AG2"]

    seen = set()
    for _ in range(20):
        response = router.completion(
            model="gpt-5",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_team_id": "team2", "user_api_key_auth": scoped_key},
        )
        seen.add(response.choices[0].message.content)

    assert seen == {"response-via-AG2"}


def test_explicit_model_access_does_not_force_access_group_filtering():
    """
    If a key has explicit model access in addition to access group entries,
    do not force access-group-only filtering for deployment selection.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "key1",
                    "mock_response": "response-via-AG1",
                },
                "model_info": {"access_groups": ["AG1"]},
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "key2",
                    "mock_response": "response-via-AG2",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
        ]
    )

    explicit_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team2",
        models=["AG2", "gpt-5"],
        team_models=["AG2", "gpt-5"],
    )

    _model, deployments = router._common_checks_available_deployment(
        model="gpt-5",
        request_kwargs={
            "metadata": {
                "user_api_key_team_id": "team2",
                "user_api_key_auth": explicit_key,
            }
        },
    )

    deployment_groups = [
        d.get("model_info", {}).get("access_groups") for d in deployments
    ]
    assert ["AG1"] in deployment_groups
    assert ["AG2"] in deployment_groups


def test_access_group_filter_empty_does_not_bypass_via_litellm_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    When access-group filtering removes all candidates, _get_deployment_by_litellm_model
    must not run: it does not re-apply access groups and could return blocked deployments
    that share the same litellm_params.model as the request model string.

    ``get_model_access_groups`` is patched to expose AG1 for the public model (so the
    access-group filter runs with a non-empty allowed set) while every deployment
    returned for that name is AG2-only — filtered to empty. Without the guard, the
    litellm-model fallback would return both rows because ``litellm_params.model`` matches.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "gpt-5",
                    "api_key": "key1",
                    "mock_response": "blocked-dep-1",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "gpt-5",
                    "api_key": "key2",
                    "mock_response": "blocked-dep-2",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
        ]
    )

    orig_groups = router.get_model_access_groups

    def fake_get_model_access_groups(
        model_name=None, model_access_group=None, team_id=None
    ):
        if model_name == "gpt-5" and model_access_group is None:
            return {"AG1": ["gpt-5"], "AG2": ["gpt-5"]}
        return orig_groups(
            model_name=model_name,
            model_access_group=model_access_group,
            team_id=team_id,
        )

    monkeypatch.setattr(router, "get_model_access_groups", fake_get_model_access_groups)

    scoped_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team2",
        models=["AG1"],
        team_models=["AG1"],
    )

    with pytest.raises(litellm.BadRequestError):
        router._common_checks_available_deployment(
            model="gpt-5",
            request_kwargs={
                "metadata": {
                    "user_api_key_team_id": "team2",
                    "user_api_key_auth": scoped_key,
                }
            },
        )


def test_access_group_block_does_not_silently_use_default_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    When access-group filtering empties candidates for model X, the router must not use
    ``fallbacks`` default ``*`` routing to model Y: Y may have no ``access_groups``, so
    ``_filter_deployments_by_model_access_groups`` would not constrain Y and the caller
    would be served despite being blocked from X.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "gpt-5",
                    "api_key": "key1",
                    "mock_response": "blocked-dep-1",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "gpt-5",
                    "api_key": "key2",
                    "mock_response": "blocked-dep-2",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
            {
                "model_name": "gpt-4-fallback",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fallback-key",
                    "mock_response": "should-not-reach",
                },
            },
        ],
        fallbacks=[{"*": ["gpt-4-fallback"]}],
    )

    orig_groups = router.get_model_access_groups

    def fake_get_model_access_groups(
        model_name=None, model_access_group=None, team_id=None
    ):
        if model_name == "gpt-5" and model_access_group is None:
            return {"AG1": ["gpt-5"], "AG2": ["gpt-5"]}
        return orig_groups(
            model_name=model_name,
            model_access_group=model_access_group,
            team_id=team_id,
        )

    monkeypatch.setattr(router, "get_model_access_groups", fake_get_model_access_groups)

    scoped_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team2",
        models=["AG1"],
        team_models=["AG1"],
    )

    with pytest.raises(litellm.BadRequestError):
        router._common_checks_available_deployment(
            model="gpt-5",
            request_kwargs={
                "metadata": {
                    "user_api_key_team_id": "team2",
                    "user_api_key_auth": scoped_key,
                }
            },
        )


def test_access_group_block_via_litellm_model_branch_does_not_use_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    When the by-name lookup returns no deployments and the litellm-model fallback
    branch finds candidates that access-group filtering then empties, the router
    must not fall through to default ``fallbacks`` routing — the default fallback
    model may have no ``access_groups`` and would short-circuit the filter,
    silently serving a caller blocked by access-group restrictions.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5-alias",
                "litellm_params": {
                    "model": "gpt-5",
                    "api_key": "key1",
                    "mock_response": "blocked-dep-1",
                },
                "model_info": {"access_groups": ["AG2"]},
            },
            {
                "model_name": "gpt-4-fallback",
                "litellm_params": {
                    "model": "gpt-4",
                    "api_key": "fallback-key",
                    "mock_response": "should-not-reach",
                },
            },
        ],
        fallbacks=[{"*": ["gpt-4-fallback"]}],
    )

    orig_groups = router.get_model_access_groups

    def fake_get_model_access_groups(
        model_name=None, model_access_group=None, team_id=None
    ):
        if model_name == "gpt-5" and model_access_group is None:
            return {"AG1": ["gpt-5"], "AG2": ["gpt-5"]}
        return orig_groups(
            model_name=model_name,
            model_access_group=model_access_group,
            team_id=team_id,
        )

    monkeypatch.setattr(router, "get_model_access_groups", fake_get_model_access_groups)

    scoped_key = UserAPIKeyAuth(
        api_key="hashed-key",
        team_id="team2",
        models=["AG1"],
        team_models=["AG1"],
    )

    with pytest.raises(litellm.BadRequestError):
        router._common_checks_available_deployment(
            model="gpt-5",
            request_kwargs={
                "metadata": {
                    "user_api_key_team_id": "team2",
                    "user_api_key_auth": scoped_key,
                }
            },
        )


def test_try_early_resolve_deployments_for_model_not_in_names():
    """
    Direct coverage for ``_try_early_resolve_deployments_for_model_not_in_names``:

    - Returns ``None`` when the requested model is already in ``self.model_names``
      (the by-name lookup path will handle it).
    - Returns ``None`` when there are no team deployments, no pattern matches, and
      no default deployment to fall back to.
    - Returns the pattern-router match when the model matches a wildcard route.
    - Returns the default deployment with the request model substituted in when one
      is configured, without mutating the stored default.
    """
    router_in_names = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5",
                "litellm_params": {
                    "model": "openai/gpt-5",
                    "api_key": "key1",
                },
            },
        ]
    )

    assert (
        router_in_names._try_early_resolve_deployments_for_model_not_in_names(
            model="gpt-5", request_team_id=None
        )
        is None
    )
    assert (
        router_in_names._try_early_resolve_deployments_for_model_not_in_names(
            model="some-unknown-model", request_team_id=None
        )
        is None
    )

    pattern_router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "key-pattern",
                },
            },
        ]
    )

    pattern_result = (
        pattern_router._try_early_resolve_deployments_for_model_not_in_names(
            model="openai/gpt-4o-mini", request_team_id=None
        )
    )
    assert pattern_result is not None
    resolved_model, pattern_deployments = pattern_result
    assert resolved_model == "openai/gpt-4o-mini"
    assert isinstance(pattern_deployments, list) and len(pattern_deployments) == 1

    default_router = litellm.Router(
        model_list=[
            {
                "model_name": "named-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "key-named",
                },
            },
        ]
    )
    default_router.default_deployment = {
        "model_name": "default",
        "litellm_params": {
            "model": "openai/will-be-overridden",
            "api_key": "key-default",
        },
    }

    default_result = (
        default_router._try_early_resolve_deployments_for_model_not_in_names(
            model="brand-new-model", request_team_id=None
        )
    )
    assert default_result is not None
    resolved_model, default_deployment = default_result
    assert resolved_model == "brand-new-model"
    assert isinstance(default_deployment, dict)
    assert default_deployment["litellm_params"]["model"] == "brand-new-model"
    # The original default_deployment must not be mutated.
    assert (
        default_router.default_deployment["litellm_params"]["model"]
        == "openai/will-be-overridden"
    )


def _router_with_two_deployments(blocked_flags):
    import litellm

    model_list = []
    for idx, blocked in enumerate(blocked_flags):
        model_list.append(
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": f"openai/gpt-4o-{idx}"},
                "model_info": {"id": f"dep-{idx}", "blocked": blocked},
            }
        )
    return litellm.Router(model_list=model_list)


def test_get_fully_blocked_model_names_marks_name_when_all_deployments_blocked():
    router = _router_with_two_deployments([True, True])
    assert router.get_fully_blocked_model_names() == {"gpt-4o"}


def test_get_fully_blocked_model_names_keeps_name_when_partial_blocked():
    router = _router_with_two_deployments([True, False])
    assert router.get_fully_blocked_model_names() == set()


def test_get_fully_blocked_model_names_treats_missing_key_as_unblocked():
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "dep-0"},
            }
        ]
    )
    assert router.get_fully_blocked_model_names() == set()


def _seed_unhealthy_states(router, unhealthy_ids, timestamp=None):
    import time

    ts = timestamp if timestamp is not None else time.time()
    router.health_state_cache.set_deployment_health_states(
        {
            uid: {"is_healthy": False, "timestamp": ts, "reason": "test_unhealthy"}
            for uid in unhealthy_ids
        }
    )


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_marks_name_when_all_unhealthy():
    router = _router_with_two_deployments([False, False])
    _seed_unhealthy_states(router, {"dep-0", "dep-1"})
    assert await router.async_get_fully_unhealthy_model_names() == {"gpt-4o"}


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_keeps_name_when_partial():
    router = _router_with_two_deployments([False, False])
    _seed_unhealthy_states(router, {"dep-0"})
    assert await router.async_get_fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_empty_without_health_state():
    router = _router_with_two_deployments([False, False])
    assert await router.async_get_fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_ignores_stale_state():
    import time

    router = _router_with_two_deployments([False, False])
    stale_ts = time.time() - (router.health_state_cache.staleness_threshold + 10)
    _seed_unhealthy_states(router, {"dep-0", "dep-1"}, timestamp=stale_ts)
    assert await router.async_get_fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_includes_team_alias():
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {
                    "id": "dep-0",
                    "team_id": "team-1",
                    "team_public_model_name": "team-gpt",
                },
            }
        ]
    )
    _seed_unhealthy_states(router, {"dep-0"})
    assert await router.async_get_fully_unhealthy_model_names() == {
        "gpt-4o",
        "team-gpt",
    }


@pytest.mark.asyncio
async def test_async_get_fully_unhealthy_model_names_noop_with_allowed_fails_policy():
    from litellm.types.router import AllowedFailsPolicy

    router = _router_with_two_deployments([False, False])
    router.allowed_fails_policy = AllowedFailsPolicy(BadRequestErrorAllowedFails=1)
    _seed_unhealthy_states(router, {"dep-0", "dep-1"})
    assert await router.async_get_fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_async_get_healthy_deployments_skips_blocked_deployment():
    router = _router_with_two_deployments([True, False])
    healthy, all_dep = await router._async_get_healthy_deployments(
        model="gpt-4o", parent_otel_span=None
    )
    healthy_ids = [d["model_info"]["id"] for d in healthy]
    assert "dep-0" not in healthy_ids
    assert "dep-1" in healthy_ids
    assert len(all_dep) == 2


def test_get_healthy_deployments_sync_skips_blocked_deployment():
    router = _router_with_two_deployments([False, True])
    healthy, all_dep = router._get_healthy_deployments(
        model="gpt-4o", parent_otel_span=None
    )
    healthy_ids = [d["model_info"]["id"] for d in healthy]
    assert "dep-0" in healthy_ids
    assert "dep-1" not in healthy_ids
    assert len(all_dep) == 2


def test_filter_blocked_deployments_drops_blocked_keeps_unblocked():
    router = _router_with_two_deployments([True, False])
    filtered = router._filter_blocked_deployments(router.get_model_list() or [])
    ids = [d["model_info"]["id"] for d in filtered]
    assert ids == ["dep-1"]


@pytest.mark.asyncio
async def test_public_async_get_healthy_deployments_skips_blocked_on_primary_path():
    router = _router_with_two_deployments([True, False])
    deployments = await router.async_get_healthy_deployments(
        model="gpt-4o", request_kwargs={}
    )
    assert isinstance(deployments, list)
    ids = [d["model_info"]["id"] for d in deployments]
    assert "dep-0" not in ids
    assert "dep-1" in ids


def test_public_get_available_deployment_skips_blocked_on_primary_path():
    router = _router_with_two_deployments([True, False])
    deployment = router.get_available_deployment(model="gpt-4o", request_kwargs={})
    assert deployment["model_info"]["id"] == "dep-1"


def test_get_available_deployment_raises_when_addressed_dict_is_blocked():
    import litellm

    router = _router_with_two_deployments([True, True])
    with pytest.raises(litellm.ServiceUnavailableError):
        router.get_available_deployment(model="dep-0", request_kwargs={})


def _router_with_two_pass_through_deployments(blocked_flags):
    import litellm

    model_list = []
    for idx, blocked in enumerate(blocked_flags):
        model_list.append(
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": f"openai/gpt-4o-{idx}",
                    "api_key": "sk-fake-for-tests",
                    "use_in_pass_through": True,
                },
                "model_info": {"id": f"pt-{idx}", "blocked": blocked},
            }
        )
    return litellm.Router(model_list=model_list)


def test_get_available_deployment_for_pass_through_skips_blocked():
    router = _router_with_two_pass_through_deployments([True, False])
    deployment = router.get_available_deployment_for_pass_through(
        model="gpt-4o", request_kwargs={}
    )
    assert deployment["model_info"]["id"] == "pt-1"


def test_get_available_deployment_for_pass_through_raises_when_dict_blocked():
    import litellm

    router = _router_with_two_pass_through_deployments([True, True])
    with pytest.raises(litellm.ServiceUnavailableError):
        router.get_available_deployment_for_pass_through(
            model="pt-0", request_kwargs={}
        )


def test_initialize_deployment_for_pass_through_keeps_bedrock_iam_deployment():
    """
    Bedrock deployments using IAM/OIDC auth have no api_key; pass-through
    init must not raise and drop them from routing (#27728).
    """
    import litellm

    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-claude",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "aws_role_name": "arn:aws:iam::123456789012:role/my-role",
                    "aws_session_name": "my-session",
                    "use_in_pass_through": True,
                },
                "model_info": {"id": "bedrock-iam-pt"},
            }
        ]
    )
    assert [m["model_info"]["id"] for m in router.get_model_list()] == [
        "bedrock-iam-pt"
    ]


def test_initialize_deployment_for_pass_through_sets_credentials_with_api_key():
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        passthrough_endpoint_router,
    )

    passthrough_endpoint_router.credentials.clear()
    router = _router_with_two_pass_through_deployments([False, False])
    assert len(router.get_model_list()) == 2
    assert (
        passthrough_endpoint_router.get_credentials(
            custom_llm_provider="openai", region_name=None
        )
        == "sk-fake-for-tests"
    )


def test_get_deployment_credentials_returns_none_for_blocked_deployment():
    router = _router_with_two_deployments([True, False])
    assert router.get_deployment_credentials(model_id="dep-0") is None
    assert router.get_deployment_credentials(model_id="dep-1") is not None


def test_get_deployment_credentials_with_provider_returns_none_for_blocked_deployment():
    router = _router_with_two_deployments([True, False])
    assert router.get_deployment_credentials_with_provider(model_id="dep-0") is None
    assert router.get_deployment_credentials_with_provider(model_id="dep-1") is not None


def test_is_deployment_blocked_static_helper_reflects_blocked_flag():
    """
    Exercises Router._is_deployment_blocked so router_code_coverage.py (AST call graph)
    marks the helper as covered by router-named tests.
    """
    import types

    import litellm

    router = _router_with_two_deployments([True, False])
    blocked_dep = router.get_deployment("dep-0")
    unblocked_dep = router.get_deployment("dep-1")
    assert blocked_dep is not None and unblocked_dep is not None
    assert litellm.Router._is_deployment_blocked(blocked_dep) is True
    assert litellm.Router._is_deployment_blocked(unblocked_dep) is False

    # No model_info on deployment object → treated as not blocked
    assert litellm.Router._is_deployment_blocked(object()) is False
    missing_blocked = types.SimpleNamespace()
    assert (
        litellm.Router._is_deployment_blocked(
            types.SimpleNamespace(model_info=missing_blocked)
        )
        is False
    )
    assert (
        litellm.Router._is_deployment_blocked(
            types.SimpleNamespace(model_info=types.SimpleNamespace(blocked=True))
        )
        is True
    )


class TestRouterRequestTimeoutPropagation:
    """litellm_settings.request_timeout must act as an independent per-attempt timeout.

    Regression for LIT-2369: request_timeout was shadowed by router_settings.timeout,
    so Bedrock (and other provider) calls fell back to the hardcoded 600s httpx
    default instead of the configured value.
    """

    def _make_router(self, timeout=None, stream_timeout=None):
        return litellm.Router(
            model_list=[
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "openai/gpt-4",
                        "api_key": "sk-test",
                    },
                }
            ],
            timeout=timeout,
            stream_timeout=stream_timeout,
        )

    @pytest.fixture
    def explicit_request_timeout(self):
        original_value = litellm.request_timeout
        original_flag = litellm.request_timeout_explicitly_set
        litellm.request_timeout = 300
        litellm.request_timeout_explicitly_set = True
        try:
            yield 300
        finally:
            litellm.request_timeout = original_value
            litellm.request_timeout_explicitly_set = original_flag

    def test_request_timeout_stored_independently_when_both_set(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330)
        assert router.timeout == 330
        assert router.request_timeout == 300

    def test_request_timeout_none_when_not_explicitly_configured(self):
        original_value = litellm.request_timeout
        original_flag = litellm.request_timeout_explicitly_set
        litellm.request_timeout = litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS
        litellm.request_timeout_explicitly_set = False
        try:
            router = self._make_router(timeout=330)
            assert router.timeout == 330
            assert router.request_timeout is None
        finally:
            litellm.request_timeout = original_value
            litellm.request_timeout_explicitly_set = original_flag

    def test_non_stream_prefers_request_timeout_over_router_timeout(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330)
        assert router._get_non_stream_timeout(kwargs={}, data={}) == 300

    def test_stream_prefers_request_timeout_over_router_timeout(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330)
        # stream=True resolves through _get_stream_timeout; request_timeout must win.
        assert router._get_timeout(kwargs={"stream": True}, data={}) == 300

    def test_explicit_stream_timeout_still_wins_over_request_timeout(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330, stream_timeout=45)
        assert router._get_stream_timeout(kwargs={}, data={}) == 45

    def test_non_stream_falls_through_to_router_timeout_without_request_timeout(self):
        original_value = litellm.request_timeout
        original_flag = litellm.request_timeout_explicitly_set
        litellm.request_timeout = litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS
        litellm.request_timeout_explicitly_set = False
        try:
            router = self._make_router(timeout=330)
            assert router._get_non_stream_timeout(kwargs={}, data={}) == 330
        finally:
            litellm.request_timeout = original_value
            litellm.request_timeout_explicitly_set = original_flag

    def test_per_deployment_timeout_overrides_request_timeout(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330)
        assert router._get_non_stream_timeout(kwargs={}, data={"timeout": 120}) == 120

    def test_per_request_timeout_overrides_request_timeout(
        self, explicit_request_timeout
    ):
        router = self._make_router(timeout=330)
        assert (
            router._get_non_stream_timeout(
                kwargs={"timeout": 60}, data={"timeout": 120}
            )
            == 60
        )


class TestAdvisorSubCallCooldown:
    """Regression for LIT-4565: an advisor orchestration failure must not cool
    down the selected (healthy) deployment, which would reject unrelated
    callers to the same model group."""

    def _router(self):
        return litellm.Router(
            model_list=[
                {
                    "model_name": "claude-sonnet-5",
                    "litellm_params": {"model": "bedrock/us.anthropic.claude-opus-4-8"},
                    "model_info": {"id": "dep-1"},
                }
            ],
        )

    def _kwargs(self, exception):
        return {
            "exception": exception,
            "litellm_params": {"model_info": {"id": "dep-1"}, "metadata": {}},
        }

    def _auth_error(self):
        return litellm.AuthenticationError(
            message="x-api-key header is required",
            llm_provider="anthropic",
            model="claude-opus-4-8",
        )

    def _cooled_down_ids(self, router):
        active = router.cooldown_cache.get_active_cooldowns(
            model_ids=["dep-1"], parent_otel_span=None
        )
        return [entry[0] for entry in active]

    @pytest.mark.asyncio
    async def test_untagged_auth_error_cools_down_deployment(self):
        from datetime import datetime

        router = self._router()
        now = datetime.now()
        assert (
            router.deployment_callback_on_failure(
                self._kwargs(self._auth_error()), None, now, now
            )
            is True
        )
        assert "dep-1" in self._cooled_down_ids(router)

    def test_advisor_orchestration_failure_does_not_cool_down_deployment(self):
        from datetime import datetime

        from litellm.router_utils.cooldown_handlers import (
            mark_advisor_orchestration_failure,
        )

        router = self._router()
        exception = self._auth_error()
        mark_advisor_orchestration_failure(exception)

        now = datetime.now()
        assert (
            router.deployment_callback_on_failure(
                self._kwargs(exception), None, now, now
            )
            is False
        )
        assert "dep-1" not in self._cooled_down_ids(router)


def test_get_configured_token_limits_reads_deployment_model_info():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-custom-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"max_input_tokens": 32000, "max_output_tokens": 8000},
            }
        ]
    )

    assert router.get_configured_token_limits("my-custom-model") == (32000, 8000)


def test_get_configured_token_limits_returns_none_for_unset_or_unknown():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "no-limits-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
            }
        ]
    )

    assert router.get_configured_token_limits("no-limits-model") == (None, None)
    assert router.get_configured_token_limits("not-a-real-model") == (None, None)


def test_get_configured_token_limits_skips_wildcard_pattern_matching():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock/*",
                "litellm_params": {"model": "bedrock/*"},
                "model_info": {"max_input_tokens": 12345},
            }
        ]
    )

    with patch.object(
        router.pattern_router, "route", side_effect=AssertionError("pattern route called")
    ):
        assert router.get_configured_token_limits(
            "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
        ) == (None, None)
