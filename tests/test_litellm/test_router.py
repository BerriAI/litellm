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
    kwargs = {}

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
async def test_router_with_tags_and_fallbacks():
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
async def test_router_acreate_file():
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
async def test_router_acreate_file_with_jsonl():
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
async def test_router_async_get_healthy_deployments():
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
async def test_router_amoderation_with_credential_name(mock_amoderation):
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


def test_router_test_team_model():
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


def test_router_ignore_invalid_deployments():
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
            litellm_params={"model": "my-bad-model"},
            model_info={"tpm": 1000, "rpm": 1000},
        )
    )

    assert router.get_model_list() == []
