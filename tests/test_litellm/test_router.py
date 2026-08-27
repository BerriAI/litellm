import asyncio
import copy
import json
import logging
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest



import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.experimental_pass_through.messages.agentic_streaming_iterator import (
    SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES,
)
from litellm.router import (
    MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS,
    FallbackAwareAnthropicMessagesStream,
    _anthropic_stream_commits_now,
    _anthropic_stream_should_decline_fallback,
    _anthropic_stream_error_is_gateway_verdict,
    _anthropic_stream_forwards_ping_live,
    _anthropic_stream_should_drop_pre_content_ping,
    _is_retriable_anthropic_status,
)


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

    with pytest.raises(litellm.InternalServerError):
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
async def test_async_router_acreate_file_does_not_fall_back_across_model_groups():
    """A file created for batches only exists under the credentials of the model group
    the caller named. A cross-group fallback silently stores it with the wrong provider
    and the later batch create against the named group permanently fails."""
    from unittest.mock import MagicMock, patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "azure-gpt",
                "litellm_params": {
                    "model": "azure/my-azure-deployment",
                    "api_base": "http://127.0.0.1:9",
                    "api_key": "dummy-key",
                    "api_version": "2024-06-01",
                },
            },
            {
                "model_name": "openai-gpt",
                "litellm_params": {"model": "gpt-4o-mini"},
            },
        ],
        fallbacks=[{"azure-gpt": ["openai-gpt"]}],
    )

    def fail_azure(*args: object, **kwargs: object) -> MagicMock:
        if kwargs.get("model") == "azure/my-azure-deployment":
            raise litellm.APIConnectionError(
                message="Connection error.",
                llm_provider="azure",
                model="azure/my-azure-deployment",
            )
        return MagicMock()

    with patch("litellm.acreate_file", side_effect=fail_azure) as mock_acreate_file:
        with pytest.raises(litellm.APIConnectionError):
            await router.acreate_file(
                model="azure-gpt",
                purpose="batch",
                file=MagicMock(),
            )

        called_models = [call.kwargs.get("model") for call in mock_acreate_file.call_args_list]
        assert "azure/my-azure-deployment" in called_models
        assert "gpt-4o-mini" not in called_models


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
async def test_async_router_acreate_file_forwards_target_model_names_to_litellm_proxy():
    import json
    from io import BytesIO
    from unittest.mock import MagicMock, patch

    jsonl_file = BytesIO(
        json.dumps({"body": {"model": "chained-batch", "messages": [{"role": "user", "content": "hi"}]}}).encode(
            "utf-8"
        )
    )
    jsonl_file.name = "test.jsonl"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "chained-batch",
                "litellm_params": {
                    "model": "litellm_proxy/gpt-4.1-batch",
                    "api_base": "http://localhost:4001/v1",
                    "api_key": "sk-proxy-b",
                },
            },
        ],
    )

    with patch("litellm.acreate_file", return_value=MagicMock()) as mock_acreate_file:
        await router.acreate_file(
            model="chained-batch",
            purpose="batch",
            file=jsonl_file,
        )

        assert mock_acreate_file.call_count == 1
        call_kwargs = mock_acreate_file.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "litellm_proxy"
        assert call_kwargs["extra_body"] == {"target_model_names": "gpt-4.1-batch"}
        uploaded_line = json.loads(call_kwargs["file"].read().decode("utf-8").split("\n")[0])
        assert uploaded_line["body"]["model"] == "gpt-4.1-batch"


@pytest.mark.asyncio
async def test_async_router_acreate_file_does_not_inject_target_model_names_for_other_providers():
    from unittest.mock import MagicMock, patch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4.1-batch",
                "litellm_params": {"model": "gpt-4.1"},
            },
        ],
    )

    with patch("litellm.acreate_file", return_value=MagicMock()) as mock_acreate_file:
        await router.acreate_file(
            model="gpt-4.1-batch",
            purpose="batch",
            file=MagicMock(),
        )

        assert mock_acreate_file.call_count == 1
        assert mock_acreate_file.call_args.kwargs.get("extra_body") is None


@pytest.mark.asyncio
async def test_async_router_acreate_file_litellm_proxy_sends_target_model_names_in_multipart_form():
    import json
    from io import BytesIO

    import httpx
    import respx

    jsonl_file = BytesIO(
        json.dumps({"body": {"model": "chained-batch", "messages": [{"role": "user", "content": "hi"}]}}).encode(
            "utf-8"
        )
    )
    jsonl_file.name = "test.jsonl"

    router = litellm.Router(
        model_list=[
            {
                "model_name": "chained-batch",
                "litellm_params": {
                    "model": "litellm_proxy/gpt-4.1-batch",
                    "api_base": "http://localhost:4001/v1",
                    "api_key": "sk-proxy-b",
                },
            },
        ],
    )

    file_object_json = {
        "id": "file-abc123",
        "object": "file",
        "bytes": 100,
        "created_at": 1700000000,
        "filename": "test.jsonl",
        "purpose": "batch",
        "status": "processed",
    }

    with respx.mock(assert_all_called=True) as respx_mock:
        create_route = respx_mock.post("http://localhost:4001/v1/files").mock(
            return_value=httpx.Response(200, json=file_object_json)
        )
        response = await router.acreate_file(
            model="chained-batch",
            purpose="batch",
            file=jsonl_file,
        )

    assert response.id == "file-abc123"
    request_body = create_route.calls.last.request.content
    assert b'name="target_model_names"' in request_body
    assert b"gpt-4.1-batch" in request_body
    assert b'name="purpose"' in request_body


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
    with pytest.raises(Exception, match='No deployments available for selected model, Try again in') as e:
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

    with pytest.raises(Exception, match='Unsupported provider - vertex_ai_eu') as e:
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

        with pytest.raises(Exception, match='No deployment available') as exc_info:
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
                    with pytest.raises(Exception, match='Mock failure') as exc_info:
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

    # Test 2: MidStreamFallbackError with generated content is re-raised, not silently continued
    print("\n=== Test 2: MidStreamFallbackError re-raises when content already generated ===")

    # Error with generated content and is_pre_first_chunk=False (the default):
    # the router must re-raise instead of attempting a continuation-prompt fallback,
    # because partial content has already been sent to the client.
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

    mock_error_response = AsyncIteratorWithError(mock_chunks, 1)  # Error after first chunk

    setattr(mock_error_response, "model", "gpt-4")
    setattr(mock_error_response, "custom_llm_provider", "openai")
    setattr(mock_error_response, "logging_obj", MagicMock())

    result = await router._acompletion_streaming_iterator(
        model_response=mock_error_response,
        messages=messages,
        initial_kwargs=initial_kwargs,
    )

    # Collect streamed chunks — the first chunk succeeds, then the error re-raises
    collected_chunks = []
    async def _drain():
        async for chunk in result:
            collected_chunks.append(chunk)

    with pytest.raises(MidStreamFallbackError):
        await _drain()

    assert len(collected_chunks) == 1, "one chunk yielded before the error"
    print("✓ MidStreamFallbackError re-raised correctly when content was already generated")

    print("\n=== All tests passed! ===")


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_reraises_original_exception_when_available():
    """Async: when the mid-stream MidStreamFallbackError wraps a real provider
    exception (original_exception), the router must re-raise that original
    exception instead of the internal wrapper, so the client sees the
    specific error type/code (e.g. RateLimitError) rather than a generic
    MidStreamFallbackError."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError, RateLimitError

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

    original_exception = RateLimitError(
        message="rate limited",
        llm_provider="vertex_ai",
        model="gpt-4",
    )
    error = MidStreamFallbackError(
        message="rate limited",
        model="gpt-4",
        llm_provider="openai",
        original_exception=original_exception,
        generated_content="Hello",
    )

    mock_chunks = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content=" there"))]),
    ]

    class AsyncIteratorWithError:
        def __init__(self, items, error_after_index):
            self.items = items
            self.index = 0
            self.error_after_index = error_after_index

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

    mock_error_response = AsyncIteratorWithError(mock_chunks, 1)
    setattr(mock_error_response, "model", "gpt-4")
    setattr(mock_error_response, "custom_llm_provider", "openai")
    setattr(mock_error_response, "logging_obj", MagicMock())

    result = await router._acompletion_streaming_iterator(
        model_response=mock_error_response,
        messages=messages,
        initial_kwargs=initial_kwargs,
    )

    with pytest.raises(RateLimitError) as exc_info:
        async for _ in result:
            pass
    assert exc_info.value is original_exception
    assert exc_info.value.type == "throttling_error"
    assert exc_info.value.code == "429"


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


def test_completion_streaming_iterator_reraises_mid_chunk_error():
    """Sync: MidStreamFallbackError with generated_content and is_pre_first_chunk=False
    must be re-raised immediately; the router cannot recover after partial content
    has already been sent to the client."""
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

    mid_chunk_error = MidStreamFallbackError(
        message="Connection reset",
        model="gpt-4",
        llm_provider="openai",
        generated_content="Hello, I am",
        is_pre_first_chunk=False,
    )

    class SyncIteratorMidChunkError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __iter__(self):
            return self

        def __next__(self):
            raise mid_chunk_error

    mock_response = SyncIteratorMidChunkError()

    result = router._completion_streaming_iterator(
        model_response=mock_response,
        messages=messages,
        initial_kwargs=initial_kwargs,
    )

    with pytest.raises(MidStreamFallbackError):
        list(result)


def test_completion_streaming_iterator_reraises_original_exception_when_available():
    """Sync: when the mid-chunk MidStreamFallbackError wraps a real provider
    exception (original_exception), the router must re-raise that original
    exception instead of the internal wrapper, so the client sees the
    specific error type/code (e.g. RateLimitError) rather than a generic
    MidStreamFallbackError."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError, RateLimitError

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

    original_exception = RateLimitError(
        message="rate limited",
        llm_provider="vertex_ai",
        model="gpt-4",
    )
    mid_chunk_error = MidStreamFallbackError(
        message="rate limited",
        model="gpt-4",
        llm_provider="openai",
        original_exception=original_exception,
        generated_content="Hello, I am",
        is_pre_first_chunk=False,
    )

    class SyncIteratorMidChunkError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = []

        def __iter__(self):
            return self

        def __next__(self):
            raise mid_chunk_error

    mock_response = SyncIteratorMidChunkError()

    result = router._completion_streaming_iterator(
        model_response=mock_response,
        messages=messages,
        initial_kwargs=initial_kwargs,
    )

    with pytest.raises(RateLimitError) as exc_info:
        list(result)
    assert exc_info.value is original_exception
    assert exc_info.value.type == "throttling_error"
    assert exc_info.value.code == "429"


def test_completion_streaming_iterator_reraises_mid_chunk_error_with_no_text_content():
    """Sync: a reasoning-only chunk sets is_pre_first_chunk=False without populating
    generated_content (which only tracks text deltas). The re-raise guard must still
    detect this via the raw chunks on the wrapper, or the router silently retries and
    the client receives duplicated/inconsistent output."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.utils import Delta, StreamingChoices

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

    mid_chunk_error = MidStreamFallbackError(
        message="Connection reset",
        model="gpt-4",
        llm_provider="openai",
        generated_content="",
        is_pre_first_chunk=False,
    )

    reasoning_chunk = litellm.ModelResponseStream(
        id="chatcmpl-partial-1",
        model="gpt-4",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(reasoning_content="Thinking about the answer", role="assistant"),
            )
        ],
    )

    class SyncIteratorNoTextChunkError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = [reasoning_chunk]

        def __iter__(self):
            return self

        def __next__(self):
            raise mid_chunk_error

    mock_response = SyncIteratorNoTextChunkError()

    with patch.object(router, "function_with_fallbacks") as mock_fallback:
        result = router._completion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        with pytest.raises(MidStreamFallbackError):
            list(result)

        assert not mock_fallback.called, (
            "fallback must not be attempted once any content, text or non-text, has already streamed"
        )


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


@pytest.mark.asyncio
async def test_acompletion_streaming_iterator_reraises_mid_chunk_error_with_no_text_content():
    """Async: a reasoning-only chunk sets is_pre_first_chunk=False without populating
    generated_content (which only tracks text deltas). The re-raise guard must still
    detect this via the raw chunks on the wrapper, or the router silently retries and
    the client receives duplicated/inconsistent output."""
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError
    from litellm.types.utils import Delta, StreamingChoices

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

    mid_chunk_error = MidStreamFallbackError(
        message="Connection reset",
        model="gpt-4",
        llm_provider="openai",
        generated_content="",
        is_pre_first_chunk=False,
    )

    reasoning_chunk = litellm.ModelResponseStream(
        id="chatcmpl-partial-1",
        model="gpt-4",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(reasoning_content="Thinking about the answer", role="assistant"),
            )
        ],
    )

    class AsyncIteratorNoTextChunkError:
        def __init__(self):
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.chunks = [reasoning_chunk]

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise mid_chunk_error

    mock_response = AsyncIteratorNoTextChunkError()

    with patch.object(router, "async_function_with_fallbacks_common_utils") as mock_fallback_utils:
        iterator = await router._acompletion_streaming_iterator(
            model_response=mock_response,
            messages=messages,
            initial_kwargs=initial_kwargs,
        )

        with pytest.raises(MidStreamFallbackError):
            async for _ in iterator:
                pass

        assert not mock_fallback_utils.called, (
            "fallback must not be attempted once any content, text or non-text, has already streamed"
        )


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


def test_pre_call_checks_uses_precounted_tokens(monkeypatch):
    """
    An async caller counts off the event loop and passes the result in. _pre_call_checks
    must filter on that count instead of re-counting on the loop.
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
        litellm, "token_counter", lambda *a, **k: calls.append(1) or 1
    )

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    with pytest.raises(litellm.ContextWindowExceededError):
        router._pre_call_checks(
            model="m",
            healthy_deployments=deployments,
            messages=[{"role": "user", "content": "hi"}],
            input_token_count=1000,
        )

    assert calls == []


async def test_async_get_healthy_deployments_counts_tokens_off_the_event_loop(monkeypatch):
    """
    The async deployment path must hand _pre_call_checks a count taken in a worker thread,
    so a multi-MB prompt never blocks the proxy during deployment selection.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(
        router, "get_router_model_info", lambda **kwargs: {"max_input_tokens": 1_000_000}
    )

    counting_threads = []
    monkeypatch.setattr(
        litellm,
        "token_counter",
        lambda *a, **k: counting_threads.append(threading.current_thread()) or 42,
    )

    counts_passed_in = []
    original_pre_call_checks = router._pre_call_checks

    def spy(**kwargs):
        counts_passed_in.append(kwargs.get("input_token_count"))
        return original_pre_call_checks(**kwargs)

    monkeypatch.setattr(router, "_pre_call_checks", spy)

    result = await router.async_get_healthy_deployments(
        model="m",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
        input=None,
        specific_deployment=False,
        parent_otel_span=None,
    )

    assert len(result) == 1
    assert counts_passed_in == [42]
    assert len(counting_threads) == 1
    assert counting_threads[0] is not threading.current_thread()


@pytest.mark.parametrize(
    "model_info,expected",
    [
        ({"max_input_tokens": 100}, True),
        ({"max_input_tokens": None}, False),
        ({}, False),
    ],
)
def test_pre_call_checks_need_token_count(monkeypatch, model_info, expected):
    """Only a deployment that declares an integer context window makes a token count worth taking."""
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )
    monkeypatch.setattr(router, "get_router_model_info", lambda **kwargs: model_info)

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    assert router._pre_call_checks_need_token_count("m", deployments) is expected


def test_deployment_max_input_tokens_survives_an_unmappable_deployment(monkeypatch):
    """
    _pre_call_checks skips a deployment it cannot resolve and carries on. The off-loop
    pre-count must do the same, or an unmapped first deployment hides the limit declared by
    a later one and the count lands back on the event loop.
    """
    router = litellm.Router(
        model_list=[
            {"model_name": "m", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ],
        enable_pre_call_checks=True,
    )

    def flaky_model_info(deployment, received_model_name, id=None):
        if deployment["model_info"]["id"] == "unmapped":
            raise ValueError("This model isn't mapped yet.")
        return {"max_input_tokens": 100}

    monkeypatch.setattr(router, "get_router_model_info", flaky_model_info)

    unmapped = {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "unmapped"}}
    mapped = {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "mapped"}}

    assert router._deployment_max_input_tokens("m", unmapped) is None
    assert router._deployment_max_input_tokens("m", mapped) == 100
    assert router._pre_call_checks_need_token_count("m", [unmapped, mapped]) is True


def test_pre_call_checks_does_not_recount_inline_after_an_off_loop_failure(monkeypatch):
    """
    When the off-loop count failed there is nothing left to filter on, so _pre_call_checks must
    return the deployments unfiltered rather than repeating the count on the event loop.
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
    ]
    result = router._pre_call_checks(
        model="m",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "hi"}],
        input_token_count=None,
        skip_inline_token_count=True,
    )

    assert calls == []
    assert len(result) == 1


async def test_async_get_healthy_deployments_never_recounts_on_the_loop(monkeypatch):
    """
    An off-loop count that raises must not send the same work back onto the event loop through
    _pre_call_checks' inline fallback.
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

    counting_threads = []

    def exploding_counter(*args, **kwargs):
        counting_threads.append(threading.current_thread())
        raise ValueError("Invalid content item type: image")

    monkeypatch.setattr(litellm, "token_counter", exploding_counter)

    result = await router.async_get_healthy_deployments(
        model="m",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
        input=None,
        specific_deployment=False,
        parent_otel_span=None,
    )

    assert len(result) == 1
    assert len(counting_threads) == 1
    assert counting_threads[0] is not threading.current_thread()


async def test_acount_pre_call_check_tokens_leaves_the_event_loop_free(monkeypatch):
    """
    A multi-MB prompt must not stall the proxy: a competing coroutine has to get
    scheduled while the router's context-window count is in flight.
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

    deployments = [
        {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
    ]
    ran = []

    async def competitor():
        ran.append("competitor")

    task = asyncio.create_task(competitor())
    count = await router._acount_pre_call_check_tokens(
        model="m",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "A" * 512 * 1024}],
        input=None,
        request_kwargs=None,
    )
    ran.append("count")
    await task

    assert count is not None and count > 0
    assert ran == ["competitor", "count"]


async def test_acount_pre_call_check_tokens_skips_without_max_input_tokens(monkeypatch):
    """No deployment limits its context window, so there is nothing to count."""
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

    count = await router._acount_pre_call_check_tokens(
        model="m",
        healthy_deployments=[
            {"litellm_params": {"model": "gpt-3.5-turbo"}, "model_info": {"id": "d1"}},
        ],
        messages=[{"role": "user", "content": "hi"}],
        input=None,
        request_kwargs=None,
    )

    assert count is None
    assert calls == []


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

    with pytest.raises(ValueError, match='Either messages or input must be provided to count tokens'):
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


def test_get_deployment_credentials_with_provider_bedrock_batch_fields():
    """
    Test that get_deployment_credentials_with_provider returns the deployment's
    model and the Bedrock batch/S3 fields (s3_region_name, s3_encryption_key_id,
    aws_batch_role_arn) instead of silently dropping them (#25104).
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-batch-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "aws_region_name": "us-west-2",
                    "s3_bucket_name": "my-batch-bucket",
                    "s3_region_name": "us-east-1",
                    "s3_encryption_key_id": "arn:aws:kms:us-west-2:123:key/abc",
                    "aws_batch_role_arn": "arn:aws:iam::123:role/batch-role",
                },
            }
        ],
    )

    credentials = router.get_deployment_credentials_with_provider(
        model_id="bedrock-batch-model"
    )

    assert credentials is not None
    assert credentials["custom_llm_provider"] == "bedrock"
    assert credentials["model"] == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert credentials["aws_region_name"] == "us-west-2"
    assert credentials["s3_bucket_name"] == "my-batch-bucket"
    assert credentials["s3_region_name"] == "us-east-1"
    assert credentials["s3_encryption_key_id"] == "arn:aws:kms:us-west-2:123:key/abc"
    assert credentials["aws_batch_role_arn"] == "arn:aws:iam::123:role/batch-role"


def test_get_deployment_credentials_with_provider_preserves_aws_auth_params():
    """
    Test that get_deployment_credentials_with_provider preserves every AWS auth
    selector (session token, assume-role, web identity, profile) so bedrock
    files/batches deployments using temporary or role-based credentials do not
    silently fall back to the server's ambient identity (#36155).
    """
    aws_auth_params = {
        "aws_access_key_id": "deployment-access-key",
        "aws_secret_access_key": "deployment-secret",
        "aws_session_token": "deployment-session-token",
        "aws_region_name": "us-west-2",
        "aws_session_name": "deployment-session",
        "aws_profile_name": "deployment-profile",
        "aws_role_name": "arn:aws:iam::123:role/deployment-role",
        "aws_web_identity_token": "deployment-web-identity",
        "aws_sts_endpoint": "https://sts.us-west-2.amazonaws.com",
        "aws_external_id": "deployment-external-id",
    }
    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-batch-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    **aws_auth_params,
                },
            }
        ],
    )

    credentials = router.get_deployment_credentials_with_provider(
        model_id="bedrock-batch-model"
    )

    assert credentials is not None
    for key, value in aws_auth_params.items():
        assert credentials.get(key) == value, key


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


def test_get_deployment_credentials_with_provider_skips_other_team_deployment():
    """
    Regression: a team-scoped deployment sharing a model_name with a global
    deployment must never resolve for another team's (or an unscoped) caller,
    even when it is indexed first; the shared global deployment wins instead.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "team-b-project",
                },
                "model_info": {
                    "id": "team-b-vertex",
                    "team_id": "team-b",
                    "team_public_model_name": "gemini-2.5-pro",
                },
            },
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "shared-project",
                },
            },
        ],
    )

    other_team_credentials = router.get_deployment_credentials_with_provider(
        model_id="gemini-2.5-pro", team_id="team-a"
    )
    assert other_team_credentials is not None
    assert other_team_credentials["vertex_project"] == "shared-project"

    unscoped_credentials = router.get_deployment_credentials_with_provider(
        model_id="gemini-2.5-pro"
    )
    assert unscoped_credentials is not None
    assert unscoped_credentials["vertex_project"] == "shared-project"

    owner_credentials = router.get_deployment_credentials_with_provider(
        model_id="gemini-2.5-pro", team_id="team-b"
    )
    assert owner_credentials is not None
    assert owner_credentials["vertex_project"] == "team-b-project"


def test_get_deployment_credentials_with_provider_no_fallback_to_other_team_only_name():
    """
    When the only deployments under a model name belong to another team, other
    callers must get None (env fallback) instead of that team's credentials.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "team-b-project",
                },
                "model_info": {
                    "id": "team-b-vertex",
                    "team_id": "team-b",
                    "team_public_model_name": "gemini-2.5-pro",
                },
            },
        ],
    )

    assert (
        router.get_deployment_credentials_with_provider(
            model_id="gemini-2.5-pro", team_id="team-a"
        )
        is None
    )
    assert (
        router.get_deployment_credentials_with_provider(model_id="gemini-2.5-pro")
        is None
    )


def test_deployment_usable_by_team_helpers():
    """
    Direct coverage of the team-ownership filter: a team-scoped deployment is
    usable only by its owning team, shared deployments by anyone, and the
    model-group picker returns the first usable deployment or None.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "team-b-project",
                },
                "model_info": {
                    "id": "team-b-vertex",
                    "team_id": "team-b",
                    "team_public_model_name": "gemini-2.5-pro",
                },
            },
            {
                "model_name": "gemini-2.5-pro",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "vertex_project": "shared-project",
                },
            },
        ],
    )

    team_owned, shared = router.model_list
    assert router._deployment_usable_by_team(team_owned, "team-b") is True
    assert router._deployment_usable_by_team(team_owned, "team-a") is False
    assert router._deployment_usable_by_team(team_owned, None) is False
    assert router._deployment_usable_by_team(shared, "team-a") is True
    assert router._deployment_usable_by_team(shared, None) is True

    picked = router._get_model_group_deployment_usable_by_team(
        model_group_name="gemini-2.5-pro", team_id="team-a"
    )
    assert picked is not None
    assert picked.litellm_params.vertex_project == "shared-project"

    owner_picked = router._get_model_group_deployment_usable_by_team(
        model_group_name="gemini-2.5-pro", team_id="team-b"
    )
    assert owner_picked is not None
    assert owner_picked.litellm_params.vertex_project == "team-b-project"

    assert (
        router._get_model_group_deployment_usable_by_team(
            model_group_name="unknown-model", team_id="team-a"
        )
        is None
    )


def test_get_deployment_credentials_with_provider_skips_other_team_wildcard():
    """
    Global wildcard resolution must skip a team-scoped wildcard deployment for
    callers outside that team, falling through to the shared wildcard entry.
    """
    router = litellm.Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "team-b-key"},
                "model_info": {
                    "id": "team-b-wildcard",
                    "team_id": "team-b",
                    "team_public_model_name": "openai/*",
                },
            },
            {
                "model_name": "openai/*",
                "litellm_params": {"model": "openai/*", "api_key": "global-key"},
            },
        ],
    )

    other_team_credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-5.2", team_id="team-a"
    )
    assert other_team_credentials is not None
    assert other_team_credentials["api_key"] == "global-key"

    owner_credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-5.2", team_id="team-b"
    )
    assert owner_credentials is not None
    assert owner_credentials["api_key"] == "team-b-key"


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
                    delta=Delta(content="The Roman Empire began when", role="assistant"),
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
        async def _drain():
            async for chunk in result:
                collected.append(chunk)

        with pytest.raises(MidStreamFallbackError):
            await _drain()

    assert len(collected) == 1
    logging_obj.dispatch_success_handlers.assert_not_called()

    # Mid-stream errors with generated content are now re-raised immediately;
    # no continuation-prompt fallback is attempted.  Success handlers must
    # still not be dispatched in this path.
    model_response, logging_obj = _make_interrupted_model_response()

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        result = await router._acompletion_streaming_iterator(
            model_response=model_response,
            messages=messages,
            initial_kwargs=dict(initial_kwargs),
        )
        collected = []
        async def _drain():
            async for chunk in result:
                collected.append(chunk)

        with pytest.raises(MidStreamFallbackError):
            await _drain()

    assert len(collected) == 1, "only the partial chunk before the error"
    mock_fallback.assert_not_called()
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

    with pytest.raises(litellm.InternalServerError):
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


def test_pass_through_deployment_api_key_resolves_via_get_credentials():
    from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
        PassthroughEndpointRouter,
    )

    router = _router_with_two_pass_through_deployments([False, False])
    passthrough_router = PassthroughEndpointRouter(llm_router_getter=lambda: router)
    assert len(router.get_model_list()) == 2
    assert (
        passthrough_router.get_credentials(
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


# ---------------------------------------------------------------------------
# Deferred-stream eager-fetch tests
# ---------------------------------------------------------------------------


def _make_deferred_stream_wrapper(make_call_fn):
    """Return a CustomStreamWrapper with completion_stream=None and the given make_call."""
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    return CustomStreamWrapper(
        completion_stream=None,
        model="vertex_ai/gemini-2.0-flash",
        logging_obj=logging_obj,
        custom_llm_provider="vertex_ai_beta",
        make_call=make_call_fn,
    )


def _make_router_with_vertex_and_fallback():
    return litellm.Router(
        model_list=[
            {
                "model_name": "my-gemini",
                "litellm_params": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "vertex_project": "test-project",
                    "vertex_location": "us-central1",
                },
            },
            {
                "model_name": "my-fallback",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fake",
                },
            },
        ],
        fallbacks=[{"my-gemini": ["my-fallback"]}],
        num_retries=0,
    )


@pytest.mark.asyncio
async def test_acompletion_deferred_stream_error_propagates_through_acompletion():
    """Regression: a deferred-stream CustomStreamWrapper whose make_call raises a 429
    must propagate the exception from within _acompletion's except block so that
    fail_calls is incremented (i.e., deployment cooldown fires) and the standard
    router fallback chain can handle it.

    Before the fix, the HTTP call happened inside __anext__ (outside the except block),
    so fail_calls was never incremented.
    """
    import litellm as _litellm

    rate_limit_err = _litellm.RateLimitError(
        message="Resource exhausted",
        llm_provider="vertex_ai",
        model="gemini-2.0-flash",
    )

    async def failing_make_call(**kwargs):
        raise rate_limit_err

    router = _make_router_with_vertex_and_fallback()
    deferred_wrapper = _make_deferred_stream_wrapper(failing_make_call)

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=deferred_wrapper,
    ):
        with pytest.raises(_litellm.RateLimitError):
            await router._acompletion(
                model="vertex_ai/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                specific_deployment=router.model_list[0],
            )

    model_name = router.model_list[0]["litellm_params"]["model"]
    assert router.fail_calls[model_name] == 1, (
        "fail_calls must be incremented when the deferred HTTP call fails; "
        "without the eager fetch_stream() fix this stays at 0"
    )


@pytest.mark.asyncio
async def test_acompletion_deferred_stream_preserves_original_headers_on_error():
    """Router is used both by the proxy and directly as an SDK. HTTP-framing headers
    (Content-Length, Transfer-Encoding, ...) must NOT be stripped at this layer, or
    direct SDK callers lose legitimate provider metadata (e.g. content-type,
    proxy-authenticate) that only the proxy's own response construction needs to
    worry about. Stripping happens in the proxy layer instead
    (_handle_llm_api_exception)."""
    import litellm as _litellm

    err = _litellm.RateLimitError(
        message="Resource exhausted",
        llm_provider="vertex_ai",
        model="gemini-2.0-flash",
    )
    err.headers = {
        "content-length": "42",
        "transfer-encoding": "chunked",
        "content-encoding": "gzip",
        "content-type": "application/json",
        "x-request-id": "abc-123",
    }

    async def failing_make_call(**kwargs):
        raise err

    router = _make_router_with_vertex_and_fallback()
    deferred_wrapper = _make_deferred_stream_wrapper(failing_make_call)

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=deferred_wrapper,
    ):
        with pytest.raises(_litellm.RateLimitError) as exc_info:
            await router._acompletion(
                model="vertex_ai/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                specific_deployment=router.model_list[0],
            )

    raised = exc_info.value
    headers = getattr(raised, "headers", {})
    assert headers.get("content-length") == "42"
    assert headers.get("transfer-encoding") == "chunked"
    assert headers.get("content-encoding") == "gzip"
    assert headers.get("content-type") == "application/json"
    assert headers.get("x-request-id") == "abc-123"


@pytest.mark.asyncio
async def test_acompletion_deferred_stream_skipped_when_stream_already_set():
    """When completion_stream is already populated (non-deferred provider), the eager
    fetch_stream() call must be skipped entirely; no exception should be raised even
    if make_call would fail.
    """
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    async def would_fail(**kwargs):
        raise RuntimeError("should not be called")

    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}

    async def noop_aiter():
        return
        yield

    noop_stream = noop_aiter()
    already_set_wrapper = CustomStreamWrapper(
        completion_stream=noop_stream,
        model="openai/gpt-4o",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        make_call=would_fail,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "api_key": "sk-fake",
                },
            }
        ],
    )

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
        return_value=already_set_wrapper,
    ):
        result = await router._acompletion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            specific_deployment=router.model_list[0],
        )

    assert result is not None, "should return a streaming wrapper without errors"
    assert already_set_wrapper.completion_stream is noop_stream, "completion_stream must not be re-fetched"
    await noop_stream.aclose()


def test_completion_deferred_stream_error_propagates_through_completion():
    """Regression: the sync router path needs the same eager fetch as the async one.

    A deferred-stream CustomStreamWrapper hands back a wrapper whose HTTP call has
    not happened yet, so without fetch_sync_stream() the provider error surfaces on
    first iteration, outside _completion's except block. The deployment is then never
    marked failed and function_with_fallbacks never sees the error.
    """
    import litellm as _litellm

    rate_limit_err = _litellm.RateLimitError(
        message="Resource exhausted",
        llm_provider="vertex_ai",
        model="gemini-2.0-flash",
    )
    make_call_invocations = []

    def failing_make_call(**kwargs):
        make_call_invocations.append(kwargs)
        raise rate_limit_err

    router = _make_router_with_vertex_and_fallback()
    deferred_wrapper = _make_deferred_stream_wrapper(failing_make_call)

    with patch("litellm.completion", return_value=deferred_wrapper):
        with pytest.raises(_litellm.RateLimitError):
            router._completion(
                model="vertex_ai/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                specific_deployment=router.model_list[0],
            )

    assert len(make_call_invocations) == 1, (
        "the deferred HTTP call must run inside _completion's try block; "
        "without the eager fetch_sync_stream() fix it is deferred to first iteration"
    )


def test_completion_deferred_stream_skipped_when_stream_already_set():
    """A non-deferred sync provider already has completion_stream populated, so the
    eager fetch must be skipped and make_call left untouched.
    """
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

    def would_fail(**kwargs):
        raise RuntimeError("should not be called")

    logging_obj = MagicMock()
    logging_obj.model_call_details = {"litellm_params": {}}
    already_set_stream = iter([])

    already_set_wrapper = CustomStreamWrapper(
        completion_stream=already_set_stream,
        model="openai/gpt-4o",
        logging_obj=logging_obj,
        custom_llm_provider="openai",
        make_call=would_fail,
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-model",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-fake"},
            }
        ],
    )

    with patch("litellm.completion", return_value=already_set_wrapper):
        result = router._completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            specific_deployment=router.model_list[0],
        )

    assert result is not None, "should return a streaming wrapper without errors"
    assert already_set_wrapper.completion_stream is already_set_stream, "completion_stream must not be re-fetched"


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


def test_stream_chunks_have_generated_content_detects_text_and_non_text():
    from litellm.router import _stream_chunks_have_generated_content
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        StreamingChoices,
    )

    def _chunk(delta):
        return litellm.ModelResponseStream(
            id="chatcmpl-1",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[StreamingChoices(finish_reason=None, index=0, delta=delta)],
        )

    assert _stream_chunks_have_generated_content([]) is False

    empty_chunk = _chunk(Delta(role="assistant"))
    assert _stream_chunks_have_generated_content([empty_chunk]) is False

    text_chunk = _chunk(Delta(content="Hello"))
    assert _stream_chunks_have_generated_content([text_chunk]) is True

    reasoning_chunk = _chunk(Delta(reasoning_content="Thinking"))
    assert _stream_chunks_have_generated_content([reasoning_chunk]) is True

    tool_call_delta = Delta(
        tool_calls=[
            ChatCompletionDeltaToolCall(
                id="call_1",
                function=Function(name="get_weather", arguments="{}"),
                type="function",
                index=0,
            )
        ]
    )
    tool_call_chunk = _chunk(tool_call_delta)
    assert _stream_chunks_have_generated_content([tool_call_chunk]) is True

    thinking_delta = Delta(thinking_blocks=[{"type": "thinking", "thinking": "Let me think..."}])
    thinking_chunk = _chunk(thinking_delta)
    assert _stream_chunks_have_generated_content([thinking_chunk]) is True

    reasoning_items_delta = Delta(reasoning_items=[{"type": "reasoning", "id": "rs_1"}])
    reasoning_items_chunk = _chunk(reasoning_items_delta)
    assert _stream_chunks_have_generated_content([reasoning_items_chunk]) is True

    audio_delta = Delta(audio={"data": "abc123", "expires_at": 1234567890, "transcript": "hello"})
    audio_chunk = _chunk(audio_delta)
    assert _stream_chunks_have_generated_content([audio_chunk]) is True

    images_delta = Delta(images=[{"image_url": {"url": "https://example.com/img.png"}, "index": 0, "type": "image_url"}])
    images_chunk = _chunk(images_delta)
    assert _stream_chunks_have_generated_content([images_chunk]) is True

    annotations_delta = Delta(
        annotations=[{"type": "url_citation", "url_citation": {"url": "https://example.com"}}]
    )
    annotations_chunk = _chunk(annotations_delta)
    assert _stream_chunks_have_generated_content([annotations_chunk]) is True


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


def test_get_configured_token_limits_treats_malformed_values_as_absent():
    malformed = ["", "unlimited", "128,000", [128000], {"max": 128000}, True]
    router = litellm.Router(
        model_list=[
            {
                "model_name": f"bad-limit-{i}",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"max_input_tokens": bad, "max_output_tokens": bad},
            }
            for i, bad in enumerate(malformed)
        ]
    )

    for i in range(len(malformed)):
        assert router.get_configured_token_limits(f"bad-limit-{i}") == (None, None)


def test_get_configured_token_limits_coerces_numeric_strings():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "quoted-limits-model",
                "litellm_params": {"model": "openai/some-unmapped-model"},
                "model_info": {"max_input_tokens": "32000", "max_output_tokens": "8000"},
            }
        ]
    )

    assert router.get_configured_token_limits("quoted-limits-model") == (32000, 8000)


@pytest.mark.asyncio
async def test_acreate_batch_disable_fallbacks_surfaces_owning_provider_error():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "owning-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-owning",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_key": "sk-fallback",
                    "api_base": "https://fallback.openai.azure.com",
                    "api_version": "2024-08-01-preview",
                },
            },
        ],
        fallbacks=[{"owning-model": ["fallback-model"]}],
        num_retries=0,
    )
    owning_provider_error = litellm.BadRequestError(
        message="completion_window must be one of: 24h",
        model="openai/gpt-4o-mini",
        llm_provider="openai",
    )
    mock_create = AsyncMock(side_effect=owning_provider_error)

    with patch.object(router, "_acreate_batch", mock_create):
        with pytest.raises(litellm.BadRequestError, match="24h"):
            await router.acreate_batch(
                model="owning-model",
                input_file_id="file-owned-by-openai",
                endpoint="/v1/chat/completions",
                completion_window="5m",
                disable_fallbacks=True,
            )

    mock_create.assert_awaited_once()
    assert mock_create.call_args.kwargs["model"] == "owning-model"


@pytest.mark.asyncio
async def test_acreate_batch_surfaces_owning_provider_error_without_disable_fallbacks():
    """The router itself has to keep a batch inside the group that owns the input file:
    the proxy only sets disable_fallbacks on the managed-files route, so the caller
    otherwise gets the fallback provider's error for a file it never received."""
    from litellm.types.utils import LiteLLMBatch

    router = litellm.Router(
        model_list=[
            {
                "model_name": "owning-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-owning",
                },
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_key": "sk-fallback",
                    "api_base": "https://fallback.openai.azure.com",
                    "api_version": "2024-08-01-preview",
                },
            },
        ],
        fallbacks=[{"owning-model": ["fallback-model"]}],
        num_retries=0,
    )
    attempted_models = []

    async def _acreate_batch(model, **kwargs):
        attempted_models.append(model)
        if model == "owning-model":
            raise litellm.APIConnectionError(
                message="Connection error - openai is unreachable",
                model="openai/gpt-4o-mini",
                llm_provider="openai",
            )
        return LiteLLMBatch(
            id="batch-created-on-the-wrong-provider",
            completion_window="24h",
            created_at=0,
            endpoint="/v1/chat/completions",
            input_file_id="file-owned-by-openai",
            object="batch",
            status="validating",
        )

    with patch.object(router, "_acreate_batch", _acreate_batch):
        with pytest.raises(litellm.APIConnectionError, match="openai is unreachable"):
            await router.acreate_batch(
                model="owning-model",
                input_file_id="file-owned-by-openai",
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"team": "batch-jobs"},
            )

    assert attempted_models == ["owning-model"]


@pytest.mark.asyncio
async def test_acreate_batch_still_falls_back_within_the_owning_model_group():
    """Holding a batch inside the model group that owns its input file must not
    disable fallbacks outright (#35359): the owning group's second deployment is
    still tried in `order`, and only the cross-group target is skipped."""
    completion_window_error = "Invalid value: '5m'. Supported values are: '24h'."
    attempted_models = []

    async def _acreate_batch(**kwargs):
        model = kwargs["model"]
        attempted_models.append(model)
        if model.startswith("azure/"):
            raise litellm.BadRequestError(
                message="Error code: 400 - {'error': {'code': 'quotaExceeded'}}",
                model=model,
                llm_provider="azure",
            )
        raise litellm.BadRequestError(
            message=completion_window_error,
            model=model,
            llm_provider="openai",
        )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "my-gpt",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-owning",
                    "order": 1,
                },
                "model_info": {"id": "my-gpt-1"},
            },
            {
                "model_name": "my-gpt",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini-backup",
                    "api_key": "sk-owning",
                    "order": 2,
                },
                "model_info": {"id": "my-gpt-2"},
            },
            {
                "model_name": "my-azure-gpt",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_key": "sk-fallback",
                    "api_base": "https://fallback.openai.azure.com",
                    "api_version": "2024-08-01-preview",
                },
                "model_info": {"id": "my-azure-gpt-1"},
            },
        ],
        fallbacks=[{"my-gpt": ["my-azure-gpt"]}],
        num_retries=0,
    )

    with patch.object(litellm, "acreate_batch", new=_acreate_batch):
        with pytest.raises(litellm.BadRequestError) as raised:
            await router.acreate_batch(
                model="my-gpt",
                input_file_id="file-owned-by-my-gpt",
                endpoint="/v1/chat/completions",
                completion_window="5m",
            )

    assert "24h" in str(raised.value)
    assert "quotaExceeded" not in str(raised.value)
    assert attempted_models == ["openai/gpt-4o-mini", "openai/gpt-4o-mini-backup"]


@pytest.mark.asyncio
async def test_acreate_batch_request_bedrock_tags_override_deployment_tags():
    import httpx

    from litellm.llms.bedrock.common_utils import CommonBatchFilesUtils

    deployment_tags = [{"key": "application", "value": "config-level"}]
    request_tags = [{"key": "application", "value": "request-level"}]
    router = litellm.Router(
        model_list=[
            {
                "model_name": "bedrock-batch-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-sonnet-5",
                    "aws_batch_role_arn": "arn:aws:iam::123:role/batch-role",
                    "aws_region_name": "us-west-2",
                    "bedrock_tags": deployment_tags,
                },
            }
        ]
    )

    def fake_response():
        return httpx.Response(
            status_code=200,
            json={
                "jobArn": "arn:aws:bedrock:us-west-2:123:model-invocation-job/abc1234567",
                "status": "Submitted",
            },
        )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=lambda *args, **kwargs: fake_response())

    with patch.object(
        CommonBatchFilesUtils,
        "sign_aws_request",
        return_value=({"Authorization": "signed"}, b"{}"),
    ) as mock_sign, patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
        return_value=mock_client,
    ):
        await router.acreate_batch(
            model="bedrock-batch-model",
            input_file_id="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        assert mock_sign.call_args.kwargs["data"]["tags"] == deployment_tags

        await router.acreate_batch(
            model="bedrock-batch-model",
            input_file_id="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            completion_window="24h",
            bedrock_tags=request_tags,
        )
        assert mock_sign.call_args.kwargs["data"]["tags"] == request_tags


class TestPreRoutingStrategyRegistryLifecycle:
    """
    Regression tests: a deployment leaving the model_list must release the
    pre-routing strategy slot it holds in `auto_routers` / `complexity_routers` /
    `adaptive_routers` / `quality_routers`.

    Before this fix, editing an auto-router-family model (a UI save, which reaches
    every other pod as an `upsert_deployment` from the periodic DB reload) popped
    the deployment out of the model_list and then failed to re-add it: registration
    raised "already exists" against the stale registry entry, and
    `ignore_invalid_deployments=True` swallowed the error. The router vanished from
    the Models page and stayed gone until a proxy restart, while the DB row and the
    "saved successfully" response both looked fine.
    """

    @staticmethod
    def _complexity_router_params(default_model: str, tags=None) -> dict:
        return {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o", "COMPLEX": "gpt-4o"}
            },
            "complexity_router_default_model": default_model,
            **({"tags": tags} if tags else {}),
        }

    @classmethod
    def _router_with_complexity_router(cls, default_model: str = "gpt-4o") -> "litellm.Router":
        return litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                {
                    "model_name": "smart-router",
                    "litellm_params": cls._complexity_router_params(default_model),
                    "model_info": {"id": "router-1", "db_model": True},
                },
            ],
            ignore_invalid_deployments=True,
        )

    @staticmethod
    def _model_names(router: "litellm.Router") -> list:
        return [model["model_name"] for model in router.model_list]

    def test_upsert_of_edited_router_keeps_it_routable(self):
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_complexity_router()

        router.upsert_deployment(
            deployment=Deployment(
                model_name="smart-router",
                litellm_params=LiteLLM_Params(**self._complexity_router_params("gpt-4o-mini")),
                model_info=ModelInfo(id="router-1", db_model=True),
            )
        )

        assert "smart-router" in self._model_names(router)
        registered = router.complexity_routers["smart-router"]
        assert len(registered) == 1
        # the surviving strategy is the edited one, not the pre-edit leftover
        assert registered[0].strategy.config.default_model == "gpt-4o-mini"

    def test_unchanged_upsert_leaves_router_untouched(self):
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_complexity_router()
        strategy_before = router.complexity_routers["smart-router"][0].strategy

        for _ in range(3):
            router.upsert_deployment(
                deployment=Deployment(
                    model_name="smart-router",
                    litellm_params=LiteLLM_Params(**self._complexity_router_params("gpt-4o")),
                    model_info=ModelInfo(id="router-1", db_model=True),
                )
            )

        assert "smart-router" in self._model_names(router)
        assert router.complexity_routers["smart-router"][0].strategy is strategy_before

    def test_delete_frees_the_name_for_a_new_router(self):
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_complexity_router()

        router.delete_deployment(id="router-1")
        assert "smart-router" not in router.complexity_routers

        router.add_deployment(
            deployment=Deployment(
                model_name="smart-router",
                litellm_params=LiteLLM_Params(**self._complexity_router_params("gpt-4o-mini")),
                model_info=ModelInfo(id="router-2", db_model=True),
            )
        )

        assert "smart-router" in self._model_names(router)
        assert router.complexity_routers["smart-router"][0].strategy.config.default_model == "gpt-4o-mini"

    def test_delete_only_frees_the_matching_tag_slot(self):
        router = litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                {
                    "model_name": "shared-router",
                    "litellm_params": self._complexity_router_params("gpt-4o", tags=["team-a"]),
                    "model_info": {"id": "router-a"},
                },
                {
                    "model_name": "shared-router",
                    "litellm_params": self._complexity_router_params("gpt-4o-mini", tags=["team-b"]),
                    "model_info": {"id": "router-b"},
                },
            ],
            ignore_invalid_deployments=True,
        )
        assert len(router.complexity_routers["shared-router"]) == 2

        router.delete_deployment(id="router-a")

        remaining = router.complexity_routers["shared-router"]
        assert len(remaining) == 1
        assert remaining[0].tags == ("team-b",)

    def test_delete_of_regular_model_preserves_router_sharing_its_name(self):
        router = litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "gpt-4o-mini"}},
                {
                    "model_name": "shared-name",
                    "litellm_params": self._complexity_router_params("gpt-4o"),
                    "model_info": {"id": "router-1"},
                },
                {
                    "model_name": "shared-name",
                    "litellm_params": {"model": "openai/gpt-4o"},
                    "model_info": {"id": "regular-1"},
                },
            ],
            ignore_invalid_deployments=True,
        )
        strategy = router.complexity_routers["shared-name"][0].strategy

        router.delete_deployment(id="regular-1")

        assert router.complexity_routers["shared-name"][0].strategy is strategy

    def test_upsert_of_edited_adaptive_router_rebuilds_it(self):
        """Adaptive routers are built by set_model_list()'s deferred pass, not by
        add_deployment(), so releasing the slot on edit must be paired with a rebuild -
        otherwise the edit silently turns adaptive routing off."""
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        def adaptive_params(available_models: list) -> dict:
            return {
                "model": "auto_router/adaptive_router",
                "adaptive_router_config": {"available_models": available_models},
            }

        router = litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}},
                {
                    "model_name": "adaptive-router",
                    "litellm_params": adaptive_params(["gpt-4o-mini"]),
                    "model_info": {"id": "router-1", "db_model": True},
                },
            ],
            ignore_invalid_deployments=True,
        )
        assert "adaptive-router" in router.adaptive_routers

        router.upsert_deployment(
            deployment=Deployment(
                model_name="adaptive-router",
                litellm_params=LiteLLM_Params(**adaptive_params(["gpt-4o", "gpt-4o-mini"])),
                model_info=ModelInfo(id="router-1", db_model=True),
            )
        )

        assert "adaptive-router" in self._model_names(router)
        registered = router.adaptive_routers["adaptive-router"]
        assert len(registered) == 1
        assert set(registered[0].strategy.config.available_models) == {"gpt-4o", "gpt-4o-mini"}

    def test_delete_repairs_indices_even_when_strategy_release_fails(self):
        """Structural removal and strategy release are not equally critical. Once the entry
        leaves model_list the index maps must be repaired no matter what, so releasing the
        registry slot runs after that repair and cannot abandon the router half-updated."""
        router = self._router_with_complexity_router()
        idx = router.model_id_to_deployment_index_map["router-1"]
        router.model_list[idx] = {"model_name": "smart-router", "litellm_params": None}

        returned = router.delete_deployment(id="router-1")

        assert returned is not None
        assert "router-1" not in router.model_id_to_deployment_index_map
        assert all(entry.get("model_info", {}).get("id") != "router-1" for entry in router.model_list)
        assert router.get_deployment(model_id="router-1") is None
        assert "gpt-4o" in self._model_names(router)

    def test_delete_of_adaptive_enabled_complexity_router_frees_both_registries(self):
        """A complexity router with adaptive set is registered in BOTH complexity_routers
        and adaptive_routers under the same (model_name, tags). Releasing only the first
        match leaves the adaptive strategy live, so a deleted alias stays routable and its
        post-call hook keeps recording."""
        import litellm as litellm_module
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

        params = {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o"},
                "adaptive": True,
            },
            "complexity_router_default_model": "gpt-4o",
        }
        router = litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}},
                {
                    "model_name": "hybrid-router",
                    "litellm_params": params,
                    "model_info": {"id": "router-1", "db_model": True},
                },
            ],
            ignore_invalid_deployments=True,
        )
        assert "hybrid-router" in router.complexity_routers
        assert "hybrid-router" in router.adaptive_routers
        hooks = litellm_module.logging_callback_manager.get_custom_loggers_for_type(AdaptiveRouterPostCallHook)
        assert len(hooks) == 1

        router.delete_deployment(id="router-1")

        assert "hybrid-router" not in router.complexity_routers
        assert "hybrid-router" not in router.adaptive_routers
        remaining_hooks = litellm_module.logging_callback_manager.get_custom_loggers_for_type(
            AdaptiveRouterPostCallHook
        )
        assert remaining_hooks == []

    def test_upsert_of_edited_quality_router_keeps_it_routable(self):
        """_unregister_pre_routing_strategy_for_deployment dispatches on four prefixes;
        quality_router is one of them and would otherwise go unexercised."""
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        def quality_params(default_model: str) -> dict:
            return {
                "model": "auto_router/quality_router",
                "quality_router_default_model": default_model,
            }

        router = litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}},
                {
                    "model_name": "quality-router",
                    "litellm_params": quality_params("gpt-4o"),
                    "model_info": {"id": "router-1", "db_model": True},
                },
            ],
            ignore_invalid_deployments=True,
        )
        assert "quality-router" in router.quality_routers

        router.upsert_deployment(
            deployment=Deployment(
                model_name="quality-router",
                litellm_params=LiteLLM_Params(**quality_params("gpt-4o-mini")),
                model_info=ModelInfo(id="router-1", db_model=True),
            )
        )

        assert "quality-router" in self._model_names(router)
        registered = router.quality_routers["quality-router"]
        assert len(registered) == 1
        assert registered[0].strategy.config.default_model == "gpt-4o-mini"

    @staticmethod
    def _hybrid_router_params(tiers: dict) -> dict:
        return {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": tiers, "adaptive": True},
            "complexity_router_default_model": "gpt-4o",
        }

    @classmethod
    def _router_with_hybrid_router(cls) -> "litellm.Router":
        return litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "gpt-4o-mini", "litellm_params": {"model": "openai/gpt-4o-mini"}},
                {
                    "model_name": "hybrid-router",
                    "litellm_params": cls._hybrid_router_params({"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o"}),
                    "model_info": {"id": "router-1", "db_model": True},
                },
            ],
            ignore_invalid_deployments=True,
        )

    def test_upsert_of_edited_hybrid_complexity_router_relinks_adaptive(self):
        """Editing an adaptive-enabled complexity router releases its adaptive companion
        along with the complexity slot; the finalize re-run must fire for it (not just for
        `auto_router/adaptive_router` deployments) or the rebuilt complexity router keeps
        routing while bandit recording, DB persistence and /adaptive_router/state all
        silently stop until the next full reload."""
        import litellm as litellm_module
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_hybrid_router()
        assert "hybrid-router" in router.adaptive_routers

        router.upsert_deployment(
            deployment=Deployment(
                model_name="hybrid-router",
                litellm_params=LiteLLM_Params(
                    **self._hybrid_router_params(
                        {"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o", "COMPLEX": "gpt-4o"}
                    )
                ),
                model_info=ModelInfo(id="router-1", db_model=True),
            )
        )

        assert "hybrid-router" in self._model_names(router)
        assert "hybrid-router" in router.complexity_routers
        assert "hybrid-router" in router.adaptive_routers
        rebuilt = router.complexity_routers["hybrid-router"][0].strategy
        assert router.adaptive_routers["hybrid-router"][0].strategy is rebuilt.adaptive_router
        hooks = litellm_module.logging_callback_manager.get_custom_loggers_for_type(AdaptiveRouterPostCallHook)
        assert len(hooks) == 1

    def test_upsert_turning_adaptive_on_builds_the_companion(self):
        """An edit that flips `adaptive: true` on an existing complexity router must
        register the companion immediately; neither side of the old prefix-only gate
        matches a complexity deployment, so the flip was a silent no-op until restart."""
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_complexity_router()
        assert "smart-router" not in router.adaptive_routers

        params = self._complexity_router_params("gpt-4o")
        params["complexity_router_config"] = {**params["complexity_router_config"], "adaptive": True}
        router.upsert_deployment(
            deployment=Deployment(
                model_name="smart-router",
                litellm_params=LiteLLM_Params(**params),
                model_info=ModelInfo(id="router-1", db_model=True),
            )
        )

        assert "smart-router" in router.adaptive_routers

    def test_unregister_pre_routing_strategy_scopes_the_drop_by_tags(self):
        """The bool return drives the hook re-sync; a tag mismatch must report False and
        leave the registry untouched, and dropping the last entry must free the key."""
        from litellm.types.router import TaggedPreRoutingStrategy

        registry = {
            "m": [
                TaggedPreRoutingStrategy(tags=("team-a",), strategy=object()),
                TaggedPreRoutingStrategy(tags=(), strategy=object()),
            ]
        }

        assert litellm.Router._unregister_pre_routing_strategy(registry, "m", ("team-b",)) is False
        assert len(registry["m"]) == 2

        assert litellm.Router._unregister_pre_routing_strategy(registry, "m", ("team-a",)) is True
        assert [entry.tags for entry in registry["m"]] == [()]

        assert litellm.Router._unregister_pre_routing_strategy(registry, "m", ()) is True
        assert "m" not in registry

    def test_unregister_for_deployment_ignores_non_router_deployments(self):
        """Direct twin of the endpoint-level test: a regular deployment that shares a
        router's model_name must not evict the router's registry slot."""
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = self._router_with_complexity_router()

        router._unregister_pre_routing_strategy_for_deployment(
            deployment=Deployment(
                model_name="smart-router",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
                model_info=ModelInfo(id="plain-1", db_model=True),
            )
        )

        assert "smart-router" in router.complexity_routers

    def test_sync_adaptive_router_hooks_keeps_one_hook_per_registered_router(self):
        """Re-syncing must replace, not accumulate: a duplicated hook double-fires
        bandit signal recording for every request."""
        import litellm as litellm_module
        from litellm.router_strategy.adaptive_router.hooks import AdaptiveRouterPostCallHook

        router = self._router_with_hybrid_router()

        router._sync_adaptive_router_hooks()
        router._sync_adaptive_router_hooks()

        hooks = litellm_module.logging_callback_manager.get_custom_loggers_for_type(AdaptiveRouterPostCallHook)
        assert len(hooks) == 1

    def test_deployment_participates_in_adaptive_routing_matrix(self):
        """The upsert finalize re-run keys off this predicate for both the incoming and
        outgoing deployment; a false negative silently strands the adaptive companion."""
        from litellm.types.router import LiteLLM_Params

        router = self._router_with_complexity_router()

        cases = [
            ({"model": "auto_router/adaptive_router", "adaptive_router_config": {}}, True),
            (self._hybrid_router_params({"SIMPLE": "gpt-4o-mini"}), True),
            (self._complexity_router_params("gpt-4o"), False),
            (
                {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {"tiers": {"SIMPLE": "gpt-4o-mini"}, "adaptive": False},
                    "complexity_router_default_model": "gpt-4o",
                },
                False,
            ),
            ({"model": "openai/gpt-4o"}, False),
        ]
        for params, expected in cases:
            actual = router._deployment_participates_in_adaptive_routing(
                litellm_params=LiteLLM_Params(**params)
            )
            assert actual is expected, params["model"]


def test_model_info_is_active_for_environment_matrix(monkeypatch):
    """The model-write endpoints consult this predicate to tell a deliberately
    environment-inactive model from one dropped by a failed reload; the Router's own
    deployment gate delegates to it, so the two can never diverge."""
    from litellm.router import model_info_is_active_for_environment

    assert model_info_is_active_for_environment(model_info=None) is True
    assert model_info_is_active_for_environment(model_info={"id": "m1"}) is True
    assert model_info_is_active_for_environment(model_info={"supported_environments": None}) is True

    monkeypatch.setenv("LITELLM_ENVIRONMENT", "development")
    assert model_info_is_active_for_environment(model_info={"supported_environments": ["development"]}) is True
    assert model_info_is_active_for_environment(model_info={"supported_environments": ["production"]}) is False

    monkeypatch.delenv("LITELLM_ENVIRONMENT")
    with pytest.raises(ValueError, match="LITELLM_ENVIRONMENT"):
        model_info_is_active_for_environment(model_info={"supported_environments": ["production"]})


def test_pre_call_checks_uses_deployment_model_when_model_info_lookup_raises(monkeypatch):
    """
    The supported-params check must run against the deployment's own
    provider-qualified model. Resolving the per-deployment model only after the
    model-info lookup leaves it unset whenever that lookup raises (an
    unregistered custom model), so the check falls back to the bare model group
    name and the request dies with 'LLM Provider NOT provided'.
    """
    monkeypatch.setattr(litellm, "drop_params", False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "custom-alias",
                "litellm_params": {"model": "hosted_vllm/not-in-the-catalog"},
            }
        ],
        enable_pre_call_checks=True,
    )

    def _raise_unmapped(**kwargs):
        raise ValueError("This model isn't mapped yet")

    monkeypatch.setattr(router, "get_router_model_info", _raise_unmapped)

    seen: list[tuple] = []
    original_get_supported_openai_params = litellm.get_supported_openai_params

    def _record(model, custom_llm_provider=None, **kwargs):
        seen.append((model, custom_llm_provider))
        return original_get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider, **kwargs)

    monkeypatch.setattr(litellm, "get_supported_openai_params", _record)

    deployments = [
        {
            "litellm_params": {"model": "hosted_vllm/not-in-the-catalog"},
            "model_info": {"id": "d1"},
        }
    ]
    result = router._pre_call_checks(
        model="custom-alias",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs={},
    )

    assert len(result) == 1
    assert seen == [("not-in-the-catalog", "hosted_vllm")]


def test_pre_call_checks_keeps_deployment_when_provider_is_unresolvable(monkeypatch):
    """
    Pre-call checks filter deployments; they must never be the thing that fails
    a request. A deployment whose provider cannot be resolved simply skips the
    supported-params check instead of raising out of deployment selection.
    """
    monkeypatch.setattr(litellm, "drop_params", False)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "custom-alias",
                "litellm_params": {"model": "gpt-3.5-turbo"},
            }
        ],
        enable_pre_call_checks=True,
    )

    def _raise_no_provider(**kwargs):
        raise litellm.BadRequestError(
            message="LLM Provider NOT provided.",
            model="custom-alias",
            llm_provider="",
        )

    monkeypatch.setattr(litellm, "get_llm_provider", _raise_no_provider)

    deployments = [
        {
            "litellm_params": {"model": "some-unresolvable-model"},
            "model_info": {"id": "d1"},
        }
    ]
    result = router._pre_call_checks(
        model="custom-alias",
        healthy_deployments=deployments,
        messages=[{"role": "user", "content": "hi"}],
        request_kwargs={},
    )

    assert len(result) == 1


class TestUpsertDeploymentRollback:
    """
    Regression tests: `upsert_deployment` pops the previous deployment before
    re-adding the edited one. When the re-add raises under
    `ignore_invalid_deployments=True`, the pop must be rolled back so this pod
    keeps serving the previous configuration instead of silently dropping a live
    deployment (the "Error upserting deployment" drop behind the access-group
    reload 500 in the 2-replica e2e suite).
    """

    def test_failed_upsert_keeps_previous_deployment_serving(self):
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = litellm.Router(
            model_list=[
                {
                    "model_name": "prod-model",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-old"},
                    "model_info": {"id": "prod-1", "db_model": True},
                }
            ],
            ignore_invalid_deployments=True,
        )

        result = router.upsert_deployment(
            deployment=Deployment(
                model_name="prod-model",
                litellm_params=LiteLLM_Params(model="auto_router/broken"),
                model_info=ModelInfo(id="prod-1", db_model=True),
            )
        )

        assert result is None
        restored = router.get_deployment(model_id="prod-1")
        assert restored is not None
        assert restored.litellm_params.model == "gpt-4o"
        assert [model["model_name"] for model in router.model_list] == ["prod-model"]

    def test_failed_fresh_add_returns_none_without_restore(self):
        from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

        router = litellm.Router(model_list=[], ignore_invalid_deployments=True)

        result = router.upsert_deployment(
            deployment=Deployment(
                model_name="fresh-router",
                litellm_params=LiteLLM_Params(model="auto_router/broken"),
                model_info=ModelInfo(id="fresh-1", db_model=True),
            )
        )

        assert result is None
        assert router.get_deployment(model_id="fresh-1") is None
        assert router.model_list == []

    def test_restore_re_adds_popped_deployment(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "prod-model",
                    "litellm_params": {"model": "gpt-4o", "api_key": "sk-old"},
                    "model_info": {"id": "prod-1", "db_model": True},
                }
            ],
            ignore_invalid_deployments=True,
        )
        previous = router.get_deployment(model_id="prod-1")
        router.delete_deployment(id="prod-1")
        assert router.has_model_id("prod-1") is False

        router._restore_deployment_after_failed_upsert(
            previous_deployment=previous, model_id="prod-1"
        )

        restored = router.get_deployment(model_id="prod-1")
        assert restored is not None
        assert restored.litellm_params.model == "gpt-4o"

        router._restore_deployment_after_failed_upsert(
            previous_deployment=previous, model_id="prod-1"
        )
        assert len(router.model_list) == 1

        router._restore_deployment_after_failed_upsert(
            previous_deployment=None, model_id="prod-1"
        )
        assert len(router.model_list) == 1


class TestConsumedRequestTagsStamp:
    """Issue #36621: when a request's tags select a tagged pre-routing strategy, those
    tags are consumed by the selection; the hook must stamp the rewritten model group so
    tag filtering skips request-body tags there, and must clear the stamp on every
    re-entry (fallbacks reuse the same request_kwargs) so it cannot leak elsewhere."""

    class _RewriteStrategy:
        def __init__(self, rewrite_to: str):
            self.rewrite_to = rewrite_to

        async def async_pre_routing_hook(
            self, model, request_kwargs, messages=None, input=None, specific_deployment=False
        ):
            from litellm.types.router import PreRoutingHookResponse

            return PreRoutingHookResponse(model=self.rewrite_to, messages=messages)

    @classmethod
    def _router(cls, marker_tags=("route",)) -> "litellm.Router":
        from litellm.types.router import TaggedPreRoutingStrategy

        router = litellm.Router(
            model_list=[
                {"model_name": "gpt4o", "litellm_params": {"model": "openai/gpt-4o"}},
                {"model_name": "gemini-flash", "litellm_params": {"model": "gemini/gemini-3.6-flash"}},
            ],
            enable_tag_filtering=True,
        )
        router.auto_routers = {
            "gpt4o": [TaggedPreRoutingStrategy(tags=marker_tags, strategy=cls._RewriteStrategy("gemini-flash"))]
        }
        return router

    @pytest.mark.asyncio
    async def test_stamps_the_rewritten_group_when_request_tags_selected_the_router(self):
        from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY
        from litellm.types.router import ConsumedRequestTagsStamp

        router = self._router()
        request_kwargs = {"metadata": {"tags": ["route"]}}

        await router.async_pre_routing_hook(model="gpt4o", request_kwargs=request_kwargs)

        assert request_kwargs["metadata"][CONSUMED_REQUEST_TAGS_METADATA_KEY] == ConsumedRequestTagsStamp(
            model_group="gemini-flash", tags=("route",)
        )

    @pytest.mark.asyncio
    async def test_stamps_into_litellm_metadata_when_the_request_uses_that_bucket(self):
        from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY
        from litellm.types.router import ConsumedRequestTagsStamp

        router = self._router()
        request_kwargs = {"litellm_metadata": {"tags": ["route"]}}

        await router.async_pre_routing_hook(model="gpt4o", request_kwargs=request_kwargs)

        assert request_kwargs["litellm_metadata"][CONSUMED_REQUEST_TAGS_METADATA_KEY] == ConsumedRequestTagsStamp(
            model_group="gemini-flash", tags=("route",)
        )

    @pytest.mark.asyncio
    async def test_fallback_reentry_with_a_plain_group_clears_the_stale_stamp(self):
        from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY

        router = self._router()
        request_kwargs = {"metadata": {"tags": ["route"]}}

        await router.async_pre_routing_hook(model="gpt4o", request_kwargs=request_kwargs)
        await router.async_pre_routing_hook(model="gemini-flash", request_kwargs=request_kwargs)

        assert CONSUMED_REQUEST_TAGS_METADATA_KEY not in request_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_no_stamp_when_the_request_is_untagged(self):
        from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY

        router = self._router()
        request_kwargs = {"metadata": {}}

        await router.async_pre_routing_hook(model="gpt4o", request_kwargs=request_kwargs)

        assert CONSUMED_REQUEST_TAGS_METADATA_KEY not in request_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_no_stamp_when_the_selected_strategy_carries_no_tags(self):
        from litellm.constants import CONSUMED_REQUEST_TAGS_METADATA_KEY

        router = self._router(marker_tags=())
        request_kwargs = {"metadata": {"tags": ["route"]}}

        await router.async_pre_routing_hook(model="gpt4o", request_kwargs=request_kwargs)

        assert CONSUMED_REQUEST_TAGS_METADATA_KEY not in request_kwargs["metadata"]


class TestAutoRouterMaxInputCharsWiring:
    """`auto_router_max_input_chars` on the deployment has to reach the AutoRouter that embeds prompts.

    Without it the cap silently reverts to the default, so an operator whose embedding model has a
    512-token window cannot lower it and every long prompt falls back to the default model instead
    of being routed.
    """

    @staticmethod
    def _router(**extra_params) -> "litellm.Router":
        pytest.importorskip("semantic_router", reason="auto-router needs the semantic-router extra")
        return litellm.Router(
            model_list=[
                {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}},
                {
                    "model_name": "my-auto-router",
                    "litellm_params": {
                        "model": "auto_router/my-auto-router",
                        "auto_router_config": json.dumps(
                            {"routes": [{"name": "gpt-4o", "utterances": ["write me code"]}]}
                        ),
                        "auto_router_default_model": "gpt-4o",
                        "auto_router_embedding_model": "text-embedding-3-small",
                        **extra_params,
                    },
                },
            ]
        )

    @staticmethod
    def _registered_auto_router(router: "litellm.Router"):
        return router.auto_routers["my-auto-router"][0].strategy

    def test_should_pass_the_configured_cap_to_the_auto_router(self):
        router = self._router(auto_router_max_input_chars=512)

        assert self._registered_auto_router(router).max_input_chars == 512

    def test_should_fall_back_to_the_shared_default_when_the_deployment_omits_it(self):
        from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS

        router = self._router()

        assert self._registered_auto_router(router).max_input_chars == DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS


class TestTaggedAutoRouterOnSharedModelName:
    """A tagged auto-router marker sharing its model_name with a plain deployment must not
    capture requests whose tags don't match it when tag filtering is enabled (#36620)."""

    class _FixedRouteLayer:
        def __call__(self, text: str):
            from semantic_router.schema import RouteChoice

            return RouteChoice(name="gemini-flash")

    @classmethod
    def _router(cls, marker_tags, include_plain_sibling: bool, enable_tag_filtering: bool) -> "litellm.Router":
        pytest.importorskip("semantic_router", reason="auto-router needs the semantic-router extra")
        marker = {
            "model_name": "gpt4o",
            "litellm_params": {
                "model": "auto_router/gpt4o-router",
                "auto_router_config": json.dumps(
                    {"routes": [{"name": "gemini-flash", "utterances": ["capital city questions"]}]}
                ),
                "auto_router_default_model": "gemini-flash",
                "auto_router_embedding_model": "text-embedding-3-small",
                **({"tags": marker_tags} if marker_tags else {}),
            },
        }
        plain = {"model_name": "gpt4o", "litellm_params": {"model": "openai/gpt-4o"}}
        tier = {"model_name": "gemini-flash", "litellm_params": {"model": "gemini/gemini-3.6-flash"}}
        router = litellm.Router(
            model_list=[plain, marker, tier] if include_plain_sibling else [marker, tier],
            enable_tag_filtering=enable_tag_filtering,
        )
        router.auto_routers["gpt4o"][0].strategy.routelayer = cls._FixedRouteLayer()
        return router

    @staticmethod
    async def _hook_response(router: "litellm.Router", request_kwargs: dict):
        return await router.async_pre_routing_hook(
            model="gpt4o",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    @pytest.mark.asyncio
    async def test_untagged_request_bypasses_the_tagged_marker_when_a_plain_deployment_shares_the_name(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=True)

        assert await self._hook_response(router, {}) is None

    @pytest.mark.asyncio
    async def test_request_tagged_for_the_marker_is_still_semantically_routed(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=True)

        response = await self._hook_response(router, {"metadata": {"tags": ["route"]}})

        assert response is not None
        assert response.model == "gemini-flash"

    @pytest.mark.asyncio
    async def test_request_level_tag_filtering_from_key_settings_bypasses_the_marker(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=False)

        assert await self._hook_response(router, {"enable_tag_filtering": True}) is None

    @pytest.mark.asyncio
    async def test_globally_disabled_filtering_still_lets_the_marker_capture_untagged_requests(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=False)

        response = await self._hook_response(router, {})

        assert response is not None
        assert response.model == "gemini-flash"

    @pytest.mark.asyncio
    async def test_marker_only_alias_still_captures_untagged_requests(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=False, enable_tag_filtering=True)

        response = await self._hook_response(router, {})

        assert response is not None
        assert response.model == "gemini-flash"

    @pytest.mark.asyncio
    async def test_untagged_marker_sharing_the_name_still_captures_untagged_requests(self):
        router = self._router(marker_tags=None, include_plain_sibling=True, enable_tag_filtering=True)

        response = await self._hook_response(router, {})

        assert response is not None
        assert response.model == "gemini-flash"

    @pytest.mark.asyncio
    async def test_untagged_selection_never_lands_on_the_marker_deployment(self):
        router = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=True)

        for _ in range(20):
            deployment = await router.async_get_available_deployment(
                model="gpt4o",
                request_kwargs={},
                messages=[{"role": "user", "content": "What is the capital of France?"}],
            )
            assert deployment["litellm_params"]["model"] == "openai/gpt-4o"

    def test_deployment_without_litellm_params_mapping_is_not_a_marker(self):
        assert litellm.Router._is_strategy_marker_deployment({"model_name": "gpt4o"}) is False

    def test_model_name_has_plain_deployments_reflects_the_pool(self):
        mixed = self._router(marker_tags=["route"], include_plain_sibling=True, enable_tag_filtering=True)
        marker_only = self._router(marker_tags=["route"], include_plain_sibling=False, enable_tag_filtering=True)

        assert mixed._model_name_has_plain_deployments("gpt4o") is True
        assert marker_only._model_name_has_plain_deployments("gpt4o") is False


class TestAutoRouterSharedModelNameConnectionParams:
    """A plain deployment sharing its model_name with an `auto_router/` marker must not have
    its api_base and api_key grafted onto the routed tier's outbound call (#36619)."""

    PLAIN_API_BASE = "https://plain-sibling.openai.example/v1"
    PLAIN_API_KEY = "sk-plain-sibling-secret"

    class _FixedRouteLayer:
        def __call__(self, text: str):
            from semantic_router.schema import RouteChoice

            return RouteChoice(name="gemini-flash")

    @classmethod
    def _router(cls, plain_entry_first: bool) -> "litellm.Router":
        pytest.importorskip("semantic_router", reason="auto-router needs the semantic-router extra")
        plain = {
            "model_name": "gpt4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": cls.PLAIN_API_KEY,
                "api_base": cls.PLAIN_API_BASE,
            },
        }
        marker = {
            "model_name": "gpt4o",
            "litellm_params": {
                "model": "auto_router/gpt4o-router",
                "auto_router_config": json.dumps(
                    {"routes": [{"name": "gemini-flash", "utterances": ["capital city questions"]}]}
                ),
                "auto_router_default_model": "gemini-flash",
                "auto_router_embedding_model": "text-embedding-3-small",
                "drop_params": True,
            },
        }
        tier = {
            "model_name": "gemini-flash",
            "litellm_params": {"model": "gemini/gemini-3.6-flash", "api_key": "sk-tier-key"},
        }
        shared_name_entries = [plain, marker] if plain_entry_first else [marker, plain]
        router = litellm.Router(model_list=[*shared_name_entries, tier])
        router.auto_routers["gpt4o"][0].strategy.routelayer = cls._FixedRouteLayer()
        return router

    @staticmethod
    def _gemini_response() -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "Paris"}], "role": "model"}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
                "modelVersion": "gemini-3.6-flash",
            },
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

    @pytest.mark.parametrize(
        "plain_entry_first", [True, False], ids=["plain_entry_first", "marker_entry_first"]
    )
    async def test_routed_tier_call_goes_out_on_its_own_endpoint_and_credentials(self, plain_entry_first):
        """The outbound provider request for the routed tier hits the tier's own Gemini host
        with the tier's own key, never the plain sibling's api_base or api_key."""
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        router = self._router(plain_entry_first)

        with patch.object(
            AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=self._gemini_response()
        ) as mock_post:
            await router.acompletion(
                model="gpt4o",
                messages=[{"role": "user", "content": "What is the capital of France?"}],
            )

        call = mock_post.call_args
        outbound_url = str(call.kwargs["url"] if "url" in call.kwargs else call.args[0])
        outbound_headers = dict(call.kwargs.get("headers") or {})

        assert "generativelanguage.googleapis.com" in outbound_url
        assert "gemini-3.6-flash" in outbound_url
        assert self.PLAIN_API_BASE not in outbound_url
        assert self.PLAIN_API_KEY not in outbound_url
        assert self.PLAIN_API_KEY not in json.dumps(outbound_headers)


class TestGetAllowedFailsFromPolicy:
    def _make_router(self, **policy_kwargs) -> litellm.Router:
        from litellm.types.router import AllowedFailsPolicy

        return litellm.Router(
            model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4", "api_key": "fake"}}],
            allowed_fails_policy=AllowedFailsPolicy(**policy_kwargs),
        )

    def test_no_policy_returns_none(self):
        router = litellm.Router(
            model_list=[{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4", "api_key": "fake"}}],
        )
        assert router.get_allowed_fails_from_policy(litellm.RateLimitError("429", "openai", "gpt-4")) is None

    def test_internal_server_error_allowed_fails(self):
        router = self._make_router(InternalServerErrorAllowedFails=7)
        exc = litellm.InternalServerError("500", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 7

    def test_service_unavailable_error_allowed_fails(self):
        router = self._make_router(ServiceUnavailableErrorAllowedFails=4)
        exc = litellm.ServiceUnavailableError("503", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 4

    def test_bad_gateway_error_allowed_fails(self):
        router = self._make_router(BadGatewayErrorAllowedFails=2)
        exc = litellm.BadGatewayError("502", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 2

    def test_not_found_error_allowed_fails(self):
        router = self._make_router(NotFoundErrorAllowedFails=1)
        exc = litellm.NotFoundError("404", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) == 1

    def test_unmatched_exception_returns_none(self):
        router = self._make_router(InternalServerErrorAllowedFails=5)
        exc = litellm.RateLimitError("429", "openai", "gpt-4")
        assert router.get_allowed_fails_from_policy(exc) is None


class _LogCapture(logging.Handler):
    def __init__(self, level):
        super().__init__(level=level)
        self._level = level
        self.messages = []

    def emit(self, record):
        if record.levelno == self._level:
            self.messages.append(record.getMessage())


class _FallbackAttemptRecorder(CustomLogger):
    def __init__(self):
        super().__init__()
        self.failed_targets = []

    async def log_failure_fallback_event(self, original_model_group, kwargs, original_exception):
        self.failed_targets.append(kwargs.get("model"))


def _cyclic_fallback_router(num_retries=0):
    groups = ["group-a", "group-b", "group-c", "group-d"]
    return litellm.Router(
        model_list=[
            {
                "model_name": group,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-fake",
                    "mock_response": "litellm.InternalServerError",
                },
            }
            for group in groups
        ],
        fallbacks=[
            {"group-a": ["group-b", "group-c"]},
            {"group-b": ["group-a", "group-c"]},
            {"group-c": ["group-d"]},
            {"group-d": ["group-b", "group-a"]},
        ],
        num_retries=num_retries,
    )


async def _drive_cyclic_fallback(router, capture, recorder=None, **request_kwargs):
    router_logger = logging.getLogger("LiteLLM Router")
    previous_level = router_logger.level
    router_logger.setLevel(capture.level)
    router_logger.addHandler(capture)
    if recorder is not None:
        litellm.callbacks.append(recorder)
    try:
        with pytest.raises(litellm.InternalServerError):
            await router.acompletion(
                model="group-a", messages=[{"role": "user", "content": "hi"}], **request_kwargs
            )
    finally:
        router_logger.removeHandler(capture)
        router_logger.setLevel(previous_level)
        if recorder is not None:
            litellm.callbacks.remove(recorder)


@pytest.mark.asyncio
async def test_cyclic_fallback_graph_does_not_amplify_one_request():
    """A fallback graph whose entries loop back on each other is easy to build by accident,
    and every group in the loop fails identically on a deterministic error, so the walk must
    not revisit a group and must not re-emit a growing chained traceback at each level. Left
    unbounded, one request blocks the event loop long enough for health probes to fail."""
    recorder = _FallbackAttemptRecorder()
    capture = _LogCapture(logging.ERROR)

    await _drive_cyclic_fallback(_cyclic_fallback_router(), capture, recorder)

    assert sorted(set(recorder.failed_targets)) == ["group-b", "group-c", "group-d"]
    assert len(recorder.failed_targets) == len(set(recorder.failed_targets))
    assert not any("Traceback (most recent call last)" in message for message in capture.messages)
    assert sum(len(message) for message in capture.messages) < 5_000


@pytest.mark.asyncio
async def test_retry_breadcrumbs_do_not_carry_the_walk_state():
    """log_retry copies every kwarg into previous_models, which reaches spend logs and
    logging callbacks. The set of already-attempted groups is router-internal walk state
    with no diagnostic value there, and it is the one entry that is not a plain scalar.
    A retry has to be configured for the walk state to reach log_retry at all."""
    router = _cyclic_fallback_router(num_retries=1)
    capture = _LogCapture(logging.ERROR)

    await _drive_cyclic_fallback(router, capture)

    assert router.previous_models, "no retry breadcrumbs were recorded"
    assert any(
        "fallback_depth" in breadcrumb for breadcrumb in router.previous_models
    ), "no breadcrumb carried router walk state, so this test cannot see the leak"
    for breadcrumb in router.previous_models:
        assert "attempted_targets" not in breadcrumb


_BREADCRUMB_CREDENTIAL_CANARY = "Bearer sk-ant-oat01-RETRY-BREADCRUMB-CANARY-doNotShip"


@pytest.mark.parametrize(
    "container_key, request_kwargs",
    [
        (
            "provider_specific_header",
            {
                "provider_specific_header": {
                    "custom_llm_provider": "openai",
                    "extra_headers": {"authorization": _BREADCRUMB_CREDENTIAL_CANARY},
                }
            },
        ),
        (
            "extra_headers",
            {"extra_headers": {"authorization": _BREADCRUMB_CREDENTIAL_CANARY}},
        ),
        (
            "api_key",
            {"api_key": _BREADCRUMB_CREDENTIAL_CANARY},
        ),
    ],
)
@pytest.mark.asyncio
async def test_retry_breadcrumbs_never_carry_a_forwarded_credential(container_key, request_kwargs):
    """log_retry copies kwargs into previous_models, which reaches spend logs and logging callbacks.
    Any of these kwargs can carry a client's forwarded Authorization token or a provider key, and a
    breadcrumb has no diagnostic use for the raw secret. A denylist of key names is always one new
    credential kwarg behind, so log_retry scrubs credential-named values by pattern instead: the
    container still reaches the breadcrumb, but the raw secret never does, whatever key holds it."""
    router = _cyclic_fallback_router(num_retries=1)
    capture = _LogCapture(logging.ERROR)

    await _drive_cyclic_fallback(router, capture, **request_kwargs)

    assert router.previous_models, "no retry breadcrumbs were recorded"
    dumped = json.dumps(router.previous_models, default=str)
    assert container_key in dumped, "the credential-bearing kwarg never reached the breadcrumb, so this test cannot see the leak"
    assert _BREADCRUMB_CREDENTIAL_CANARY not in dumped


@pytest.mark.asyncio
async def test_fallback_traceback_stays_available_at_debug_level():
    """Dropping the stack from the ERROR line is only safe because the fallback path still
    emits it once per level at DEBUG, which is what an operator needs to diagnose why every
    fallback failed. This pins that remaining debug traceback."""
    capture = _LogCapture(logging.DEBUG)

    await _drive_cyclic_fallback(_cyclic_fallback_router(), capture)

    assert any("Traceback (most recent call last)" in message for message in capture.messages)


@pytest.mark.asyncio
async def test_fallback_failure_detail_from_upstream_is_bounded():
    """The detail each level records about the level below it is attacker-influenced, since
    it carries whatever the upstream error said. It has to be bounded on its own, so a walk
    over several groups cannot compound one large message into the log or into the message
    handed back to the caller."""
    huge_message = "z" * 50_000
    capture = _LogCapture(logging.ERROR)

    await _drive_cyclic_fallback(
        _cyclic_fallback_router(),
        capture,
        mock_response=litellm.InternalServerError(
            message=huge_message, llm_provider="openai", model="group-a"
        ),
    )

    assert capture.messages, "the fallback failure path did not log at ERROR"
    assert huge_message not in "".join(capture.messages)
    assert max(len(message) for message in capture.messages) < 5_000


def test_stamp_or_clear_metadata_key_writes_and_clears_both_buckets():
    request_kwargs = {"metadata": {}}
    litellm.Router._stamp_or_clear_metadata_key(request_kwargs=request_kwargs, key="probe", value=7)
    assert request_kwargs["metadata"]["probe"] == 7

    stale_kwargs = {"metadata": {"probe": 7}, "litellm_metadata": {"probe": 7}}
    litellm.Router._stamp_or_clear_metadata_key(request_kwargs=stale_kwargs, key="probe", value=None)
    assert "probe" not in stale_kwargs["metadata"]
    assert "probe" not in stale_kwargs["litellm_metadata"]


@pytest.mark.parametrize(
    "complexity_router_config,expect_callback",
    [
        ({"tiers": {"SIMPLE": "gpt-4o"}}, True),
        ({"tiers": {"SIMPLE": "gpt-4o"}, "deployment_affinity": False}, False),
        ({"tiers": {"SIMPLE": "gpt-4o"}, "deployment_affinity": False, "session_affinity": True}, True),
    ],
)
def test_complexity_router_registers_affinity_callback_for_deployment_pin(complexity_router_config, expect_callback):
    """The marker the complexity router stamps is inert unless a DeploymentAffinityCheck is
    registered to read it, so deployment_affinity has to pull the callback in, and its default-on
    means a bare config registers one. Opting out must skip the callback entirely rather than
    register a filter that can never fire, including when session_affinity is on, since the two
    pins are independent."""
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    router = litellm.Router(
        model_list=[
            {"model_name": "gpt-4o", "litellm_params": {"model": "gpt-4o"}},
            {
                "model_name": "my-complexity-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": complexity_router_config,
                },
            },
        ]
    )
    try:
        registered = any(isinstance(cb, DeploymentAffinityCheck) for cb in router.optional_callbacks or [])
        assert registered is expect_callback
    finally:
        for cb in router.optional_callbacks or []:
            litellm.logging_callback_manager.remove_callback_from_all_lists(cb)


def test_ensure_deployment_affinity_callback_is_idempotent():
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    router = litellm.Router(model_list=[])
    try:
        router._ensure_deployment_affinity_callback()
        router._ensure_deployment_affinity_callback()
        affinity_callbacks = [
            cb for cb in router.optional_callbacks or [] if isinstance(cb, DeploymentAffinityCheck)
        ]
        assert len(affinity_callbacks) == 1
    finally:
        for cb in router.optional_callbacks or []:
            litellm.logging_callback_manager.remove_callback_from_all_lists(cb)


def test_get_router_model_info_does_not_wipe_cached_pricing():
    """A Deployment's model_info declares the mirrored pricing fields with None defaults;
    merging it must not write those Nones into the lru_cache'd dict get_model_info() owns,
    or /model/info loses built-in prices for every model a worker serves."""
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    litellm.get_model_info.cache_clear()
    expected = copy.deepcopy(litellm.get_model_info(model="anthropic/claude-sonnet-4-5"))

    router = litellm.Router(model_list=[])
    merged = router.get_router_model_info(
        deployment=Deployment(
            model_name="sonnet",
            litellm_params=LiteLLM_Params(model="claude-sonnet-4-5", custom_llm_provider="anthropic"),
            model_info=ModelInfo(id="sonnet-1"),
        ),
        received_model_name="sonnet",
    )

    assert litellm.get_model_info(model="anthropic/claude-sonnet-4-5") == expected
    for field in ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost"):
        assert merged[field] == expected[field]


def test_get_router_model_info_keeps_explicit_pricing_overrides():
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    litellm.get_model_info.cache_clear()
    router = litellm.Router(model_list=[])
    merged = router.get_router_model_info(
        deployment=Deployment(
            model_name="sonnet",
            litellm_params=LiteLLM_Params(model="claude-sonnet-4-5", custom_llm_provider="anthropic"),
            model_info=ModelInfo(id="sonnet-1", input_cost_per_token=1e-08),
        ),
        received_model_name="sonnet",
    )

    assert merged["input_cost_per_token"] == 1e-08
    assert litellm.get_model_info(model="anthropic/claude-sonnet-4-5")["input_cost_per_token"] != 1e-08


class TestModelGroupAliasReachesPreRoutingStrategies:
    """A `model_group_alias` whose target is a strategy router must dispatch exactly like the
    router's own model_name. The four strategy registries are keyed by the marker deployment's
    model_name, so the alias has to be resolved before the pre-routing hook looks anything up,
    and a group that resolves only to markers is not callable at all (LIT-4664)."""

    MARKER_TIMEOUT = 42.0
    REGISTRY_NAMES = ("auto_routers", "complexity_routers", "adaptive_routers", "quality_routers")

    class _RewriteStrategy:
        async def async_pre_routing_hook(
            self, model, request_kwargs, messages=None, input=None, specific_deployment=False
        ):
            from litellm.types.router import PreRoutingHookResponse

            return PreRoutingHookResponse(model="gemini-flash", messages=messages)

    @classmethod
    def _router(cls, registry_name: str | None) -> "litellm.Router":
        from litellm.types.router import TaggedPreRoutingStrategy

        tiers = dict.fromkeys(("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"), "gemini-flash")
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "smart-route",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_config": {"tiers": tiers},
                        "complexity_router_default_model": "gemini-flash",
                        "timeout": cls.MARKER_TIMEOUT,
                    },
                },
                {
                    "model_name": "gemini-flash",
                    "litellm_params": {"model": "gemini/gemini-3.6-flash", "mock_response": "routed by the tier"},
                },
            ],
            model_group_alias={"smart-alias": "smart-route"},
        )
        for name in cls.REGISTRY_NAMES:
            setattr(router, name, {})
        if registry_name is not None:
            setattr(
                router,
                registry_name,
                {"smart-route": [TaggedPreRoutingStrategy(tags=(), strategy=cls._RewriteStrategy())]},
            )
        return router

    @staticmethod
    def _messages() -> list[dict[str, str]]:
        return [{"role": "user", "content": "What is the capital of France?"}]

    @pytest.mark.parametrize("registry_name", REGISTRY_NAMES)
    @pytest.mark.asyncio
    async def test_alias_dispatches_to_the_strategy_registered_under_the_target(self, registry_name):
        router = self._router(registry_name)
        request_kwargs = {"metadata": {}}

        response = await router.async_pre_routing_hook(
            model="smart-alias", request_kwargs=request_kwargs, messages=self._messages()
        )

        assert response is not None
        assert response.model == "gemini-flash"

    @pytest.mark.asyncio
    async def test_alias_call_still_forwards_the_marker_own_params_to_the_routed_tier(self):
        router = self._router("auto_routers")
        request_kwargs = {"metadata": {}}

        await router.async_pre_routing_hook(
            model="smart-alias", request_kwargs=request_kwargs, messages=self._messages()
        )

        assert request_kwargs["timeout"] == self.MARKER_TIMEOUT

    @pytest.mark.asyncio
    async def test_alias_deployment_selection_lands_on_the_tier_never_the_marker(self):
        router = self._router("auto_routers")

        deployment = await router.async_get_available_deployment(
            model="smart-alias", request_kwargs={"metadata": {}}, messages=self._messages()
        )

        assert deployment["litellm_params"]["model"] == "gemini/gemini-3.6-flash"

    @pytest.mark.asyncio
    async def test_alias_call_completes_and_still_bills_the_name_the_caller_sent(self):
        router = self._router("auto_routers")
        metadata: dict = {}

        response = await router.acompletion(
            model="smart-alias", messages=self._messages(), metadata=metadata
        )

        assert response.choices[0].message.content == "routed by the tier"
        assert metadata["model_group"] == "smart-alias"
        assert metadata["model_group_alias"] == "smart-alias"

    def test_a_group_of_only_markers_is_not_a_callable_model(self):
        router = self._router(None)

        with pytest.raises(litellm.BadRequestError, match="strategy router marker"):
            router.get_available_deployment(
                model="smart-route", messages=self._messages(), request_kwargs={"metadata": {}}
            )


@pytest.mark.usefixtures("local_model_cost_map")
class TestAzureBaseModelFallbackLogging:
    """When an azure deployment has no base_model but its model name is a known
    azure key in the cost map, get_router_model_info resolves it via the
    fallback, so it must not log the per-request 'Could not identify azure
    model' ERROR. The ERROR must remain for genuinely unmappable deployment
    names. Issue #33172."""

    def _router_with_azure_deployment(self, deployment_model: str):
        return litellm.Router(
            model_list=[
                {
                    "model_name": "my-group",
                    "litellm_params": {
                        "model": deployment_model,
                        "api_key": "fake-key",
                        "api_base": "https://fake.openai.azure.com",
                    },
                    "model_info": {"id": "azure-base-model-test-id"},
                }
            ]
        )

    def test_map_known_deployment_name_resolves_without_error_log(self):
        router = self._router_with_azure_deployment("azure/gpt-4o")

        with patch(
            "litellm.router.verbose_router_logger.error"
        ) as mock_error:
            model_info = router.get_router_model_info(
                deployment=None, received_model_name="my-group", id="azure-base-model-test-id"
            )

        assert not any(
            "Could not identify azure model" in str(call)
            for call in mock_error.call_args_list
        ), f"unexpected error log: {mock_error.call_args_list}"
        # the fallback resolution must actually surface the map values
        assert model_info["max_input_tokens"] == litellm.model_cost["azure/gpt-4o"]["max_input_tokens"]
        assert model_info["input_cost_per_token"] == litellm.model_cost["azure/gpt-4o"]["input_cost_per_token"]

    def test_unmappable_deployment_name_still_logs_error(self):
        router = self._router_with_azure_deployment("azure/my-custom-deployment-name")

        with patch(
            "litellm.router.verbose_router_logger.error"
        ) as mock_error:
            model_info = router.get_router_model_info(
                deployment=None, received_model_name="my-group", id="azure-base-model-test-id"
            )

        assert any(
            "Could not identify azure model" in str(call)
            for call in mock_error.call_args_list
        ), "expected the error log for an unmappable azure deployment name"
        # unmappable names resolve to a zeroed stub — unchanged behavior
        assert model_info.get("max_input_tokens") is None

    def test_explicit_base_model_still_wins(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "my-group",
                    "litellm_params": {
                        "model": "azure/some-deployment",
                        "api_key": "fake-key",
                        "api_base": "https://fake.openai.azure.com",
                    },
                    "model_info": {
                        "id": "azure-base-model-test-id",
                        "base_model": "azure/gpt-4o-mini",
                    },
                }
            ]
        )

        model_info = router.get_router_model_info(
            deployment=None, received_model_name="my-group", id="azure-base-model-test-id"
        )
        assert model_info["max_input_tokens"] == litellm.model_cost["azure/gpt-4o-mini"]["max_input_tokens"]

def test_model_group_info_intersects_supported_reasoning_efforts():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart-group",
                "litellm_params": {"model": "anthropic/opus-like"},
                "model_info": {"id": "opus-like-deployment"},
            },
            {
                "model_name": "smart-group",
                "litellm_params": {"model": "openai/mini-like"},
                "model_info": {"id": "mini-like-deployment"},
            },
        ]
    )

    def _model_info(model_id: str, model_name: str):
        if model_id == "opus-like-deployment":
            return {
                "key": model_name,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_reasoning": True,
                "supports_xhigh_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
            }
        return {
            "key": model_name,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_reasoning": True,
            "supports_none_reasoning_effort": False,
            "supports_minimal_reasoning_effort": True,
            "supports_xhigh_reasoning_effort": False,
        }

    with patch.object(router, "get_deployment_model_info", side_effect=_model_info):
        result = router._set_model_group_info(
            model_group="smart-group",
            user_facing_model_group_name="smart-group",
        )

    assert result is not None
    # opus-like offers all seven levels, mini-like lacks none/xhigh/max; only the common set survives,
    # so the group never advertises an effort routing could hand to a deployment that rejects it.
    assert result.supported_reasoning_efforts == ("minimal", "low", "medium", "high")


def test_model_group_info_reasoning_efforts_ignore_a_deployment_off_the_map():
    """The router fills every ModelInfo key, so a deployment absent from the model map arrives with
    supports_reasoning None rather than with the key missing. Its synthesized entry carries no mode,
    which is what separates it from a mapped non-reasoning model, and nothing being known about it is
    no reason to drop the levels the rest of the group agrees on."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart-group",
                "litellm_params": {"model": "anthropic/opus-like"},
                "model_info": {"id": "opus-like-deployment"},
            },
            {
                "model_name": "smart-group",
                "litellm_params": {"model": "openai/unmapped-model"},
                "model_info": {"id": "unmapped-deployment"},
            },
        ]
    )

    def _model_info(model_id: str, model_name: str):
        if model_id == "opus-like-deployment":
            return {
                "key": model_name,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_reasoning": True,
                "supports_max_reasoning_effort": True,
            }
        return {"key": model_name, "litellm_provider": "openai", "mode": None, "supports_reasoning": None}

    with patch.object(router, "get_deployment_model_info", side_effect=_model_info):
        result = router._set_model_group_info(
            model_group="smart-group",
            user_facing_model_group_name="smart-group",
        )

    assert result is not None
    assert result.supported_reasoning_efforts == ("none", "minimal", "low", "medium", "high", "max")


def test_model_group_info_reasoning_efforts_empty_on_a_mapped_non_reasoning_deployment():
    """A group mixing a reasoning model with one the map knows is not a reasoning model shares no
    level, so it advertises none and the picker offers nothing rather than a level routing would
    hand to a deployment that rejects it."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "mixed-group",
                "litellm_params": {"model": "anthropic/opus-like"},
                "model_info": {"id": "opus-like-deployment"},
            },
            {
                "model_name": "mixed-group",
                "litellm_params": {"model": "openai/plain-chat"},
                "model_info": {"id": "plain-chat-deployment"},
            },
        ]
    )

    def _model_info(model_id: str, model_name: str):
        if model_id == "opus-like-deployment":
            return {
                "key": model_name,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_reasoning": True,
                "supports_max_reasoning_effort": True,
            }
        return {"key": model_name, "litellm_provider": "openai", "mode": "chat", "supports_reasoning": None}

    with patch.object(router, "get_deployment_model_info", side_effect=_model_info):
        result = router._set_model_group_info(
            model_group="mixed-group",
            user_facing_model_group_name="mixed-group",
        )

    assert result is not None
    assert result.supported_reasoning_efforts == ()


def test_model_group_info_reasoning_efforts_ignore_a_value_declared_in_model_info():
    """The group's levels are computed from its deployments, so a value an operator left in one
    deployment's model_info must not seed them. Seeding let the first deployment read narrow the
    whole group while the same value on any other deployment was silently ignored."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "declared-group",
                "litellm_params": {"model": "openai/first-reasoner"},
                "model_info": {"id": "first-deployment"},
            },
            {
                "model_name": "declared-group",
                "litellm_params": {"model": "openai/second-reasoner"},
                "model_info": {"id": "second-deployment"},
            },
        ]
    )

    def _model_info(model_id: str, model_name: str):
        info = {
            "key": model_name,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_reasoning": True,
            "supports_none_reasoning_effort": True,
        }
        if model_id == "first-deployment":
            info["supported_reasoning_efforts"] = ("high",)
        return info

    with patch.object(router, "get_deployment_model_info", side_effect=_model_info):
        result = router._set_model_group_info(
            model_group="declared-group",
            user_facing_model_group_name="declared-group",
        )

    assert result is not None
    assert result.supported_reasoning_efforts == ("none", "minimal", "low", "medium", "high")


def test_model_group_info_survives_a_junk_typed_operator_effort_value():
    """A deployment's registered model_info reads back with whatever the operator wrote under any
    key, so a wrong-typed supported_reasoning_efforts must not fail the group's info. Only the
    constructor's trailing override keeps the junk away from ModelGroupInfo validation."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "junk-declared-group",
                "litellm_params": {"model": "openai/lone-reasoner"},
                "model_info": {"id": "junk-deployment"},
            },
        ]
    )

    def _model_info(model_id: str, model_name: str):
        return {
            "key": model_name,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_reasoning": True,
            "supports_none_reasoning_effort": True,
            "supported_reasoning_efforts": "high",
        }

    with patch.object(router, "get_deployment_model_info", side_effect=_model_info):
        result = router._set_model_group_info(
            model_group="junk-declared-group",
            user_facing_model_group_name="junk-declared-group",
        )

    assert result is not None
    assert result.supported_reasoning_efforts == ("none", "minimal", "low", "medium", "high")


def test_model_group_info_reasoning_efforts_ignore_a_mode_the_operator_declared():
    """A deployment is registered in the cost map under its own id with whatever model_info the
    operator wrote, so a mode they set themselves reads back exactly like one the map supplied. Only
    a mode the map supplied marks the deployment as known, or an off-map deployment carrying any
    mode empties the group it sits in."""
    from litellm.router_utils.reasoning_effort_capability import resolve_supported_reasoning_efforts

    mapped_model = "openai/gpt-5.6-sol"
    expected = resolve_supported_reasoning_efforts(
        litellm.get_model_info(model=mapped_model),
        deployment_is_mapped=True,
    )
    assert expected

    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart-group",
                "litellm_params": {"model": mapped_model, "api_key": "sk-fake"},
                "model_info": {"id": "mapped-deployment"},
            },
            {
                "model_name": "smart-group",
                "litellm_params": {"model": "openai/a-model-the-map-never-heard-of", "api_key": "sk-fake"},
                "model_info": {"id": "off-map-deployment", "mode": "chat"},
            },
        ]
    )

    result = router._set_model_group_info(
        model_group="smart-group",
        user_facing_model_group_name="smart-group",
    )

    assert result is not None
    assert result.supported_reasoning_efforts == expected


class TestAddDeploymentApiBaseProviderResolution:
    def test_bare_model_with_known_api_base_initializes(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "groq-pinned",
                    "litellm_params": {
                        "model": "llama-3.3-70b-versatile",
                        "api_base": "https://api.groq.com/openai/v1",
                        "api_key": "fake-key",
                    },
                },
                {
                    "model_name": "deepseek-pinned",
                    "litellm_params": {
                        "model": "deepseek-chat",
                        "api_base": "https://api.deepseek.com/v1",
                        "api_key": "fake-key",
                    },
                },
            ]
        )

        model_list = router.get_model_list()
        assert model_list is not None
        assert {m["model_name"] for m in model_list} == {"groq-pinned", "deepseek-pinned"}

    def test_bare_model_with_unknown_api_base_still_raises(self):
        with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
            litellm.Router(
                model_list=[
                    {
                        "model_name": "mystery",
                        "litellm_params": {
                            "model": "some-unknown-model",
                            "api_base": "https://llm.internal.example.com/v1",
                            "api_key": "fake-key",
                        },
                    }
                ]
            )

    def test_explicit_custom_llm_provider_beats_api_base_endpoint_match(self):
        router = litellm.Router(
            model_list=[
                {
                    "model_name": "openai-via-gateway",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "custom_llm_provider": "openai",
                        "api_base": "https://api.groq.com/openai/v1",
                        "api_key": "fake-key",
                    },
                }
            ]
        )

        deployment = router.get_deployment_by_model_group_name("openai-via-gateway")
        assert deployment is not None
        assert deployment.litellm_params.custom_llm_provider == "openai"

# =====================================================================
# anthropic_messages mid-stream-fallback helpers, added for #24004
# (mid-stream fallback not supported for anthropic_messages route type).
#
# anthropic_messages goes through _ageneric_api_call_with_fallbacks rather
# than _acompletion, so its returned iterator was never wrapped by the chat
# completions fallback handler: an SSE `event: error` frame from a native
# Anthropic/Bedrock passthrough passed through to the client silently, and a
# MidStreamFallbackError raised by the completion-bridge path's
# CustomStreamWrapper (e.g. a Vertex AI transport drop) propagated
# unhandled.
#
# Targets the helpers introduced on Router:
#   - _aanthropic_messages_streaming_iterator
#   - _aanthropic_messages_fallback_attempt
#   - _aanthropic_messages_with_streaming_fallbacks
#   - _dispatch_generic_call_type
# =====================================================================


async def _anthropic_messages_empty_generator():
    return
    yield  # pragma: no cover - makes this an async generator


def _anthropic_messages_make_wrapper() -> FallbackAwareAnthropicMessagesStream:
    """A minimal wrapper for tests that call _aanthropic_messages_fallback_attempt
    directly, bypassing _aanthropic_messages_streaming_iterator."""
    return FallbackAwareAnthropicMessagesStream(_anthropic_messages_empty_generator(), object())


def _anthropic_messages_make_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": "anthropic/claude-sonnet-4-5",
                    "api_key": "sk-test",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-sonnet-4-5",
                },
            },
        ]
    )


class _AnthropicMessagesFakeByteStream:
    """Minimal AsyncIterator[bytes], carrying _hidden_params like
    AnthropicMessagesStreamingResponse does."""

    def __init__(self, chunks: list) -> None:
        self._chunks = list(chunks)
        self._hidden_params = {"additional_headers": {"x-amzn-requestid": "req-1"}}
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class _AnthropicMessagesRaisingByteStream:
    """Simulates the completion-bridge path: no error SSE chunk is ever
    yielded, the underlying CustomStreamWrapper raises MidStreamFallbackError
    directly out of the iterator instead (a Vertex AI transport drop)."""

    def __init__(self, chunks: list, error: Exception) -> None:
        self._chunks = list(chunks)
        self._error = error
        self._hidden_params: dict = {}
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        raise self._error

    async def aclose(self) -> None:
        self.closed = True


class _AnthropicMessagesFallbackByteStream:
    def __init__(self, chunks: list, hidden_params: dict | None = None) -> None:
        self._chunks = list(chunks)
        self._hidden_params = hidden_params if hidden_params is not None else {}

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _anthropic_messages_overloaded_error_chunk() -> bytes:
    return (
        b"event: error\n"
        b'data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}\n\n'
    )


def _anthropic_messages_invalid_request_error_chunk() -> bytes:
    return (
        b"event: error\n"
        b'data: {"type": "error", "error": {"type": "invalid_request_error", "message": "bad request"}}\n\n'
    )


def _anthropic_messages_rate_limit_error_chunk() -> bytes:
    return (
        b"event: error\n"
        b'data: {"type": "error", "error": {"type": "rate_limit_error", "message": "Too many requests"}}\n\n'
    )


def _anthropic_messages_content_chunk(text: str = "hi") -> bytes:
    payload = f'{{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": "{text}"}}}}'
    return f"event: content_block_delta\ndata: {payload}\n\n".encode()


def _anthropic_messages_message_start_chunk() -> bytes:
    """A lifecycle/bookkeeping frame Anthropic sends before any real content -
    routinely the very first event before an overload error."""
    return b'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_1"}}\n\n'


def _anthropic_messages_ping_chunk() -> bytes:
    return b'event: ping\ndata: {"type": "ping"}\n\n'


# -------- _aanthropic_messages_streaming_iterator (passthrough) --------


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_iterator_passthrough():
    """Without any error chunk, the wrapper forwards every chunk unchanged
    and carries the source iterator's _hidden_params through (so response
    headers like Bedrock's request-id keep flowing to the client)."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_content_chunk("hi"), _anthropic_messages_content_chunk(" there")]
    )

    wrapped = await router._aanthropic_messages_streaming_iterator(
        response=source, initial_kwargs={"model": "primary"}
    )

    collected = [chunk async for chunk in wrapped]
    assert collected == [_anthropic_messages_content_chunk("hi"), _anthropic_messages_content_chunk(" there")]
    assert wrapped._hidden_params["additional_headers"]["x-amzn-requestid"] == "req-1"


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_iterator_flushes_buffered_lifecycle_frames_in_order():
    """Regression: lifecycle frames held back to guard against a mid-stream
    fallback must still reach the client, in order, once real content
    arrives - buffering them for the fallback-safety check must not silently
    drop them on the happy path."""
    router = _anthropic_messages_make_router()
    message_stop = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_message_start_chunk(), _anthropic_messages_content_chunk("hi"), message_stop]
    )

    wrapped = await router._aanthropic_messages_streaming_iterator(
        response=source, initial_kwargs={"model": "primary"}
    )

    collected = [chunk async for chunk in wrapped]
    assert collected == [_anthropic_messages_message_start_chunk(), _anthropic_messages_content_chunk("hi"), message_stop]


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_iterator_flushes_buffered_frames_on_stream_end():
    """Regression: if the primary stream ends with only lifecycle frames and
    no content and no error, the buffered frames must still reach the
    client rather than being silently swallowed."""
    router = _anthropic_messages_make_router()
    message_stop = b'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    source = _AnthropicMessagesFakeByteStream([_anthropic_messages_message_start_chunk(), message_stop])

    wrapped = await router._aanthropic_messages_streaming_iterator(
        response=source, initial_kwargs={"model": "primary"}
    )

    collected = [chunk async for chunk in wrapped]
    assert collected == [_anthropic_messages_message_start_chunk(), message_stop]

    with pytest.raises(StopAsyncIteration):
        await wrapped.__anext__()


@pytest.mark.asyncio
async def test_anthropic_messages_content_coalesced_with_error_in_one_physical_chunk_skips_fallback():
    """Greptile review round: transport-level buffering can coalesce a real
    content_block_delta and a following retriable error into ONE physical
    read from the source iterator. Since the whole chunk (content and error
    together) is forwarded to the client atomically, the client genuinely
    receives the content - so no fallback must be attempted, exactly as if
    the two events had arrived as separate reads."""
    router = _anthropic_messages_make_router()
    coalesced_chunk = _anthropic_messages_content_chunk("partial") + _anthropic_messages_overloaded_error_chunk()
    source = _AnthropicMessagesFakeByteStream([coalesced_chunk])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_AnthropicMessagesFallbackByteStream([])),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [coalesced_chunk]
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_ping_behind_buffered_lifecycle_frame_is_dropped():
    """Bugbot regression: a `ping` keepalive behind buffered lifecycle frames
    carries no content and is dropped outright rather than buffered -
    otherwise a slow-starting connection sending many pings could grow the
    pre-content buffer without bound."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [
            _anthropic_messages_message_start_chunk(),
            _anthropic_messages_ping_chunk(),
            _anthropic_messages_content_chunk("hi"),
        ]
    )

    wrapped = await router._aanthropic_messages_streaming_iterator(response=source, initial_kwargs={"model": "primary"})
    collected = [chunk async for chunk in wrapped]

    assert collected == [_anthropic_messages_message_start_chunk(), _anthropic_messages_content_chunk("hi")]


@pytest.mark.asyncio
async def test_anthropic_messages_leading_ping_keepalive_is_forwarded_live():
    """A `ping` that no lifecycle frame precedes is how a hold-back turn keeps
    its connection alive (AgenticAnthropicStreamingIterator), so it must reach
    the client at once rather than wait behind the pre-content buffer."""
    router = _anthropic_messages_make_router()
    content_released = asyncio.Event()

    async def source():
        yield _anthropic_messages_ping_chunk()
        await content_released.wait()
        yield _anthropic_messages_message_start_chunk()
        yield _anthropic_messages_content_chunk("hi")

    wrapped = await router._aanthropic_messages_streaming_iterator(response=source(), initial_kwargs={"model": "primary"})

    assert await asyncio.wait_for(wrapped.__anext__(), timeout=1) == _anthropic_messages_ping_chunk()
    content_released.set()
    assert [chunk async for chunk in wrapped] == [
        _anthropic_messages_message_start_chunk(),
        _anthropic_messages_content_chunk("hi"),
    ]


@pytest.mark.asyncio
async def test_anthropic_messages_leading_ping_does_not_disqualify_fallback():
    """A live-forwarded leading `ping` commits nothing: a retriable error after
    it still falls back, and the fallback's own lifecycle follows the ping cleanly."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_ping_chunk(), _anthropic_messages_overloaded_error_chunk()]
    )
    fallback_message_start = b'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_2"}}\n\n'
    fallback_stream = _AnthropicMessagesFallbackByteStream(
        [fallback_message_start, _anthropic_messages_content_chunk("fallback answer")]
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ):
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [
        _anthropic_messages_ping_chunk(),
        fallback_message_start,
        _anthropic_messages_content_chunk("fallback answer"),
    ]


@pytest.mark.asyncio
async def test_anthropic_messages_hold_back_retrieval_failure_reaches_client_without_fallback():
    """The hold-back iterator's own retrieval-failure frame is the gateway's verdict, not a
    provider failure: a configured fallback stays untouched and the client reads the error
    right after the live keepalive."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_ping_chunk(), SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES]
    )
    fallback = AsyncMock(
        return_value=_AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])
    )

    with patch.object(router, "async_function_with_fallbacks_common_utils", new=fallback):
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    fallback.assert_not_called()
    assert collected == [_anthropic_messages_ping_chunk(), SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES]


@pytest.mark.asyncio
async def test_anthropic_messages_pre_content_buffer_cap_forces_commit():
    """Bugbot regression: a hostile or pathological upstream that never emits
    real content or an error must not grow the pre-content lifecycle buffer
    without bound - hitting MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS commits
    to the primary stream early, exactly as real content arriving would."""
    router = _anthropic_messages_make_router()
    lifecycle_chunk = _anthropic_messages_message_start_chunk()
    error_chunk = _anthropic_messages_overloaded_error_chunk()
    chunks = [lifecycle_chunk] * (MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS + 5) + [error_chunk]
    source = _AnthropicMessagesFakeByteStream(chunks)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=_AnthropicMessagesFallbackByteStream([])),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source, initial_kwargs={"model": "primary"}
        )
        collected = [chunk async for chunk in wrapped]

    mock_fallback.assert_not_awaited()
    assert collected.count(lifecycle_chunk) == MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS + 5
    assert collected[-1] == error_chunk


@pytest.mark.asyncio
async def test_anthropic_messages_ping_coalesced_with_content_in_one_physical_chunk_is_forwarded():
    """Greptile/Bugbot regression: transport-level buffering can coalesce a
    `ping` keepalive and a real content_block_delta into ONE physical read.
    The pre-content ping-drop must only discard PURE ping frames - dropping
    the whole coalesced chunk would silently lose generated content."""
    router = _anthropic_messages_make_router()
    coalesced_chunk = _anthropic_messages_ping_chunk() + _anthropic_messages_content_chunk("hi")
    source = _AnthropicMessagesFakeByteStream([coalesced_chunk])

    wrapped = await router._aanthropic_messages_streaming_iterator(response=source, initial_kwargs={"model": "primary"})
    collected = [chunk async for chunk in wrapped]

    assert collected == [coalesced_chunk]


@pytest.mark.asyncio
async def test_anthropic_messages_ping_coalesced_with_retriable_error_still_falls_back():
    """Greptile/Bugbot regression: a physical chunk coalescing a `ping` with a
    retriable `event: error` must not be discarded as a keepalive - the error
    inside it must still trigger the mid-stream fallback."""
    router = _anthropic_messages_make_router()
    coalesced_chunk = _anthropic_messages_ping_chunk() + _anthropic_messages_overloaded_error_chunk()
    source = _AnthropicMessagesFakeByteStream([coalesced_chunk])
    fallback_stream = _AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source, initial_kwargs={"model": "primary"}
        )
        collected = [chunk async for chunk in wrapped]

    mock_fallback.assert_awaited_once()
    assert collected == [_anthropic_messages_content_chunk("fallback answer")]


# -------- _aanthropic_messages_fallback_attempt --------


@pytest.mark.asyncio
async def test_aanthropic_messages_fallback_attempt_yields_fallback_stream():
    """Direct-call regression: the fallback-attempt helper re-enters the
    Router's fallback chain and forwards whatever the fallback produces."""
    router = _anthropic_messages_make_router()
    fallback_stream = _AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])
    error = MidStreamFallbackError(message="overloaded", model="primary", llm_provider="anthropic")

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        collected = [
            chunk
            async for chunk in router._aanthropic_messages_fallback_attempt(
                error,
                {"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
                _anthropic_messages_make_wrapper(),
            )
        ]

    assert collected == [_anthropic_messages_content_chunk("fallback answer")]
    mock_fallback.assert_awaited_once()
    assert mock_fallback.await_args.kwargs["e"] is error


@pytest.mark.asyncio
async def test_aanthropic_messages_fallback_attempt_raises_original_exception_on_double_failure():
    """Direct-call regression: when the fallback attempt itself fails with a
    MidStreamFallbackError wrapping a real provider exception, that real
    exception must surface rather than the internal wrapper exception."""
    router = _anthropic_messages_make_router()
    error = MidStreamFallbackError(message="overloaded", model="primary", llm_provider="anthropic")
    original_exception = litellm.APIError(
        status_code=503, message="fallback also overloaded", llm_provider="bedrock", model="fallback"
    )
    fallback_failure = MidStreamFallbackError(
        message="fallback failed", model="fallback", llm_provider="bedrock", original_exception=original_exception
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=fallback_failure),
    ):
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in router._aanthropic_messages_fallback_attempt(
                error, {"model": "primary"}, _anthropic_messages_make_wrapper()
            ):
                pass

    assert exc_info.value is original_exception


@pytest.mark.asyncio
async def test_aanthropic_messages_fallback_attempt_yields_non_streaming_fallback_response():
    """Bugbot regression: a fallback that resolves to a non-streaming
    response (no __aiter__, e.g. an agentic tool-use interception loop) must
    be synthesized into a valid SSE byte sequence, not yielded as a raw dict
    into a byte stream - the generator is typed AsyncGenerator[bytes, None]
    and every item reaching the client must be a real SSE frame."""
    router = _anthropic_messages_make_router()
    error = MidStreamFallbackError(message="overloaded", model="primary", llm_provider="anthropic")
    non_streaming_response = {"id": "msg_1", "type": "message", "content": [{"type": "text", "text": "hi"}]}

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=non_streaming_response),
    ):
        collected = [
            item
            async for item in router._aanthropic_messages_fallback_attempt(
                error, {"model": "primary"}, _anthropic_messages_make_wrapper()
            )
        ]

    assert all(isinstance(item, bytes) for item in collected)
    event_types = [item.split(b"\n")[0].removeprefix(b"event: ") for item in collected]
    assert event_types == [
        b"message_start",
        b"content_block_start",
        b"content_block_delta",
        b"content_block_stop",
        b"message_delta",
        b"message_stop",
    ]
    assert b'"text": "hi"' in collected[2]


@pytest.mark.asyncio
async def test_aanthropic_messages_fallback_attempt_reraises_plain_exception_on_double_failure():
    """Direct-call regression: when the fallback attempt fails with a plain
    exception (not a MidStreamFallbackError), that exception itself must
    propagate unchanged."""
    router = _anthropic_messages_make_router()
    error = MidStreamFallbackError(message="overloaded", model="primary", llm_provider="anthropic")
    fallback_failure = ValueError("no healthy deployments")

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=fallback_failure),
    ):
        with pytest.raises(ValueError, match="no healthy deployments") as exc_info:
            async for _ in router._aanthropic_messages_fallback_attempt(
                error, {"model": "primary"}, _anthropic_messages_make_wrapper()
            ):
                pass

    assert exc_info.value is fallback_failure


# -------- _aanthropic_messages_with_streaming_fallbacks --------


@pytest.mark.asyncio
async def test_aanthropic_messages_with_streaming_fallbacks_non_streaming_passthrough():
    """A non-streaming response (plain dict) is returned unchanged, never wrapped."""
    router = _anthropic_messages_make_router()
    plain_response = {"id": "msg_1", "type": "message"}

    async def fake_original(**_kwargs):
        return plain_response

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks",
        new=AsyncMock(return_value=plain_response),
    ):
        out = await router._aanthropic_messages_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=False,
        )
    assert out is plain_response


@pytest.mark.asyncio
async def test_aanthropic_messages_with_streaming_fallbacks_wraps_streaming_iterator():
    """A streaming response is wrapped via _aanthropic_messages_streaming_iterator."""
    router = _anthropic_messages_make_router()
    streaming_iter = _AnthropicMessagesFakeByteStream([_anthropic_messages_content_chunk()])
    wrapped_marker = object()

    async def fake_original(**_kwargs):
        return streaming_iter

    with (
        patch.object(
            router,
            "_ageneric_api_call_with_fallbacks",
            new=AsyncMock(return_value=streaming_iter),
        ),
        patch.object(
            router,
            "_aanthropic_messages_streaming_iterator",
            new=AsyncMock(return_value=wrapped_marker),
        ) as mock_wrap,
    ):
        out = await router._aanthropic_messages_with_streaming_fallbacks(
            original_function=fake_original,
            model="primary",
            stream=True,
        )
    assert out is wrapped_marker
    mock_wrap.assert_awaited_once()


# -------- mid-stream error handling --------


@pytest.mark.asyncio
async def test_anthropic_messages_fallback_on_pre_first_chunk_error_event():
    """Regression for #24004: a retriable SSE `event: error` frame
    (overloaded_error/internal_server_error) that arrives before any real
    content must trigger the router's fallback chain instead of passing
    through to the client silently."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream([_anthropic_messages_overloaded_error_chunk()])
    fallback_stream = _AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [_anthropic_messages_content_chunk("fallback answer")]
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == 503
    assert raised.is_pre_first_chunk is True
    assert source.closed is True


@pytest.mark.asyncio
async def test_anthropic_messages_mid_stream_error_preserves_real_status_code():
    """Bugbot regression: the MidStreamFallbackError raised for a detected SSE
    `event: error` frame must carry the error's REAL parsed status code
    (via original_exception), not silently default to 503 for every error
    type - a rate_limit_error (429) must surface as 429, not 503."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream([_anthropic_messages_rate_limit_error_chunk()])
    fallback_stream = _AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary", "messages": [{"role": "user", "content": "hi"}]},
        )
        [chunk async for chunk in wrapped]

    raised = mock_fallback.await_args.kwargs["e"]
    assert isinstance(raised, MidStreamFallbackError)
    assert raised.status_code == 429
    assert raised.original_exception is not None
    assert raised.original_exception.status_code == 429
    assert raised.original_exception.llm_provider == "anthropic"


def test_merge_fallback_hidden_params_direct_call():
    """Direct-call regression: merge_fallback_hidden_params combines the
    fallback's hidden params/headers with whatever was already present,
    with the fallback's values winning on key collisions."""
    wrapper = FallbackAwareAnthropicMessagesStream(
        _anthropic_messages_empty_generator(),
        _AnthropicMessagesFakeByteStream([]),  # carries {"additional_headers": {"x-amzn-requestid": "req-1"}}
    )
    wrapper.merge_fallback_hidden_params(
        {"model_id": "fallback-deployment"},
        {"x-amzn-requestid": "req-2", "x-fallback-only": "yes"},
    )
    assert wrapper._hidden_params["model_id"] == "fallback-deployment"
    assert wrapper._hidden_params["additional_headers"] == {
        "x-amzn-requestid": "req-2",
        "x-fallback-only": "yes",
    }


def test_anthropic_stream_should_drop_pre_content_ping_direct_call():
    ping = _anthropic_messages_ping_chunk()
    content = _anthropic_messages_content_chunk("hi")
    assert _anthropic_stream_should_drop_pre_content_ping(ping, has_generated_content=False) is True
    assert _anthropic_stream_should_drop_pre_content_ping(ping, has_generated_content=True) is False
    assert _anthropic_stream_should_drop_pre_content_ping(content, has_generated_content=False) is False


def test_anthropic_stream_forwards_ping_live_direct_call():
    ping = _anthropic_messages_ping_chunk()
    content = _anthropic_messages_content_chunk("hi")
    assert _anthropic_stream_forwards_ping_live(ping, has_generated_content=False, buffered_chunk_count=0) is True
    assert _anthropic_stream_forwards_ping_live(ping, has_generated_content=False, buffered_chunk_count=1) is False
    assert _anthropic_stream_forwards_ping_live(ping, has_generated_content=True, buffered_chunk_count=0) is False
    assert _anthropic_stream_forwards_ping_live(content, has_generated_content=False, buffered_chunk_count=0) is False


def test_anthropic_stream_error_is_gateway_verdict_direct_call():
    assert _anthropic_stream_error_is_gateway_verdict(SERVER_FULFILLED_TOOL_LEAK_ERROR_SSE_BYTES) is True
    assert _anthropic_stream_error_is_gateway_verdict(_anthropic_messages_overloaded_error_chunk()) is False
    assert _anthropic_stream_error_is_gateway_verdict(_anthropic_messages_ping_chunk()) is False


def test_fallback_aware_stream_reports_withheld_output_of_its_current_source():
    """The proxy's cancel-refund guard reads this flag off the router wrapper, so it
    must reflect the stream actually being drained: the primary, then the fallback."""

    class _HoldingBack:
        _hidden_params = {"additional_headers": {}}
        has_buffered_provider_output = True

    wrapper = FallbackAwareAnthropicMessagesStream(_anthropic_messages_empty_generator(), _HoldingBack())
    assert wrapper.has_buffered_provider_output is True

    wrapper.adopt_fallback_source(_AnthropicMessagesFakeByteStream([]))
    assert wrapper.has_buffered_provider_output is False


def test_is_retriable_anthropic_status_direct_call():
    assert _is_retriable_anthropic_status(429) is True
    assert _is_retriable_anthropic_status(503) is True
    assert _is_retriable_anthropic_status(500) is True
    assert _is_retriable_anthropic_status(400) is False
    assert _is_retriable_anthropic_status(404) is False


def test_anthropic_stream_should_decline_fallback_direct_call():
    pre_first_chunk_error = MidStreamFallbackError(
        message="overloaded", model="primary", llm_provider="anthropic", is_pre_first_chunk=True
    )
    post_first_chunk_error = MidStreamFallbackError(
        message="overloaded", model="primary", llm_provider="anthropic", is_pre_first_chunk=False
    )
    assert _anthropic_stream_should_decline_fallback(False, pre_first_chunk_error) is False
    assert _anthropic_stream_should_decline_fallback(True, pre_first_chunk_error) is True
    assert _anthropic_stream_should_decline_fallback(False, post_first_chunk_error) is True


def test_anthropic_stream_commits_now_direct_call():
    content = _anthropic_messages_content_chunk("hi")
    lifecycle_chunk = _anthropic_messages_message_start_chunk()
    assert _anthropic_stream_commits_now(content, has_generated_content=False, buffered_chunk_count=0) is True
    assert _anthropic_stream_commits_now(content, has_generated_content=True, buffered_chunk_count=0) is False
    assert (
        _anthropic_stream_commits_now(
            lifecycle_chunk,
            has_generated_content=False,
            buffered_chunk_count=MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS,
        )
        is True
    )
    assert (
        _anthropic_stream_commits_now(
            lifecycle_chunk,
            has_generated_content=False,
            buffered_chunk_count=MAX_BUFFERED_PRE_CONTENT_ANTHROPIC_CHUNKS - 1,
        )
        is False
    )


@pytest.mark.asyncio
async def test_anthropic_messages_fallback_merges_fallback_hidden_params():
    """Bugbot regression: after a successful mid-stream fallback, the
    wrapper's _hidden_params must reflect the FALLBACK deployment's own
    provider headers (e.g. a different Bedrock request-id), not stay
    frozen on the primary's - raw bytes can't carry per-item _hidden_params
    the way a ModelResponseStream/ResponsesAPI event can, so the wrapper
    itself is the only place left to expose them."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_overloaded_error_chunk()]
    )  # carries x-amzn-requestid: req-1
    fallback_stream = _AnthropicMessagesFallbackByteStream(
        [_anthropic_messages_content_chunk("fallback answer")],
        hidden_params={"additional_headers": {"x-amzn-requestid": "req-2", "x-fallback-only": "yes"}},
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ):
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        _ = [chunk async for chunk in wrapped]

    headers = wrapped._hidden_params["additional_headers"]
    assert headers["x-amzn-requestid"] == "req-2"
    assert headers["x-fallback-only"] == "yes"


@pytest.mark.asyncio
async def test_aanthropic_messages_with_streaming_fallbacks_deep_copies_nested_metadata():
    """Bugbot regression: a shallow .copy() of kwargs still shares the
    nested litellm_metadata/metadata dict objects with the primary attempt.
    _update_kwargs_with_deployment mutates that dict in place with
    deployment-specific fields, which must not leak into the fallback
    request's metadata."""
    router = _anthropic_messages_make_router()
    primary_metadata = {"model_group": "primary"}
    streaming_iter_kwargs = {}

    async def fake_original(**_kwargs):
        # Simulate _update_kwargs_with_deployment mutating the primary's
        # litellm_metadata in place, as the real helper does.
        primary_metadata["deployment"] = "primary-deployment-object"
        return _AnthropicMessagesFakeByteStream([_anthropic_messages_content_chunk("hi")])

    with patch.object(
        router,
        "_aanthropic_messages_streaming_iterator",
        new=AsyncMock(side_effect=lambda **kwargs: streaming_iter_kwargs.update(kwargs) or "wrapped"),
    ):
        with patch.object(
            router,
            "_ageneric_api_call_with_fallbacks",
            new=AsyncMock(side_effect=fake_original),
        ):
            await router._aanthropic_messages_with_streaming_fallbacks(
                original_function=fake_original,
                model="primary",
                stream=True,
                litellm_metadata=primary_metadata,
            )

    fallback_kwargs = streaming_iter_kwargs["initial_kwargs"]
    assert fallback_kwargs["litellm_metadata"] is not primary_metadata
    assert "deployment" not in fallback_kwargs["litellm_metadata"]


@pytest.mark.asyncio
async def test_aanthropic_messages_with_streaming_fallbacks_deep_copies_metadata_field():
    """Same regression as above for the (separate) `metadata` kwarg some
    call sites use instead of `litellm_metadata`."""
    router = _anthropic_messages_make_router()
    primary_metadata = {"tag": "primary"}
    streaming_iter_kwargs = {}

    async def fake_original(**_kwargs):
        primary_metadata["deployment"] = "primary-deployment-object"
        return _AnthropicMessagesFakeByteStream([_anthropic_messages_content_chunk("hi")])

    with patch.object(
        router,
        "_aanthropic_messages_streaming_iterator",
        new=AsyncMock(side_effect=lambda **kwargs: streaming_iter_kwargs.update(kwargs) or "wrapped"),
    ):
        with patch.object(
            router,
            "_ageneric_api_call_with_fallbacks",
            new=AsyncMock(side_effect=fake_original),
        ):
            await router._aanthropic_messages_with_streaming_fallbacks(
                original_function=fake_original,
                model="primary",
                stream=True,
                metadata=primary_metadata,
            )

    fallback_kwargs = streaming_iter_kwargs["initial_kwargs"]
    assert fallback_kwargs["metadata"] is not primary_metadata
    assert "deployment" not in fallback_kwargs["metadata"]


@pytest.mark.asyncio
async def test_anthropic_messages_fallback_triggers_after_lifecycle_only_frame():
    """Regression: Anthropic routinely sends a message_start lifecycle frame
    before an overload error even fires. A lifecycle-only frame (no real
    content) must not disqualify the fallback attempt, and must not reach
    the client either - forwarding it and then appending the fallback's own
    message_start would produce two overlapping message lifecycles on one
    SSE stream. The primary's buffered lifecycle frame is discarded and the
    client sees only the fallback's own, single, clean lifecycle."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream(
        [_anthropic_messages_message_start_chunk(), _anthropic_messages_overloaded_error_chunk()]
    )
    fallback_message_start = b'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_2"}}\n\n'
    fallback_stream = _AnthropicMessagesFallbackByteStream(
        [fallback_message_start, _anthropic_messages_content_chunk("fallback answer")]
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [fallback_message_start, _anthropic_messages_content_chunk("fallback answer")]
    assert collected.count(_anthropic_messages_message_start_chunk()) == 0, (
        "the primary's message_start must never reach the client"
    )
    assert sum(1 for c in collected if c.startswith(b"event: message_start")) == 1, (
        "exactly one message_start must reach the client"
    )
    mock_fallback.assert_awaited_once()
    raised = mock_fallback.await_args.kwargs["e"]
    assert raised.is_pre_first_chunk is True


@pytest.mark.asyncio
async def test_anthropic_messages_raised_error_after_real_content_does_not_restart_stream():
    """Regression: a MidStreamFallbackError raised directly by the source
    iterator (the completion-bridge path's CustomStreamWrapper, e.g. a
    transport drop) must not trigger a fallback once real content already
    reached the client - that would append a second, overlapping message
    lifecycle onto the same SSE stream. The original exception must
    propagate to the caller instead."""
    router = _anthropic_messages_make_router()
    content = _anthropic_messages_content_chunk("partial answer")
    original_exception = litellm.APIError(
        status_code=503,
        message="stream reset",
        llm_provider="vertex_ai",
        model="primary",
    )
    raised_error = MidStreamFallbackError(
        message="stream reset",
        model="primary",
        llm_provider="vertex_ai",
        original_exception=original_exception,
        is_pre_first_chunk=False,
    )
    source = _AnthropicMessagesRaisingByteStream([content], raised_error)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = []

        async def _consume():
            async for chunk in wrapped:
                collected.append(chunk)

        with pytest.raises(litellm.APIError) as exc_info:
            await _consume()

    assert collected == [content]
    assert exc_info.value is original_exception
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_fallback_also_catches_raised_midstream_error():
    """Regression for the completion-bridge path (deployments with no native
    /v1/messages endpoint): its CustomStreamWrapper raises
    MidStreamFallbackError directly (e.g. on a Vertex AI transport drop)
    instead of yielding an SSE error chunk - the wrapper must catch that too."""
    router = _anthropic_messages_make_router()
    raised_error = MidStreamFallbackError(
        message="stream reset",
        model="primary",
        llm_provider="vertex_ai",
        is_pre_first_chunk=True,
    )
    source = _AnthropicMessagesRaisingByteStream([], raised_error)
    fallback_stream = _AnthropicMessagesFallbackByteStream([_anthropic_messages_content_chunk("fallback answer")])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(return_value=fallback_stream),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [_anthropic_messages_content_chunk("fallback answer")]
    mock_fallback.assert_awaited_once()
    assert mock_fallback.await_args.kwargs["e"] is raised_error


@pytest.mark.asyncio
async def test_anthropic_messages_non_retriable_client_error_skips_fallback():
    """A 4xx (non-429) error type (e.g. invalid_request_error) is a client
    error a fallback attempt cannot fix, so it must be forwarded to the
    client as-is rather than burning a fallback attempt."""
    router = _anthropic_messages_make_router()
    error_chunk = _anthropic_messages_invalid_request_error_chunk()
    source = _AnthropicMessagesFakeByteStream([error_chunk])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [error_chunk]
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_post_first_chunk_error_skips_fallback():
    """Once content has already reached the caller, retrying would start a
    second, overlapping Anthropic message lifecycle on the same SSE stream -
    the error must be forwarded instead of triggering an invisible retry."""
    router = _anthropic_messages_make_router()
    content = _anthropic_messages_content_chunk("partial answer")
    error_chunk = _anthropic_messages_overloaded_error_chunk()
    source = _AnthropicMessagesFakeByteStream([content, error_chunk])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [content, error_chunk]
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_non_retriable_error_flushes_buffered_lifecycle_frames():
    """A non-retriable error arriving while lifecycle frames are still
    buffered (no content seen yet) must flush those buffered frames before
    forwarding the error, so the client still sees the whole primary
    attempt rather than losing the buffered message_start silently."""
    router = _anthropic_messages_make_router()
    error_chunk = _anthropic_messages_invalid_request_error_chunk()
    source = _AnthropicMessagesFakeByteStream([_anthropic_messages_message_start_chunk(), error_chunk])

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = [chunk async for chunk in wrapped]

    assert collected == [_anthropic_messages_message_start_chunk(), error_chunk]
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_raised_error_declined_flushes_buffered_lifecycle_frames():
    """When a raised MidStreamFallbackError is declined (source says content
    was not pre-first-chunk) while lifecycle frames are still buffered, they
    must be flushed to the client before the exception propagates."""
    router = _anthropic_messages_make_router()
    raised_error = MidStreamFallbackError(
        message="stream reset",
        model="primary",
        llm_provider="vertex_ai",
        is_pre_first_chunk=False,
    )
    source = _AnthropicMessagesRaisingByteStream([_anthropic_messages_message_start_chunk()], raised_error)

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(),
    ) as mock_fallback:
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        collected = []

        async def _consume():
            async for chunk in wrapped:
                collected.append(chunk)

        with pytest.raises(MidStreamFallbackError) as exc_info:
            await _consume()

    assert collected == [_anthropic_messages_message_start_chunk()]
    assert exc_info.value is raised_error
    mock_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_messages_raised_error_without_original_exception_reraises_itself():
    """When a declined MidStreamFallbackError carries no original_exception,
    the bare exception itself must propagate rather than being swallowed."""
    router = _anthropic_messages_make_router()
    content = _anthropic_messages_content_chunk("partial answer")
    raised_error = MidStreamFallbackError(
        message="stream reset",
        model="primary",
        llm_provider="vertex_ai",
        is_pre_first_chunk=False,
    )
    source = _AnthropicMessagesRaisingByteStream([content], raised_error)

    wrapped = await router._aanthropic_messages_streaming_iterator(
        response=source,
        initial_kwargs={"model": "primary"},
    )
    collected = []

    async def _consume():
        async for chunk in wrapped:
            collected.append(chunk)

    with pytest.raises(MidStreamFallbackError) as exc_info:
        await _consume()

    assert collected == [content]
    assert exc_info.value is raised_error


@pytest.mark.asyncio
async def test_anthropic_messages_fallback_also_failing_raises_original_exception():
    """If the fallback attempt itself fails with a MidStreamFallbackError
    wrapping a real provider exception, the client must see that real
    exception, not the internal MidStreamFallbackError."""
    router = _anthropic_messages_make_router()
    source = _AnthropicMessagesFakeByteStream([_anthropic_messages_overloaded_error_chunk()])
    original_exception = litellm.APIError(
        status_code=503,
        message="fallback also overloaded",
        llm_provider="bedrock",
        model="fallback",
    )
    fallback_failure = MidStreamFallbackError(
        message="fallback failed",
        model="fallback",
        llm_provider="bedrock",
        original_exception=original_exception,
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        new=AsyncMock(side_effect=fallback_failure),
    ):
        wrapped = await router._aanthropic_messages_streaming_iterator(
            response=source,
            initial_kwargs={"model": "primary"},
        )
        with pytest.raises(litellm.APIError) as exc_info:
            async for _ in wrapped:
                pass

    assert exc_info.value is original_exception


# -------- _dispatch_generic_call_type --------


@pytest.mark.asyncio
async def test_dispatch_generic_call_type_routes_anthropic_messages_through_streaming_fallbacks():
    router = _anthropic_messages_make_router()

    async def fake_original(**_kwargs):
        return {"id": "msg_1"}

    with patch.object(
        router,
        "_aanthropic_messages_with_streaming_fallbacks",
        new=AsyncMock(return_value="anthropic-result"),
    ) as mock_anthropic:
        out = await router._dispatch_generic_call_type(
            call_type="anthropic_messages",
            original_function=fake_original,
            model="primary",
        )
    assert out == "anthropic-result"
    mock_anthropic.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_generic_call_type_other_call_types_use_generic_fallback():
    router = _anthropic_messages_make_router()

    async def fake_original(**_kwargs):
        return {"id": "file_1"}

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks",
        new=AsyncMock(return_value="generic-result"),
    ) as mock_generic:
        out = await router._dispatch_generic_call_type(
            call_type="afile_delete",
            original_function=fake_original,
            model="primary",
        )
    assert out == "generic-result"
    mock_generic.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_function_anthropic_messages_uses_streaming_fallback_dispatch():
    """anthropic_messages must be wired through the mid-stream-fallback-aware
    path rather than the bare generic dispatch every other call type without
    special handling uses."""
    router = _anthropic_messages_make_router()
    wrapped = router.factory_function(litellm.anthropic_messages, call_type="anthropic_messages")
    assert callable(wrapped)

    with patch.object(
        router,
        "_aanthropic_messages_with_streaming_fallbacks",
        new=AsyncMock(return_value="ok"),
    ) as mock_anthropic:
        result = await wrapped(model="primary")
    assert result == "ok"
    mock_anthropic.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_stamps_zero_attempted_fallbacks():
    """A request served by the primary model group records attempted_fallbacks=0 and
    the requested model group in metadata, mirroring the x-litellm-attempted-fallbacks header."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {}

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
    )

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_stamps_route_bucket_not_litellm_metadata():
    """A chat completion carrying both metadata buckets gets stamped in the route's bucket
    (metadata), matching where run_async_fallback rewrites, so the two never diverge."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {}
    litellm_metadata = {"client_key": "client_value"}

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
        litellm_metadata=litellm_metadata,
    )

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "gpt-3.5-turbo"
    assert litellm_metadata["client_key"] == "client_value"
    assert "attempted_fallbacks" not in litellm_metadata
    assert "original_model_group" not in litellm_metadata


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_overrides_client_supplied_stamp_values():
    """Client-supplied attempted_fallbacks and original_model_group are replaced on entry,
    so a reused metadata dict or a spoofed value cannot leak stale attribution into logs."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {"attempted_fallbacks": 99, "original_model_group": "stale-group"}

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
    )

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_stamps_despite_forged_reentry_params():
    """A client injecting fallback_depth or a JSON-shaped attempted_targets via request
    litellm params cannot skip the entry stamp; only the router's own in-process
    AttemptedFallbackTargets instance marks a genuine re-entrant hop."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {"attempted_fallbacks": 99, "original_model_group": "spoofed-group"}

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
        fallback_depth=3,
        attempted_targets={"keys": ["spoofed-group"]},
    )

    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_skips_stamp_on_genuine_reentrant_hop():
    """A re-entrant hop carrying the router's own AttemptedFallbackTargets instance keeps
    the per-hop metadata that run_async_fallback wrote instead of resetting it to zero."""
    from litellm.router_utils.fallback_event_handlers import AttemptedFallbackTargets

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {"attempted_fallbacks": 1, "original_model_group": "prod-chat"}

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
        attempted_targets=AttemptedFallbackTargets(keys=frozenset(("prod-chat",))),
    )

    assert metadata["attempted_fallbacks"] == 1
    assert metadata["original_model_group"] == "prod-chat"


@pytest.mark.asyncio
async def test_async_function_with_fallbacks_scrubs_spoofed_values_from_sibling_bucket():
    """Spend logs read a truthy litellm_metadata dict in preference to metadata, so spoofed
    stamp keys planted in the bucket the route does not own are removed on entry instead of
    flowing into the spend log row."""
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "hi"},
            }
        ]
    )
    metadata = {}
    litellm_metadata = {
        "attempted_fallbacks": 99,
        "original_model_group": "spoofed-group",
        "client_key": "client_value",
    }

    await router.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hey"}],
        metadata=metadata,
        litellm_metadata=litellm_metadata,
    )

    assert "attempted_fallbacks" not in litellm_metadata
    assert "original_model_group" not in litellm_metadata
    assert litellm_metadata["client_key"] == "client_value"
    assert metadata["attempted_fallbacks"] == 0
    assert metadata["original_model_group"] == "gpt-3.5-turbo"


def _permission_denied_error() -> litellm.PermissionDeniedError:
    return litellm.PermissionDeniedError(
        message="OpenrouterException - this key has no access to the model",
        llm_provider="openrouter",
        model="openrouter/openai/gpt-4o",
        response=httpx.Response(status_code=403, request=httpx.Request(method="POST", url="https://openrouter.ai")),
    )


def test_permission_denied_error_is_not_retried_against_a_single_deployment():
    router = litellm.Router(
        model_list=[
            {"model_name": "gpt-4o", "litellm_params": {"model": "openrouter/openai/gpt-4o", "api_key": "sk-test"}},
        ]
    )

    with pytest.raises(litellm.PermissionDeniedError):
        router.should_retry_this_error(
            error=_permission_denied_error(),
            healthy_deployments=router.model_list,
            all_deployments=router.model_list,
        )


def test_permission_denied_error_is_retried_when_other_deployments_exist():
    router = litellm.Router(
        model_list=[
            {"model_name": "gpt-4o", "litellm_params": {"model": "openrouter/openai/gpt-4o", "api_key": "sk-test"}},
            {"model_name": "gpt-4o", "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"}},
        ]
    )

    assert (
        router.should_retry_this_error(
            error=_permission_denied_error(),
            healthy_deployments=router.model_list,
            all_deployments=router.model_list,
        )
        is True
    )
