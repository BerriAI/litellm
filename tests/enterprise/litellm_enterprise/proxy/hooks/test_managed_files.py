import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

from litellm.caching import DualCache
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
)


def test_get_file_ids_from_messages():
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=MagicMock()
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this recording?"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg",
                    },
                },
            ],
        },
    ]
    file_ids = proxy_managed_files.get_file_ids_from_messages(messages)
    assert file_ids == [
        "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCxmYzdmMmVhNS0wZjUwLTQ5ZjYtODljMS03ZTZhNTRiMTIxMzg"
    ]


@pytest.mark.asyncio
async def test_async_pre_call_hook_batch_retrieve():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    return_value = MagicMock()
    return_value.created_by = "123"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = return_value
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    data = {
        "user_api_key_dict": UserAPIKeyAuth(
            user_id="123", parent_otel_span=MagicMock()
        ),
        "data": {
            "batch_id": "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDpteS1nZW5lcmFsLWF6dXJlLWRlcGxveW1lbnQ7bGxtX2JhdGNoX2lkOmJhdGNoX2EzMjJiNmJhLWFjN2UtNDg4OC05MjljLTFhZDM0NDJmMDZlZA",
        },
        "call_type": "aretrieve_batch",
        "cache": MagicMock(),
    }
    response = await proxy_managed_files.async_pre_call_hook(**data)
    assert response["batch_id"] == "batch_a322b6ba-ac7e-4888-929c-1ad3442f06ed"
    assert response["model"] == "my-general-azure-deployment"


# def test_list_managed_files():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create some test files
#     file1 = proxy_managed_files.create_file(
#         file=("test1.txt", b"test content 1", "text/plain"),
#         purpose="assistants"
#     )
#     file2 = proxy_managed_files.create_file(
#         file=("test2.pdf", b"test content 2", "application/pdf"),
#         purpose="assistants"
#     )

#     # List all files
#     files = proxy_managed_files.list_files()

#     # Verify response
#     assert len(files) == 2
#     assert all(f.id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value) for f in files)
#     assert any(f.filename == "test1.txt" for f in files)
#     assert any(f.filename == "test2.pdf" for f in files)
#     assert all(f.purpose == "assistants" for f in files)

# def test_retrieve_managed_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create a test file
#     test_content = b"test content for retrieve"
#     created_file = proxy_managed_files.create_file(
#         file=("test.txt", test_content, "text/plain"),
#         purpose="assistants"
#     )

#     # Retrieve the file
#     retrieved_file = proxy_managed_files.retrieve_file(created_file.id)

#     # Verify response
#     assert retrieved_file.id == created_file.id
#     assert retrieved_file.filename == "test.txt"
#     assert retrieved_file.purpose == "assistants"
#     assert retrieved_file.bytes == len(test_content)
#     assert retrieved_file.status == "uploaded"

# def test_delete_managed_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create a test file
#     created_file = proxy_managed_files.create_file(
#         file=("test.txt", b"test content", "text/plain"),
#         purpose="assistants"
#     )

#     # Delete the file
#     deleted_file = proxy_managed_files.delete_file(created_file.id)

#     # Verify deletion
#     assert deleted_file.id == created_file.id
#     assert deleted_file.deleted == True

#     # Verify file is no longer retrievable
#     with pytest.raises(Exception):
#         proxy_managed_files.retrieve_file(created_file.id)

#     # Verify file is not in list
#     files = proxy_managed_files.list_files()
#     assert created_file.id not in [f.id for f in files]

# def test_retrieve_nonexistent_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Try to retrieve a non-existent file
#     with pytest.raises(Exception):
#         proxy_managed_files.retrieve_file("nonexistent-file-id")

# def test_delete_nonexistent_file():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Try to delete a non-existent file
#     with pytest.raises(Exception):
#         proxy_managed_files.delete_file("nonexistent-file-id")

# def test_list_files_with_purpose_filter():
#     proxy_managed_files = _PROXY_LiteLLMManagedFiles(DualCache())

#     # Create files with different purposes
#     file1 = proxy_managed_files.create_file(
#         file=("test1.txt", b"test content 1", "text/plain"),
#         purpose="assistants"
#     )
#     file2 = proxy_managed_files.create_file(
#         file=("test2.pdf", b"test content 2", "application/pdf"),
#         purpose="batch"
#     )

#     # List files with purpose filter
#     assistant_files = proxy_managed_files.list_files(purpose="assistants")
#     batch_files = proxy_managed_files.list_files(purpose="batch")

#     # Verify filtering
#     assert len(assistant_files) == 1
#     assert len(batch_files) == 1
#     assert assistant_files[0].id == file1.id
#     assert batch_files[0].id == file2.id


@pytest.mark.asyncio
async def test_async_post_call_success_hook_for_unified_finetuning_job():
    from litellm.types.utils import LiteLLMFineTuningJob

    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW07dW5pZmllZF9pZCxiZTQ0ZDVlYi1mNDU3LTRiNzktOWM4My01N2QxMTMxYWM0YzY7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00LjEtb3BlbmFpO2xsbV9vdXRwdXRfZmlsZV9pZCxmaWxlLURKMnQ0OWZlQ2NTQk5vNG9oekZ6NGc7bGxtX291dHB1dF9maWxlX21vZGVsX2lkLGRiNjY5ODcwNzdkZTdmYzZjNzAzY2Y1MDczMGU2MmNkOWQ3YTU1N2NlNjVmMDUzNTFkYTM4YTA3ZjBlZDEyNzQ"
    provider_ft_job = LiteLLMFineTuningJob(
        object="fine_tuning.job",
        id="ftjob-0kEBV5b4sPrFcMnuzmYSzU1G",
        model="gpt-3.5-turbo-0613",
        created_at=1692779769,
        finished_at=None,
        fine_tuned_model=None,
        organization_id="org-dUVLhaAQ37YCGwVC2QVY8sdB",
        result_files=[],
        status="validating_files",
        validation_file=None,
        training_file="file-azQuKMLAmiFdEjxpCcbI11zF",
        hyperparameters={"n_epochs": 8},
        trained_tokens=None,
        seed=0,
    )
    provider_ft_job._hidden_params = {
        "unified_file_id": unified_file_id,
        "model_id": "gpt-3.5-turbo-0613",
    }
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=AsyncMock()
    )
    data = {
        "user_api_key_dict": {"parent_otel_span": MagicMock()},
    }

    response = await proxy_managed_files.async_post_call_success_hook(
        data=data,
        user_api_key_dict=MagicMock(),
        response=provider_ft_job,
    )

    assert isinstance(response, LiteLLMFineTuningJob)
    assert _is_base64_encoded_unified_file_id(response.id)


@pytest.mark.asyncio
async def test_async_pre_call_hook_for_unified_finetuning_job():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    return_value = MagicMock()
    return_value.created_by = "123"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = return_value
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    data = {
        "user_api_key_dict": UserAPIKeyAuth(
            user_id="123", parent_otel_span=MagicMock()
        ),
        "data": {
            "fine_tuning_job_id": "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDo0OTIxODU4MWY3OGViZTllZjE4NDE0ZmE0ZjdmYjlmYTc0YzA5NWVkMTEyY2E4NDBkZDU2ZGZmZTliZDMwZGQxO2dlbmVyaWNfcmVzcG9uc2VfaWQ6ZnRqb2ItalRCeXM3YlZzYnlaRE93TDlHbHBZcVhS",
        },
        "call_type": "acancel_fine_tuning_job",
        "cache": MagicMock(),
    }

    response = await proxy_managed_files.async_pre_call_hook(**data)
    assert response["fine_tuning_job_id"] == "ftjob-jTBys7bVsbyZDOwL9GlpYqXR"


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["afile_content", "afile_delete", "afile_retrieve"])
async def test_can_user_call_unified_file_id(call_type):
    """
    Test that on file retrieve, delete, and content we check if the user has access to the file
    """
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    return_value = MagicMock()
    return_value.created_by = "123"
    prisma_client.db.litellm_managedfiletable.find_first.return_value = return_value
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        MagicMock(), prisma_client=prisma_client
    )
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW07dW5pZmllZF9pZCxmMTNlNDAzZS01YWM3LTRhZjktOGQzNS0wNDgwZDMxOTgyYTg7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00by1taW5pLW9wZW5haTtsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1Ib3UxZDFXc3c1SDNKcjFMYllpZDJiO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxmODBiNWU2NzQ1NzdkNjkyMjM4YmVhNTIxZDdiMGI5ZGYyY2FmMTEwMTU2YmU5YzBjM2NjMmNkNTBjOTM1ZDI0"

    with pytest.raises(HTTPException):
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="456", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"file_id": unified_file_id},
            call_type=call_type,
        )


@pytest.mark.asyncio
async def test_router_acreate_batch_only_selects_from_file_id_mapping(monkeypatch):
    """
    Test that router.acreate_batch only selects model_id from the file_id_mapping
    """
    import litellm

    prisma_client = AsyncMock()
    return_value = MagicMock()
    return_value.created_by = "123"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = return_value
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )

    monkeypatch.setattr(
        litellm,
        "callbacks",
        [proxy_managed_files],
    )

    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "1234"},
            },
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {"model": "gpt-3.5-turbo"},
                "model_info": {"id": "5678"},
            },
        ],
    )

    file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW07dW5pZmllZF9pZCw2YmQ4ZjhhYS02NmEzLTRmY2MtOTIxZS1lMTYwYzIzZWZjNzU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1MTENVRkI1MnVUTWE5aE5ZanRldzlWO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxmMzJlNWQ0OC05YWZmLTQ5YjMtOWE1Ny0zYzJhN2JjN2NjMmE"

    model_file_id_mapping = {file_id: {"5678": "file-LLCUFB52uTMa9hNYjtew9V"}}

    with patch.object(
        litellm, "acreate_batch", return_value=AsyncMock()
    ) as mock_acreate_batch:
        for _ in range(1000):
            await router.acreate_batch(
                model="gpt-3.5-turbo",
                input_file_id=file_id,
                model_file_id_mapping=model_file_id_mapping,
            )

            mock_acreate_batch.assert_called()
            assert "5678" in json.dumps(mock_acreate_batch.call_args.kwargs)


@pytest.mark.asyncio
async def test_output_file_id_for_batch_retrieve():
    """
    Test that the output file id is the same as the input file id
    """
    from typing import cast

    from openai.types.batch import BatchRequestCounts

    from litellm.types.utils import LiteLLMBatch

    batch = LiteLLMBatch(
        id="bGl0ZWxsbV9wcm94eTttb2RlbF9pZDoxMjM0NTY3OTtsbG1fYmF0Y2hfaWQ6YmF0Y2hfNjg1YzVlNWQ2Mzk4ODE5MGI4NWJkYjIxNDdiYTEzMWQ",
        completion_window="24h",
        created_at=1750883933,
        endpoint="/v1/chat/completions",
        input_file_id="file-8ci8gux8s7oES7GydYvnMG",
        object="batch",
        status="completed",
        cancelled_at=None,
        cancelling_at=None,
        completed_at=1750883939,
        error_file_id=None,
        errors=None,
        expired_at=None,
        expires_at=1750970333,
        failed_at=None,
        finalizing_at=1750883938,
        in_progress_at=1750883934,
        metadata={"description": "nightly eval job"},
        output_file_id="file-3BZYhmdJQ3V2oZPAtQsEax",
        request_counts=BatchRequestCounts(completed=1, failed=0, total=1),
        usage=None,
    )

    batch._hidden_params = {
        "litellm_call_id": "dcd789e0-c0ad-4244-9564-4e611448d650",
        "api_base": "https://api.openai.com",
        "model_id": "12345679",
        "response_cost": 0.0,
        "additional_headers": {},
        "litellm_model_name": "gpt-4o",
        "unified_batch_id": "litellm_proxy;model_id:12345679;llm_batch_id:batch_685c5e5d63988190b85bdb2147ba131d",
    }
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=AsyncMock()
    )

    response = await proxy_managed_files.async_post_call_success_hook(
        data={},
        user_api_key_dict=MagicMock(),
        response=batch,
    )

    assert not cast(LiteLLMBatch, response).output_file_id.startswith("file-")


@pytest.mark.asyncio
async def test_error_file_id_for_failed_batch():
    """
    Test that the error_file_id is properly managed when a batch fails
    """
    from typing import cast

    from openai.types.batch import BatchRequestCounts

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import OpenAIFileObject
    from litellm.types.utils import LiteLLMBatch

    batch = LiteLLMBatch(
        id="bGl0ZWxsbV9wcm94eTttb2RlbF9pZDoxMjM0NTY3OTtsbG1fYmF0Y2hfaWQ6YmF0Y2hfYWJjMTIz",
        completion_window="24h",
        created_at=1714508499,
        endpoint="/v1/chat/completions",
        input_file_id="file-abc123",
        object="batch",
        status="failed",
        cancelled_at=None,
        cancelling_at=None,
        completed_at=None,
        error_file_id="error-abc123",
        errors=None,
        expired_at=None,
        expires_at=1714536634,
        failed_at=None,
        finalizing_at=None,
        in_progress_at=None,
        metadata=None,
        output_file_id=None,
        request_counts=BatchRequestCounts(completed=0, failed=0, total=0),
        usage=None,
    )

    batch._hidden_params = {
        "litellm_call_id": "test-call-id",
        "api_base": "https://api.openai.com",
        "model_id": "test-model-id",
        "model_name": "gpt-4o",
        "response_cost": 0.0,
        "additional_headers": {},
        "litellm_model_name": "gpt-4o",
        "unified_batch_id": "litellm_proxy;model_id:test-model-id;llm_batch_id:batch_abc123",
    }
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=AsyncMock()
    )

    # Create a proper OpenAIFileObject for the error file
    error_file_object = OpenAIFileObject(
        id="error-abc123",
        object="file",
        bytes=1234,
        created_at=1714508500,
        filename="error.jsonl",
        purpose="batch_output",
        status="processed",
    )

    # Mock the afile_retrieve to simulate retrieving error file metadata
    with patch("litellm.afile_retrieve", new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = error_file_object
        
        user_api_key_dict = UserAPIKeyAuth(
            user_id="test-user-123",
            parent_otel_span=MagicMock()
        )
        
        response = await proxy_managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=user_api_key_dict,
            response=batch,
        )

    # Verify that error_file_id was transformed to a managed file ID
    assert cast(LiteLLMBatch, response).error_file_id is not None
    assert not cast(LiteLLMBatch, response).error_file_id.startswith("error-")
    # Verify it's a base64 encoded managed file ID
    assert _is_base64_encoded_unified_file_id(cast(LiteLLMBatch, response).error_file_id)


@pytest.mark.asyncio
async def test_async_post_call_success_hook_twice_assert_no_unique_violation():
    import asyncio

    from openai.types.batch import BatchRequestCounts

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import LiteLLMBatch

    # Use AsyncMock instead of real database connection
    prisma_client = AsyncMock()
    
    batch = LiteLLMBatch(
        id="bGl0ZWxsbV9wcm94eTttb2RlbF9pZDoxMjM0NTY3OTtsbG1fYmF0Y2hfaWQ6YmF0Y2hfNjg1YzVlNWQ2Mzk4ODE5MGI4NWJkYjIxNDdiYTEzMWQ",
        completion_window="24h",
        created_at=1750883933,
        endpoint="/v1/chat/completions",
        input_file_id="file-8ci8gux8s7oES7GydYvnMG",
        object="batch",
        status="completed",
        metadata={"description": "nightly eval job"},
        request_counts=BatchRequestCounts(completed=1, failed=0, total=1),
        usage=None,
    )

    batch._hidden_params = {
        "model_id": "12345679",
        "response_cost": 0.0,
        "litellm_model_name": "gpt-4o",
        "unified_batch_id": "litellm_proxy;model_id:12345679;llm_batch_id:batch_685c5e5d63988190b85bdb2147ba131d",
    }

    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )

    # first retrieve batch
    tasks = []
    first_create_task = asyncio.create_task
    with patch('asyncio.create_task') as mock_create_task:
        mock_create_task.side_effect = lambda coro: tasks.append(first_create_task(coro)) or tasks[-1]

        response = await proxy_managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=UserAPIKeyAuth(user_id="default_id"),
            response=batch.copy(),
        )

        if tasks:
            # make sure asyncio(db create) is finished
            await asyncio.sleep(0.02)
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                assert task.exception() is None, f"Error: {task.exception()}"

            assert isinstance(response, LiteLLMBatch)
            assert _is_base64_encoded_unified_file_id(response.id)

    # second retrieve batch
    tasks = []
    second_create_task = asyncio.create_task
    with patch('asyncio.create_task') as mock_create_task:
        mock_create_task.side_effect = lambda coro: tasks.append(second_create_task(coro)) or tasks[-1]

        await proxy_managed_files.async_post_call_success_hook(
            data={},
            user_api_key_dict=UserAPIKeyAuth(user_id="default_id"),
            response=batch.copy(),
        )

        if tasks:
            await asyncio.sleep(0.01)
            await asyncio.gather(*tasks, return_exceptions=True)
            for task in tasks:
                assert task.exception() is None, f"Error: {task.exception()}"


def test_update_responses_input_with_unified_file_id():
    """
    Test that update_responses_input_with_model_file_ids correctly decodes
    unified file IDs and extracts llm_output_file_id from responses API input.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    # Create a base64-encoded unified file ID
    # This decodes to: litellm_proxy:application/pdf;unified_id,6c0b5890-8914-48e0-b8f4-0ae5ed3c14a5;target_model_names,gpt-4o;llm_output_file_id,file-ECBPW7ML9g7XHdwGgUPZaM;llm_output_file_model_id,e26453f9e76e7993680d0068d98c1f4cc205bbad0967a33c664893568ca743c2
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    
    # Test input with unified file ID in content array
    input_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": unified_file_id,
                },
                {
                    "type": "input_text",
                    "text": "What is the first dragon in the book?",
                },
            ],
        }
    ]
    
    # Update the input
    updated_input = update_responses_input_with_model_file_ids(input=input_data)
    
    # Verify the file_id was updated to the provider-specific file ID
    assert updated_input[0]["content"][0]["type"] == "input_file"
    assert updated_input[0]["content"][0]["file_id"] == "file-ECBPW7ML9g7XHdwGgUPZaM"
    assert updated_input[0]["content"][1]["type"] == "input_text"
    assert updated_input[0]["content"][1]["text"] == "What is the first dragon in the book?"


def test_update_responses_input_with_regular_file_id():
    """
    Test that update_responses_input_with_model_file_ids keeps regular
    OpenAI file IDs unchanged.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    # Regular OpenAI file ID (not a unified file ID)
    regular_file_id = "file-abc123xyz"
    
    input_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": regular_file_id,
                },
                {
                    "type": "input_text",
                    "text": "What is this file?",
                },
            ],
        }
    ]
    
    # Update the input
    updated_input = update_responses_input_with_model_file_ids(input=input_data)
    
    # Verify the file_id was kept unchanged (regular OpenAI file ID)
    assert updated_input[0]["content"][0]["type"] == "input_file"
    assert updated_input[0]["content"][0]["file_id"] == regular_file_id
    assert updated_input[0]["content"][1]["type"] == "input_text"


def test_update_responses_input_with_string_input():
    """
    Test that update_responses_input_with_model_file_ids returns string input unchanged.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )
    
    input_data = "What is AI?"
    
    updated_input = update_responses_input_with_model_file_ids(input=input_data)
    
    assert updated_input == input_data
    assert isinstance(updated_input, str)


def test_update_responses_input_with_multiple_file_ids():
    """
    Test that update_responses_input_with_model_file_ids handles multiple file IDs
    (both unified and regular) in the same input.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    # Unified file ID
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    # Regular OpenAI file ID
    regular_file_id = "file-regular123"
    
    input_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": unified_file_id,
                },
                {
                    "type": "input_text",
                    "text": "Compare these files",
                },
                {
                    "type": "input_file",
                    "file_id": regular_file_id,
                },
            ],
        }
    ]
    
    updated_input = update_responses_input_with_model_file_ids(input=input_data)
    
    # Verify unified file ID was updated
    assert updated_input[0]["content"][0]["file_id"] == "file-ECBPW7ML9g7XHdwGgUPZaM"
    # Verify regular file ID was kept unchanged
    assert updated_input[0]["content"][2]["file_id"] == regular_file_id
    # Verify text content was preserved
    assert updated_input[0]["content"][1]["text"] == "Compare these files"


def test_update_responses_input_with_model_file_id_mapping():
    """
    Test that update_responses_input_with_model_file_ids correctly uses
    model_file_id_mapping to map managed file IDs to provider-specific file IDs.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_input_with_model_file_ids,
    )

    # Managed file ID (unified)
    managed_file_id = "litellm_proxy_file_123"
    
    # Model file ID mapping
    model_file_id_mapping = {
        managed_file_id: {
            "model_id_1": "openai_file_abc",
            "model_id_2": "azure_file_xyz",
        }
    }
    
    input_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_id": managed_file_id,
                },
                {
                    "type": "input_text",
                    "text": "Analyze this file",
                },
            ],
        }
    ]
    
    # Update input with model_id_1 mapping
    updated_input = update_responses_input_with_model_file_ids(
        input=input_data,
        model_id="model_id_1",
        model_file_id_mapping=model_file_id_mapping,
    )
    
    # Verify the file_id was mapped to the correct provider-specific file ID
    assert updated_input[0]["content"][0]["file_id"] == "openai_file_abc"
    
    # Test with different model_id
    updated_input_2 = update_responses_input_with_model_file_ids(
        input=input_data,
        model_id="model_id_2",
        model_file_id_mapping=model_file_id_mapping,
    )
    
    assert updated_input_2[0]["content"][0]["file_id"] == "azure_file_xyz"


def test_update_responses_tools_with_model_file_id_mapping():
    """
    Test that update_responses_tools_with_model_file_ids correctly maps
    file IDs in code_interpreter tools with container.file_ids.
    
    This is a regression test for the issue where managed file IDs in
    tools.container.file_ids were not being replaced with provider-specific
    file IDs, causing "string too long" errors from OpenAI.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_tools_with_model_file_ids,
    )

    # Managed file IDs
    managed_file_id_1 = "litellm_proxy_file_123"
    managed_file_id_2 = "litellm_proxy_file_456"
    
    # Model file ID mapping
    model_file_id_mapping = {
        managed_file_id_1: {
            "model_id_1": "openai_file_abc",
        },
        managed_file_id_2: {
            "model_id_1": "openai_file_def",
        },
    }
    
    tools = [
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": [managed_file_id_1, managed_file_id_2],
            },
        }
    ]
    
    # Update tools with model mapping
    updated_tools = update_responses_tools_with_model_file_ids(
        tools=tools,
        model_id="model_id_1",
        model_file_id_mapping=model_file_id_mapping,
    )
    
    # Verify the file IDs were mapped to provider-specific file IDs
    assert updated_tools[0]["type"] == "code_interpreter"
    assert updated_tools[0]["container"]["file_ids"] == ["openai_file_abc", "openai_file_def"]


def test_update_responses_tools_without_mapping():
    """
    Test that update_responses_tools_with_model_file_ids keeps file IDs
    unchanged when no mapping is provided.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_tools_with_model_file_ids,
    )

    regular_file_id = "file-abc123"
    
    tools = [
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": [regular_file_id],
            },
        }
    ]
    
    # Update tools without mapping
    updated_tools = update_responses_tools_with_model_file_ids(
        tools=tools,
        model_id=None,
        model_file_id_mapping=None,
    )
    
    # Verify the file ID was kept unchanged
    assert updated_tools[0]["container"]["file_ids"] == [regular_file_id]


def test_update_responses_tools_with_mixed_file_ids():
    """
    Test that update_responses_tools_with_model_file_ids correctly handles
    a mix of managed and regular file IDs.
    """
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        update_responses_tools_with_model_file_ids,
    )

    managed_file_id = "litellm_proxy_file_123"
    regular_file_id = "file-abc123"
    
    model_file_id_mapping = {
        managed_file_id: {
            "model_id_1": "openai_file_abc",
        },
    }
    
    tools = [
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": [managed_file_id, regular_file_id],
            },
        }
    ]
    
    # Update tools
    updated_tools = update_responses_tools_with_model_file_ids(
        tools=tools,
        model_id="model_id_1",
        model_file_id_mapping=model_file_id_mapping,
    )
    
    # Verify managed file ID was mapped and regular file ID was kept
    assert updated_tools[0]["container"]["file_ids"] == ["openai_file_abc", regular_file_id]


def test_get_file_ids_from_responses_tools():
    """
    Test that get_file_ids_from_responses_tools correctly extracts
    file IDs from the tools parameter.
    """
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=MagicMock()
    )
    
    tools = [
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file-123", "file-456"],
            },
        }
    ]
    
    file_ids = proxy_managed_files.get_file_ids_from_responses_tools(tools)
    
    assert file_ids == ["file-123", "file-456"]


def test_get_file_ids_from_responses_tools_multiple_tools():
    """
    Test that get_file_ids_from_responses_tools handles multiple tools.
    """
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=MagicMock()
    )
    
    tools = [
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file-123"],
            },
        },
        {
            "type": "file_search",
        },
        {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file-456", "file-789"],
            },
        },
    ]
    
    file_ids = proxy_managed_files.get_file_ids_from_responses_tools(tools)
    
    # Should extract file IDs only from code_interpreter tools
    assert file_ids == ["file-123", "file-456", "file-789"]


def test_get_file_ids_from_responses_tools_empty():
    """
    Test that get_file_ids_from_responses_tools handles empty or None tools.
    """
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=MagicMock()
    )
    
    # Test with None
    file_ids = proxy_managed_files.get_file_ids_from_responses_tools(None)
    assert file_ids == []
    
    # Test with empty list
    file_ids = proxy_managed_files.get_file_ids_from_responses_tools([])
    assert file_ids == []
    
    # Test with tools without file_ids
    tools = [{"type": "file_search"}]
    file_ids = proxy_managed_files.get_file_ids_from_responses_tools(tools)
    assert file_ids == []


@pytest.mark.asyncio
async def test_check_file_ids_access_with_unified_file_ids():
    """
    Test that check_file_ids_access validates user access to managed file IDs.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    # Create a unified file ID
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    regular_file_id = "file-abc123"
    
    # Mock the access check to return True
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock can_user_call_unified_file_id to return True
    proxy_managed_files.can_user_call_unified_file_id = AsyncMock(return_value=True)
    
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test_user_123",
        parent_otel_span=MagicMock(),
    )
    
    # Should not raise an exception for accessible files
    await proxy_managed_files.check_file_ids_access(
        [unified_file_id, regular_file_id],
        user_api_key_dict,
    )
    
    # Verify can_user_call_unified_file_id was called for the unified file ID
    proxy_managed_files.can_user_call_unified_file_id.assert_called_once_with(
        unified_file_id, user_api_key_dict
    )


@pytest.mark.asyncio
async def test_check_file_ids_access_denied():
    """
    Test that check_file_ids_access raises HTTPException when user doesn't have access.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock can_user_call_unified_file_id to return False (access denied)
    proxy_managed_files.can_user_call_unified_file_id = AsyncMock(return_value=False)
    
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test_user_123",
        parent_otel_span=MagicMock(),
    )
    
    # Should raise HTTPException with 403 status code
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.check_file_ids_access(
            [unified_file_id],
            user_api_key_dict,
        )
    
    assert exc_info.value.status_code == 403
    assert "does not have access to the file" in exc_info.value.detail


@pytest.mark.asyncio
async def test_check_file_ids_access_with_regular_files_only():
    """
    Test that check_file_ids_access doesn't check access for regular (non-unified) file IDs.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    regular_file_id_1 = "file-abc123"
    regular_file_id_2 = "file-xyz789"
    
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock can_user_call_unified_file_id (should not be called for regular files)
    proxy_managed_files.can_user_call_unified_file_id = AsyncMock()
    
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test_user_123",
        parent_otel_span=MagicMock(),
    )
    
    # Should not raise exception and should not call can_user_call_unified_file_id
    await proxy_managed_files.check_file_ids_access(
        [regular_file_id_1, regular_file_id_2],
        user_api_key_dict,
    )
    
    # Verify can_user_call_unified_file_id was NOT called
    proxy_managed_files.can_user_call_unified_file_id.assert_not_called()


@pytest.mark.asyncio
async def test_completion_with_file_access_check():
    """
    Test that completion call type checks file access before processing.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    
    prisma_client = AsyncMock()
    prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    
    internal_usage_cache = MagicMock()
    internal_usage_cache.async_get_cache = AsyncMock(return_value=None)
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock the get_model_file_id_mapping to return empty dict
    proxy_managed_files.get_model_file_id_mapping = AsyncMock(return_value={})
    
    # Mock access check to allow access
    proxy_managed_files.can_user_call_unified_file_id = AsyncMock(return_value=True)
    
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test_user_123",
        parent_otel_span=MagicMock(),
    )
    
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this file?"},
                    {
                        "type": "file",
                        "file": {"file_id": unified_file_id},
                    },
                ],
            }
        ],
        "model": "gpt-4",
    }
    
    # Should not raise exception
    result = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="acompletion",
    )
    
    # Verify access check was called
    proxy_managed_files.can_user_call_unified_file_id.assert_called_once()


@pytest.mark.asyncio
async def test_responses_with_file_access_check():
    """
    Test that responses API checks file access for files in both input and tools.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    unified_file_id_1 = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9wZGY7dW5pZmllZF9pZCw2YzBiNTg5MC04OTE0LTQ4ZTAtYjhmNC0wYWU1ZWQzYzE0YTU7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1FQ0JQVzdNTDlnN1hIZHdHZ1VQWmFNO2xsbV9vdXRwdXRfZmlsZV9tb2RlbF9pZCxlMjY0NTNmOWU3NmU3OTkzNjgwZDAwNjhkOThjMWY0Y2MyMDViYmFkMDk2N2EzM2M2NjQ4OTM1NjhjYTc0M2My"
    unified_file_id_2 = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsNzc3Nzc3Nzc7dGFyZ2V0X21vZGVsX25hbWVzLGdwdC00bztsbG1fb3V0cHV0X2ZpbGVfaWQsZmlsZS1YWVo7bGxtX291dHB1dF9maWxlX21vZGVsX2lkLG1vZGVsXzEyMw"
    
    prisma_client = AsyncMock()
    prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(return_value=None)
    
    internal_usage_cache = MagicMock()
    internal_usage_cache.async_get_cache = AsyncMock(return_value=None)
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock the get_model_file_id_mapping to return empty dict
    proxy_managed_files.get_model_file_id_mapping = AsyncMock(return_value={})
    
    # Mock access check to allow access
    proxy_managed_files.can_user_call_unified_file_id = AsyncMock(return_value=True)
    
    user_api_key_dict = UserAPIKeyAuth(
        user_id="test_user_123",
        parent_otel_span=MagicMock(),
    )
    
    data = {
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Analyze this"},
                    {"type": "input_file", "file_id": unified_file_id_1},
                ],
            }
        ],
        "tools": [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": [unified_file_id_2],
                },
            }
        ],
        "model": "gpt-4",
    }
    
    # Should not raise exception
    result = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="aresponses",
    )
    
    # Verify access check was called for both file IDs
    assert proxy_managed_files.can_user_call_unified_file_id.call_count == 2


@pytest.mark.asyncio
async def test_store_unified_file_id_with_none_file_object():
    """
    Test that store_unified_file_id works when file_object is None
    (e.g., for batch output files that are stored before file metadata is available).
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    prisma_client.db.litellm_managedfiletable.create = AsyncMock(return_value=MagicMock())
    internal_usage_cache = MagicMock()
    internal_usage_cache.async_set_cache = AsyncMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Store with file_object=None (simulating batch output file storage)
    await proxy_managed_files.store_unified_file_id(
        file_id="test-unified-file-id",
        file_object=None,
        litellm_parent_otel_span=None,
        model_mappings={"model-123": "file-provider-xyz"},
        user_api_key_dict=UserAPIKeyAuth(user_id="test-user"),
    )
    
    # Verify DB create was called with expected data (without file_object)
    prisma_client.db.litellm_managedfiletable.create.assert_called_once()
    call_args = prisma_client.db.litellm_managedfiletable.create.call_args
    assert call_args.kwargs["data"]["unified_file_id"] == "test-unified-file-id"
    assert "file_object" not in call_args.kwargs["data"]


@pytest.mark.asyncio
async def test_afile_delete_returns_provider_response_when_stored_file_object_none():
    """
    Test that afile_delete returns the provider's delete response when the
    stored file_object is None (e.g., for batch output files).
    """
    from litellm.types.llms.openai import OpenAIFileObject
    
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsdGVzdC1pZDt0YXJnZXRfbW9kZWxfbmFtZXMsZ3B0LTRvO2xsbV9vdXRwdXRfZmlsZV9pZCxmaWxlLXByb3ZpZGVyLXh5ejtsbG1fb3V0cHV0X2ZpbGVfbW9kZWxfaWQsbW9kZWwtMTIz"
    
    prisma_client = AsyncMock()
    db_record = MagicMock()
    db_record.model_mappings = '{"model-123": "file-provider-xyz"}'
    prisma_client.db.litellm_managedfiletable.find_first = AsyncMock(return_value=db_record)
    prisma_client.db.litellm_managedfiletable.delete = AsyncMock()
    
    internal_usage_cache = MagicMock()
    internal_usage_cache.async_get_cache = AsyncMock(return_value={
        "unified_file_id": unified_file_id,
        "model_mappings": {"model-123": "file-provider-xyz"},
        "flat_model_file_ids": ["file-provider-xyz"],
        "file_object": None,
        "created_by": "test-user",
        "updated_by": "test-user",
    })
    internal_usage_cache.async_set_cache = AsyncMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock the delete_unified_file_id to return None (simulating file_object=None)
    proxy_managed_files.delete_unified_file_id = AsyncMock(return_value=None)
    
    # Mock router response
    provider_delete_response = OpenAIFileObject(
        id="file-provider-xyz",
        object="file",
        bytes=1234,
        created_at=1234567890,
        filename="test.jsonl",
        purpose="batch",
    )
    
    mock_router = MagicMock()
    mock_router.afile_delete = AsyncMock(return_value=provider_delete_response)
    
    result = await proxy_managed_files.afile_delete(
        file_id=unified_file_id,
        litellm_parent_otel_span=None,
        llm_router=mock_router,
    )
    
    # Should return the provider response with the unified file ID
    assert result is not None
    assert result.id == unified_file_id


@pytest.mark.asyncio
async def test_afile_retrieve_fetches_from_provider_when_file_object_none():
    """
    Test that afile_retrieve fetches from the provider when the stored
    file_object is None (e.g., for batch output files).
    """
    from litellm.types.llms.openai import OpenAIFileObject
    
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock get_unified_file_id to return a stored object with file_object=None
    stored_file = MagicMock()
    stored_file.file_object = None
    stored_file.model_mappings = {"model-123": "file-provider-xyz"}
    proxy_managed_files.get_unified_file_id = AsyncMock(return_value=stored_file)
    
    # Mock the router and provider response
    provider_file_response = OpenAIFileObject(
        id="file-provider-xyz",
        object="file",
        bytes=5678,
        created_at=1234567890,
        filename="output.jsonl",
        purpose="batch_output",
    )
    
    mock_router = MagicMock()
    mock_router.get_deployment_credentials_with_provider = MagicMock(return_value={
        "api_key": "test-key",
        "api_base": "https://api.openai.com",
    })
    
    with patch("litellm.afile_retrieve", new_callable=AsyncMock) as mock_afile_retrieve:
        mock_afile_retrieve.return_value = provider_file_response
        
        unified_file_id = "test-unified-file-id"
        result = await proxy_managed_files.afile_retrieve(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=mock_router,
        )
        
        # Should return the provider response with the unified file ID
        assert result is not None
        assert result.id == unified_file_id
        mock_afile_retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_afile_retrieve_raises_error_when_no_router_and_file_object_none():
    """
    Test that afile_retrieve raises an appropriate error when file_object is None
    and no llm_router is provided to fetch from the provider.
    """
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock get_unified_file_id to return a stored object with file_object=None
    stored_file = MagicMock()
    stored_file.file_object = None
    stored_file.model_mappings = {"model-123": "file-provider-xyz"}
    proxy_managed_files.get_unified_file_id = AsyncMock(return_value=stored_file)
    
    unified_file_id = "test-unified-file-id"
    
    with pytest.raises(Exception) as exc_info:
        await proxy_managed_files.afile_retrieve(
            file_id=unified_file_id,
            litellm_parent_otel_span=None,
            llm_router=None,
        )
    
    assert "llm_router is required" in str(exc_info.value)


@pytest.mark.asyncio
async def test_afile_retrieve_returns_stored_file_object_when_exists():
    """
    Test that afile_retrieve returns the stored file_object directly when it exists
    (the normal case for user-uploaded files).
    """
    from litellm.types.llms.openai import OpenAIFileObject
    
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock get_unified_file_id to return a stored object WITH file_object
    stored_file_object = OpenAIFileObject(
        id="test-unified-file-id",
        object="file",
        bytes=1234,
        created_at=1234567890,
        filename="input.jsonl",
        purpose="batch",
    )
    stored_file = MagicMock()
    stored_file.file_object = stored_file_object
    proxy_managed_files.get_unified_file_id = AsyncMock(return_value=stored_file)
    
    result = await proxy_managed_files.afile_retrieve(
        file_id="test-unified-file-id",
        litellm_parent_otel_span=None,
        llm_router=None,
    )
    
    # Should return the stored file object directly
    assert result == stored_file_object


@pytest.mark.asyncio
async def test_afile_retrieve_raises_error_for_non_managed_file():
    """
    Test that afile_retrieve raises an error when the file_id is not found
    in the managed files table.
    """
    prisma_client = AsyncMock()
    internal_usage_cache = MagicMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=internal_usage_cache,
        prisma_client=prisma_client,
    )
    
    # Mock get_unified_file_id to return None (file not found)
    proxy_managed_files.get_unified_file_id = AsyncMock(return_value=None)
    
    with pytest.raises(Exception) as exc_info:
        await proxy_managed_files.afile_retrieve(
            file_id="non-existent-file-id",
            litellm_parent_otel_span=None,
        )
    
    assert "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_batches_from_managed_objects_table():
    from openai.types.batch import BatchRequestCounts

    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    
    batch_record_1 = MagicMock()
    batch_record_1.unified_object_id = "unified-batch-id-1"
    batch_record_1.file_object = json.dumps({
        "id": "batch_abc123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1234567890,
        "input_file_id": "file-input-1",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
    })
    
    batch_record_2 = MagicMock()
    batch_record_2.unified_object_id = "unified-batch-id-2"
    batch_record_2.file_object = json.dumps({
        "id": "batch_xyz789",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": "in_progress",
        "created_at": 1234567891,
        "input_file_id": "file-input-2",
        "request_counts": {"total": 5, "completed": 2, "failed": 0},
    })
    
    prisma_client.db.litellm_managedobjecttable.find_many.return_value = [
        batch_record_1,
        batch_record_2,
    ]
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    result = await proxy_managed_files.list_user_batches(
        user_api_key_dict=UserAPIKeyAuth(user_id="test-user"),
        limit=10,
    )
    
    assert result["object"] == "list"
    assert len(result["data"]) == 2
    assert result["data"][0].id == "unified-batch-id-1"
    assert result["data"][1].id == "unified-batch-id-2"
    assert result["first_id"] == "unified-batch-id-1"
    assert result["last_id"] == "unified-batch-id-2"
    
    # Should filter by user_id (created_by)
    prisma_client.db.litellm_managedobjecttable.find_many.assert_called_once_with(
        where={"file_purpose": "batch", "created_by": "test-user"},
        take=10,
        order={"created_at": "desc"},
    )


@pytest.mark.asyncio
async def test_list_batches_from_managed_objects_table_empty_list():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    prisma_client.db.litellm_managedobjecttable.find_many.return_value = []
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    result = await proxy_managed_files.list_user_batches(
        user_api_key_dict=UserAPIKeyAuth(user_id="test-user"),
    )
    
    assert result["object"] == "list"
    assert len(result["data"]) == 0
    assert result["first_id"] is None
    assert result["last_id"] is None
    assert result["has_more"] is False
    
    # Verify where clause includes created_by filter
    # Default take is 20 when no limit is provided
    prisma_client.db.litellm_managedobjecttable.find_many.assert_called_once_with(
        where={"file_purpose": "batch", "created_by": "test-user"},
        take=20,
        order={"created_at": "desc"},
    )


def _create_unified_batch_id(model_id: str, batch_id: str) -> str:
    import base64
    unified_str = f"litellm_proxy;model_id:{model_id};llm_batch_id:{batch_id}"
    return base64.urlsafe_b64encode(unified_str.encode()).decode().rstrip("=")


@pytest.mark.asyncio
async def test_list_batches_from_managed_objects_table_provider_filter_raises_exception():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # Filtering by provider should raise Exception
    with pytest.raises(Exception) as exc_info:
        await proxy_managed_files.list_user_batches(
            user_api_key_dict=UserAPIKeyAuth(user_id="test-user"),
            limit=10,
            provider="openai",
        )
    
    assert str(exc_info.value) == (
        "Filtering by 'provider' is not supported when using managed batches."
    )
    
    # Verify find_many was NOT called since exception is raised before database query
    prisma_client.db.litellm_managedobjecttable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_list_batches_from_managed_objects_table_target_model_name_filter_raises_exception():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )

    # Filtering by provider should raise Exception
    with pytest.raises(Exception) as exc_info:
        await proxy_managed_files.list_user_batches(
            user_api_key_dict=UserAPIKeyAuth(user_id="test-user"),
            limit=10,
            target_model_names="gpt-4o,gpt-3.5",
        )
    
    assert str(exc_info.value) == (
        "Filtering by 'target_model_names' is not supported when using managed batches."
    )
    
    # Verify find_many was NOT called since exception is raised before database query
    prisma_client.db.litellm_managedobjecttable.find_many.assert_not_called()

@pytest.mark.asyncio
async def test_list_batches_from_managed_objects_table_filters_by_created_by():
    from litellm.proxy._types import UserAPIKeyAuth

    prisma_client = AsyncMock()
    
    # Create batch for user1
    batch_user1 = MagicMock()
    batch_user1.unified_object_id = "unified-batch-user1"
    batch_user1.file_object = json.dumps({
        "id": "batch_user1_abc",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1234567890,
        "input_file_id": "file-input-user1",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
    })
    
    # Create batch for user2
    batch_user2 = MagicMock()
    batch_user2.unified_object_id = "unified-batch-user2"
    batch_user2.file_object = json.dumps({
        "id": "batch_user2_xyz",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1234567891,
        "input_file_id": "file-input-user2",
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
    })
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # Query with user1's API key - should only return user1's batch
    prisma_client.db.litellm_managedobjecttable.find_many.return_value = [batch_user1]
    result_user1 = await proxy_managed_files.list_user_batches(
        user_api_key_dict=UserAPIKeyAuth(user_id="user1"),
        limit=10,
    )
    
    assert len(result_user1["data"]) == 1
    assert result_user1["data"][0].id == "unified-batch-user1"
    prisma_client.db.litellm_managedobjecttable.find_many.assert_called_with(
        where={"file_purpose": "batch", "created_by": "user1"},
        take=10,
        order={"created_at": "desc"},
    )
    
    # Query with user2's API key - should only return user2's batch
    prisma_client.db.litellm_managedobjecttable.find_many.return_value = [batch_user2]
    result_user2 = await proxy_managed_files.list_user_batches(
        user_api_key_dict=UserAPIKeyAuth(user_id="user2"),
        limit=10,
    )
    
    assert len(result_user2["data"]) == 1
    assert result_user2["data"][0].id == "unified-batch-user2"
    prisma_client.db.litellm_managedobjecttable.find_many.assert_called_with(
        where={"file_purpose": "batch", "created_by": "user2"},
        take=10,
        order={"created_at": "desc"},
    )


@pytest.mark.asyncio
async def test_return_unified_file_id_includes_expires_at():
    from litellm.types.llms.openai import OpenAIFileObject

    # Create a mock file object with expires_at set
    file_object = OpenAIFileObject(
        id="file-abc123",
        object="file",
        bytes=1234,
        created_at=1234567890,
        filename="test.jsonl",
        purpose="batch",
        status="uploaded",
        expires_at=1234657890,  
    )
    file_object._hidden_params = {"model_id": "test-model-id"}

    create_file_request = {
        "file": ("test.jsonl", b"test content", "application/jsonl"),
        "purpose": "batch",
    }

    internal_usage_cache = MagicMock()

    result = await _PROXY_LiteLLMManagedFiles.return_unified_file_id(
        file_objects=[file_object],
        create_file_request=create_file_request,
        internal_usage_cache=internal_usage_cache,
        litellm_parent_otel_span=None,
        target_model_names_list=["gpt-4o"],
    )

    # Verify expires_at is passed through
    assert result.expires_at == 1234657890
    assert result.purpose == "batch"
    assert result.filename == "test.jsonl"
    assert result.bytes == 1234
    assert result.created_at == 1234567890
    assert _is_base64_encoded_unified_file_id(result.id)


# ============================================================================
# Permission Tests - Cross-User Batch Access
# ============================================================================
# These tests verify that batches and files created by one user
# cannot be accessed, modified, or cancelled by a different user.
# Reference: https://github.com/BerriAI/litellm/pull/17401/files


@pytest.mark.asyncio
async def test_user_b_cannot_retrieve_user_a_batch():
    """
    Test that User B cannot retrieve a batch created by User A.
    
    This verifies batch isolation between users at the database/hook level.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    batch_record = MagicMock()
    batch_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = batch_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # User B tries to retrieve User A's batch
    unified_batch_id = "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDpteS1tb2RlbDtsbG1fYmF0Y2hfaWQ6YmF0Y2hfYWJjMTIz"
    
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user_b_id", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"batch_id": unified_batch_id},
            call_type="aretrieve_batch",
        )
    
    # Should raise 403 Permission Denied
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_b_cannot_cancel_user_a_batch():
    """
    Test that User B cannot cancel a batch created by User A.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    batch_record = MagicMock()
    batch_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = batch_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # User B tries to cancel User A's batch
    unified_batch_id = "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDpteS1tb2RlbDtsbG1fYmF0Y2hfaWQ6YmF0Y2hfYWJjMTIz"
    
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user_b_id", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"batch_id": unified_batch_id},
            call_type="acancel_batch",
        )
    
    # Should raise 403 Permission Denied
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_a_can_retrieve_own_batch():
    """
    Test that User A can successfully retrieve their own batch.
    
    This is a positive test case to ensure permission checks don't block
    legitimate access.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    batch_record = MagicMock()
    batch_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = batch_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # User A retrieves their own batch
    unified_batch_id = "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDpteS1tb2RlbDtsbG1fYmF0Y2hfaWQ6YmF0Y2hfYWJjMTIz"
    
    # Should not raise an exception
    result = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="user_a_id", parent_otel_span=MagicMock()
        ),
        cache=MagicMock(),
        data={"batch_id": unified_batch_id},
        call_type="aretrieve_batch",
    )
    
    # Should successfully return the decoded batch_id
    assert "batch_id" in result
    assert result["model"] == "my-model"


@pytest.mark.asyncio
async def test_user_b_cannot_retrieve_user_a_file():
    """
    Test that User B cannot retrieve a file created by User A.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    file_record = MagicMock()
    file_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedfiletable.find_first.return_value = file_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        MagicMock(), prisma_client=prisma_client
    )
    
    # User B tries to retrieve User A's file
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsZmlsZS1hYmMxMjM"
    
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user_b_id", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"file_id": unified_file_id},
            call_type="afile_retrieve",
        )
    
    # Should raise 403 Permission Denied
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_b_cannot_download_user_a_file_content():
    """
    Test that User B cannot download file content for User A's file.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    file_record = MagicMock()
    file_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedfiletable.find_first.return_value = file_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        MagicMock(), prisma_client=prisma_client
    )
    
    # User B tries to download User A's file content
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsZmlsZS1hYmMxMjM"
    
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user_b_id", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"file_id": unified_file_id},
            call_type="afile_content",
        )
    
    # Should raise 403 Permission Denied
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_b_cannot_delete_user_a_file():
    """
    Test that User B cannot delete a file created by User A.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    file_record = MagicMock()
    file_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedfiletable.find_first.return_value = file_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        MagicMock(), prisma_client=prisma_client
    )
    
    # User B tries to delete User A's file
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsZmlsZS1hYmMxMjM"
    
    with pytest.raises(HTTPException) as exc_info:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="user_b_id", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"file_id": unified_file_id},
            call_type="afile_delete",
        )
    
    # Should raise 403 Permission Denied
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_user_a_can_retrieve_own_file():
    """
    Test that User A can successfully retrieve their own file.
    
    Positive test case to ensure permission checks work correctly for the owner.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return User A as the creator
    file_record = MagicMock()
    file_record.created_by = "user_a_id"
    file_record.model_mappings = '{"model-123": "file-abc123"}'
    file_record.file_object = json.dumps({
        "id": "file-abc123",
        "object": "file",
        "bytes": 1234,
        "created_at": 1234567890,
        "filename": "test.jsonl",
        "purpose": "batch",
    })
    prisma_client.db.litellm_managedfiletable.find_first.return_value = file_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        MagicMock(), prisma_client=prisma_client
    )
    
    # User A retrieves their own file
    unified_file_id = "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9qc29uO3VuaWZpZWRfaWQsZmlsZS1hYmMxMjM"
    
    # Should not raise an exception
    result = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="user_a_id", parent_otel_span=MagicMock()
        ),
        cache=MagicMock(),
        data={"file_id": unified_file_id},
        call_type="afile_retrieve",
    )
    
    # Should successfully return the decoded file_id
    assert "file_id" in result


@pytest.mark.asyncio
async def test_list_batches_only_returns_user_own_batches():
    """
    Test that list_user_batches only returns batches created by the requesting user.
    
    This ensures users cannot see other users' batches in list operations.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Create batches for User A
    batch_user_a = MagicMock()
    batch_user_a.unified_object_id = "batch-user-a"
    batch_user_a.file_object = json.dumps({
        "id": "batch_a",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "status": "completed",
        "created_at": 1234567890,
        "input_file_id": "file-a",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
    })
    
    # Mock database to only return User A's batches
    prisma_client.db.litellm_managedobjecttable.find_many.return_value = [batch_user_a]
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    # User A requests their batches
    result = await proxy_managed_files.list_user_batches(
        user_api_key_dict=UserAPIKeyAuth(user_id="user_a_id"),
        limit=10,
    )
    
    # Should only return User A's batches
    assert len(result["data"]) == 1
    assert result["data"][0].id == "batch-user-a"
    
    # Verify the database query filtered by user_id
    prisma_client.db.litellm_managedobjecttable.find_many.assert_called_once_with(
        where={"file_purpose": "batch", "created_by": "user_a_id"},
        take=10,
        order={"created_at": "desc"},
    )


@pytest.mark.asyncio
async def test_same_user_different_keys_can_access_batch():
    """
    Test that different API keys for the same user can access the same batch.
    
    This verifies that permission checks are based on user_id, not API key,
    allowing users to have multiple keys that can all access their resources.
    """
    from litellm.proxy._types import UserAPIKeyAuth
    
    prisma_client = AsyncMock()
    
    # Mock database to return the user_id as creator
    batch_record = MagicMock()
    batch_record.created_by = "user_a_id"
    prisma_client.db.litellm_managedobjecttable.find_first.return_value = batch_record
    
    proxy_managed_files = _PROXY_LiteLLMManagedFiles(
        DualCache(), prisma_client=prisma_client
    )
    
    unified_batch_id = "bGl0ZWxsbV9wcm94eTttb2RlbF9pZDpteS1tb2RlbDtsbG1fYmF0Y2hfaWQ6YmF0Y2hfYWJjMTIz"
    
    # First API key for User A retrieves the batch
    result1 = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="user_a_id", 
            api_key="key-1",
            parent_otel_span=MagicMock()
        ),
        cache=MagicMock(),
        data={"batch_id": unified_batch_id},
        call_type="aretrieve_batch",
    )
    
    assert "batch_id" in result1
    
    # Second API key for the same User A retrieves the batch
    result2 = await proxy_managed_files.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="user_a_id",
            api_key="key-2",
            parent_otel_span=MagicMock()
        ),
        cache=MagicMock(),
        data={"batch_id": unified_batch_id},
        call_type="aretrieve_batch",
    )
    
    assert "batch_id" in result2
    # Both keys should get the same result
    assert result1["batch_id"] == result2["batch_id"]
