import json
import os
import sys

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

from enterprise.enterprise_hooks.managed_files import _PROXY_LiteLLMManagedFiles
from litellm.caching import DualCache
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
)
from litellm.types.utils import SpecialEnums


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
        DualCache(), prisma_client=MagicMock()
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
@pytest.mark.parametrize("call_type", ["afile_content", "afile_delete"])
async def test_can_user_call_unified_file_id(call_type):
    """
    Test that on file retrieve, delete we check if the user has access to the file
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

    with pytest.raises(HTTPException) as e:
        await proxy_managed_files.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="456", parent_otel_span=MagicMock()
            ),
            cache=MagicMock(),
            data={"file_id": unified_file_id},
            call_type=call_type,
        )
