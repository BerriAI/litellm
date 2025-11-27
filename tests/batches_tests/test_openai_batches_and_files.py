# What is this?
## Unit Tests for OpenAI Batches API
import asyncio
import json
import os
import sys
import traceback
import tempfile
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path

import logging
import time
import asyncio

import pytest
from typing import Optional
import litellm
from litellm import create_batch, create_file
from litellm._logging import verbose_logger
import openai

verbose_logger.setLevel(logging.DEBUG)

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import StandardLoggingPayload
import random
from unittest.mock import patch, MagicMock


def load_vertex_ai_credentials():
    # Define the path to the vertex_key.json file
    print("loading vertex ai credentials")
    os.environ["GCS_FLUSH_INTERVAL"] = "1"
    filepath = os.path.dirname(os.path.abspath(__file__))
    vertex_key_path = filepath + "/pathrise-convert-1606954137718.json"

    # Read the existing content of the file or create an empty dictionary
    try:
        with open(vertex_key_path, "r") as file:
            # Read the file content
            print("Read vertexai file path")
            content = file.read()

            # If the file is empty or not valid JSON, create an empty dictionary
            if not content or not content.strip():
                service_account_key_data = {}
            else:
                # Attempt to load the existing JSON content
                file.seek(0)
                service_account_key_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty dictionary
        service_account_key_data = {}

    # Update the service_account_key_data with environment variables
    private_key_id = os.environ.get("GCS_PRIVATE_KEY_ID", "")
    private_key = os.environ.get("GCS_PRIVATE_KEY", "")
    private_key = private_key.replace("\\n", "\n")
    service_account_key_data["private_key_id"] = private_key_id
    service_account_key_data["private_key"] = private_key

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        # Write the updated content to the temporary files
        json.dump(service_account_key_data, temp_file, indent=2)

    # Export the temporary file as GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GCS_PATH_SERVICE_ACCOUNT"] = os.path.abspath(temp_file.name)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(temp_file.name)
    print("created gcs path service account=", os.environ["GCS_PATH_SERVICE_ACCOUNT"])


@pytest.mark.parametrize("provider", ["openai"])  # , "azure"
@pytest.mark.asyncio
async def test_create_batch(provider):
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    if provider == "azure":
        # Don't have anymore Azure Quota
        return
    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)

    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=provider,
    )
    print("Response from creating file=", file_obj)

    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    await asyncio.sleep(1)
    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider=provider,
        metadata={"key1": "value1", "key2": "value2"},
    )

    print("response from litellm.create_batch=", create_batch_response)
    await asyncio.sleep(6)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
        or create_batch_response.endpoint == "/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    retrieved_batch = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider=provider
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    # list all batches
    list_batches = await litellm.alist_batches(custom_llm_provider=provider, limit=2)
    print("list_batches=", list_batches)

    file_content = await litellm.afile_content(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    result = file_content.content

    result_file_name = "batch_job_results_furniture.jsonl"

    with open(result_file_name, "wb") as file:
        file.write(result)

    # Cancel Batch - handle race condition where batch may already be completed
    try:
        cancel_batch_response = await litellm.acancel_batch(
            batch_id=create_batch_response.id,
            custom_llm_provider=provider,
        )
        print("cancel_batch_response=", cancel_batch_response)
    except openai.ConflictError as e:
        # Only allow to pass if it's specifically the "batch already completed" error
        if "Cannot cancel a batch with status 'completed'" in str(e):
            print(f"Batch already completed, cannot cancel: {e}")
        else:
            # Re-raise other ConflictError types
            raise
    except Exception as e:
        # Re-raise any other unexpected errors
        print(f"Unexpected error during batch cancellation: {e}")
        raise

    pass


class TestCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.standard_logging_object: Optional[StandardLoggingPayload] = None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(
            "Success event logged with kwargs=",
            kwargs,
            "and response_obj=",
            response_obj,
        )
        self.standard_logging_object = kwargs["standard_logging_object"]


def cleanup_azure_files():
    """
    Delete all files for Azure - helper for when we run out of Azure Files Quota
    """
    azure_files = litellm.file_list(
        custom_llm_provider="azure",
        api_key=os.getenv("AZURE_FT_API_KEY"),
        api_base=os.getenv("AZURE_FT_API_BASE"),
    )
    print("azure_files=", azure_files)
    for _file in azure_files:
        print("deleting file=", _file)
        delete_file_response = litellm.file_delete(
            file_id=_file.id,
            custom_llm_provider="azure",
            api_key=os.getenv("AZURE_FT_API_KEY"),
            api_base=os.getenv("AZURE_FT_API_BASE"),
        )
        print("delete_file_response=", delete_file_response)
        assert delete_file_response.id == _file.id


def cleanup_azure_ft_models():
    """
    Test CLEANUP: Delete all existing fine tuning jobs for Azure
    """
    try:
        from openai import AzureOpenAI
        import requests

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_FT_API_KEY"),
            azure_endpoint=os.getenv("AZURE_FT_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )

        _list_ft_jobs = client.fine_tuning.jobs.list()
        print("_list_ft_jobs=", _list_ft_jobs)

        # delete all ft jobs make post request to this
        # Delete all fine-tuning jobs
        for job in _list_ft_jobs:
            try:
                endpoint = os.getenv("AZURE_FT_API_BASE").rstrip("/")
                url = f"{endpoint}/openai/fine_tuning/jobs/{job.id}?api-version=2024-10-21"
                print("url=", url)

                headers = {
                    "api-key": os.getenv("AZURE_FT_API_KEY"),
                    "Content-Type": "application/json",
                }

                response = requests.delete(url, headers=headers)
                print(f"Deleting job {job.id}: Status {response.status_code}")
                if response.status_code != 204:
                    print(f"Error deleting job {job.id}: {response.text}")

            except Exception as e:
                print(f"Error deleting job {job.id}: {str(e)}")
    except Exception as e:
        print(f"Error on cleanup_azure_ft_models: {str(e)}")


@pytest.mark.parametrize("provider", ["openai"])
@pytest.mark.asyncio()
@pytest.mark.flaky(retries=3, delay=1)
async def test_async_create_batch(provider):
    """
    1. Create File for Batch completion
    2. Create Batch Request
    3. Retrieve the specific batch
    """
    litellm._turn_on_debug()
    print("Testing async create batch")
    litellm.logging_callback_manager._reset_all_callbacks()
    custom_logger = TestCustomLogger()
    litellm.callbacks = [custom_logger, "datadog"]

    file_name = "openai_batch_completions.jsonl"
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(_current_dir, file_name)
    file_obj = await litellm.acreate_file(
        file=open(file_path, "rb"),
        purpose="batch",
        custom_llm_provider=provider,
    )
    print("Response from creating file=", file_obj)

    await asyncio.sleep(10)
    batch_input_file_id = file_obj.id
    assert (
        batch_input_file_id is not None
    ), "Failed to create file, expected a non null file_id but got {batch_input_file_id}"

    extra_metadata_field = {
        "user_api_key_alias": "special_api_key_alias",
        "user_api_key_team_alias": "special_team_alias",
    }
    create_batch_response = await litellm.acreate_batch(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=batch_input_file_id,
        custom_llm_provider=provider,
        metadata={"key1": "value1", "key2": "value2"},
        # litellm specific param - used for logging metadata on logging callback
        litellm_metadata=extra_metadata_field,
    )

    print("response from litellm.create_batch=", create_batch_response)

    assert (
        create_batch_response.id is not None
    ), f"Failed to create batch, expected a non null batch_id but got {create_batch_response.id}"
    assert (
        create_batch_response.endpoint == "/v1/chat/completions"
        or create_batch_response.endpoint == "/chat/completions"
    ), f"Failed to create batch, expected endpoint to be /v1/chat/completions but got {create_batch_response.endpoint}"
    assert (
        create_batch_response.input_file_id == batch_input_file_id
    ), f"Failed to create batch, expected input_file_id to be {batch_input_file_id} but got {create_batch_response.input_file_id}"

    await asyncio.sleep(6)
    # Assert that the create batch event is logged on CustomLogger
    assert custom_logger.standard_logging_object is not None
    print(
        "standard_logging_object=",
        json.dumps(custom_logger.standard_logging_object, indent=4, default=str),
    )
    assert (
        custom_logger.standard_logging_object["metadata"]["user_api_key_alias"]
        == extra_metadata_field["user_api_key_alias"]
    )
    assert (
        custom_logger.standard_logging_object["metadata"]["user_api_key_team_alias"]
        == extra_metadata_field["user_api_key_team_alias"]
    )

    retrieved_batch = await litellm.aretrieve_batch(
        batch_id=create_batch_response.id, custom_llm_provider=provider
    )
    print("retrieved batch=", retrieved_batch)
    # just assert that we retrieved a non None batch

    assert retrieved_batch.id == create_batch_response.id

    # list all batches
    list_batches = await litellm.alist_batches(custom_llm_provider=provider, limit=2)
    print("list_batches=", list_batches)

    # try to get file content for our original file

    file_content = await litellm.afile_content(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    print("file content = ", file_content)

    # file obj
    file_obj = await litellm.afile_retrieve(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )
    print("file obj = ", file_obj)
    assert file_obj.id == batch_input_file_id

    # delete file
    delete_file_response = await litellm.afile_delete(
        file_id=batch_input_file_id, custom_llm_provider=provider
    )

    print("delete file response = ", delete_file_response)

    assert delete_file_response.id == batch_input_file_id

    all_files_list = await litellm.afile_list(
        custom_llm_provider=provider,
    )

    print("all_files_list = ", all_files_list)

    result_file_name = "batch_job_results_furniture.jsonl"

    with open(result_file_name, "wb") as file:
        file.write(file_content.content)

    # Cancel Batch - handle race condition where batch may already be completed
    try:
        cancel_batch_response = await litellm.acancel_batch(
            batch_id=create_batch_response.id,
            custom_llm_provider=provider,
        )
        print("cancel_batch_response=", cancel_batch_response)
    except openai.ConflictError as e:
        # Only allow to pass if it's specifically the "batch already completed" error
        if "Cannot cancel a batch with status 'completed'" in str(e):
            print(f"Batch already completed, cannot cancel: {e}")
        else:
            # Re-raise other ConflictError types
            raise
    except Exception as e:
        # Re-raise any other unexpected errors
        print(f"Unexpected error during batch cancellation: {e}")
        raise

    if random.randint(1, 3) == 1:
        print("Running random cleanup of Azure files and models...")
        cleanup_azure_files()
        cleanup_azure_ft_models()


mock_file_response = {
    "kind": "storage#object",
    "id": "litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb/1739598666670574",
    "selfLink": "https://www.googleapis.com/storage/v1/b/litellm-local/o/litellm-vertex-files%2Fpublishers%2Fgoogle%2Fmodels%2Fgemini-1.5-flash-001%2F5f7b99ad-9203-4430-98bf-3b45451af4cb",
    "mediaLink": "https://storage.googleapis.com/download/storage/v1/b/litellm-local/o/litellm-vertex-files%2Fpublishers%2Fgoogle%2Fmodels%2Fgemini-1.5-flash-001%2F5f7b99ad-9203-4430-98bf-3b45451af4cb?generation=1739598666670574&alt=media",
    "name": "litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb",
    "bucket": "litellm-local",
    "generation": "1739598666670574",
    "metageneration": "1",
    "contentType": "application/json",
    "storageClass": "STANDARD",
    "size": "416",
    "md5Hash": "hbBNj7C8KJ7oVH+JmyRM6A==",
    "crc32c": "oDmiUA==",
    "etag": "CO7D0IT+xIsDEAE=",
    "timeCreated": "2025-02-15T05:51:06.741Z",
    "updated": "2025-02-15T05:51:06.741Z",
    "timeStorageClassUpdated": "2025-02-15T05:51:06.741Z",
    "timeFinalized": "2025-02-15T05:51:06.741Z",
}

mock_vertex_batch_response = {
    "name": "projects/123456789/locations/us-central1/batchPredictionJobs/test-batch-id-456",
    "displayName": "litellm_batch_job",
    "model": "projects/123456789/locations/us-central1/models/gemini-1.5-flash-001",
    "modelVersionId": "v1",
    "inputConfig": {
        "gcsSource": {
            "uris": [
                "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb"
            ]
        }
    },
    "outputConfig": {
        "gcsDestination": {"outputUriPrefix": "gs://litellm-local/batch-outputs/"}
    },
    "dedicatedResources": {
        "machineSpec": {
            "machineType": "n1-standard-4",
            "acceleratorType": "NVIDIA_TESLA_T4",
            "acceleratorCount": 1,
        },
        "startingReplicaCount": 1,
        "maxReplicaCount": 1,
    },
    "state": "JOB_STATE_RUNNING",
    "createTime": "2025-02-15T05:51:06.741Z",
    "startTime": "2025-02-15T05:51:07.741Z",
    "updateTime": "2025-02-15T05:51:08.741Z",
    "labels": {"key1": "value1", "key2": "value2"},
    "completionStats": {"successfulCount": 0, "failedCount": 0, "remainingCount": 100},
}

mock_vertex_list_response = {
    "batchPredictionJobs": [
        mock_vertex_batch_response,
        {
            **mock_vertex_batch_response,
            "name": "projects/123456789/locations/us-central1/batchPredictionJobs/test-batch-id-789",
            "state": "JOB_STATE_SUCCEEDED",
        },
    ],
    "nextPageToken": "",
}


@pytest.mark.asyncio
async def test_avertex_batch_prediction(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "litellm-local")
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    client = AsyncHTTPHandler()

    async def mock_side_effect(*args, **kwargs):
        print("args", args, "kwargs", kwargs)
        url = kwargs.get("url", "")
        if "files" in url:
            mock_response.json.return_value = mock_file_response
        elif "batch" in url:
            mock_response.json.return_value = mock_vertex_batch_response
            mock_response.status_code = 200
        return mock_response

    with patch.object(
        client, "post", side_effect=mock_side_effect
    ) as mock_post, patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_global_post:
        # Configure mock responses
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        # Set up different responses for different API calls
        
        mock_post.side_effect = mock_side_effect
        mock_global_post.side_effect = mock_side_effect

        # load_vertex_ai_credentials()
        litellm.set_verbose = True
        litellm._turn_on_debug()
        file_name = "vertex_batch_completions.jsonl"
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(_current_dir, file_name)

        # Create file
        file_obj = await litellm.acreate_file(
            file=open(file_path, "rb"),
            purpose="batch",
            custom_llm_provider="vertex_ai",
            client=client
        )
        print("Response from creating file=", file_obj)

        assert (
            file_obj.id
            == "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb"
        )

        # Create batch
        create_batch_response = await litellm.acreate_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=file_obj.id,
            custom_llm_provider="vertex_ai",
            metadata={"key1": "value1", "key2": "value2"},
        )
        print("create_batch_response=", create_batch_response)

        assert create_batch_response.id == "test-batch-id-456"
        assert (
            create_batch_response.input_file_id
            == "gs://litellm-local/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/5f7b99ad-9203-4430-98bf-3b45451af4cb"
        )

        # Mock the retrieve batch response
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
        ) as mock_get:
            mock_get_response = MagicMock()
            mock_get_response.json.return_value = mock_vertex_batch_response
            mock_get_response.status_code = 200
            mock_get_response.raise_for_status.return_value = None
            mock_get.return_value = mock_get_response

            retrieved_batch = await litellm.aretrieve_batch(
                batch_id=create_batch_response.id,
                custom_llm_provider="vertex_ai",
            )
            print("retrieved_batch=", retrieved_batch)

            assert retrieved_batch.id == "test-batch-id-456"


@pytest.mark.asyncio
async def test_vertex_list_batches(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "litellm-local")
    monkeypatch.setenv("VERTEXAI_PROJECT", "litellm-test-project")
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-central1")

    monkeypatch.setattr(
        "litellm.llms.vertex_ai.batches.handler.VertexAIBatchPrediction._ensure_access_token",
        lambda self, credentials, project_id, custom_llm_provider: ("mock-token", "litellm-test-project"),
    )

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.get"
    ) as mock_get:
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = mock_vertex_list_response
        mock_get_response.status_code = 200
        mock_get_response.raise_for_status.return_value = None
        mock_get.return_value = mock_get_response

        list_response = await litellm.alist_batches(
            custom_llm_provider="vertex_ai",
            limit=2,
        )

        assert list_response["object"] == "list"
        assert list_response["has_more"] is False
        assert len(list_response["data"]) == 2
        assert list_response["data"][0].id == "test-batch-id-456"
        assert list_response["data"][1].id == "test-batch-id-789"
