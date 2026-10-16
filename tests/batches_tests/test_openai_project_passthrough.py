# What is this?
## Unit tests: OpenAI `project` passthrough for the Batches and Files APIs.
## https://github.com/BerriAI/litellm/issues/41803
import os
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.openai.common_utils import get_openai_credentials


def test_get_openai_credentials_project_param():
    creds = get_openai_credentials(api_key="sk-test", organization="org-abc", project="proj-xyz")
    assert creds.organization == "org-abc"
    assert creds.project == "proj-xyz"


def test_get_openai_credentials_project_env_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_PROJECT", "proj-from-env")
    creds = get_openai_credentials(api_key="sk-test")
    assert creds.project == "proj-from-env"


def test_openai_batches_api_create_batch_forwards_project():
    from litellm.llms.openai.openai import OpenAIBatchesAPI
    from litellm.types.utils import LiteLLMBatch

    batch_response = LiteLLMBatch.model_validate(
        {
            "id": "batch_abc123",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "file-abc123",
            "status": "validating",
            "completion_window": "24h",
            "created_at": 1700000000,
        }
    )
    with patch("litellm.llms.openai.openai.OpenAI") as mock_openai:
        mock_openai.return_value.batches.create.return_value = batch_response
        OpenAIBatchesAPI().create_batch(
            _is_async=False,
            create_batch_data={
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
                "input_file_id": "file-abc123",
            },
            api_key="sk-test",
            api_base=None,
            timeout=600.0,
            max_retries=2,
            organization="org-abc",
            project="proj-xyz",
        )

    kwargs = mock_openai.call_args.kwargs
    assert kwargs["project"] == "proj-xyz"
    assert kwargs["organization"] == "org-abc"


def test_openai_batches_api_create_batch_omits_project_when_none():
    from litellm.llms.openai.openai import OpenAIBatchesAPI
    from litellm.types.utils import LiteLLMBatch

    batch_response = LiteLLMBatch.model_validate(
        {
            "id": "batch_abc123",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "file-abc123",
            "status": "validating",
            "completion_window": "24h",
            "created_at": 1700000000,
        }
    )
    with patch("litellm.llms.openai.openai.OpenAI") as mock_openai:
        mock_openai.return_value.batches.create.return_value = batch_response
        OpenAIBatchesAPI().create_batch(
            _is_async=False,
            create_batch_data={
                "completion_window": "24h",
                "endpoint": "/v1/chat/completions",
                "input_file_id": "file-abc123",
            },
            api_key="sk-test",
            api_base=None,
            timeout=600.0,
            max_retries=2,
            organization=None,
            project=None,
        )

    kwargs = mock_openai.call_args.kwargs
    assert "project" not in kwargs
    assert "organization" not in kwargs


def test_openai_files_api_retrieve_file_forwards_project():
    from litellm.llms.openai.openai import OpenAIFilesAPI

    with patch("litellm.llms.openai.openai.OpenAI") as mock_openai:
        OpenAIFilesAPI().retrieve_file(
            _is_async=False,
            file_id="file-abc123",
            api_base=None,
            api_key="sk-test",
            timeout=600.0,
            max_retries=2,
            organization=None,
            project="proj-xyz",
        )

    kwargs = mock_openai.call_args.kwargs
    assert kwargs["project"] == "proj-xyz"


def test_litellm_create_batch_project_param(monkeypatch):
    from litellm.batches import main as batches_main

    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    mock_instance = MagicMock()
    with patch.object(batches_main, "openai_batches_instance", mock_instance):
        litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file-abc123",
            custom_llm_provider="openai",
            api_key="sk-test",
            project="proj-xyz",
        )

    kwargs = mock_instance.create_batch.call_args.kwargs
    assert kwargs["project"] == "proj-xyz"


def test_litellm_create_batch_project_env_fallback(monkeypatch):
    from litellm.batches import main as batches_main

    monkeypatch.setenv("OPENAI_PROJECT", "proj-from-env")
    mock_instance = MagicMock()
    with patch.object(batches_main, "openai_batches_instance", mock_instance):
        litellm.create_batch(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id="file-abc123",
            custom_llm_provider="openai",
            api_key="sk-test",
        )

    kwargs = mock_instance.create_batch.call_args.kwargs
    assert kwargs["project"] == "proj-from-env"


def test_litellm_file_retrieve_project_param(monkeypatch):
    from litellm.files import main as files_main

    monkeypatch.delenv("OPENAI_PROJECT", raising=False)
    mock_instance = MagicMock()
    with patch.object(files_main, "openai_files_instance", mock_instance):
        files_main.file_retrieve(
            file_id="file-abc123",
            custom_llm_provider="openai",
            api_key="sk-test",
            project="proj-xyz",
        )

    kwargs = mock_instance.retrieve_file.call_args.kwargs
    assert kwargs["project"] == "proj-xyz"
