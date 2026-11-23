import os
from collections.abc import Mapping
from typing import Final, Optional
from unittest.mock import MagicMock
import openai
import requests

import litellm
from litellm.llms.azure.common_utils import BaseAzureLLM


def test_default_max_retries_env_var_reaches_azure_sdk_client():
    import subprocess
    import sys

    code = (
        "from litellm.llms.azure.common_utils import BaseAzureLLM\n"
        "client = BaseAzureLLM().get_azure_openai_client("
        "api_key='test-api-key', api_base='https://test.openai.azure.com', api_version='2024-02-01',"
        " client=None, _is_async=True, litellm_params={}, model='gpt-4')\n"
        "print(client.max_retries)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DEFAULT_MAX_RETRIES": "0"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "0"


# Test parameters for different API functions with Azure models
AZURE_API_FUNCTION_PARAMS = [
    # (function_name, is_async, args)
    (
        "completion",
        False,
        {
            "model": "azure/gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "completion",
        True,
        {
            "model": "azure/gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": True,
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "embedding",
        False,
        {
            "model": "azure/text-embedding-ada-002",
            "input": "Hello world",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "embedding",
        True,
        {
            "model": "azure/text-embedding-ada-002",
            "input": "Hello world",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "speech",
        False,
        {
            "model": "azure/tts",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "speech",
        True,
        {
            "model": "azure/tts-1",
            "input": "Hello, this is a test of text to speech",
            "voice": "alloy",
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "transcription",
        False,
        {
            "model": "azure/whisper-1",
            "file": MagicMock(),
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
    (
        "transcription",
        True,
        {
            "model": "azure/whisper-1",
            "file": MagicMock(),
            "api_key": "test-api-key",
            "api_base": "https://test.openai.azure.com",
            "api_version": "2023-05-15",
        },
    ),
]
_ENTRA_TRACE_ID: Final = "f0c149d3-ef20-4194-8b81-c81a5e002500"


def _entra_error(aadsts_code: str, description: str) -> Mapping[str, object]:
    return {
        "error": "invalid_client",
        "error_description": f"{aadsts_code}: {description} Trace ID: {_ENTRA_TRACE_ID}",
        "error_codes": [int(aadsts_code.removeprefix("AADSTS"))],
    }


_REAL_CREDENTIAL_NON_REJECTIONS: Final = {
    "unreachable_through_a_credential_chain": ("default_azure_credential_chain", None, ""),
    "html_503_through_a_credential_chain": (
        "default_azure_credential_chain",
        503,
        "<html><body>Service Unavailable</body></html>",
    ),
    "json_503_through_a_credential_chain": (
        "default_azure_credential_chain",
        503,
        _entra_error("AADSTS90033", "A transient error has occurred."),
    ),
    "throttled_client_secret": (
        "client_secret",
        429,
        _entra_error("AADSTS50196", "The server terminated an operation because it encountered a client request loop."),
    ),
    "tenant_throttled_client_secret": (
        "client_secret",
        429,
        _entra_error("AADSTS90055", "There are too many incoming requests."),
    ),
    "request_budget_exceeded_client_secret": (
        "client_secret",
        400,
        _entra_error("AADSTS90024", "A transient error has occurred. Try again."),
    ),
    "retryable_error_client_secret": (
        "client_secret",
        400,
        _entra_error("AADSTS50033", "A transient error has occurred."),
    ),
    "external_server_retryable_client_secret": (
        "client_secret",
        400,
        _entra_error("AADSTS90006", "The service is temporarily unavailable."),
    ),
    "graph_retryable_client_secret": (
        "client_secret",
        400,
        _entra_error("AADSTS90090", "The service is temporarily unavailable."),
    ),
    "token_issuance_error_client_secret": (
        "client_secret",
        400,
        _entra_error("AADSTS50000", "There was an error issuing a token or an issue with our sign-in service."),
    ),
    "throttled_without_a_code_through_a_credential_chain": (
        "default_azure_credential_chain",
        429,
        {"error": "temporarily_unavailable", "error_description": "Too many requests."},
    ),
    "json_503_through_workload_identity": (
        "workload_identity",
        503,
        _entra_error("AADSTS90033", "A transient error has occurred."),
    ),
    "throttled_workload_identity_naming_a_rejection_code": (
        "workload_identity",
        429,
        _entra_error("AADSTS7000215", "Invalid client secret provided."),
    ),
}
