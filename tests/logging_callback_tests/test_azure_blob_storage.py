import base64
import hashlib
import hmac
import os
import sys
from datetime import datetime, timedelta

from _pytest.monkeypatch import MonkeyPatch
import httpx

from litellm.integrations.azure_storage._azure_storage_auth import AzureAuthSharedKey, AzureADTokenAuth

sys.path.insert(0, os.path.abspath("../.."))

import asyncio
from unittest.mock import AsyncMock, patch
import re

import pytest

import litellm

from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger

import respx


def test_azure_shared_key_auth_simple_get():

    def get_date_str():
        return "Mon, 13 Apr 2025 12:00:00 GMT"

    key_bytes = "testkey".encode("utf-8")
    key = base64.b64encode(key_bytes)
    auth = AzureAuthSharedKey("testaccount", key, get_date_str)
    request = httpx.Request(
        "GET", "https://testaccount.dfs.core.windows.net/testfilesystem/testfile.json?some_query_param=1"
    )
    request = auth(request)

    expected_signature = hmac_sign(
        "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Mon, 13 Apr 2025 12:00:00 GMT\nx-ms-version:2021-04-10\n/testaccount/testfilesystem/testfile.json\nsome_query_param:1",
        key_bytes,
    )

    # then validate the signing uses the same signature
    assert request.headers["x-ms-date"] == "Mon, 13 Apr 2025 12:00:00 GMT"
    assert request.headers["x-ms-version"] == "2021-04-10"
    assert (
        request.headers["Authorization"]
        == f"SharedKey testaccount:{expected_signature}"
    )


def test_azure_shared_key_auth_complex_post():
    def get_date_str():
        return "Mon, 13 Apr 2025 12:00:00 GMT"

    key_bytes = "testkey".encode("utf-8")
    key = base64.b64encode(key_bytes)
    auth = AzureAuthSharedKey("testaccount", key, get_date_str)

    request = httpx.Request(
        "POST",
        "https://testaccount.dfs.core.windows.net/testfilesystem/testfile.json",
        headers={
            "hi": "there",
            "Content-Type": "application/json",
            "Content-MD5": "testmd5",
        },
        content=b'{"hello": "world"}',
    )
    request = auth(request)

    expected_signature = hmac_sign(
        "POST\n\n\n18\ntestmd5\napplication/json\n\n\n\n\n\n\nx-ms-date:Mon, 13 Apr 2025 12:00:00 GMT\nx-ms-version:2021-04-10\n/testaccount/testfilesystem/testfile.json",
        key_bytes,
    )

    assert (
        request.headers["Authorization"]
        == f"SharedKey testaccount:{expected_signature}"
    )
    
def test_azure_shared_key_query_parameter_sorting():
    def get_date_str():
        return "Mon, 13 Apr 2025 12:00:00 GMT"

    key_bytes = "testkey".encode("utf-8")
    key = base64.b64encode(key_bytes)
    auth = AzureAuthSharedKey("testaccount", key, get_date_str)

    # Create a request with unsorted query parameters
    request = httpx.Request(
        "GET", "https://testaccount.dfs.core.windows.net/testfilesystem/testfile.json?b=2&b=1&a=1"
    )
    request = auth(request)

    expected_signature = hmac_sign(
        "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Mon, 13 Apr 2025 12:00:00 GMT\nx-ms-version:2021-04-10\n/testaccount/testfilesystem/testfile.json\na:1\nb:1,2",
        key_bytes,
    )

    # Validate that the signature is correct and the query parameters are sorted
    assert (
        request.headers["Authorization"]
        == f"SharedKey testaccount:{expected_signature}"
    )

def test_azure_shared_key_all_the_headers():
    x_ms_date = "Mon, 13 Apr 2025 12:00:00 GMT"
    def get_date_str():
        return x_ms_date

    key_bytes = "testkey".encode("utf-8")
    key = base64.b64encode(key_bytes)
    auth = AzureAuthSharedKey("testaccount", key, get_date_str)
    content_encoding = "gzip"
    content_language = "en-US"
    content_length = "0"
    if_modified_since = "Mon, 13 Apr 2025 12:00:00 GMT"
    if_match = "*"
    if_none_match = "*"
    if_unmodified_since = "Sun, 12 Apr 2025 12:00:00 GMT"
    range = "0-123"
    content_md5 = "testmd5"
    content_type = "application/json"
    date = "Tue, 14 Apr 2025 12:00:00 GMT"
    if content_length == "0":
        content_length = ""
    # Create a request with unsorted query parameters
    request = httpx.Request(
        "GET", "https://testaccount.dfs.core.windows.net/testfilesystem/testfile.json",
        headers={
            "Content-Encoding": content_encoding,
            "Content-Language": content_language,
            "Content-Length": content_length,
            "Date": date,
            "If-Modified-Since": if_modified_since,
            "If-Match": if_match,
            "If-None-Match": if_none_match,
            "If-Unmodified-Since": if_unmodified_since,
            "Range": range,
            "Content-MD5": content_md5,
            "Content-Type": content_type,
        }
    )
    request = auth(request)

    expected_string_to_sign = f"GET\n{content_encoding}\n{content_language}\n\n{content_md5}\n{content_type}\n\n{if_modified_since}\n{if_match}\n{if_none_match}\n{if_unmodified_since}\n{range}\nx-ms-date:{x_ms_date}\nx-ms-version:2021-04-10\n/testaccount/testfilesystem/testfile.json"
    expected_signature = hmac_sign(
        expected_string_to_sign,
        key_bytes,
    )

    # Validate that the signature is correct and the query parameters are sorted
    assert (
        request.headers["Authorization"]
        == f"SharedKey testaccount:{expected_signature}"
    )

def hmac_sign(string: str, key: bytes) -> str:
    signed_hmac_sha256 = hmac.new(key, string.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(signed_hmac_sha256).decode()


def test_test_hmac_sign():
    # double check assumptions about the signing here in the test. Kind of just testing the test here.
    key_bytes = "testkey".encode("utf-8")
    expected_string_to_sign = "GET\n\n\n\n\n\n\n\n\n\n\n\nx-ms-date:Mon, 13 Apr 2025 12:00:00 GMT\nx-ms-version:2021-04-10\n/testaccount/testfilesystem/testfile.json"
    signed_hmac_sha256 = hmac.new(
        key_bytes, expected_string_to_sign.encode("utf-8"), hashlib.sha256
    ).digest()
    signature = base64.b64encode(signed_hmac_sha256).decode()
    assert signature == "iNSUkZfnd0VgI/bl8qyx1MiH01O/yA9vYfl2KNQLnRE="


def setup_azure_storage_mocks(
    respx_mock: respx.MockRouter,
    account_name: str,
    file_system: str,
) -> None:
    """
    Sets up common mocks for Azure Storage API calls and OpenAI completion.
    
    Args:
        respx_mock: The respx mock router
        account_name: Azure storage account name
        file_system: Azure storage file system name
    """
    # Regex to match the base URL structure and filename
    base_pattern = re.compile(
        rf"https://{account_name}\.dfs\.core\.windows\.net/{file_system}/.+\.json.*"
    )

    # Mock PUT request for file creation
    respx_mock.put(base_pattern, params={"resource": "file"}).respond(
        status_code=201  # Created
    )

    # Mock PATCH request for appending data
    respx_mock.patch(
        base_pattern, params={"action": "append", "position": "0"}
    ).respond(
        status_code=202
    )  # Accepted

    # Mock PATCH request for flushing data - allow any position value
    respx_mock.patch(base_pattern, params={"action": "flush"}).respond(
        status_code=202
    )  # Accepted

    # Mock OpenAI call - Ensure it returns an ID for filename generation
    respx_mock.post("https://api.openai.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello, world!"}}],
        }
    )


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.premium_user", True)
async def test_azure_blob_storage_shared_key(
    respx_mock: respx.MockRouter,
    monkeypatch,
):
    # Mock Azure Blob Storage API calls
    account_name = "testaccount"
    file_system = "testfilesystem"
    test_key_bytes = "testkey".encode("utf-8")
    test_key = base64.b64encode(test_key_bytes)

    # Mock environment variables for Shared Key Auth
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", account_name)
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", file_system)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", test_key)

    setup_azure_storage_mocks(respx_mock, account_name, file_system)

    azure_storage_logger = AzureBlobStorageLogger(flush_interval=0.1)
    monkeypatch.setattr(litellm, "callbacks", [azure_storage_logger])

    response = await litellm.acompletion(
        model="gpt-4o-mini",  # Use a model litellm recognizes by default
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_key="sk-12345",  # Dummy key
    )

    # Wait up to 1.5 seconds for the call count to increase to 3
    for _ in range(150):  # Check every 0.1 seconds
        if respx_mock.calls.call_count >= 3:
            break
        await asyncio.sleep(0.01)

    # Assert that the mocked Azure endpoints were called (at least create, append, flush)
    assert respx_mock.calls.call_count >= 3
    call_log: list[respx.models.Call] = list(respx_mock.calls)

    put_calls = [call for call in call_log if call.request.method == "PUT"]
    patch_calls = [call for call in call_log if call.request.method == "PATCH"]

    assert len(put_calls) == 1
    assert "resource=file" in str(put_calls[0].request.url)

    assert len(patch_calls) == 2
    append_call_found = any(
        "action=append" in str(call.request.url) for call in patch_calls
    )
    flush_call_found = any(
        "action=flush" in str(call.request.url) for call in patch_calls
    )
    assert append_call_found
    assert flush_call_found


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.premium_user", True)
async def test_azure_blob_storage_ad_auth(
    respx_mock: respx.MockRouter,
    monkeypatch: MonkeyPatch,
):
    # Mock Azure Blob Storage API calls
    account_name = "testaccount"
    file_system = "testfilesystem"
    test_token = "test_azure_ad_token"

    # Mock environment variables for AD Auth
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", account_name)
    monkeypatch.setenv("AZURE_STORAGE_FILE_SYSTEM", file_system)
    monkeypatch.setenv("AZURE_STORAGE_TENANT_ID", "test-tenant-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("AZURE_STORAGE_CLIENT_SECRET", "test-client-secret")
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)

    # Mock the Azure AD token provider
    def mock_token_provider():
        return test_token

    with patch(
        "litellm.integrations.azure_storage._azure_storage_auth.get_azure_ad_token_from_entrata_id",
        return_value=mock_token_provider,
    ):
        setup_azure_storage_mocks(respx_mock, account_name, file_system)

        azure_storage_logger = AzureBlobStorageLogger(flush_interval=0.1)
        monkeypatch.setattr(litellm, "callbacks", [azure_storage_logger])

        response = await litellm.acompletion(
            model="gpt-4o-mini",  # Use a model litellm recognizes by default
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_key="sk-12345",  # Dummy key
        )

        # Wait up to 1.5 seconds for the call count to increase to 3
        for _ in range(150):  # Check every 0.1 seconds
            if respx_mock.calls.call_count >= 3:
                break
            await asyncio.sleep(0.01)

        # Assert that the mocked Azure endpoints were called (at least create, append, flush)
        assert respx_mock.calls.call_count == 4
        call_log: list[respx.models.Call] = list(respx_mock.calls)
        azure_calls = [call for call in call_log if call.request.url.host == f"{account_name}.dfs.core.windows.net"]
        assert len(azure_calls) == 3
        
        for call in azure_calls:
            assert call.request.headers.get("Authorization") == f"Bearer {test_token}", f"call {call.request.url} with headers {call.request.headers} missing the correct token"

        put_calls = [call for call in call_log if call.request.method == "PUT"]
        patch_calls = [call for call in call_log if call.request.method == "PATCH"]

        assert len(put_calls) >= 1
        assert "resource=file" in str(put_calls[0].request.url)

        assert len(patch_calls) >= 2
        append_call_found = any(
            "action=append" in str(call.request.url) for call in patch_calls
        )
        flush_call_found = any(
            "action=flush" in str(call.request.url) for call in patch_calls
        )
        assert append_call_found
        assert flush_call_found


@pytest.mark.asyncio
async def test_azure_ad_token_caching():
    """Test that the Azure AD token is cached and reused for multiple requests"""
    account_name = "testaccount"
    test_token = "test_azure_ad_token"
    token_fetch_count = 0

    def mock_token_provider():
        nonlocal token_fetch_count
        token_fetch_count += 1
        return test_token

    with patch(
        "litellm.integrations.azure_storage._azure_storage_auth.get_azure_ad_token_from_entrata_id",
        return_value=mock_token_provider,
    ) as mock_get_token:
        auth1 = AzureADTokenAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )

        # Make requests with both auth instances
        request1 = httpx.Request("GET", f"https://{account_name}.dfs.core.windows.net/test")
        request2 = httpx.Request("GET", f"https://{account_name}.dfs.core.windows.net/test")

        auth1(request1)
        auth1(request2)

        # Verify token was only fetched once
        assert token_fetch_count == 1
        assert request1.headers["Authorization"] == f"Bearer {test_token}"
        assert request2.headers["Authorization"] == f"Bearer {test_token}"


@pytest.mark.asyncio
async def test_azure_ad_token_expiry():
    """Test that the Azure AD token is refreshed after expiry"""
    account_name = "testaccount"
    test_token_1 = "test_azure_ad_token_1"
    test_token_2 = "test_azure_ad_token_2"
    token_fetch_count = 0
    
    # Use a mutable object to track time
    mock_time = {"now": datetime(2024, 1, 1, 12, 0, 0)}

    def mock_token_provider():
        nonlocal token_fetch_count
        token_fetch_count += 1
        # Return different tokens to verify refresh
        return test_token_1 if token_fetch_count == 1 else test_token_2

    def get_mock_now():
        return mock_time["now"]

    # Mock time to control token expiry
    with patch(
        "litellm.integrations.azure_storage._azure_storage_auth.get_azure_ad_token_from_entrata_id",
        return_value=mock_token_provider,
    ) as mock_get_token, patch(
        "litellm.integrations.azure_storage._azure_storage_auth.datetime"
    ) as mock_datetime:
        # Setup datetime mock
        mock_datetime.now.side_effect = get_mock_now

        auth = AzureADTokenAuth(
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret",
        )

        # First request at t=0
        request1 = httpx.Request("GET", f"https://{account_name}.dfs.core.windows.net/test")
        auth(request1)
        assert token_fetch_count == 1
        assert request1.headers["Authorization"] == f"Bearer {test_token_1}"

        # Move time forward past token expiry (default 1 hour)
        mock_time["now"] = mock_time["now"] + timedelta(hours=1, minutes=6)

        # Second request after expiry
        request2 = httpx.Request("GET", f"https://{account_name}.dfs.core.windows.net/test")
        auth(request2)

        # Verify new token was fetched
        assert token_fetch_count == 2
        assert request2.headers["Authorization"] == f"Bearer {test_token_2}"

