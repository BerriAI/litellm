"""
Test Azure Sentinel logging integration
"""

import gzip
import json
import os

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.azure_sentinel.azure_sentinel import AzureSentinelLogger
from litellm.types.utils import StandardLoggingPayload


def _make_logger(**overrides):
    """Helper to create an AzureSentinelLogger with mocked asyncio.create_task"""
    defaults = dict(
        dcr_immutable_id="dcr-test123456789",
        endpoint="https://test-dce.eastus-1.ingest.monitor.azure.com",
        tenant_id="test-tenant-id",
        client_id="test-client-id",
        client_secret="test-client-secret",
    )
    defaults.update(overrides)
    with patch("asyncio.create_task", side_effect=lambda coro: coro.close()):
        return AzureSentinelLogger(**defaults)


def _make_payload(**overrides):
    defaults = dict(
        id="test_id",
        call_type="completion",
        model="gpt-3.5-turbo",
        status="success",
        messages=[{"role": "user", "content": "Hello"}],
        response={"choices": [{"message": {"content": "Hi"}}]},
    )
    defaults.update(overrides)
    return StandardLoggingPayload(**defaults)


def _mock_http_client(logger):
    """Wire up mocked token + API responses and return the mock."""
    mock_token_response = MagicMock()
    mock_token_response.status_code = 200
    mock_token_response.json = MagicMock(
        return_value={"access_token": "test-bearer-token", "expires_in": 3600}
    )
    mock_token_response.text = "Success"

    mock_api_response = MagicMock()
    mock_api_response.status_code = 204
    mock_api_response.text = "Success"

    async def mock_post(*args, **kwargs):
        if "oauth2/v2.0/token" in kwargs.get("url", ""):
            return mock_token_response
        return mock_api_response

    logger.async_httpx_client.post = AsyncMock(side_effect=mock_post)
    return logger.async_httpx_client.post


@pytest.mark.asyncio
async def test_azure_sentinel_oauth_and_send_batch():
    """Test that Azure Sentinel logger gets OAuth token and sends batch to API with gzip"""
    logger = _make_logger()
    logger.log_queue.append(_make_payload())
    mock_post = _mock_http_client(logger)

    await logger.async_send_batch()

    # Verify OAuth token + at least one API call
    assert mock_post.called
    assert mock_post.call_count >= 2

    # Get the API call (last call)
    api_call = mock_post.call_args_list[-1]
    assert "dcr-test123456789" in api_call.kwargs["url"]

    # Verify gzip Content-Encoding header
    headers = api_call.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert headers["Content-Encoding"] == "gzip"
    assert headers["Authorization"].startswith("Bearer ")

    # Verify body is valid gzip containing JSON array
    compressed_body = api_call.kwargs["content"]
    decompressed = gzip.decompress(compressed_body)
    parsed = json.loads(decompressed)
    assert isinstance(parsed, list)
    assert len(parsed) == 1

    # Queue should be cleared
    assert len(logger.log_queue) == 0


@pytest.mark.asyncio
async def test_azure_sentinel_batch_splitting():
    """Test that large batches are split into multiple requests under 1MB"""
    logger = _make_logger()

    # Create payloads with large content to force splitting.
    # Each payload will be ~100KB so 15 of them (~1.5MB) should trigger a split.
    large_content = "x" * 100_000
    for i in range(15):
        logger.log_queue.append(
            _make_payload(
                id=f"test_{i}",
                messages=[{"role": "user", "content": large_content}],
            )
        )

    mock_post = _mock_http_client(logger)
    await logger.async_send_batch()

    # Should have token call + multiple API calls (more than 1 batch)
    api_calls = [
        c for c in mock_post.call_args_list if "oauth2/v2.0/token" not in c.kwargs.get("url", "")
    ]
    assert len(api_calls) >= 2, f"Expected multiple batches, got {len(api_calls)}"

    # Each batch body should decompress to a valid JSON array
    total_events = 0
    for call in api_calls:
        compressed = call.kwargs["content"]
        decompressed = gzip.decompress(compressed)
        parsed = json.loads(decompressed)
        assert isinstance(parsed, list)
        total_events += len(parsed)

    # All 15 events accounted for
    assert total_events == 15
    assert len(logger.log_queue) == 0


@pytest.mark.asyncio
async def test_azure_sentinel_split_into_batches_single_oversized_entry():
    """Test that a single entry larger than MAX_BATCH_SIZE_BYTES is sent alone"""
    logger = _make_logger()

    # One very large payload that exceeds the batch size on its own.
    # Use varied content so JSON serialization keeps it large.
    import hashlib
    chunks = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(20_000)]
    huge_content = " ".join(chunks)  # ~1.3MB of hex digests
    batches = logger._split_into_batches(
        [
            _make_payload(
                id="small_1",
                messages=[{"role": "user", "content": "hi"}],
            ),
            _make_payload(
                id="huge",
                messages=[{"role": "user", "content": huge_content}],
            ),
            _make_payload(
                id="small_2",
                messages=[{"role": "user", "content": "hello"}],
            ),
        ]
    )

    # Should produce at least 2 batches (the oversized one isolated, plus the smalls)
    assert len(batches) >= 2

    # Verify all entries present across batches
    total_ids = []
    for compressed in batches:
        parsed = json.loads(gzip.decompress(compressed))
        for entry in parsed:
            total_ids.append(entry.get("id"))
    assert set(total_ids) == {"small_1", "huge", "small_2"}


def test_column_limit_truncates_large_fields():
    """Test that fields exceeding the configured limit are truncated (keeping the tail)"""
    with patch.dict(os.environ, {"AZURE_SENTINEL_TRUNCATE_BYTES": "262144"}):
        logger = _make_logger()

    # Content larger than the configured limit
    big_messages = "A" * 300_000
    big_response = "B" * 300_000

    payload = _make_payload(
        id="big_entry",
        messages=[{"role": "user", "content": big_messages}],
        response=big_response,
    )

    result = logger._enforce_column_limits(payload)

    # Should be a new object (deep copy)
    assert result is not payload

    # Messages field should be truncated, keeping tail
    msg_str = str(result["messages"])
    assert msg_str.startswith("[truncated by litellm]...")
    assert len(msg_str) <= logger.truncate_max_chars

    # Response field should be truncated, keeping tail
    resp_str = str(result["response"])
    assert resp_str.startswith("[truncated by litellm]...")
    assert len(resp_str) <= logger.truncate_max_chars

    # Truncation metadata present
    metadata = result.get("metadata", {})
    trunc_info = metadata.get("litellm_content_truncated") if isinstance(metadata, dict) else None
    if trunc_info is None:
        trunc_info = result.get("litellm_content_truncated")
    assert trunc_info is not None
    assert trunc_info["truncated"] is True
    assert trunc_info["truncate_reason"] == "azure_column_limit"
    assert "messages" in trunc_info["truncated_fields"]
    assert "response" in trunc_info["truncated_fields"]
    assert trunc_info["original_messages_chars"] == len(str(payload["messages"]))
    assert trunc_info["max_column_chars"] == 262144

    # Original payload not mutated
    assert len(str(payload["messages"])) > 262144


def test_column_limit_preserves_small_payloads():
    """Test that payloads under the configured limit are returned unchanged"""
    with patch.dict(os.environ, {"AZURE_SENTINEL_TRUNCATE_BYTES": "262144"}):
        logger = _make_logger()

    payload = _make_payload(
        id="small_entry",
        messages=[{"role": "user", "content": "hello world"}],
    )

    result = logger._enforce_column_limits(payload)

    # Should be the exact same object (no copy needed)
    assert result is payload
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict):
        assert "litellm_content_truncated" not in metadata


def test_truncate_disabled_when_env_unset():
    """Test that truncation is skipped when AZURE_SENTINEL_TRUNCATE_BYTES is not set"""
    with patch.dict(os.environ, {}, clear=False):
        # Ensure the env var is absent
        os.environ.pop("AZURE_SENTINEL_TRUNCATE_BYTES", None)
        logger = _make_logger()
    assert logger.truncate_content is False

    # Create payload with content exceeding 256 KB
    huge_content = "z" * 300_000
    payloads = [
        _make_payload(
            id="huge_no_truncate",
            messages=[{"role": "user", "content": huge_content}],
        ),
    ]

    batches = logger._split_into_batches(payloads)
    assert len(batches) >= 1
    # Collect all entries
    all_entries = []
    for compressed in batches:
        all_entries.extend(json.loads(gzip.decompress(compressed)))
    entry = all_entries[0]
    # The messages should be the full original content (not truncated)
    assert "[truncated by litellm]" not in str(entry.get("messages", ""))


def test_truncate_enabled_in_split_batches():
    """Test that _split_into_batches truncates large fields when enabled"""
    with patch.dict(os.environ, {"AZURE_SENTINEL_TRUNCATE_BYTES": "262144"}):
        logger = _make_logger()
    assert logger.truncate_content is True
    assert logger.truncate_max_chars == 262144

    # Content exceeding the configured limit
    huge_content = "X" * 400_000
    payloads = [
        _make_payload(
            id="small_before",
            messages=[{"role": "user", "content": "hi"}],
        ),
        _make_payload(
            id="huge_truncated",
            messages=[{"role": "user", "content": huge_content}],
        ),
        _make_payload(
            id="small_after",
            messages=[{"role": "user", "content": "bye"}],
        ),
    ]

    batches = logger._split_into_batches(payloads)

    # Collect all entries
    all_entries = {}
    for compressed in batches:
        for entry in json.loads(gzip.decompress(compressed)):
            all_entries[entry["id"]] = entry

    assert set(all_entries.keys()) == {"small_before", "huge_truncated", "small_after"}

    # The huge entry should have truncation metadata
    huge_entry = all_entries["huge_truncated"]
    metadata = huge_entry.get("metadata", {})
    trunc_info = metadata.get("litellm_content_truncated") if isinstance(metadata, dict) else None
    if trunc_info is None:
        trunc_info = huge_entry.get("litellm_content_truncated")
    assert trunc_info is not None
    assert trunc_info["truncated"] is True
    assert trunc_info["truncate_reason"] == "azure_column_limit"
    # Messages field should be capped near 262,144 chars
    msg_str = str(huge_entry["messages"])
    assert msg_str.startswith("[truncated by litellm]...")

    # Small entries should have no truncation metadata
    for entry_id in ("small_before", "small_after"):
        entry = all_entries[entry_id]
        meta = entry.get("metadata", {})
        if isinstance(meta, dict):
            assert "litellm_content_truncated" not in meta
