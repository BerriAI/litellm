"""
Test Model-Based Credential Routing for Files and Batches Endpoints

This test suite validates that files and batches can be routed to different
provider accounts based on model configuration in model_list.

Tests cover 3 scenarios:
1. Encoded file/batch ID with model (auto-routing)
2. Model specified via header/query/body parameter
3. Fallback to custom_llm_provider (environment variables)
"""
import pytest
import asyncio
import aiohttp
import json
import os
from typing import Optional
from openai import AsyncOpenAI

# Test configuration
BASE_URL = "http://localhost:4000"
API_KEY = "sk-1234"

# Mock model names configured in proxy config
MODEL_ACCOUNT_1 = "gpt-4o-test-account-1"
MODEL_ACCOUNT_2 = "gpt-4o-test-account-2"


class TestFilesModelBasedRouting:
    """Test model-based credential routing for files endpoints"""

    @pytest.mark.asyncio
    async def test_scenario_1_encoded_file_id_routing(self):
        """
        Scenario 1: File ID with embedded model info
        - Upload file with model parameter
        - Verify file ID is encoded with model info
        - Retrieve file using encoded ID (no model param needed)
        - Get file content using encoded ID
        - Delete file using encoded ID
        All operations should auto-route to the correct account
        """
        async with aiohttp.ClientSession() as session:
            # Upload file with model header
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": MODEL_ACCOUNT_1,
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field(
                "file",
                b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}}',
                filename="batch.jsonl",
            )

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                
                # Verify file ID starts with "file-" and contains encoded data
                assert file_id.startswith("file-"), f"File ID should start with 'file-', got: {file_id}"
                assert len(file_id) > 10, "File ID should be encoded with model info"
                
                print(f"✅ Created file with encoded ID: {file_id}")

            # Retrieve file without specifying model (should auto-route)
            url = f"{BASE_URL}/v1/files/{file_id}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            async with session.get(url, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert result["id"] == file_id
                assert result["purpose"] == "batch"
                print(f"✅ Retrieved file using auto-routing from encoded ID")

            # Get file content without specifying model
            url = f"{BASE_URL}/v1/files/{file_id}/content"
            async with session.get(url, headers=headers) as response:
                assert response.status == 200
                content = await response.text()
                assert len(content) > 0
                print(f"✅ Got file content using auto-routing from encoded ID")

            # Delete file without specifying model
            url = f"{BASE_URL}/v1/files/{file_id}"
            async with session.delete(url, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert result["deleted"] is True
                print(f"✅ Deleted file using auto-routing from encoded ID")

    @pytest.mark.asyncio
    async def test_scenario_2_model_via_header(self):
        """
        Scenario 2: Model specified via header
        - Upload file with model header
        - Retrieve file with model header
        - List files with model query param
        All operations should use credentials from the specified model
        """
        async with aiohttp.ClientSession() as session:
            # Upload file with model header
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": MODEL_ACCOUNT_2,
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "fine-tune")
            data.add_field(
                "file", b'{"prompt": "Hello", "completion": "Hi"}', filename="data.jsonl"
            )

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                print(f"✅ Created file with model header: {file_id}")

            # List files with model query parameter
            url = f"{BASE_URL}/v1/files?model={MODEL_ACCOUNT_2}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            async with session.get(url, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert "data" in result
                print(f"✅ Listed files with model query param")

            # Clean up
            url = f"{BASE_URL}/v1/files/{file_id}"
            await session.delete(url, headers={"Authorization": f"Bearer {API_KEY}", "x-litellm-model": MODEL_ACCOUNT_2})

    @pytest.mark.asyncio
    async def test_scenario_2_model_via_query_param(self):
        """
        Scenario 2: Model specified via query parameter
        - Upload file with model query param
        - Operations should use credentials from the specified model
        """
        async with aiohttp.ClientSession() as session:
            # Upload file with model query param
            url = f"{BASE_URL}/v1/files?model={MODEL_ACCOUNT_1}"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field("file", b'{"test": "data"}', filename="test.jsonl")

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                print(f"✅ Created file with model query param: {file_id}")

            # Clean up
            url = f"{BASE_URL}/v1/files/{file_id}"
            await session.delete(url, headers={"Authorization": f"Bearer {API_KEY}"})

    @pytest.mark.asyncio
    async def test_scenario_3_fallback_to_provider(self):
        """
        Scenario 3: No model specified, fallback to environment variables
        - Upload file without model parameter
        - Should use credentials from OPENAI_API_KEY env var
        """
        # This test requires OPENAI_API_KEY to be set
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set, skipping fallback test")

        async with aiohttp.ClientSession() as session:
            url = f"{BASE_URL}/v1/files"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("purpose", "fine-tune")
            data.add_field("file", b'{"prompt": "Test"}', filename="fallback.jsonl")

            async with session.post(url, headers=headers, data=data) as response:
                # Should succeed with environment credentials
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                # File ID should NOT be encoded (no model info)
                assert not file_id.startswith("file-bGl0ZWxs")  # Not base64 encoded
                print(f"✅ Created file with fallback credentials: {file_id}")

                # Clean up
                url = f"{BASE_URL}/v1/files/{file_id}"
                await session.delete(url, headers=headers)


class TestBatchesModelBasedRouting:
    """Test model-based credential routing for batches endpoints"""

    @pytest.mark.asyncio
    async def test_scenario_1_encoded_batch_id_routing(self):
        """
        Scenario 1: Batch with encoded file ID (auto-routing)
        - Create file with model parameter (gets encoded ID)
        - Create batch using encoded file ID (no model param needed)
        - Verify batch ID is also encoded
        - Retrieve batch using encoded ID
        - Cancel batch using encoded ID
        All operations should auto-route to the correct account
        """
        async with aiohttp.ClientSession() as session:
            # Step 1: Create file with model
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": MODEL_ACCOUNT_1,
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field(
                "file",
                b'{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}}',
                filename="batch_test.jsonl",
            )

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                assert file_id.startswith("file-")
                print(f"✅ Created file with encoded ID: {file_id}")

            # Step 2: Create batch using encoded file ID (no model param)
            url = f"{BASE_URL}/v1/batches"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }

            async with session.post(url, headers=headers, json=body) as response:
                assert response.status == 200
                result = await response.json()
                batch_id = result["id"]
                
                # Verify batch ID starts with "batch_" and is encoded
                assert batch_id.startswith("batch_"), f"Batch ID should start with 'batch_', got: {batch_id}"
                assert len(batch_id) > 10, "Batch ID should be encoded with model info"
                assert result["input_file_id"] == file_id
                print(f"✅ Created batch with encoded ID: {batch_id}")

            # Step 3: Retrieve batch without model param (auto-routing)
            url = f"{BASE_URL}/v1/batches/{batch_id}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            async with session.get(url, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert result["id"] == batch_id
                assert result["input_file_id"] == file_id
                print(f"✅ Retrieved batch using auto-routing from encoded ID")

            # Step 4: Cancel batch without model param
            url = f"{BASE_URL}/v1/batches/{batch_id}/cancel"
            async with session.post(url, headers=headers) as response:
                # Cancelling might fail if batch already completed, that's ok
                print(f"✅ Attempted to cancel batch using auto-routing")

            # Clean up file
            url = f"{BASE_URL}/v1/files/{file_id}"
            await session.delete(url, headers=headers)

    @pytest.mark.asyncio
    async def test_scenario_2_batch_with_model_header(self):
        """
        Scenario 2: Batch with model specified via header
        - Create batch with model header
        - List batches with model query param
        Operations should use credentials from the specified model
        """
        async with aiohttp.ClientSession() as session:
            # First create a file
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": MODEL_ACCOUNT_2,
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field(
                "file",
                b'{"custom_id": "test-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Test"}]}}',
                filename="test_batch.jsonl",
            )

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]

            # Create batch with model header
            url = f"{BASE_URL}/v1/batches"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "x-litellm-model": MODEL_ACCOUNT_2,
            }
            body = {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }

            async with session.post(url, headers=headers, json=body) as response:
                assert response.status == 200
                result = await response.json()
                batch_id = result["id"]
                print(f"✅ Created batch with model header: {batch_id}")

            # List batches with model query param
            url = f"{BASE_URL}/v1/batches?model={MODEL_ACCOUNT_2}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            async with session.get(url, headers=headers) as response:
                assert response.status == 200
                result = await response.json()
                assert "data" in result
                print(f"✅ Listed batches with model query param")

            # Clean up
            url = f"{BASE_URL}/v1/files/{file_id}"
            await session.delete(url, headers={"Authorization": f"Bearer {API_KEY}"})

    @pytest.mark.asyncio
    async def test_id_encoding_format(self):
        """
        Verify that file and batch IDs are correctly encoded with proper prefixes
        - File IDs should start with "file-"
        - Batch IDs should start with "batch_"
        """
        async with aiohttp.ClientSession() as session:
            # Create file with model
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": MODEL_ACCOUNT_1,
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field("file", b'{"test": "data"}', filename="format_test.jsonl")

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 200
                result = await response.json()
                file_id = result["id"]
                
                # Verify file ID format
                assert file_id.startswith("file-"), f"File ID must start with 'file-', got: {file_id}"
                print(f"✅ File ID format correct: {file_id}")

                # Extract base64 part (after "file-")
                base64_part = file_id[5:]
                assert len(base64_part) > 20, "Encoded data should be substantial"

            # Create batch
            url = f"{BASE_URL}/v1/batches"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }

            async with session.post(url, headers=headers, json=body) as response:
                assert response.status == 200
                result = await response.json()
                batch_id = result["id"]
                
                # Verify batch ID format
                assert batch_id.startswith("batch_"), f"Batch ID must start with 'batch_', got: {batch_id}"
                print(f"✅ Batch ID format correct: {batch_id}")

                # Verify input_file_id is preserved
                assert result["input_file_id"] == file_id
                
                # Extract base64 part (after "batch_")
                base64_part = batch_id[6:]
                assert len(base64_part) > 20, "Encoded data should be substantial"

            # Clean up
            url = f"{BASE_URL}/v1/files/{file_id}"
            await session.delete(url, headers={"Authorization": f"Bearer {API_KEY}"})


class TestModelNotFound:
    """Test error handling when model is not found in config"""

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """
        Test that using a non-existent model name returns proper error
        """
        async with aiohttp.ClientSession() as session:
            url = f"{BASE_URL}/v1/files"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "x-litellm-model": "non-existent-model-12345",
            }
            data = aiohttp.FormData()
            data.add_field("purpose", "batch")
            data.add_field("file", b'{"test": "data"}', filename="error_test.jsonl")

            async with session.post(url, headers=headers, data=data) as response:
                assert response.status == 400
                result = await response.json()
                assert "error" in result
                assert "not found in model_list" in result["error"]["message"]
                print(f"✅ Correct error for invalid model name")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

