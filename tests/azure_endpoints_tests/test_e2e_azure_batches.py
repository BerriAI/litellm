import os
import json
import requests
import sys
import tempfile
import time

sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv

# Load environment variables from .env.test
load_dotenv(
    dotenv_path=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../.env.test")
    ),
    override=True,
)
PROXY_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")


# Test cases for Azure OpenAI Batch API via LiteLLM proxy
class TestAzureBatches:
    """Test Azure OpenAI Batch API endpoint via LiteLLM proxy"""

    @classmethod
    def setup_class(cls):
        """Set up class-level resources before running tests"""
        cls.temp_file_paths = []  # Track temp files for cleanup
        cls.test_file_id = cls._create_persistent_batch_file()

    @classmethod
    def teardown_class(cls):
        """Clean up class-level resources after running tests"""
        # Clean up any temporary files created during tests
        for temp_file_path in cls.temp_file_paths:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    @classmethod
    def _create_persistent_batch_file(cls):
        """
        Create a persistent batch file for the test suite
        Uses the same logic as _create_test_batch_file but keeps the file for all tests
        """
        # Create test batch data in JSONL format
        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 50,
                },
            },
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Count to 3"}],
                    "max_tokens": 50,
                },
            },
        ]

        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
            temp_file_path = f.name
            cls.temp_file_paths.append(temp_file_path)  # Track for cleanup

        # Upload file via files API
        with open(temp_file_path, "rb") as f:
            files = {"file": ("batch_requests.jsonl", f, "application/json")}
            data = {"purpose": "batch"}
            headers = {"Authorization": "Bearer sk-1234"}

            response = requests.post(
                f"{PROXY_URL}/v1/files", files=files, data=data, headers=headers
            )

            if response.status_code == 200:
                file_data = response.json()
                return file_data["id"]
            else:
                # Throw error if file upload fails
                raise Exception(
                    f"File upload failed: {response.status_code} - {response.text}"
                )

    def _validate_batch_response(self, data):
        """
        Validate OpenAI Batch API response structure
        """
        assert isinstance(data, dict), f"Response should be dict, got {type(data)}"

        # Required fields for batch response
        required_fields = [
            "id",
            "object",
            "endpoint",
            "completion_window",
            "status",
            "created_at",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate specific field types
        assert isinstance(
            data["id"], str
        ), f"'id' must be string, got {type(data['id'])}"
        assert (
            data["object"] == "batch"
        ), f"Expected object 'batch', got {data['object']}"
        assert isinstance(
            data["endpoint"], str
        ), f"'endpoint' must be string, got {type(data['endpoint'])}"
        assert isinstance(
            data["completion_window"], str
        ), f"'completion_window' must be string, got {type(data['completion_window'])}"
        assert isinstance(
            data["status"], str
        ), f"'status' must be string, got {type(data['status'])}"
        assert isinstance(
            data["created_at"], int
        ), f"'created_at' must be integer, got {type(data['created_at'])}"

    def test_azure_batches_happy_path(self):
        """
        /batches happy path
        Input: valid batch request with properly uploaded file
        Output: 200, valid batch response structure
        """
        # Use the persistent test batch file created during setup
        file_id = self.test_file_id

        payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/batches"
        response = requests.post(url, json=payload, headers=headers)

        # Should get successful batch creation or appropriate error
        assert response.status_code == 200
        data = response.json()
        self._validate_batch_response(data)

        # Additional validations for successful batch creation
        assert (
            data["endpoint"] == "/v1/chat/completions"
        ), "Endpoint should match request"
        assert (
            data["completion_window"] == "24h"
        ), "Completion window should match request"
        assert data["status"] in [
            "validating",
            "in_progress",
            "completed",
        ], "Status should be valid"

    def test_azure_batches_malformed_payload(self):
        """
        /batches with malformed payload
        Input: missing required fields
        Output: Error response
        """
        payload = {
            "endpoint": "/v1/chat/completions"
            # Missing input_file_id and completion_window - INTENTIONALLY WRONG
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/batches"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_batches_invalid_endpoint(self):
        """
        /batches with invalid endpoint
        Input: unsupported endpoint
        Output: Error response
        """
        payload = {
            "input_file_id": self.test_file_id,
            "endpoint": "/v1/invalid-endpoint",
            "completion_window": "24h",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/batches"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_batches_invalid_completion_window(self):
        """
        /batches with invalid completion window
        Input: invalid completion window value
        Output: Error response
        """
        payload = {
            "input_file_id": self.test_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "invalid-window",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/batches"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_batches_get_nonexistent(self):
        """
        /batches/{id} GET for non-existent batch
        Input: non-existent batch ID
        Output: 404 or appropriate error
        """
        batch_id = "batch-nonexistent123"
        headers = {"Authorization": "Bearer sk-1234"}
        url = f"{PROXY_URL}/v1/batches/{batch_id}"
        response = requests.get(url, headers=headers)

        # Should return 404 or similar error
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_batches_error_shape(self):
        """
        /batches error response structure
        Input: various invalid requests
        Output: consistent error structure
        """
        invalid_payloads = [
            {
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },  # missing input_file_id
            {
                "input_file_id": "file-123",
                "completion_window": "24h",
            },  # missing endpoint
            {
                "input_file_id": "file-123",
                "endpoint": "/v1/chat/completions",
            },  # missing completion_window
            {},  # empty payload
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/batches"

        for payload in invalid_payloads:
            response = requests.post(url, json=payload, headers=headers)

            # Should return error status
            assert response.status_code >= 400, f"Expected error for payload {payload}"

            data = response.json()
            assert "error" in data, "Response should have error field"

            # Basic error structure
            error = data["error"]
            assert "message" in error, "Error should have message field"
            assert isinstance(error["message"], str), "Error message should be string"

    def test_azure_batches_create_and_retrieve_success(self):
        """
        REGRESSION TEST: End-to-end batch processing with data consistency validation
        1) POST /v1/batches to create batch with known input file
        2) Poll batch status until completion (up to 3 minutes, checking every 5 seconds)
        3) Download and validate output file contains actual LLM responses for all requests
        4) GET /v1/batches/{id} to retrieve final completed batch metadata
        5) Verify final batch data is consistent with initial creation (same endpoint, completion_window, file_id)
        6) Confirm batch shows "completed" status with valid output_file_id and completed_at timestamp

        This test ensures the full batch workflow works correctly and data remains consistent throughout.
        """
        # Step 1: Create batch
        payload = {
            "input_file_id": self.test_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }

        create_response = requests.post(
            f"{PROXY_URL}/v1/batches", json=payload, headers=headers
        )
        assert (
            create_response.status_code == 200
        ), f"Batch creation failed: {create_response.status_code} - {create_response.text}"

        create_data = create_response.json()
        self._validate_batch_response(create_data)
        batch_id = create_data["id"]

        # Step 2: Poll for batch completion
        final_batch_data = self._poll_batch_completion(batch_id, timeout=180)

        # Step 3: Download and validate output file
        output_results = self._download_and_validate_output(final_batch_data)

        # Step 4: Final GET to verify batch data consistency after completion
        self._verify_batch_consistency(
            batch_id, payload, final_batch_data["output_file_id"], output_results
        )

    def _poll_batch_completion(self, batch_id, timeout=180):
        """Poll batch until completion with timeout"""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                assert (
                    False
                ), f"Batch {batch_id} did not complete within {timeout} seconds"

            batch_response = requests.get(
                f"{PROXY_URL}/v1/batches/{batch_id}",
                headers={"Authorization": "Bearer sk-1234"},
            )
            assert (
                batch_response.status_code == 200
            ), f"Batch polling failed: {batch_response.status_code}"

            batch_data = batch_response.json()
            current_status = batch_data["status"]

            # Log progress
            progress_info = [f"status: {current_status}"]
            if "request_counts" in batch_data:
                counts = batch_data["request_counts"]
                progress_info.append(
                    f"progress: {counts.get('completed', 0)}/{counts.get('total', 0)}"
                )

            if current_status == "completed":
                return batch_data
            elif current_status in ["failed", "cancelled"]:
                assert False, f"Batch failed with status: {current_status}"
            elif current_status in ["validating", "in_progress", "finalizing"]:
                time.sleep(5)
                continue
            else:
                assert False, f"Unexpected batch status: {current_status}"

    def _download_and_validate_output(self, batch_data):
        """Download and validate batch output file"""
        assert (
            batch_data["status"] == "completed"
        ), "Batch should be completed at this point"
        assert batch_data.get(
            "output_file_id"
        ), "Completed batch should have output_file_id"

        output_file_id = batch_data["output_file_id"]

        # Download the output file
        file_response = requests.get(
            f"{PROXY_URL}/v1/files/{output_file_id}/content",
            headers={"Authorization": "Bearer sk-1234"},
        )
        assert (
            file_response.status_code == 200
        ), f"File download failed: {file_response.status_code}"

        # Parse JSONL output file
        output_lines = file_response.text.strip().split("\n")
        output_results = [json.loads(line) for line in output_lines if line.strip()]

        assert len(output_results) > 0, "Output file should contain results"

        # Validate each result
        for result in output_results:
            assert "custom_id" in result, "Result should have custom_id"
            assert "response" in result, "Result should have response"

            response_obj = result["response"]
            if response_obj.get("status_code") == 200:
                body = response_obj.get("body", {})
                assert "choices" in body, "Successful response should have choices"
                assert len(body["choices"]) > 0, "Should have at least one choice"

                choice = body["choices"][0]
                assert "message" in choice, "Choice should have message"
                assert "content" in choice["message"], "Message should have content"
                assert (
                    len(choice["message"]["content"].strip()) > 0
                ), "Content should not be empty"

        return output_results

    def _verify_batch_consistency(
        self, batch_id, original_payload, expected_output_file_id, output_results
    ):
        """Verify final batch data consistency"""
        retrieve_response = requests.get(
            f"{PROXY_URL}/v1/batches/{batch_id}",
            headers={"Authorization": "Bearer sk-1234"},
        )
        assert (
            retrieve_response.status_code == 200
        ), f"Final batch retrieval failed: {retrieve_response.status_code} - {retrieve_response.text}"

        retrieve_data = retrieve_response.json()
        self._validate_batch_response(retrieve_data)

        # Verify consistency with original request
        assert (
            retrieve_data["id"] == batch_id
        ), f"Batch ID mismatch: expected {batch_id}, got {retrieve_data['id']}"
        assert (
            retrieve_data["endpoint"] == original_payload["endpoint"]
        ), f"Endpoint mismatch: expected {original_payload['endpoint']}, got {retrieve_data['endpoint']}"
        assert (
            retrieve_data["completion_window"] == original_payload["completion_window"]
        ), f"Completion window mismatch: expected {original_payload['completion_window']}, got {retrieve_data['completion_window']}"
        assert (
            retrieve_data["input_file_id"] == original_payload["input_file_id"]
        ), f"Input file ID mismatch: expected {original_payload['input_file_id']}, got {retrieve_data['input_file_id']}"

        # Verify completion status and fields
        assert (
            retrieve_data["status"] == "completed"
        ), f"Final status should be completed, got: {retrieve_data['status']}"
        assert (
            retrieve_data.get("output_file_id") is not None
        ), "Completed batch must have output_file_id"
        assert (
            retrieve_data.get("completed_at") is not None
        ), "Completed batch must have completed_at timestamp"
        assert isinstance(
            retrieve_data["completed_at"], int
        ), "completed_at should be integer timestamp"
        assert (
            retrieve_data["output_file_id"] == expected_output_file_id
        ), f"Output file ID mismatch: expected {expected_output_file_id}, got {retrieve_data['output_file_id']}"

        # Validate request counts
        if "request_counts" in retrieve_data:
            counts = retrieve_data["request_counts"]
            assert (
                counts.get("total", 0) > 0
            ), "Completed batch should have processed requests"
            assert (
                counts.get("completed", 0) > 0
            ), "Completed batch should have successfully completed requests"
            assert counts.get("completed", 0) == len(
                output_results
            ), f"Completed count ({counts.get('completed', 0)}) should match output results ({len(output_results)})"

    def test_azure_batches_cancel_workflow(self):
        """
        REGRESSION TEST: Create batch with long completion window, immediately cancel, verify cancellation
        1) POST /v1/batches with 24h completion window
        2) Immediately POST /v1/batches/{id}/cancel (without await)
        3) GET /v1/batches/{id} to verify cancellation status

        NOTE: This test may fail initially to reveal the cancel payload structure
        """
        # Create a batch with long completion window
        payload = {
            "input_file_id": self.test_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",  # Long window to ensure batch doesn't complete immediately
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }

        # Step 1: Create batch
        create_response = requests.post(
            f"{PROXY_URL}/v1/batches", json=payload, headers=headers
        )
        assert (
            create_response.status_code == 200
        ), f"Batch creation failed: {create_response.status_code} - {create_response.text}"

        create_data = create_response.json()
        batch_id = create_data["id"]

        # Step 2: Immediately cancel the batch (without awaiting processing)
        cancel_response = requests.post(
            f"{PROXY_URL}/v1/batches/{batch_id}/cancel",
            headers={"Authorization": "Bearer sk-1234"},
        )

        if cancel_response.status_code == 200:
            cancel_data = cancel_response.json()

            # Validate cancel response structure
            assert "status" in cancel_data, "Cancel response should include status"
            assert cancel_data["status"] in [
                "cancelled",
                "cancelling",
            ], f"Expected cancelled/cancelling status, got: {cancel_data['status']}"

        # Step 3: Verify cancellation by retrieving batch status
        retrieve_response = requests.get(
            f"{PROXY_URL}/v1/batches/{batch_id}",
            headers={"Authorization": "Bearer sk-1234"},
        )
        assert (
            retrieve_response.status_code == 200
        ), f"Batch retrieval after cancel failed: {retrieve_response.status_code}"

        final_data = retrieve_response.json()
        final_status = final_data["status"]

        # Verify the batch was actually cancelled and didn't process
        assert final_status in [
            "cancelled",
            "cancelling",
        ], f"Expected batch to be cancelled, but status is: {final_status}"
        assert (
            final_status != "completed"
        ), "Batch should not have completed after being cancelled"
        assert (
            final_status != "in_progress"
        ), "Batch should not be in progress after being cancelled"
