import os
import requests
import sys
import re

sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv

"""
Azure OpenAI Responses API E2E Tests

WARNING: The Azure Responses API is very specific about:
1. Model compatibility - Only certain models support the Responses API (currently o4-mini)
2. API version requirements - Requires api-version "2025-03-01-preview" or later
3. Model deployments may change frequently with Azure updates

If tests fail, check:
- Azure model deployment status and supported models
- API version compatibility
- Model name changes in Azure portal
"""

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

# Load model configuration from YAML - find o4-mini model
# TODO: Update config file to have a specific "responses_api_model" descriptor
# instead of hardcoding "o4-mini", as the specific model will likely change
# in future Azure updates and deployments
config_path = os.path.join(os.path.dirname(__file__), "azure_testing_config.yaml")
DEFAULT_RESPONSES_MODEL = None

with open(config_path, 'r') as f:
    config_content = f.read()
    # Look for o4-mini model definition
    if 'model_name: o4-mini' in config_content:
        DEFAULT_RESPONSES_MODEL = "o4-mini"

if DEFAULT_RESPONSES_MODEL is None:
    raise ValueError("o4-mini model not found in azure_testing_config.yaml")


# Test cases for Azure OpenAI Responses API via LiteLLM proxy
class TestAzureResponsesAPI:
    """Test Azure OpenAI Responses API endpoint via LiteLLM proxy"""

    def _validate_responses_api_response(self, response):
        """Validate OpenAI Responses API response structure"""
        assert isinstance(response, dict), "Response must be dict"

        # Required fields for responses API
        required_fields = ["id", "object", "created_at", "model", "output"]
        for field in required_fields:
            assert field in response, f"Response missing required field: {field}"

        # Validate id format
        assert isinstance(response["id"], str), "id must be string"

        # Validate object type
        assert "object" in response, "Response missing object field"

        # Validate created_at timestamp
        assert isinstance(response["created_at"], int), "created_at must be integer"

        # Validate model
        assert isinstance(response["model"], str), "model must be string"

        # Validate output field (responses API uses 'output' instead of 'choices')
        if "output" in response:
            assert isinstance(response["output"], list), "output must be list"
            assert len(response["output"]) > 0, "output must not be empty"

        # Optional usage field
        if "usage" in response:
            usage = response["usage"]
            assert isinstance(usage, dict), "usage must be dict"
            # Responses API may use different usage field names
            expected_usage_fields = ["input_tokens", "output_tokens", "total_tokens"]
            for field in expected_usage_fields:
                if field in usage:
                    assert isinstance(
                        usage[field], int
                    ), f"usage.{field} must be integer"

    def test_basic_responses_request(self):
        """Test basic responses API request"""
        response = requests.post(
            f"{PROXY_URL}/responses",
            json={
                "model": DEFAULT_RESPONSES_MODEL,
                "input": "Write a haiku about coding",
                "max_output_tokens": 50,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-1234",
            },
            timeout=5,
        )

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        self._validate_responses_api_response(data)

    def test_responses_malformed_payload(self):
        """Test responses API with malformed payload"""
        # TODO: Update test assertion for proper error message once Azure error format is stabilized
        response = requests.post(
            f"{PROXY_URL}/responses",
            json={
                "model": DEFAULT_RESPONSES_MODEL
                # Missing required "input" field
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-1234",
            },
            timeout=5,
        )

        # Should return 400 or 500 for malformed payload (implementations vary)
        assert (
            response.status_code >= 400
        ), f"Expected error status for malformed payload, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_responses_invalid_model(self):
        """Test responses API with invalid model"""
        # TODO: Update test assertion for proper error message once Azure error format is stabilized
        response = requests.post(
            f"{PROXY_URL}/responses",
            json={
                "model": "non-existent-model",
                "input": "This should fail",
                "max_output_tokens": 20,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-1234",
            },
            timeout=5,
        )

        # Should return 400 for invalid model
        assert (
            response.status_code == 400
        ), f"Expected 400 for invalid model, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_responses_with_max_tokens(self):
        """Test responses API with max_output_tokens parameter"""
        response = requests.post(
            f"{PROXY_URL}/responses",
            json={
                "model": DEFAULT_RESPONSES_MODEL,
                "input": "Write a very long story about...",
                "max_output_tokens": 20,  # Use valid minimum value (16+)
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-1234",
            },
            timeout=5,
        )

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        self._validate_responses_api_response(data)

    def test_responses_error_handling(self):
        """Test responses API error handling with various invalid requests"""
        # TODO: Update test assertions for proper error messages once Azure error format is stabilized
        invalid_payloads = [
            {"input": "test"},  # missing model
            {"model": DEFAULT_RESPONSES_MODEL},  # missing input
        ]

        for payload in invalid_payloads:
            response = requests.post(
                f"{PROXY_URL}/responses",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-1234",
                },
                timeout=5,
            )

            # Should return error status for invalid payloads
            assert (
                response.status_code >= 400
            ), f"Expected error for payload {payload}, got {response.status_code}"

            data = response.json()
            assert (
                "error" in data
            ), f"Response should have error field for payload {payload}"
