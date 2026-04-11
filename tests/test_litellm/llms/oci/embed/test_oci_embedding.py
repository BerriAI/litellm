import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.oci.embed.transformation import OCIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse

# Test constants
TEST_MODEL_NAME = "cohere.embed-english-v3.0"
TEST_MODEL = f"oci/{TEST_MODEL_NAME}"
TEST_COMPARTMENT_ID = "ocid1.compartment.oc1..xxxxxx"
BASE_OCI_PARAMS = {
    "oci_region": "us-ashburn-1",
    "oci_user": "ocid1.user.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_fingerprint": "4f:29:77:cc:b1:3e:55:ab:61:2a:de:47:f1:38:4c:90",
    "oci_tenancy": "ocid1.tenancy.oc1..xxxxxxEXAMPLExxxxxx",
    "oci_compartment_id": TEST_COMPARTMENT_ID,
}

TEST_OCI_PARAMS_KEY = {
    **BASE_OCI_PARAMS,
    "oci_key": "<private_key.pem as string>",
}

TEST_OCI_PARAMS_KEY_FILE = {
    **BASE_OCI_PARAMS,
    "oci_key_file": "<private_key.pem as a Path>",
}

# Mock OCI embedding response
MOCK_OCI_EMBEDDING_RESPONSE = {
    "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
    "modelId": "cohere.embed-english-v3.0",
    "modelVersion": "3.0",
    "inputTextTokenCounts": [5, 4],
}


@pytest.fixture(params=[TEST_OCI_PARAMS_KEY, TEST_OCI_PARAMS_KEY_FILE])
def supplied_params(request):
    """Fixture for passing in optional_parameters"""
    return request.param


class TestOCIEmbeddingConfig:
    def test_get_complete_url_default_region(self):
        """test_get_complete_url returns URL with us-ashburn-1 when no api_base is given."""
        config = OCIEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=TEST_MODEL_NAME,
            optional_params={},
            litellm_params={},
        )
        assert "us-ashburn-1" in url
        assert "embedText" in url

    def test_get_complete_url_custom_region(self):
        """test_get_complete_url uses region from optional_params."""
        config = OCIEmbeddingConfig()
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=TEST_MODEL_NAME,
            optional_params={"oci_region": "us-chicago-1"},
            litellm_params={},
        )
        assert "us-chicago-1" in url
        assert "embedText" in url

    def test_get_complete_url_custom_api_base(self):
        """test_get_complete_url returns api_base as-is when provided."""
        config = OCIEmbeddingConfig()
        custom_base = "https://custom.oci.example.com/embed"
        url = config.get_complete_url(
            api_base=custom_base,
            api_key=None,
            model=TEST_MODEL_NAME,
            optional_params={},
            litellm_params={},
        )
        assert url == custom_base

    def test_get_supported_openai_params(self):
        """test_get_supported_openai_params returns expected params list."""
        config = OCIEmbeddingConfig()
        params = config.get_supported_openai_params(model=TEST_MODEL_NAME)
        assert "dimensions" in params
        assert "encoding_format" not in params

    def test_map_openai_params_dimensions(self):
        """test dimensions is mapped correctly."""
        config = OCIEmbeddingConfig()
        optional_params = {}
        result = config.map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params=optional_params,
            model=TEST_MODEL_NAME,
            drop_params=False,
        )
        assert result["dimensions"] == 512

    def test_validate_environment_with_credentials(self, supplied_params):
        """test validate_environment returns content-type and user-agent headers when credentials are supplied."""
        config = OCIEmbeddingConfig()
        headers = {}
        result = config.validate_environment(
            headers=headers,
            model=TEST_MODEL,
            messages=[],
            optional_params=supplied_params,
            litellm_params={},
        )
        assert result["content-type"] == "application/json"
        assert "litellm" in result["user-agent"]

    def test_validate_environment_missing_credentials(self):
        """test validate_environment raises Exception with 'Missing required parameters' when credentials are incomplete."""
        config = OCIEmbeddingConfig()
        incomplete_params = {
            "oci_user": "ocid1.user.oc1..xxx",
            # Missing oci_fingerprint, oci_tenancy, oci_key/oci_key_file, oci_compartment_id
        }
        with pytest.raises(Exception) as excinfo:
            config.validate_environment(
                headers={},
                model=TEST_MODEL,
                messages=[],
                optional_params=incomplete_params,
                litellm_params={},
            )
        assert "Missing required parameters" in str(excinfo.value)

    def test_validate_environment_with_signer(self):
        """test validate_environment passes when oci_signer is provided."""
        config = OCIEmbeddingConfig()

        class MockSigner:
            def do_request_sign(self, request, enforce_content_headers=True):
                request.headers["authorization"] = 'Signature version="1"'

        optional_params = {
            "oci_signer": MockSigner(),
            "oci_region": "us-ashburn-1",
        }
        result = config.validate_environment(
            headers={},
            model=TEST_MODEL,
            messages=[],
            optional_params=optional_params,
            litellm_params={},
        )
        assert result["content-type"] == "application/json"

    def test_transform_embedding_request_on_demand(self):
        """test transform_embedding_request builds correct ON_DEMAND OCI request body."""
        config = OCIEmbeddingConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
        }
        with patch.object(config, "sign_request", return_value=({}, "{}")):
            result = config.transform_embedding_request(
                model=TEST_MODEL_NAME,
                input=["Hello world", "Goodbye world"],
                optional_params=optional_params,
                headers={},
            )

        assert result["compartmentId"] == TEST_COMPARTMENT_ID
        assert result["servingMode"]["servingType"] == "ON_DEMAND"
        assert result["servingMode"]["modelId"] == TEST_MODEL_NAME
        assert result["inputs"] == ["Hello world", "Goodbye world"]
        assert result["truncate"] == "END"

    def test_transform_embedding_request_dedicated(self):
        """test transform_embedding_request builds DEDICATED servingMode with endpointId."""
        config = OCIEmbeddingConfig()
        test_endpoint_id = "ocid1.generativeaiendpoint.oc1.us-chicago-1.xxxxxx"
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "oci_serving_mode": "DEDICATED",
            "oci_endpoint_id": test_endpoint_id,
        }
        with patch.object(config, "sign_request", return_value=({}, "{}")):
            result = config.transform_embedding_request(
                model=TEST_MODEL_NAME,
                input=["Hello world"],
                optional_params=optional_params,
                headers={},
            )

        assert result["servingMode"]["servingType"] == "DEDICATED"
        assert result["servingMode"]["endpointId"] == test_endpoint_id

    def test_transform_embedding_request_input_type(self):
        """test input_type=search_query is mapped to SEARCH_QUERY in request data."""
        config = OCIEmbeddingConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
            "input_type": "search_query",
        }
        with patch.object(config, "sign_request", return_value=({}, "{}")):
            result = config.transform_embedding_request(
                model=TEST_MODEL_NAME,
                input=["What is the capital of Brazil?"],
                optional_params=optional_params,
                headers={},
            )

        assert result["inputType"] == "SEARCH_QUERY"

    def test_transform_embedding_request_string_input(self):
        """test single string input is wrapped in a list."""
        config = OCIEmbeddingConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
        }
        with patch.object(config, "sign_request", return_value=({}, "{}")):
            result = config.transform_embedding_request(
                model=TEST_MODEL_NAME,
                input="Hello world",
                optional_params=optional_params,
                headers={},
            )

        assert isinstance(result["inputs"], list)
        assert result["inputs"] == ["Hello world"]

    def test_transform_embedding_request_token_list_raises(self):
        """test token-array inputs raise ValueError instead of silent conversion."""
        config = OCIEmbeddingConfig()
        optional_params = {
            "oci_compartment_id": TEST_COMPARTMENT_ID,
        }
        with patch.object(config, "sign_request", return_value=({}, "{}")):
            with pytest.raises(ValueError, match="does not support token-array"):
                config.transform_embedding_request(
                    model=TEST_MODEL_NAME,
                    input=[[1234, 5678]],
                    optional_params=optional_params,
                    headers={},
                )

    def test_transform_embedding_response(self):
        """test OCI embedding response is correctly transformed into EmbeddingResponse."""
        config = OCIEmbeddingConfig()
        mock_response = httpx.Response(
            status_code=200,
            json=MOCK_OCI_EMBEDDING_RESPONSE,
            request=httpx.Request("POST", "https://test.com"),
        )
        mock_logging = MagicMock()
        model_response = EmbeddingResponse()

        result = config.transform_embedding_response(
            model=TEST_MODEL_NAME,
            raw_response=mock_response,
            model_response=model_response,
            logging_obj=mock_logging,
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.model == "cohere.embed-english-v3.0"
        assert len(result.data) == 2
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
        assert result.data[1]["embedding"] == [0.5, 0.6, 0.7, 0.8]
        assert result.data[0]["index"] == 0
        assert result.data[1]["index"] == 1
        # Total tokens: 5 + 4 = 9
        assert result.usage.prompt_tokens == 9
        assert result.usage.total_tokens == 9

    def test_transform_embedding_response_error(self):
        """test non-200 status code raises OCIError."""
        from litellm.llms.oci.common_utils import OCIError

        config = OCIEmbeddingConfig()
        mock_response = httpx.Response(
            status_code=400,
            text="Bad Request",
            request=httpx.Request("POST", "https://test.com"),
        )
        mock_logging = MagicMock()
        model_response = EmbeddingResponse()

        with pytest.raises(OCIError):
            config.transform_embedding_response(
                model=TEST_MODEL_NAME,
                raw_response=mock_response,
                model_response=model_response,
                logging_obj=mock_logging,
            )

    def test_model_prices_embedding_models(self):
        """test all 8 OCI embedding models exist in model_prices_and_context_window.json with mode=embedding."""
        model_prices_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "model_prices_and_context_window.json",
        )
        with open(model_prices_path) as f:
            model_prices = json.load(f)

        expected_embedding_models = [
            "oci/cohere.embed-english-v3.0",
            "oci/cohere.embed-english-light-v3.0",
            "oci/cohere.embed-multilingual-v3.0",
            "oci/cohere.embed-multilingual-light-v3.0",
            "oci/cohere.embed-english-image-v3.0",
            "oci/cohere.embed-english-light-image-v3.0",
            "oci/cohere.embed-multilingual-light-image-v3.0",
            "oci/cohere.embed-v4.0",
        ]

        for model_key in expected_embedding_models:
            assert model_key in model_prices, f"Missing model: {model_key}"
            assert (
                model_prices[model_key].get("mode") == "embedding"
            ), f"Model {model_key} does not have mode='embedding'"

    def test_model_prices_new_chat_models(self):
        """test the 16 new OCI chat models exist in model_prices_and_context_window.json with mode=chat."""
        model_prices_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "..",
            "model_prices_and_context_window.json",
        )
        with open(model_prices_path) as f:
            model_prices = json.load(f)

        expected_chat_models = [
            "oci/xai.grok-3",
            "oci/xai.grok-3-fast",
            "oci/xai.grok-3-mini",
            "oci/xai.grok-3-mini-fast",
            "oci/xai.grok-4",
            "oci/xai.grok-4-fast",
            "oci/xai.grok-4.1-fast",
            "oci/xai.grok-4.20",
            "oci/xai.grok-4.20-multi-agent",
            "oci/xai.grok-code-fast-1",
            "oci/cohere.command-a-03-2025",
            "oci/cohere.command-a-reasoning-08-2025",
            "oci/cohere.command-a-vision-07-2025",
            "oci/cohere.command-a-translate-08-2025",
            "oci/google.gemini-2.5-pro",
            "oci/google.gemini-2.5-flash",
        ]

        for model_key in expected_chat_models:
            assert model_key in model_prices, f"Missing model: {model_key}"
            assert (
                model_prices[model_key].get("mode") == "chat"
            ), f"Model {model_key} does not have mode='chat'"
