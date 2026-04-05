"""
Unit tests for OCI Generative AI embedding transformation.

These tests exercise the transformation layer only — no real OCI calls are made.
"""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.oci.common_utils import OCIError
from litellm.llms.oci.embed.transformation import OCI_EMBED_BATCH_LIMIT, OCIEmbedConfig
from litellm.types.utils import EmbeddingResponse, Usage

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

COMPARTMENT_ID = "ocid1.compartment.oc1..test"
BASE_PARAMS = {
    "oci_region": "us-ashburn-1",
    "oci_user": "ocid1.user.oc1..test",
    "oci_fingerprint": "aa:bb:cc:dd",
    "oci_tenancy": "ocid1.tenancy.oc1..test",
    "oci_compartment_id": COMPARTMENT_ID,
    "oci_key": "-----BEGIN RSA PRIVATE KEY-----\nfakekey\n-----END RSA PRIVATE KEY-----",
}


class TestOCIEmbedConfig:
    def _config(self) -> OCIEmbedConfig:
        return OCIEmbedConfig()

    # ------------------------------------------------------------------
    # validate_environment
    # ------------------------------------------------------------------

    def test_validate_environment_sets_headers(self):
        cfg = self._config()
        headers = cfg.validate_environment(
            headers={},
            model="oci/cohere.embed-v3.0",
            messages=[],
            optional_params=BASE_PARAMS,
            litellm_params={},
        )
        assert headers["content-type"] == "application/json"
        assert "litellm/" in headers["user-agent"]

    # ------------------------------------------------------------------
    # get_complete_url
    # ------------------------------------------------------------------

    def test_get_complete_url_default_region(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base=None,
            api_key=None,
            model="cohere.embed-v3.0",
            optional_params={"oci_region": "us-chicago-1"},
            litellm_params={},
        )
        assert url == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/embedText"

    def test_get_complete_url_respects_api_base(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base="https://custom.endpoint.example.com",
            api_key=None,
            model="cohere.embed-v3.0",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.endpoint.example.com/20231130/actions/embedText"

    def test_get_complete_url_strips_trailing_slash(self):
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base="https://custom.endpoint.example.com/",
            api_key=None,
            model="cohere.embed-v3.0",
            optional_params={},
            litellm_params={},
        )
        assert not url.endswith("//")
        assert url.endswith("/20231130/actions/embedText")

    # ------------------------------------------------------------------
    # transform_embedding_request
    # ------------------------------------------------------------------

    def test_transform_request_single_string(self):
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input="hello world",
            optional_params={"oci_compartment_id": COMPARTMENT_ID},
            headers={},
        )
        assert result["compartmentId"] == COMPARTMENT_ID
        assert result["servingMode"]["servingType"] == "ON_DEMAND"
        assert result["servingMode"]["modelId"] == "cohere.embed-v3.0"
        assert result["inputs"] == ["hello world"]

    def test_transform_request_list_of_texts(self):
        cfg = self._config()
        texts = ["hello", "world"]
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=texts,
            optional_params={"oci_compartment_id": COMPARTMENT_ID},
            headers={},
        )
        assert result["inputs"] == texts

    def test_transform_request_with_input_type(self):
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=["query"],
            optional_params={
                "oci_compartment_id": COMPARTMENT_ID,
                "input_type": "SEARCH_QUERY",
            },
            headers={},
        )
        assert result["inputType"] == "SEARCH_QUERY"

    def test_transform_request_with_output_dimensions(self):
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v4.0",
            input=["text"],
            optional_params={
                "oci_compartment_id": COMPARTMENT_ID,
                "outputDimensions": 512,
            },
            headers={},
        )
        assert result["outputDimensions"] == 512

    def test_transform_request_dedicated_serving_mode(self):
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=["text"],
            optional_params={
                "oci_compartment_id": COMPARTMENT_ID,
                "oci_serving_mode": "DEDICATED",
                "oci_endpoint_id": "ocid1.genaiendpoint.oc1..test",
            },
            headers={},
        )
        assert result["servingMode"]["servingType"] == "DEDICATED"
        assert result["servingMode"]["endpointId"] == "ocid1.genaiendpoint.oc1..test"
        assert "modelId" not in result["servingMode"]

    def test_transform_request_missing_compartment_id_raises(self):
        cfg = self._config()
        with pytest.raises(OCIError) as exc_info:
            cfg.transform_embedding_request(
                model="cohere.embed-v3.0",
                input=["text"],
                optional_params={},
                headers={},
            )
        assert exc_info.value.status_code == 400
        assert "oci_compartment_id" in str(exc_info.value)

    def test_transform_request_batch_limit_exceeded_raises(self):
        cfg = self._config()
        texts = ["text"] * (OCI_EMBED_BATCH_LIMIT + 1)
        with pytest.raises(OCIError) as exc_info:
            cfg.transform_embedding_request(
                model="cohere.embed-v3.0",
                input=texts,
                optional_params={"oci_compartment_id": COMPARTMENT_ID},
                headers={},
            )
        assert exc_info.value.status_code == 400
        assert str(OCI_EMBED_BATCH_LIMIT) in str(exc_info.value)

    def test_transform_request_invalid_serving_mode_raises(self):
        cfg = self._config()
        with pytest.raises(OCIError) as exc_info:
            cfg.transform_embedding_request(
                model="cohere.embed-v3.0",
                input=["text"],
                optional_params={
                    "oci_compartment_id": COMPARTMENT_ID,
                    "oci_serving_mode": "INVALID",
                },
                headers={},
            )
        assert exc_info.value.status_code == 400

    def test_transform_request_none_input_becomes_string(self):
        """Non-list, non-string inputs are coerced to str."""
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=42,  # type: ignore
            optional_params={"oci_compartment_id": COMPARTMENT_ID},
            headers={},
        )
        assert result["inputs"] == ["42"]

    # ------------------------------------------------------------------
    # transform_embedding_response
    # ------------------------------------------------------------------

    def _mock_response(self, status_code: int, body: dict) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            content=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )

    def test_transform_response_success(self):
        cfg = self._config()
        model_response = EmbeddingResponse()
        raw = self._mock_response(
            200,
            {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "modelId": "cohere.embed-v3.0",
                "modelVersion": "3.0.0",
                "usage": {"promptTokens": 10, "totalTokens": 10},
            },
        )
        result = cfg.transform_embedding_response(
            model="cohere.embed-v3.0",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            api_key=None,
            request_data={},
            optional_params={},
            litellm_params={},
        )
        assert len(result.data) == 2
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[1]["index"] == 1
        assert result.model == "cohere.embed-v3.0"
        assert result.usage.prompt_tokens == 10

    def test_transform_response_no_usage(self):
        cfg = self._config()
        model_response = EmbeddingResponse()
        raw = self._mock_response(
            200,
            {
                "embeddings": [[0.1]],
                "modelId": "cohere.embed-v3.0",
                "modelVersion": "3.0.0",
            },
        )
        result = cfg.transform_embedding_response(
            model="cohere.embed-v3.0",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            api_key=None,
            request_data={},
            optional_params={},
            litellm_params={},
        )
        assert len(result.data) == 1

    def test_transform_response_http_error_raises(self):
        cfg = self._config()
        raw = self._mock_response(401, {"error": "Unauthorized"})
        with pytest.raises(OCIError) as exc_info:
            cfg.transform_embedding_response(
                model="cohere.embed-v3.0",
                raw_response=raw,
                model_response=EmbeddingResponse(),
                logging_obj=MagicMock(),
                api_key=None,
                request_data={},
                optional_params={},
                litellm_params={},
            )
        assert exc_info.value.status_code == 401

    def test_transform_response_invalid_json_raises(self):
        cfg = self._config()
        raw = httpx.Response(
            status_code=200,
            content=b"not-json",
            headers={"content-type": "text/plain"},
        )
        with pytest.raises(OCIError):
            cfg.transform_embedding_response(
                model="cohere.embed-v3.0",
                raw_response=raw,
                model_response=EmbeddingResponse(),
                logging_obj=MagicMock(),
                api_key=None,
                request_data={},
                optional_params={},
                litellm_params={},
            )

    # ------------------------------------------------------------------
    # map_openai_params
    # ------------------------------------------------------------------

    def test_map_openai_params_dimensions(self):
        cfg = self._config()
        result = cfg.map_openai_params(
            non_default_params={"dimensions": 512},
            optional_params={},
            model="cohere.embed-v4.0",
        )
        assert result["outputDimensions"] == 512

    def test_map_openai_params_encoding_format_raises_without_drop(self):
        cfg = self._config()
        with pytest.raises(OCIError):
            cfg.map_openai_params(
                non_default_params={"encoding_format": "float"},
                optional_params={},
                model="cohere.embed-v3.0",
            )

    def test_map_openai_params_encoding_format_dropped_silently(self):
        cfg = self._config()
        result = cfg.map_openai_params(
            non_default_params={"encoding_format": "float"},
            optional_params={},
            model="cohere.embed-v3.0",
            drop_params=True,
        )
        assert "encoding_format" not in result

    # ------------------------------------------------------------------
    # env var credential resolution
    # ------------------------------------------------------------------

    def test_env_var_compartment_id(self, monkeypatch):
        monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.from.env")
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=["hello"],
            optional_params={},  # no compartment_id in params
            headers={},
        )
        assert result["compartmentId"] == "ocid1.compartment.from.env"

    def test_explicit_param_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("OCI_COMPARTMENT_ID", "ocid1.compartment.from.env")
        cfg = self._config()
        result = cfg.transform_embedding_request(
            model="cohere.embed-v3.0",
            input=["hello"],
            optional_params={"oci_compartment_id": "ocid1.compartment.explicit"},
            headers={},
        )
        assert result["compartmentId"] == "ocid1.compartment.explicit"

    def test_env_var_region_used_in_url(self, monkeypatch):
        monkeypatch.setenv("OCI_REGION", "eu-frankfurt-1")
        cfg = self._config()
        url = cfg.get_complete_url(
            api_base=None,
            api_key=None,
            model="cohere.embed-v3.0",
            optional_params={},  # no explicit region
            litellm_params={},
        )
        assert "eu-frankfurt-1" in url
