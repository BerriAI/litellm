"""
Unit tests for Vertex AI Private Service Connect (PSC) endpoint support

Tests that LiteLLM properly constructs URLs when using custom api_base
for PSC endpoints.
"""

import os
import sys

import pytest

# Add the litellm package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from litellm.llms.vertex_ai.vertex_llm_base import VertexBase


class TestVertexAIPSCEndpointSupport:
    """Test cases for PSC endpoint URL construction"""

    def test_psc_endpoint_url_construction_basic(self):
        """Test basic PSC endpoint URL construction for predict endpoint"""
        vertex_base = VertexBase()
        psc_api_base = "http://10.96.32.8"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True

        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header="test-token",
            url="",  # This will be replaced
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        expected_url = f"{psc_api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_psc_endpoint_url_construction_with_streaming(self):
        """Test PSC endpoint URL construction with streaming enabled"""
        vertex_base = VertexBase()
        psc_api_base = "http://10.96.32.8"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="streamGenerateContent",
            stream=True,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        expected_url = f"{psc_api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:streamGenerateContent?alt=sse"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_psc_endpoint_url_construction_v1beta1(self):
        """Test PSC endpoint URL construction with v1beta1 API version"""
        vertex_base = VertexBase()
        psc_api_base = "http://10.96.32.8"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1beta1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        expected_url = f"{psc_api_base}/v1beta1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_psc_endpoint_url_with_https(self):
        """Test PSC endpoint URL construction with HTTPS"""
        vertex_base = VertexBase()
        psc_api_base = "https://10.96.32.8"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        expected_url = f"{psc_api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_psc_endpoint_with_trailing_slash(self):
        """Test that trailing slashes in api_base are handled correctly"""
        vertex_base = VertexBase()
        psc_api_base = "http://10.96.32.8/"
        endpoint_id = "1234567890"
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        # rstrip('/') should remove the trailing slash
        expected_url = f"{psc_api_base.rstrip('/')}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_standard_proxy_with_googleapis(self):
        """Test that standard proxies with googleapis.com in URL use simple format"""
        vertex_base = VertexBase()
        proxy_api_base = "https://my-proxy.googleapis.com"
        endpoint_id = "gemini-pro"  # Not numeric
        project_id = "test-project"
        location = "us-central1"
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=proxy_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="generateContent",
            stream=False,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
        )

        # Should use simple format: api_base:endpoint
        expected_url = f"{proxy_api_base}:generateContent"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_custom_proxy_with_numeric_model(self):
        """Test that numeric model IDs trigger PSC-style URL construction"""
        vertex_base = VertexBase()
        proxy_api_base = "https://my-custom-proxy.example.com"
        endpoint_id = "9876543210"  # Numeric endpoint ID
        project_id = "test-project"
        location = "us-central1"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=proxy_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header="test-token",
            url="",
            model=endpoint_id,
            vertex_project=project_id,
            vertex_location=location,
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        # Numeric model should trigger full path construction
        expected_url = f"{proxy_api_base}/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}:predict"
        assert (
            url == expected_url
        ), f"Expected {expected_url}, but got {url}"

    def test_no_api_base_returns_original_url(self):
        """Test that when api_base is None, the original URL is returned"""
        vertex_base = VertexBase()
        original_url = "https://us-central1-aiplatform.googleapis.com/v1/projects/test/locations/us-central1/publishers/google/models/gemini-pro:generateContent"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=None,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="generateContent",
            stream=False,
            auth_header="test-token",
            url=original_url,
            model="gemini-pro",
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        # When api_base is None, original URL should be returned unchanged
        assert url == original_url, f"Expected {original_url}, but got {url}"

    def test_auth_header_preserved(self):
        """Test that auth_header is properly preserved"""
        vertex_base = VertexBase()
        psc_api_base = "http://10.96.32.8"
        test_auth_header = "Bearer test-token-12345"
        use_psc_endpoint_format = True
        auth_header, url = vertex_base._check_custom_proxy(
            api_base=psc_api_base,
            custom_llm_provider="vertex_ai",
            gemini_api_key=None,
            endpoint="predict",
            stream=False,
            auth_header=test_auth_header,
            url="",
            model="1234567890",
            vertex_project="test-project",
            vertex_location="us-central1",
            vertex_api_version="v1",
            use_psc_endpoint_format=use_psc_endpoint_format,
        )

        assert (
            auth_header == test_auth_header
        ), f"Auth header should be preserved, got {auth_header}"

