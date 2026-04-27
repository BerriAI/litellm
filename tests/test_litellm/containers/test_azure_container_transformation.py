import os
import sys
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../"))

import litellm
from litellm.llms.azure.containers.transformation import AzureContainerConfig
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerListResponse,
    ContainerObject,
    DeleteContainerResult,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging


class TestAzureContainerConfig:
    """Test suite for Azure container transformation functionality."""

    def setup_method(self):
        self.config = AzureContainerConfig()
        self.logging_obj = LiteLLMLogging(
            model="",
            messages=[],
            stream=False,
            call_type="create_container",
            start_time=None,
            litellm_call_id="test_call_id",
            function_id="test_function_id",
        )

    def test_inherits_base_container_config(self):
        assert isinstance(self.config, BaseContainerConfig)

    def test_get_supported_openai_params(self):
        supported_params = self.config.get_supported_openai_params()
        assert "name" in supported_params
        assert "expires_after" in supported_params
        assert "file_ids" in supported_params

    def test_validate_environment_with_api_key(self):
        headers = {}
        api_key = "test-azure-key"

        validated_headers = self.config.validate_environment(
            headers=headers, api_key=api_key
        )

        assert "api-key" in validated_headers
        assert validated_headers["api-key"] == api_key

    def test_validate_environment_uses_azure_env_var(self, monkeypatch):
        monkeypatch.setenv("AZURE_API_KEY", "env-azure-key")
        headers = {}

        validated_headers = self.config.validate_environment(headers=headers)

        assert "api-key" in validated_headers
        assert validated_headers["api-key"] == "env-azure-key"

    def test_validate_environment_no_bearer_token(self):
        """Azure uses api-key header, not Authorization: Bearer."""
        headers = {}
        api_key = "azure-test-key"

        validated_headers = self.config.validate_environment(
            headers=headers, api_key=api_key
        )

        assert "Authorization" not in validated_headers
        assert "api-key" in validated_headers

    def test_get_complete_url_default_v1(self):
        """With default_api_version='v1', URL should include /openai/v1/containers."""
        api_base = "https://my-resource.openai.azure.com"
        litellm_params = {}

        url = self.config.get_complete_url(
            api_base=api_base, litellm_params=litellm_params
        )

        assert "/openai/v1/containers" in url
        assert "my-resource.openai.azure.com" in url

    def test_get_complete_url_with_explicit_api_version(self):
        api_base = "https://my-resource.openai.azure.com"
        litellm_params = {"api_version": "2025-01-01"}

        url = self.config.get_complete_url(
            api_base=api_base, litellm_params=litellm_params
        )

        assert "api-version=2025-01-01" in url
        assert "/openai/containers" in url

    def test_get_complete_url_with_latest_api_version(self):
        api_base = "https://my-resource.openai.azure.com"
        litellm_params = {"api_version": "latest"}

        url = self.config.get_complete_url(
            api_base=api_base, litellm_params=litellm_params
        )

        assert "/openai/v1/containers" in url

    def test_get_complete_url_raises_without_api_base(self, monkeypatch):
        monkeypatch.delenv("AZURE_API_BASE", raising=False)
        monkeypatch.setattr(litellm, "api_base", None)
        with pytest.raises(ValueError, match="api_base is required"):
            self.config.get_complete_url(api_base=None, litellm_params={})

    def test_transform_container_create_request(self):
        from litellm.types.router import GenericLiteLLMParams

        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-key"}
        name = "My Azure Container"
        optional_params = {
            "expires_after": {"anchor": "last_active_at", "minutes": 30},
            "file_ids": ["file_abc"],
        }

        data = self.config.transform_container_create_request(
            name=name,
            container_create_optional_request_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert data["name"] == name
        assert data["expires_after"]["minutes"] == 30
        assert data["file_ids"] == ["file_abc"]

    def test_transform_container_create_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_azure_123",
            "object": "container",
            "created_at": 1747857508,
            "status": "running",
            "expires_after": {"anchor": "last_active_at", "minutes": 30},
            "last_active_at": 1747857508,
            "name": "My Azure Container",
        }

        container = self.config.transform_container_create_response(
            raw_response=mock_response, logging_obj=self.logging_obj
        )

        assert isinstance(container, ContainerObject)
        assert container.id == "cntr_azure_123"
        assert container.name == "My Azure Container"
        assert container.status == "running"

    def test_transform_container_list_request(self):
        from litellm.types.router import GenericLiteLLMParams

        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-key"}

        url, params = self.config.transform_container_list_request(
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            limit=5,
            order="desc",
        )

        assert url == api_base
        assert params["limit"] == "5"
        assert params["order"] == "desc"

    def test_transform_container_list_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "object": "list",
            "data": [
                {
                    "id": "cntr_1",
                    "object": "container",
                    "created_at": 1747857508,
                    "status": "running",
                    "expires_after": {"anchor": "last_active_at", "minutes": 20},
                    "last_active_at": 1747857508,
                    "name": "Container 1",
                }
            ],
            "first_id": "cntr_1",
            "last_id": "cntr_1",
            "has_more": False,
        }

        container_list = self.config.transform_container_list_response(
            raw_response=mock_response, logging_obj=self.logging_obj
        )

        assert isinstance(container_list, ContainerListResponse)
        assert len(container_list.data) == 1
        assert container_list.first_id == "cntr_1"

    def test_transform_container_retrieve_request(self):
        from litellm.types.router import GenericLiteLLMParams

        container_id = "cntr_azure_abc"
        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-key"}

        url, params = self.config.transform_container_retrieve_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert url == f"{api_base}/{container_id}"
        assert params == {}

    def test_transform_container_delete_request(self):
        from litellm.types.router import GenericLiteLLMParams

        container_id = "cntr_azure_del"
        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-key"}

        url, params = self.config.transform_container_delete_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        assert url == f"{api_base}/{container_id}"
        assert params == {}

    def test_transform_container_delete_response(self):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.return_value = {
            "id": "cntr_azure_del",
            "object": "container.deleted",
            "deleted": True,
        }

        delete_result = self.config.transform_container_delete_response(
            raw_response=mock_response, logging_obj=self.logging_obj
        )

        assert isinstance(delete_result, DeleteContainerResult)
        assert delete_result.id == "cntr_azure_del"
        assert delete_result.deleted is True

    def test_transform_container_file_list_request(self):
        from litellm.types.router import GenericLiteLLMParams

        container_id = "cntr_azure_files"
        api_base = "https://my-resource.openai.azure.com/openai/v1/containers"
        litellm_params = GenericLiteLLMParams()
        headers = {"api-key": "test-key"}

        url, params = self.config.transform_container_file_list_request(
            container_id=container_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            limit=10,
        )

        assert url == f"{api_base}/{container_id}/files"
        assert params["limit"] == "10"

    def test_transform_requests_preserve_query_string_after_path(self):
        """api-version must not appear before /{container_id}/... (Azure bases include ?)."""
        from litellm.types.router import GenericLiteLLMParams

        api_base = (
            "https://my-resource.openai.azure.com/openai/v1/containers"
            "?api-version=v1"
        )
        litellm_params = GenericLiteLLMParams()
        headers: dict = {}

        url_r, _ = self.config.transform_container_retrieve_request(
            container_id="cntr_x",
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        assert (
            url_r
            == "https://my-resource.openai.azure.com/openai/v1/containers/cntr_x?api-version=v1"
        )

        url_fl, _ = self.config.transform_container_file_list_request(
            container_id="cntr_x",
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        assert (
            url_fl
            == "https://my-resource.openai.azure.com/openai/v1/containers/cntr_x/files?api-version=v1"
        )

        url_fc, _ = self.config.transform_container_file_content_request(
            container_id="cntr_x",
            file_id="cfile_y",
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        expected_fc = (
            "https://my-resource.openai.azure.com/openai/v1/containers/"
            "cntr_x/files/cfile_y/content?api-version=v1"
        )
        assert url_fc == expected_fc
        assert url_fc.index("/content") < url_fc.index("?")

    def test_provider_config_manager_returns_azure_config(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_container_config(
            provider=LlmProviders.AZURE
        )

        assert config is not None
        assert isinstance(config, AzureContainerConfig)

    def test_proxy_handler_factory_returns_azure_config(self):
        from litellm.proxy.container_endpoints.handler_factory import (
            _get_container_provider_config,
        )

        config = _get_container_provider_config("azure")

        assert config is not None
        assert isinstance(config, AzureContainerConfig)

    def test_proxy_handler_factory_raises_for_unsupported_provider(self):
        from litellm.proxy.container_endpoints.handler_factory import (
            _get_container_provider_config,
        )

        with pytest.raises(ValueError, match="Container API not supported"):
            _get_container_provider_config("anthropic")


class TestAzureContainerKnownFailureRegressions:
    """Regression tests for real production / proxy failures (Azure containers).

    1. **URL / api-version** — ``get_complete_url`` appends ``?api-version=…`` to the
       container base. Naïve ``f\"{api_base}/…\"`` put the query *before* path segments,
       e.g. ``…/containers?api-version=v1/cntr_…/files``, which Azure rejects
       ("API version not supported" / 404-style routing).

    2. **Bare resource root** — ``AZURE_API_BASE`` is only the host (no ``?``). The
       query appears only after LiteLLM builds the full container base; downstream
       transforms must still append ``/cntr_…/files/…`` *before* the query string.

    3. **File content path** — The worst case in logs was POST/GET logging showing
       ``…containers?api-version=v1/cntr_…/files/cfile_…/content``; correct wire shape is
       ``…containers/cntr_…/files/cfile_…/content?api-version=v1``.
    """

    def setup_method(self):
        self.config = AzureContainerConfig()

    def test_regression_query_never_splits_before_container_segment(self):
        """Forbid the broken shape: …/containers?api-version=v1/cntr_…"""
        from litellm.types.router import GenericLiteLLMParams

        api_base = (
            "https://my-resource.openai.azure.com/openai/v1/containers"
            "?api-version=v1"
        )
        cid = "cntr_69d4f27de324819082c54f6aeaab6391056f5dbdf1fe2b02"
        fid = "cfile_69d4f283bac0819094bfe7805a4f3ce8"
        litellm_params = GenericLiteLLMParams()
        headers: dict = {}

        url_fc, _ = self.config.transform_container_file_content_request(
            container_id=cid,
            file_id=fid,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        # Exact substring seen in broken logs
        assert "containers?api-version=v1/" + cid not in url_fc
        assert "containers?api-version=v1/cntr_" not in url_fc

        parsed = urlparse(url_fc)
        assert parsed.path == (f"/openai/v1/containers/{cid}/files/{fid}/content")
        assert parse_qs(parsed.query).get("api-version") == ["v1"]
        assert url_fc.index("/content") < url_fc.index("?")

    def test_regression_full_chain_bare_resource_root_like_env(self):
        """Mimics AZURE_API_BASE=https://resource.openai.azure.com — no ? in env."""
        from litellm.types.router import GenericLiteLLMParams

        resource_root = "https://my-resource.openai.azure.com"
        container_base = self.config.get_complete_url(
            api_base=resource_root,
            litellm_params={},
        )
        assert "openai.azure.com" in container_base
        assert (
            "openai/v1/containers" in container_base
            or "/openai/containers" in container_base
        )

        cid = "cntr_livepath123"
        fid = "cfile_live456"
        url_fc, params = self.config.transform_container_file_content_request(
            container_id=cid,
            file_id=fid,
            api_base=container_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert cid in url_fc
        assert fid in url_fc
        parsed = urlparse(url_fc)
        assert cid in parsed.path
        assert "?" not in parsed.path
        assert "/content" in parsed.path
        assert url_fc.index(cid) < (url_fc.index("?") if "?" in url_fc else len(url_fc))
        assert params == {}

    def test_regression_all_crud_urls_with_azure_style_api_base(self):
        """Retrieve, delete, list files, and file content all keep ?api-version last."""
        from litellm.types.router import GenericLiteLLMParams

        api_base = (
            "https://iamkankute-5584-resource.openai.azure.com/openai/v1/containers"
            "?api-version=v1"
        )
        cid = "cntr_69d4f1c5c6448190930a444af3f84f670b35dc2ee845cd1b"
        fid = "cfile_69d4f1c97a1081908d22a9f56268c743"
        litellm_params = GenericLiteLLMParams()
        headers: dict = {}

        url_r, _ = self.config.transform_container_retrieve_request(
            container_id=cid,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        url_d, _ = self.config.transform_container_delete_request(
            container_id=cid,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        url_lf, _ = self.config.transform_container_file_list_request(
            container_id=cid,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )
        url_fc, _ = self.config.transform_container_file_content_request(
            container_id=cid,
            file_id=fid,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        for name, u in (
            ("retrieve", url_r),
            ("delete", url_d),
            ("list_files", url_lf),
            ("file_content", url_fc),
        ):
            assert f"containers?api-version=v1/{cid}" not in u, name
            p = urlparse(u)
            assert cid in p.path, name
            assert "api-version" in p.query or "api-version=v1" in u, name

        assert urlparse(url_fc).path.endswith(f"/{cid}/files/{fid}/content")

    def test_regression_api_base_with_extra_query_params(self):
        """Multiple query params must stay at the end after path join."""
        from litellm.types.router import GenericLiteLLMParams

        api_base = (
            "https://my-resource.openai.azure.com/openai/v1/containers"
            "?api-version=v1&foo=bar"
        )
        cid = "cntr_x"
        url_lf, _ = self.config.transform_container_file_list_request(
            container_id=cid,
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        p = urlparse(url_lf)
        assert p.path == f"/openai/v1/containers/{cid}/files"
        qs = parse_qs(p.query)
        assert qs.get("api-version") == ["v1"]
        assert qs.get("foo") == ["bar"]

    def test_regression_proxy_resolves_azure_text_same_as_azure(self):
        """Router/proxy treat azure_text like azure for container config."""
        from litellm.proxy.container_endpoints.handler_factory import (
            _get_container_provider_config,
        )

        c1 = _get_container_provider_config("azure")
        c2 = _get_container_provider_config("azure_text")
        assert type(c1) is type(c2)
        assert isinstance(c1, AzureContainerConfig)
