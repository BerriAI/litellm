import os
import sys
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../"))

import litellm
from litellm.llms.azure.containers.transformation import AzureContainerConfig
from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.containers.main import (
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

    def test_get_complete_url_strips_responses_path_and_preserves_api_version(self):
        """When api_base is the responses endpoint URL, get_complete_url must:
        - strip /openai/responses (no double-path)
        - use the api-version from api_base query string, NOT the deployment's
          older api_version (e.g. 2024-08-01-preview → containers need 2025-04-01-preview)
        """
        api_base = "https://my-resource.cognitiveservices.azure.com/openai/responses?api-version=2025-04-01-preview"

        url = self.config.get_complete_url(
            api_base=api_base,
            litellm_params={"api_version": "2024-08-01-preview"},
        )

        assert (
            "/openai/responses/openai/containers" not in url
        ), "path must not double /openai/responses"
        assert "my-resource.cognitiveservices.azure.com" in url
        assert "/openai/containers" in url or "/openai/v1/containers" in url
        assert (
            "2025-04-01-preview" in url
        ), "must use version from api_base, not litellm_params"
        assert (
            "2024-08-01-preview" not in url
        ), "must not fall back to older chat api_version"

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

    def test_transform_requests_encode_path_ids_before_query_string(self):
        from litellm.types.router import GenericLiteLLMParams

        api_base = (
            "https://my-resource.openai.azure.com/openai/v1/containers"
            "?api-version=v1"
        )

        url, _ = self.config.transform_container_file_content_request(
            container_id="../../other",
            file_id="file?download=1#frag",
            api_base=api_base,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        expected_url = (
            "https://my-resource.openai.azure.com/openai/v1/containers/"
            "..%2F..%2Fother/files/file%3Fdownload%3D1%23frag/content"
            "?api-version=v1"
        )
        assert url == expected_url

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

    @pytest.mark.asyncio
    async def test_regression_no_container_id_does_not_use_user_supplied_model_id(
        self, monkeypatch
    ):
        """Operations without container_id (create, list) must NOT route via
        _ageneric_api_call_with_fallbacks using a caller-supplied model_id.

        Security boundary: only the path that holds a validated container_id
        is trusted to fall back to the forwarded model_id.  A caller setting
        model_id without container_id on POST /v1/containers must not gain
        access to an arbitrary deployment UUID.
        """
        from litellm.router import Router

        router = Router(
            model_list=[
                {
                    "model_name": "azure-model",
                    "litellm_params": {
                        "model": "azure/gpt-4",
                        "api_base": "https://my-resource.cognitiveservices.azure.com",
                        "api_key": "test-key",
                        "api_version": "2025-04-01-preview",
                    },
                    "model_info": {"id": "deployment-uuid-123"},
                }
            ]
        )

        fallback_called = {"called": False}

        async def _mock_fallback(original_function, **kwargs):
            fallback_called["called"] = True
            return {}

        monkeypatch.setattr(router, "_ageneric_api_call_with_fallbacks", _mock_fallback)

        original_called = {"called": False}

        async def _noop(**kwargs):
            original_called["called"] = True
            return {}

        # No container_id — simulates create/list; caller injects a model_id
        await router._init_containers_api_endpoints(
            original_function=_noop,
            model_id="deployment-uuid-123",
            custom_llm_provider="azure",
        )

        assert not fallback_called["called"], (
            "_ageneric_api_call_with_fallbacks must NOT be called when "
            "container_id is absent, even if model_id is supplied"
        )
        assert original_called["called"], "original_function must be called directly"

    def test_regression_httpx_empty_params_strips_query_string(self):
        """httpx erases the URL query-string when params={} (empty dict) is passed.

        Root cause of the Azure container 404s on POST/DELETE:
          _build_query_params returns {} when the endpoint has no extra params;
          passing that {} as params= to httpx wiped ?api-version=2025-04-01-preview.

        Fix: every container httpx call now uses `params or None` so an empty
        dict falls back to None, which tells httpx to leave the URL untouched.
        """
        url = (
            "https://resource.cognitiveservices.azure.com"
            "/openai/containers/cntr_123?api-version=2025-04-01-preview"
        )
        client = httpx.AsyncClient()

        req_none = client.build_request("DELETE", url, params=None)
        assert "api-version=2025-04-01-preview" in str(req_none.url)

        req_empty = client.build_request("DELETE", url, params={})
        assert "api-version" not in str(
            req_empty.url
        ), "Documents root cause: params={} strips the query string"

        effective: dict = {}
        req_guarded = client.build_request("DELETE", url, params=effective or None)
        assert "api-version=2025-04-01-preview" in str(
            req_guarded.url
        ), "`params or None` must preserve ?api-version"

    def test_regression_proxy_resolves_azure_text_same_as_azure(self):
        """Router/proxy treat azure_text like azure for container config."""
        from litellm.proxy.container_endpoints.handler_factory import (
            _get_container_provider_config,
        )

        c1 = _get_container_provider_config("azure")
        c2 = _get_container_provider_config("azure_text")
        assert type(c1) is type(c2)
        assert isinstance(c1, AzureContainerConfig)

    @pytest.mark.asyncio
    async def test_proxy_process_request_forwards_decoded_container_id(
        self, monkeypatch
    ):
        from starlette.requests import Request

        from litellm.proxy.container_endpoints import handler_factory

        encoded_id = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id="model_abc123",
            container_id="cntr_123",
        )
        captured = {}

        async def _mock_base_process_llm_request(
            self,
            request,
            fastapi_response,
            user_api_key_dict,
            route_type,
            **kwargs,
        ):
            captured["data"] = self.data
            captured["route_type"] = route_type
            return {"id": "cfile_abc"}

        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        monkeypatch.setattr(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            _mock_base_process_llm_request,
        )
        access_check = AsyncMock(return_value=("cntr_123", "azure"))
        monkeypatch.setattr(
            handler_factory,
            "assert_user_can_access_container",
            access_check,
        )

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/v1/containers/id/files/id/content",
                "headers": [],
                "query_string": b"",
            }
        )
        fastapi_response = MagicMock()

        await handler_factory._process_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=MagicMock(),
            route_type="alist_container_files",
            path_params={"container_id": encoded_id},
        )

        access_check.assert_awaited_once()
        assert access_check.await_args.kwargs["container_id"] == encoded_id
        assert captured["route_type"] == "alist_container_files"
        assert captured["data"]["container_id"] == "cntr_123"
        assert captured["data"]["custom_llm_provider"] == "azure"
        assert captured["data"]["model_id"] == "model_abc123"
        assert "api_base" not in captured["data"]

    @pytest.mark.asyncio
    async def test_regression_binary_file_request_routes_through_proxy_processor(
        self, monkeypatch
    ):
        from fastapi import Response
        from starlette.requests import Request

        from litellm.proxy.container_endpoints import handler_factory

        encoded_id = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id="model_abc123",
            container_id="cntr_123",
        )
        captured = {}

        async def _mock_base_process_llm_request(
            self,
            request,
            fastapi_response,
            user_api_key_dict,
            route_type,
            **kwargs,
        ):
            captured["data"] = self.data
            captured["route_type"] = route_type
            fastapi_response.headers["x-litellm-call-id"] = "call-123"
            return b"csv-bytes"

        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )

        monkeypatch.setattr(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            _mock_base_process_llm_request,
        )
        access_check = AsyncMock(return_value=("cntr_123", "azure"))
        monkeypatch.setattr(
            handler_factory,
            "assert_user_can_access_container",
            access_check,
        )

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/v1/containers/id/files/id/content",
                "headers": [],
                "query_string": b"",
            }
        )
        fastapi_response = Response()

        response = await handler_factory._process_binary_request(
            request=request,
            fastapi_response=fastapi_response,
            container_id=encoded_id,
            file_id="cfile_abc",
            user_api_key_dict=MagicMock(),
        )

        access_check.assert_awaited_once()
        assert access_check.await_args.kwargs["container_id"] == encoded_id
        assert captured["route_type"] == "aretrieve_container_file_content"
        assert captured["data"]["container_id"] == "cntr_123"
        assert captured["data"]["file_id"] == "cfile_abc"
        assert captured["data"]["custom_llm_provider"] == "azure"
        assert captured["data"]["model_id"] == "model_abc123"
        assert response.status_code == 200
        assert response.body == b"csv-bytes"
        assert response.headers["x-litellm-call-id"] == "call-123"

    @pytest.mark.asyncio
    async def test_regression_multipart_upload_request_uses_provider_from_managed_id(
        self, monkeypatch
    ):
        from starlette.requests import Request

        from litellm.proxy.common_request_processing import (
            ProxyBaseLLMRequestProcessing,
        )
        from litellm.proxy.common_utils import http_parsing_utils
        from litellm.proxy.container_endpoints import handler_factory

        encoded_id = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id="model_abc123",
            container_id="cntr_123",
        )
        captured = {}

        async def _mock_get_form_data(request):
            return {"file": "ignored"}

        async def _mock_convert_upload_files_to_file_data(form_data):
            return {"file": [("data.csv", b"csv-bytes", "text/csv")]}

        async def _mock_base_process_llm_request(
            self,
            request,
            fastapi_response,
            user_api_key_dict,
            route_type,
            **kwargs,
        ):
            captured["data"] = self.data
            captured["route_type"] = route_type
            return {"id": "cfile_abc"}

        monkeypatch.setattr(
            http_parsing_utils,
            "get_form_data",
            _mock_get_form_data,
        )
        monkeypatch.setattr(
            http_parsing_utils,
            "convert_upload_files_to_file_data",
            _mock_convert_upload_files_to_file_data,
        )
        monkeypatch.setattr(
            ProxyBaseLLMRequestProcessing,
            "base_process_llm_request",
            _mock_base_process_llm_request,
        )
        access_check = AsyncMock(return_value=("cntr_123", "azure"))
        monkeypatch.setattr(
            handler_factory,
            "assert_user_can_access_container",
            access_check,
        )

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/containers/id/files",
                "headers": [],
                "query_string": b"",
            }
        )

        await handler_factory._process_multipart_upload_request(
            request=request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            route_type="aupload_container_file",
            container_id=encoded_id,
        )

        access_check.assert_awaited_once()
        assert access_check.await_args.kwargs["container_id"] == encoded_id
        assert captured["route_type"] == "aupload_container_file"
        assert captured["data"]["container_id"] == "cntr_123"
        assert captured["data"]["custom_llm_provider"] == "azure"
        assert captured["data"]["model_id"] == "model_abc123"

    @pytest.mark.asyncio
    async def test_regression_get_container_forwarding_params_sets_model_id_for_managed_id(
        self,
    ):
        """get_container_forwarding_params must extract model_id from a
        LiteLLM-managed encoded container ID and include it in the forwarding
        dict.  This is the proxy-side half of the native-Azure-ID routing fix:
        the router's _init_containers_api_endpoints reads kwargs["model_id"]
        which is set here.
        """
        from litellm.proxy.container_endpoints.ownership import (
            get_container_forwarding_params,
        )

        encoded_id = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id="deployment-uuid-123",
            container_id="cntr_6a058b43d24c8190a226cfb1d35405b20115fb7875ff11df",
        )

        params = await get_container_forwarding_params(
            container_id=encoded_id,
            original_container_id="cntr_6a058b43d24c8190a226cfb1d35405b20115fb7875ff11df",
            custom_llm_provider="azure",
        )

        assert (
            params.get("model_id") == "deployment-uuid-123"
        ), "model_id must be forwarded to the router for managed container IDs"
        assert params.get("container_id") == (
            "cntr_6a058b43d24c8190a226cfb1d35405b20115fb7875ff11df"
        )
        assert params.get("custom_llm_provider") == "azure"

    @pytest.mark.asyncio
    async def test_regression_get_container_forwarding_params_recovers_model_id_for_native_id(
        self, monkeypatch
    ):
        """Native Azure IDs (``cntr_<hex>``) cannot be decoded, so model_id
        must be recovered from the ownership row's ``unified_object_id`` —
        the encoded form captured at create time when the router selected a
        specific deployment. Without this, the router-side fallback for
        native IDs in ``_init_containers_api_endpoints`` is dead code.
        """
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from litellm.proxy.container_endpoints import ownership
        from litellm.proxy.container_endpoints.ownership import (
            get_container_forwarding_params,
        )

        native_id = "cntr_6a058b43d24c8190a226cfb1d35405b20115fb7875ff11df"
        encoded_stored_id = ResponsesAPIRequestUtils._build_container_id(
            custom_llm_provider="azure",
            model_id="deployment-uuid-123",
            container_id=native_id,
        )

        ownership._CONTAINER_STORED_ID_CACHE.flush_cache()
        ownership._CONTAINER_OWNER_CACHE.flush_cache()

        table = AsyncMock()
        table.find_first.return_value = SimpleNamespace(
            created_by="user-1",
            file_purpose=ownership.CONTAINER_OBJECT_PURPOSE,
            unified_object_id=encoded_stored_id,
        )
        prisma_client = SimpleNamespace(
            db=SimpleNamespace(litellm_managedobjecttable=table)
        )
        monkeypatch.setattr(
            ownership,
            "_get_prisma_client",
            AsyncMock(return_value=prisma_client),
        )

        params = await get_container_forwarding_params(
            container_id=native_id,
            original_container_id=native_id,
            custom_llm_provider="azure",
        )

        assert params.get("model_id") == "deployment-uuid-123", (
            "model_id must be recovered from the stored unified_object_id "
            "for native upstream container IDs"
        )
        assert params.get("container_id") == native_id
        assert params.get("custom_llm_provider") == "azure"

    @pytest.mark.asyncio
    async def test_regression_native_azure_container_id_uses_forwarded_model_id(
        self, monkeypatch
    ):
        """Native Azure container IDs (cntr_ + hex, no LiteLLM payload) must
        still route through _ageneric_api_call_with_fallbacks using the
        model_id forwarded from the proxy ownership check so that deployment
        credentials (api_base) are applied."""
        from litellm.router import Router

        router = Router(
            model_list=[
                {
                    "model_name": "azure-model",
                    "litellm_params": {
                        "model": "azure/gpt-4",
                        "api_base": "https://my-resource.cognitiveservices.azure.com",
                        "api_key": "test-key",
                        "api_version": "2025-04-01-preview",
                    },
                    "model_info": {"id": "deployment-uuid-123"},
                }
            ]
        )

        called_with: dict = {}

        async def _mock_fallback(original_function, **kwargs):
            called_with.update(kwargs)
            return {}

        monkeypatch.setattr(router, "_ageneric_api_call_with_fallbacks", _mock_fallback)

        native_azure_id = "cntr_6a058b43d24c8190a226cfb1d35405b20115fb7875ff11df"

        async def _noop(**kwargs):
            return {}

        await router._init_containers_api_endpoints(
            original_function=_noop,
            container_id=native_azure_id,
            model_id="deployment-uuid-123",
            custom_llm_provider="azure",
        )

        assert called_with.get("model") == "deployment-uuid-123", (
            "_ageneric_api_call_with_fallbacks must be called with the forwarded "
            "model_id when the container_id carries no LiteLLM routing payload"
        )
