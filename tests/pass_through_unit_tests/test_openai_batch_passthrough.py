import base64
import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)


def _make_logging_obj(model: str = "gpt-4o") -> LiteLLMLoggingObj:
    obj = LiteLLMLoggingObj(
        model=model,
        messages=[],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-func-id",
    )
    obj.async_success_handler = AsyncMock()
    return obj


def _make_batch_httpx_response(batch_id: str = "batch_abc123") -> Mock:
    mock_resp = Mock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-xyz",
        "completion_window": "24h",
        "status": "validating",
        "output_file_id": None,
        "error_file_id": None,
        "created_at": 1711471533,
        "expires_at": 1711557933,
        "in_progress_at": None,
        "finalizing_at": None,
        "completed_at": None,
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": None,
    }
    mock_request = Mock()
    mock_request.method = "POST"
    mock_resp.request = mock_request
    return mock_resp


class TestIsOpenAIBatchRoute:
    def test_openai_batch_post(self):
        url = "https://api.openai.com/v1/batches"
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_batch_route(url, "POST") is True
        )

    def test_openai_batch_get_ignored(self):
        url = "https://api.openai.com/v1/batches"
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_batch_route(url, "GET") is False
        )

    def test_azure_batch_post(self):
        url = "https://myresource.openai.azure.com/openai/deployments/gpt-4o/batches"
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_batch_route(url, "POST") is True
        )

    def test_non_batch_route(self):
        url = "https://api.openai.com/v1/chat/completions"
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route(url) is False

    def test_batch_with_trailing_slash_not_matched(self):
        url = "https://api.openai.com/v1/batches/"
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route(url) is False

    def test_empty_url(self):
        assert OpenAIPassthroughLoggingHandler.is_openai_batch_route("") is False


class TestExtractModelFromBatchUrl:
    def test_openai_url_returns_openai(self):
        url = "https://api.openai.com/v1/batches"
        result = OpenAIPassthroughLoggingHandler._extract_model_from_batch_url(
            url, "openai"
        )
        assert result == "openai"

    def test_azure_url_extracts_deployment(self):
        url = "https://myresource.openai.azure.com/openai/deployments/gpt-4o/batches"
        result = OpenAIPassthroughLoggingHandler._extract_model_from_batch_url(
            url, "azure"
        )
        assert result == "azure/gpt-4o"

    def test_azure_url_with_version_prefix_extracts_deployment(self):
        url = (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4-turbo/batches"
        )
        result = OpenAIPassthroughLoggingHandler._extract_model_from_batch_url(
            url, "azure"
        )
        assert result == "azure/gpt-4-turbo"


class TestBatchCreationHandler:
    def test_stores_managed_object_on_success(self):
        logging_obj = _make_logging_obj()
        httpx_response = _make_batch_httpx_response("batch_abc123")
        url_route = "https://api.openai.com/v1/batches"

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.store_batch_managed_object"
            ) as mock_store,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.get_actual_model_id_from_router",
                return_value="openai",
            ),
        ):
            result = OpenAIPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=datetime.now(),
                request_body={},
                litellm_params={"metadata": {}},
            )

        mock_store.assert_called_once()
        call_kwargs = mock_store.call_args
        assert call_kwargs.kwargs["model_object_id"] == "batch_abc123"
        assert result["kwargs"]["response_cost"] == 0.0
        assert result["kwargs"]["batch_id"] == "batch_abc123"
        assert result["kwargs"]["batch_job_state"] == "in_progress"

    def test_unified_object_id_encodes_model_and_batch_id(self):
        logging_obj = _make_logging_obj()
        httpx_response = _make_batch_httpx_response("batch_xyz")
        url_route = "https://api.openai.com/v1/batches"

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.store_batch_managed_object"
            ) as mock_store,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.get_actual_model_id_from_router",
                return_value="openai",
            ),
        ):
            result = OpenAIPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=datetime.now(),
                request_body={},
                litellm_params={"metadata": {}},
            )

        unified_object_id = result["kwargs"]["unified_object_id"]
        padding = (
            "=" * (4 - len(unified_object_id) % 4) if len(unified_object_id) % 4 else ""
        )
        decoded = base64.urlsafe_b64decode(unified_object_id + padding).decode()
        assert "openai" in decoded
        assert "batch_xyz" in decoded

    def test_non_200_response_returns_zero_cost(self):
        logging_obj = _make_logging_obj()
        mock_resp = Mock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "bad request"}
        mock_request = Mock()
        mock_request.method = "POST"
        mock_resp.request = mock_request

        result = OpenAIPassthroughLoggingHandler.batch_creation_handler(
            httpx_response=mock_resp,
            logging_obj=logging_obj,
            url_route="https://api.openai.com/v1/batches",
            start_time=datetime.now(),
            request_body={},
        )

        assert result["kwargs"]["response_cost"] == 0.0
        assert result["result"] is None

    def test_azure_batch_extracts_deployment_as_model(self):
        logging_obj = _make_logging_obj()
        httpx_response = _make_batch_httpx_response("batch_azure_001")
        url_route = (
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o/batches"
        )

        with (
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.store_batch_managed_object"
            ) as mock_store,
            patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.get_actual_model_id_from_router",
                side_effect=lambda m: m,
            ),
        ):
            result = OpenAIPassthroughLoggingHandler.batch_creation_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                start_time=datetime.now(),
                request_body={},
                litellm_params={"metadata": {}},
            )

        call_kwargs = mock_store.call_args
        unified_object_id = call_kwargs.kwargs["unified_object_id"]
        padding = (
            "=" * (4 - len(unified_object_id) % 4) if len(unified_object_id) % 4 else ""
        )
        decoded = base64.urlsafe_b64decode(unified_object_id + padding).decode()
        assert "azure/gpt-4o" in decoded
        assert "batch_azure_001" in decoded


class TestOpenAIPassthroughHandlerBatchDispatch:
    def test_batch_creation_dispatched_from_main_handler(self):
        logging_obj = _make_logging_obj()
        httpx_response = _make_batch_httpx_response("batch_dispatch_test")
        url_route = "https://api.openai.com/v1/batches"

        with patch.object(
            OpenAIPassthroughLoggingHandler,
            "batch_creation_handler",
            return_value={"result": None, "kwargs": {"response_cost": 0.0}},
        ) as mock_batch_handler:
            OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=httpx_response,
                response_body={},
                logging_obj=logging_obj,
                url_route=url_route,
                result="",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                request_body={},
            )

        mock_batch_handler.assert_called_once()

    def test_non_batch_post_not_dispatched_to_batch_handler(self):
        url_route = "https://api.openai.com/v1/chat/completions"
        assert not OpenAIPassthroughLoggingHandler.is_openai_batch_route(
            url_route, "POST"
        )
