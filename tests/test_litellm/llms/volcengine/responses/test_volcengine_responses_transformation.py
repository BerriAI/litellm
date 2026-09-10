"""
Tests for Volcengine Responses API transformation.
"""

from typing import List, Literal, Optional, Union

import httpx
import pytest
from pydantic import BaseModel, Field


import litellm
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestVolcengineResponsesAPITransformation:
    """Test Volcengine Responses API configuration and transformations."""

    def test_provider_config_registration(self):
        """Provider registry should return VolcEngineResponsesAPIConfig."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="volcengine/demo-model",
            provider=LlmProviders.VOLCENGINE,
        )

        assert config is not None, "Config should not be None for Volcengine provider"
        assert isinstance(config, VolcEngineResponsesAPIConfig), (
            f"Expected VolcEngineResponsesAPIConfig, got {type(config)}"
        )
        assert config.custom_llm_provider == LlmProviders.VOLCENGINE, "custom_llm_provider should be VOLCENGINE"

    def test_parallel_tool_calls_dropped(self):
        """Volcengine does not list parallel_tool_calls; ensure it is removed."""
        config = VolcEngineResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(
            parallel_tool_calls=True,
            temperature=0.5,
            metadata={"k": "v"},
        )

        mapped = config.map_openai_params(
            response_api_optional_params=params,
            model="volcengine/demo-model",
            drop_params=False,
        )

        assert "parallel_tool_calls" not in mapped, "parallel_tool_calls must be dropped"
        assert mapped.get("temperature") == 0.5
        assert "metadata" not in mapped, "Undocumented params should not be included"

    def test_unsupported_params_are_dropped(self):
        """Unknown fields should be dropped before send, including nested extra_body."""
        config = VolcEngineResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "unsupported_custom_param": 0.1,
                "temperature": 0.2,
                "metadata": {"k": "v"},
                "extra_body": {"unsupported_custom_param": 1, "temperature": 0.3},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "unsupported_custom_param" not in request
        assert request["temperature"] == 0.2
        assert "metadata" not in request
        assert "extra_body" in request
        assert "unsupported_custom_param" not in request["extra_body"]
        assert request["extra_body"]["temperature"] == 0.3

    def test_get_complete_url_variants(self):
        """Ensure Volcengine endpoint construction handles different bases."""
        config = VolcEngineResponsesAPIConfig()

        default_url = config.get_complete_url(api_base=None, litellm_params={})
        assert default_url == "https://ark.cn-beijing.volces.com/api/v3/responses"

        api_base_with_api = config.get_complete_url(api_base="https://custom.volc.com/api/v3", litellm_params={})
        assert api_base_with_api == "https://custom.volc.com/api/v3/responses"

        api_base_full = config.get_complete_url(api_base="https://custom.volc.com/api/v3/responses", litellm_params={})
        assert api_base_full == "https://custom.volc.com/api/v3/responses"

    def test_response_id_path_requests_encode_response_id(self):
        """response_id should be encoded before building Volcengine URLs."""
        config = VolcEngineResponsesAPIConfig()

        url, params = config.transform_cancel_response_api_request(
            response_id="../../responses/other?x=1#frag",
            api_base="https://custom.volc.com/api/v3/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://custom.volc.com/api/v3/responses/..%2F..%2Fresponses%2Fother%3Fx%3D1%23frag/cancel"
        assert params == {}

    @pytest.mark.parametrize(
        "litellm_params, expected_key",
        [
            ({"api_key": "dict-key"}, "dict-key"),
            (GenericLiteLLMParams(api_key="attr-key"), "attr-key"),
        ],
    )
    def test_validate_environment_uses_api_key(self, monkeypatch, litellm_params, expected_key):
        """validate_environment should pull api key from params/env and attach headers."""
        config = VolcEngineResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

        headers = config.validate_environment(headers={}, model="volcengine/demo-model", litellm_params=litellm_params)

        assert headers.get("Authorization") == f"Bearer {expected_key}"
        assert headers.get("Content-Type") == "application/json"

    def test_validate_environment_raises_without_key(self, monkeypatch):
        """validate_environment should error when no key is available."""
        config = VolcEngineResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

        with pytest.raises(ValueError, match='Volcengine API key is required\\. Set ARK_API_KEY /'):
            config.validate_environment(headers={}, model="volcengine/demo", litellm_params={})

    def test_unsupported_params_are_dropped_with_extra_body(self):
        """Unknown fields (including extra_body) should be dropped before send."""
        config = VolcEngineResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "unsupported_custom_param": 0.1,
                "temperature": 0.2,
                "metadata": {"k": "v"},
                "extra_body": {"unsupported_custom_param": 1, "temperature": 0.3},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "unsupported_custom_param" not in request
        assert "metadata" not in request
        assert request["temperature"] == 0.2
        assert "extra_body" in request
        assert "unsupported_custom_param" not in request["extra_body"]
        assert request["extra_body"]["temperature"] == 0.3

    def test_valid_thinking_caching_and_expire_at_pass(self):
        """Documented params should pass through without validation errors."""
        config = VolcEngineResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "instructions": "do X",
                "thinking": {"type": "enabled"},
                "caching": {"type": "enabled"},
                "expire_at": 1234567890,
                "temperature": 0.5,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["thinking"]["type"] == "enabled"
        assert request["caching"]["type"] == "enabled"
        assert request["expire_at"] == 1234567890
        assert request["instructions"] == "do X"

    def test_supported_params_limited_to_docs(self):
        """Supported params should match documented Volcengine surface."""
        config = VolcEngineResponsesAPIConfig()
        supported = set(config.get_supported_openai_params("volcengine/demo-model"))

        expected = {
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "store",
            "reasoning",
            "stream",
            "temperature",
            "top_p",
            "text",
            "tools",
            "tool_choice",
            "max_tool_calls",
            "thinking",
            "caching",
            "expire_at",
            "context_management",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }

        assert supported == expected

    def test_error_class_returns_volcengine_error(self):
        """Errors should be wrapped with VolcEngineError for consistent handling."""
        config = VolcEngineResponsesAPIConfig()
        error = config.get_error_class("bad request", 400, headers={"x": "y"})

        # Use class name comparison instead of isinstance to avoid issues with
        # module reloading during parallel test execution (conftest reloads litellm)
        assert type(error).__name__ == "VolcEngineError", f"Expected VolcEngineError, got {type(error).__name__}"
        assert error.status_code == 400
        assert error.message == "bad request"
        assert error.headers.get("x") == "y"

    def test_transform_response_api_response_sets_headers_and_created_at(self):
        """Responses should include processed headers and keep created_at intact."""
        config = VolcEngineResponsesAPIConfig()
        response_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "status": "completed",
            "output": [],
            "model": "demo-model",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        http_response = httpx.Response(
            status_code=200,
            json=response_payload,
            request=httpx.Request("POST", "https://example.com/responses"),
            headers={"x-test": "1"},
        )

        result = config.transform_response_api_response(
            model="volcengine/demo-model",
            raw_response=http_response,
            logging_obj=type(
                "Logger",
                (),
                {"post_call": staticmethod(lambda **kwargs: None)},
            ),
        )

        assert result.created_at == 123
        assert result._hidden_params["headers"].get("x-test") == "1"
        assert "additional_headers" in result._hidden_params

    def test_transform_delete_response_api_response_parses_json(self):
        """DELETE response parsing should return DeleteResponseResult."""
        config = VolcEngineResponsesAPIConfig()
        http_response = httpx.Response(
            status_code=200,
            json={"id": "resp_123", "deleted": True},
            request=httpx.Request("DELETE", "https://example.com/responses/resp_123"),
        )

        result = config.transform_delete_response_api_response(
            raw_response=http_response,
            logging_obj=None,
        )

        assert isinstance(result, DeleteResponseResult)
        assert result.deleted is True

    def test_transform_streaming_response_fills_missing_required_fields(self):
        config = VolcEngineResponsesAPIConfig()

        event = config.transform_streaming_response(
            model="volcengine/demo-model",
            parsed_chunk={"type": "response.completed", "response": {"id": "resp_1"}},
            logging_obj=None,
        )

        assert type(event).__name__ == "ResponseCompletedEvent"
        assert event.type == "response.completed"
        assert event.response.id == "resp_1"
        assert event.response.output == []
        assert event.response.created_at == 0

    def test_transform_response_api_response_falls_back_to_model_construct(self):
        config = VolcEngineResponsesAPIConfig()
        http_response = httpx.Response(
            status_code=200,
            json={"id": "resp_fallback", "created_at": 123, "output": "not-a-list"},
            request=httpx.Request("POST", "https://example.com/responses"),
            headers={"x-test": "1"},
        )

        result = config.transform_response_api_response(
            model="volcengine/demo-model",
            raw_response=http_response,
            logging_obj=type(
                "Logger",
                (),
                {"post_call": staticmethod(lambda **kwargs: None)},
            ),
        )

        assert result.id == "resp_fallback"
        assert result.output == "not-a-list"
        assert result._hidden_params["headers"].get("x-test") == "1"

    def test_transform_delete_response_api_request_builds_url(self):
        config = VolcEngineResponsesAPIConfig()

        url, data = config.transform_delete_response_api_request(
            response_id="resp_123",
            api_base="https://custom.volc.com/api/v3/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://custom.volc.com/api/v3/responses/resp_123"
        assert data == {}

    def test_transform_get_response_api_request_and_response(self):
        config = VolcEngineResponsesAPIConfig()

        url, data = config.transform_get_response_api_request(
            response_id="resp 123",
            api_base="https://custom.volc.com/api/v3/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == "https://custom.volc.com/api/v3/responses/resp%20123"
        assert data == {}

        http_response = httpx.Response(
            status_code=200,
            json={
                "id": "resp_123",
                "object": "response",
                "created_at": 123,
                "status": "completed",
                "output": [],
                "model": "demo-model",
            },
            request=httpx.Request("GET", url),
            headers={"x-test": "1"},
        )

        result = config.transform_get_response_api_response(
            raw_response=http_response,
            logging_obj=None,
        )

        assert result.id == "resp_123"
        assert result._hidden_params["headers"].get("x-test") == "1"

    def test_transform_cancel_response_api_response_parses_json(self):
        config = VolcEngineResponsesAPIConfig()
        http_response = httpx.Response(
            status_code=200,
            json={
                "id": "resp_123",
                "object": "response",
                "created_at": 123,
                "status": "cancelled",
                "output": [],
                "model": "demo-model",
            },
            request=httpx.Request("POST", "https://example.com/responses/resp_123/cancel"),
            headers={"x-test": "1"},
        )

        result = config.transform_cancel_response_api_response(
            raw_response=http_response,
            logging_obj=None,
        )

        assert result.id == "resp_123"
        assert result.status == "cancelled"
        assert result._hidden_params["headers"].get("x-test") == "1"

    def test_transform_list_input_items_request_builds_query_params(self):
        config = VolcEngineResponsesAPIConfig()

        url, params = config.transform_list_input_items_request(
            response_id="resp_123",
            api_base="https://custom.volc.com/api/v3/responses",
            litellm_params=GenericLiteLLMParams(),
            headers={},
            after="item_a",
            before="item_b",
            include=["metadata", "usage"],
            limit=5,
            order="asc",
        )

        assert url == "https://custom.volc.com/api/v3/responses/resp_123/input_items"
        assert params == {
            "after": "item_a",
            "before": "item_b",
            "include": "metadata,usage",
            "limit": 5,
            "order": "asc",
        }

    def test_transform_list_input_items_response_returns_parsed_body(self):
        config = VolcEngineResponsesAPIConfig()
        payload = {"object": "list", "data": [{"id": "item_1"}]}
        http_response = httpx.Response(
            status_code=200,
            json=payload,
            request=httpx.Request("GET", "https://example.com/responses/resp_123/input_items"),
        )

        result = config.transform_list_input_items_response(
            raw_response=http_response,
            logging_obj=None,
        )

        assert result == payload


class _FillWidget(BaseModel):
    type: Literal["widget"]
    count: int
    parts: List[str]
    label: Optional[str]


class _FillGadget(BaseModel):
    type: Literal["gadget"]
    name: str


class _FillEnvelope(BaseModel):
    kind: str = "envelope"
    tags: List[str] = Field(default_factory=lambda: ["default-tag"])
    payload: Union[_FillWidget, _FillGadget]
    entries: List[_FillWidget]
    note: Optional[str]
    values: Union[List[str], str]


class TestVolcengineStreamingFieldFill:
    def test_fill_uses_defaults_factories_and_heuristics(self):
        filled = VolcEngineResponsesAPIConfig._fill_missing_fields(
            {"payload": {"type": "gadget", "name": "g"}, "entries": [{"type": "widget"}]},
            _FillEnvelope,
        )

        assert filled["kind"] == "envelope"
        assert filled["tags"] == ["default-tag"]
        assert filled["note"] is None
        assert filled["values"] == []

        validated = _FillEnvelope.model_validate(filled)
        assert isinstance(validated.payload, _FillGadget)
        assert validated.entries[0].count == 0
        assert validated.entries[0].parts == []
        assert validated.entries[0].label is None

    def test_fill_selects_union_member_by_type_literal(self):
        filled = VolcEngineResponsesAPIConfig._fill_missing_fields(
            {"payload": {"type": "widget"}, "entries": []},
            _FillEnvelope,
        )

        validated = _FillEnvelope.model_validate(filled)
        assert isinstance(validated.payload, _FillWidget)
        assert validated.payload.count == 0
        assert validated.payload.parts == []
        assert validated.payload.label is None


class _Pep604Envelope(BaseModel):
    payload: _FillWidget | _FillGadget
    note: str | None
    values: list[str] | str


class TestVolcenginePep604FieldFill:
    def test_fill_handles_pep604_union_spellings(self):
        filled = VolcEngineResponsesAPIConfig._fill_missing_fields(
            {"payload": {"type": "widget"}},
            _Pep604Envelope,
        )

        assert filled["note"] is None
        assert filled["values"] == []

        validated = _Pep604Envelope.model_validate(filled)
        assert isinstance(validated.payload, _FillWidget)
        assert validated.payload.count == 0
        assert validated.payload.parts == []
        assert validated.payload.label is None
