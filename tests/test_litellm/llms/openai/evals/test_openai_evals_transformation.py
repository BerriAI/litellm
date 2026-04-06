"""
Unit tests for OpenAI Evals API transformation
"""

import httpx
import pytest

from litellm.llms.openai.evals.transformation import OpenAIEvalsConfig
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture()
def config() -> OpenAIEvalsConfig:
    return OpenAIEvalsConfig()


def test_validate_environment_sets_headers(config: OpenAIEvalsConfig):
    """Test that validate_environment correctly sets authorization headers"""
    headers: dict = {}
    params = GenericLiteLLMParams(api_key="sk-test-12345")

    result = config.validate_environment(headers=headers, litellm_params=params)

    assert result["Authorization"] == "Bearer sk-test-12345"
    assert result["Content-Type"] == "application/json"


def test_validate_environment_requires_api_key(config: OpenAIEvalsConfig, monkeypatch):
    """Test that validate_environment raises error when no API key is provided"""
    import os

    # Ensure OPENAI_API_KEY environment variable is None before validation
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    headers: dict = {}
    params = GenericLiteLLMParams()

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        config.validate_environment(headers=headers, litellm_params=params)


def test_get_complete_url_with_eval_id(config: OpenAIEvalsConfig):
    """Test URL construction with eval_id"""
    url = config.get_complete_url(
        api_base="https://api.openai.com",
        endpoint="evals",
        eval_id="eval_123",
    )
    assert url == "https://api.openai.com/v1/evals/eval_123"


def test_get_complete_url_without_eval_id(config: OpenAIEvalsConfig):
    """Test URL construction without eval_id"""
    url = config.get_complete_url(
        api_base="https://api.openai.com",
        endpoint="evals",
    )
    assert url == "https://api.openai.com/v1/evals"


def test_transform_create_eval_request(config: OpenAIEvalsConfig):
    """Test transformation of create eval request"""
    create_request = {
        "name": "Test Eval",
        "data_source_config": {
            "type": "stored_completions",
            "metadata": {"usecase": "chatbot"}
        },
        "testing_criteria": [
            {
                "type": "label_model",
                "model": "gpt-4o",
                "input": [{"role": "user", "content": "Test"}],
                "passing_labels": ["positive"],
                "labels": ["positive", "negative"],
                "name": "Test Grader"
            }
        ],
    }

    result = config.transform_create_eval_request(
        create_request=create_request,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert result["name"] == "Test Eval"
    assert result["data_source_config"]["type"] == "stored_completions"
    assert len(result["testing_criteria"]) == 1
    assert result["testing_criteria"][0]["type"] == "label_model"


def test_transform_create_eval_response(config: OpenAIEvalsConfig):
    """Test transformation of create eval response"""
    response = httpx.Response(
        status_code=200,
        json={
            "id": "eval_123",
            "object": "eval",
            "created_at": 1234567890,
            "name": "Test Eval",
            "data_source_config": {"type": "stored_completions"},
            "testing_criteria": [],
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/evals"),
    )

    result = config.transform_create_eval_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert result.id == "eval_123"
    assert result.object == "eval"
    assert result.name == "Test Eval"


def test_transform_list_evals_request(config: OpenAIEvalsConfig):
    """Test transformation of list evals request"""
    list_params = {
        "limit": 10,
        "after": "eval_123",
        "order": "desc",
    }

    url, query_params = config.transform_list_evals_request(
        list_params=list_params,
        litellm_params=GenericLiteLLMParams(api_base="https://api.openai.com"),
        headers={},
    )

    assert url == "https://api.openai.com/v1/evals"
    assert query_params["limit"] == 10
    assert query_params["after"] == "eval_123"
    assert query_params["order"] == "desc"


def test_transform_list_evals_response(config: OpenAIEvalsConfig):
    """Test transformation of list evals response"""
    response = httpx.Response(
        status_code=200,
        json={
            "object": "list",
            "data": [
                {
                    "id": "eval_123",
                    "object": "eval",
                    "created_at": 1234567890,
                    "name": "Test Eval",
                    "data_source_config": {"type": "stored_completions"},
                    "testing_criteria": [],
                }
            ],
            "first_id": "eval_123",
            "last_id": "eval_123",
            "has_more": False,
        },
        request=httpx.Request("GET", "https://api.openai.com/v1/evals"),
    )

    result = config.transform_list_evals_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert result.object == "list"
    assert len(result.data) == 1
    assert result.data[0].id == "eval_123"
    assert result.has_more is False


def test_transform_update_eval_request(config: OpenAIEvalsConfig):
    """Test transformation of update eval request"""
    update_request = {
        "name": "Updated Eval Name",
        "metadata": {"key": "value"},
    }

    url, headers, request_body = config.transform_update_eval_request(
        eval_id="eval_123",
        update_request=update_request,
        api_base="https://api.openai.com",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://api.openai.com/v1/evals/eval_123"
    assert request_body["name"] == "Updated Eval Name"
    assert request_body["metadata"]["key"] == "value"


def test_transform_delete_eval_request(config: OpenAIEvalsConfig):
    """Test transformation of delete eval request"""
    url, headers = config.transform_delete_eval_request(
        eval_id="eval_123",
        api_base="https://api.openai.com",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://api.openai.com/v1/evals/eval_123"


def test_transform_delete_eval_response(config: OpenAIEvalsConfig):
    """Test transformation of delete eval response"""
    response = httpx.Response(
        status_code=200,
        json={
            "object": "eval.deleted",
            "deleted": True,
            "eval_id": "eval_abc123"
            },
        request=httpx.Request("DELETE", "https://api.openai.com/v1/evals/eval_123"),
    )

    result = config.transform_delete_eval_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert result.eval_id == "eval_abc123"
    assert result.object == "eval.deleted"
    assert result.deleted is True


def test_transform_cancel_eval_request(config: OpenAIEvalsConfig):
    """Test transformation of cancel eval request"""
    url, headers, request_body = config.transform_cancel_eval_request(
        eval_id="eval_123",
        api_base="https://api.openai.com",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://api.openai.com/v1/evals/eval_123/cancel"
    assert request_body == {}


def test_transform_cancel_eval_response(config: OpenAIEvalsConfig):
    """Test transformation of cancel eval response"""
    response = httpx.Response(
        status_code=200,
        json={
            "id": "eval_123",
            "object": "eval",
            "status": "cancelled",
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/evals/eval_123/cancel"),
    )

    result = config.transform_cancel_eval_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert result.id == "eval_123"
    assert result.object == "eval"
