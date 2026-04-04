"""
Unit tests for OpenAI Evals API transformation
"""

import httpx
import pytest

from litellm.llms.openai.evals.transformation import OpenAIEvalsConfig
from litellm.types.llms.openai_evals import ListRunsResponse, Run
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


def test_transform_update_eval_request_converts_non_string_metadata(
    config: OpenAIEvalsConfig,
):
    """Test that non-string metadata values (e.g. queue_time_seconds float) are
    converted to strings before forwarding to OpenAI, which requires all metadata
    values to be strings.  Regression test for #23329 Bug 1."""
    update_request = {
        "name": "Test Eval",
        "metadata": {
            "env": "production",
            "queue_time_seconds": 0.042,
            "retry_count": 3,
        },
    }

    _, _, request_body = config.transform_update_eval_request(
        eval_id="eval_456",
        update_request=update_request,
        api_base="https://api.openai.com",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    # All metadata values must be strings
    for k, v in request_body["metadata"].items():
        assert isinstance(v, str), f"metadata[{k!r}] should be str, got {type(v)}"
    # String values should be preserved as-is
    assert request_body["metadata"]["env"] == "production"
    # Float/int values should be stringified
    assert request_body["metadata"]["queue_time_seconds"] == "0.042"
    assert request_body["metadata"]["retry_count"] == "3"


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


def test_list_runs_response_with_missing_criteria_fields(config: OpenAIEvalsConfig):
    """Test that ListRunsResponse accepts per_testing_criteria_results items
    that are missing testing_criteria_index and result_counts, matching what
    OpenAI actually returns.  Regression test for #23329 Bug 2."""
    response = httpx.Response(
        status_code=200,
        json={
            "object": "list",
            "data": [
                {
                    "id": "evalrun_abc",
                    "object": "eval.run",
                    "created_at": 1234567890,
                    "status": "completed",
                    "data_source": {"type": "jsonl"},
                    "eval_id": "eval_123",
                    "per_testing_criteria_results": [
                        {
                            "failed": 1,
                            "passed": 1,
                            "sample_id": "550e8400-e29b-41d4-a716-446655440000",
                        }
                    ],
                    "result_counts": {"passed": 1, "failed": 1, "total": 2},
                }
            ],
            "first_id": "evalrun_abc",
            "last_id": "evalrun_abc",
            "has_more": False,
        },
        request=httpx.Request("GET", "https://api.openai.com/v1/evals/eval_123/runs"),
    )

    result = config.transform_list_runs_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert len(result.data) == 1
    run = result.data[0]
    assert run.id == "evalrun_abc"
    assert run.per_testing_criteria_results is not None
    assert len(run.per_testing_criteria_results) == 1
    # Fields should default to None when missing from OpenAI response
    assert run.per_testing_criteria_results[0].testing_criteria_index is None
    assert run.per_testing_criteria_results[0].result_counts is None


def test_get_run_response_with_missing_criteria_fields(config: OpenAIEvalsConfig):
    """Test that Run model accepts per_testing_criteria_results items without
    testing_criteria_index and result_counts.  Regression test for #23329 Bug 3."""
    response = httpx.Response(
        status_code=200,
        json={
            "id": "evalrun_xyz",
            "object": "eval.run",
            "created_at": 1234567890,
            "status": "completed",
            "data_source": {"type": "jsonl"},
            "eval_id": "eval_456",
            "per_testing_criteria_results": [
                {
                    "failed": 0,
                    "passed": 2,
                    "sample_id": "660e8400-e29b-41d4-a716-446655440000",
                }
            ],
            "result_counts": {"passed": 2, "failed": 0, "total": 2},
        },
        request=httpx.Request(
            "GET", "https://api.openai.com/v1/evals/eval_456/runs/evalrun_xyz"
        ),
    )

    result = config.transform_get_run_response(
        raw_response=response,
        logging_obj=None,  # type: ignore
    )

    assert result.id == "evalrun_xyz"
    assert result.status == "completed"
    assert result.per_testing_criteria_results is not None
    assert result.per_testing_criteria_results[0].testing_criteria_index is None
    assert result.per_testing_criteria_results[0].result_counts is None
