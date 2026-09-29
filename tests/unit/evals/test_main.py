from typing import Literal, TypeAlias, assert_never

import httpx
import pytest
from pydantic import BaseModel

from litellm.evals.main import (
    cancel_eval,
    cancel_run,
    create_eval,
    create_run,
    delete_eval,
    delete_run,
    get_eval,
    get_run,
    list_evals,
    list_runs,
    update_eval,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler

EvalOperation: TypeAlias = Literal[
    "create_eval",
    "list_evals",
    "get_eval",
    "update_eval",
    "delete_eval",
    "cancel_eval",
    "create_run",
    "list_runs",
    "get_run",
    "cancel_run",
    "delete_run",
]


def _call_operation(operation: EvalOperation, http_client: HTTPHandler) -> object:
    match operation:
        case "create_eval":
            return create_eval(
                data_source_config={"type": "stored_completions"},
                testing_criteria=[],
                name="unit eval",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "list_evals":
            return list_evals(
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "get_eval":
            return get_eval(
                eval_id="eval_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "update_eval":
            return update_eval(
                eval_id="eval_123",
                name="unit eval updated",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "delete_eval":
            return delete_eval(
                eval_id="eval_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "cancel_eval":
            return cancel_eval(
                eval_id="eval_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "create_run":
            return create_run(
                eval_id="eval_123",
                data_source={"type": "dataset", "dataset_id": "dataset_123"},
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "list_runs":
            return list_runs(
                eval_id="eval_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "get_run":
            return get_run(
                eval_id="eval_123",
                run_id="run_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "cancel_run":
            return cancel_run(
                eval_id="eval_123",
                run_id="run_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
        case "delete_run":
            return delete_run(
                eval_id="eval_123",
                run_id="run_123",
                custom_llm_provider="openai",
                api_key="sk-unit-test",
                api_base="https://api.openai.com",
                client=http_client,
            )
    assert_never(operation)


@pytest.mark.parametrize(
    ("operation", "request_method", "request_path", "response_json"),
    [
        pytest.param(
            "create_eval",
            "POST",
            "/v1/evals",
            {
                "id": "eval_123",
                "object": "eval",
                "created_at": 1,
                "name": "unit eval",
                "data_source_config": {"type": "stored_completions"},
                "testing_criteria": [],
            },
            id="create-eval",
        ),
        pytest.param(
            "list_evals",
            "GET",
            "/v1/evals",
            {"object": "list", "data": []},
            id="list-evals",
        ),
        pytest.param(
            "get_eval",
            "GET",
            "/v1/evals/eval_123",
            {
                "id": "eval_123",
                "object": "eval",
                "created_at": 1,
                "name": "unit eval",
                "data_source_config": {"type": "stored_completions"},
                "testing_criteria": [],
            },
            id="get-eval",
        ),
        pytest.param(
            "update_eval",
            "POST",
            "/v1/evals/eval_123",
            {
                "id": "eval_123",
                "object": "eval",
                "created_at": 1,
                "name": "unit eval updated",
                "data_source_config": {"type": "stored_completions"},
                "testing_criteria": [],
            },
            id="update-eval",
        ),
        pytest.param(
            "delete_eval",
            "DELETE",
            "/v1/evals/eval_123",
            {"eval_id": "eval_123", "object": "eval.deleted", "deleted": True},
            id="delete-eval",
        ),
        pytest.param(
            "cancel_eval",
            "POST",
            "/v1/evals/eval_123/cancel",
            {"id": "eval_123", "object": "eval", "status": "cancelled"},
            id="cancel-eval",
        ),
        pytest.param(
            "create_run",
            "POST",
            "/v1/evals/eval_123/runs",
            {
                "id": "run_123",
                "object": "eval.run",
                "created_at": 1,
                "status": "queued",
                "data_source": {"type": "dataset", "dataset_id": "dataset_123"},
                "eval_id": "eval_123",
            },
            id="create-run",
        ),
        pytest.param(
            "list_runs",
            "GET",
            "/v1/evals/eval_123/runs",
            {"object": "list", "data": []},
            id="list-runs",
        ),
        pytest.param(
            "get_run",
            "GET",
            "/v1/evals/eval_123/runs/run_123",
            {
                "id": "run_123",
                "object": "eval.run",
                "created_at": 1,
                "status": "queued",
                "data_source": {"type": "dataset", "dataset_id": "dataset_123"},
                "eval_id": "eval_123",
            },
            id="get-run",
        ),
        pytest.param(
            "cancel_run",
            "POST",
            "/v1/evals/eval_123/runs/run_123/cancel",
            {"id": "run_123", "object": "eval.run", "status": "cancelled"},
            id="cancel-run",
        ),
        pytest.param(
            "delete_run",
            "DELETE",
            "/v1/evals/eval_123/runs/run_123",
            {"run_id": "run_123", "object": "eval.run.deleted", "deleted": True},
            id="delete-run",
        ),
    ],
)
def test_sync_eval_operations_return_parsed_provider_responses(
    operation: EvalOperation,
    request_method: str,
    request_path: str,
    response_json: dict[str, object],
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == request_method
        assert request.url.path == request_path
        return httpx.Response(status_code=200, json=response_json, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        http_client = HTTPHandler(client=client)
        result = _call_operation(operation, http_client)

    assert isinstance(result, BaseModel)
    assert result.model_dump(exclude_unset=True) == response_json
