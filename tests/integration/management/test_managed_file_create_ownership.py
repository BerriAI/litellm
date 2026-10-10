from __future__ import annotations

import base64
import json
import uuid
from dataclasses import dataclass
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, string_value
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse
from pydantic import JsonValue

_FILE: Final = {
    "id": "file-in-$REQUEST_ID",
    "object": "file",
    "purpose": "batch",
    "bytes": 100,
    "created_at": 1,
    "filename": "input.jsonl",
    "status": "processed",
}
_BATCH: Final = {
    "id": "batch-$REQUEST_ID",
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "errors": None,
    "input_file_id": "file-in-$REQUEST_ID",
    "completion_window": "24h",
    "status": "validating",
    "output_file_id": None,
    "error_file_id": None,
    "created_at": 1,
    "expires_at": 1,
    "request_counts": {"total": 1, "completed": 0, "failed": 0},
    "metadata": None,
}
_JOB: Final = {
    "id": "ftjob-$REQUEST_ID",
    "object": "fine_tuning.job",
    "model": "gpt-4o-mini-2024-07-18",
    "created_at": 1,
    "fine_tuned_model": None,
    "organization_id": "org-integration",
    "result_files": [],
    "status": "validating_files",
    "validation_file": None,
    "training_file": "file-in-$REQUEST_ID",
    "hyperparameters": {"n_epochs": "auto"},
    "seed": 1,
}


@dataclass(frozen=True, slots=True)
class _Tenants:
    model: str
    handle: ScenarioHandle
    owner_key: str
    teammate_key: str
    outsider_key: str
    owner_file_id: str


def _routes() -> RoutedResponse:
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /files": JsonResponse(content_type="application/json", body=_FILE),
            "POST /batches": JsonResponse(content_type="application/json", body=_BATCH),
            "POST /fine_tuning/jobs": JsonResponse(content_type="application/json", body=_JOB),
        },
    )


def _upload(scenario: Scenario, key: str, model: str, purpose: str) -> str:
    line: Final = json.dumps(
        {
            "custom_id": "req-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": "ping"}]},
        }
    ).encode()
    uploaded: Final = scenario.gateway.request_multipart(
        "/v1/files",
        {"purpose": purpose, "target_model_names": model},
        {"file": ("input.jsonl", line + b"\n", "application/jsonl")},
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    return string_value(JSON_OBJECT.validate_json(uploaded.content)["id"])


def _tenants(scenario: Scenario, purpose: str) -> _Tenants:
    handle: Final = register_scenario(f"managed-file-create-ownership-{uuid.uuid4().hex}", _routes())
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(api_base=handle.api_base())
    team_a: Final = scenario.team(models=[model])
    team_b: Final = scenario.team(models=[model])
    owner_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), team_id=team_a, models=[model])
    teammate_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), team_id=team_a, models=[model])
    outsider_key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"), team_id=team_b, models=[model])
    return _Tenants(model, handle, owner_key, teammate_key, outsider_key, _upload(scenario, owner_key, model, purpose))


def _upstream_creates(gateway: Gateway, handle: ScenarioHandle, path: str) -> int:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    assert response.status_code == 200, response.text
    requests: Final = JSON_OBJECT.validate_json(response.content)["requests"]
    assert isinstance(requests, list), response.text
    return sum(
        1
        for request in requests
        if isinstance(request, dict)
        and request.get("method") == "POST"
        and handle.scenario_id in str(request.get("path"))
        and str(request.get("path")).endswith(path)
    )


def _create_batch(gateway: Gateway, key: str, model: str, input_file_id: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )


def _create_job(gateway: Gateway, key: str, model: str, files: dict[str, JsonValue]) -> httpx.Response:
    return gateway.request("POST", "/v1/fine_tuning/jobs", {"model": model, **files}, key=key)


def test_batch_create_from_another_teams_managed_file_is_403_and_never_reaches_the_provider(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tenants: Final = _tenants(scenario, "batch")
        denied: Final = _create_batch(gateway, tenants.outsider_key, tenants.model, tenants.owner_file_id)
        assert denied.status_code == 403, denied.text
        assert tenants.owner_file_id in denied.text, denied.text
        assert _upstream_creates(gateway, tenants.handle, "/batches") == 0


def test_batch_create_from_own_or_teammates_managed_file_still_dispatches(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tenants: Final = _tenants(scenario, "batch")
        owner: Final = _create_batch(gateway, tenants.owner_key, tenants.model, tenants.owner_file_id)
        teammate: Final = _create_batch(gateway, tenants.teammate_key, tenants.model, tenants.owner_file_id)
        admin: Final = _create_batch(gateway, gateway.key, tenants.model, tenants.owner_file_id)
        assert (owner.status_code, teammate.status_code, admin.status_code) == (200, 200, 200), (
            owner.text,
            teammate.text,
            admin.text,
        )
        assert JSON_OBJECT.validate_json(owner.content)["input_file_id"] == tenants.owner_file_id, owner.text
        assert _upstream_creates(gateway, tenants.handle, "/batches") == 3


def test_batch_create_from_a_managed_file_id_with_no_owner_row_is_404(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tenants: Final = _tenants(scenario, "batch")
        forged_raw: Final = (
            f"litellm_proxy:application/octet-stream;unified_id,{uuid.uuid4()};target_model_names,{tenants.model};"
            "llm_output_file_id,file-forged;llm_output_file_model_id,model-forged"
        )
        forged: Final = base64.urlsafe_b64encode(forged_raw.encode()).decode().rstrip("=")
        missing: Final = _create_batch(gateway, tenants.outsider_key, tenants.model, forged)
        assert missing.status_code == 404, missing.text
        assert _upstream_creates(gateway, tenants.handle, "/batches") == 0


@pytest.mark.parametrize("field", ["training_file", "validation_file"])
def test_fine_tuning_create_from_another_teams_managed_file_is_403(gateway: Gateway, field: str) -> None:
    with gateway.scenario() as scenario:
        tenants: Final = _tenants(scenario, "fine-tune")
        own_training: Final = _upload(scenario, tenants.outsider_key, tenants.model, "fine-tune")
        files: Final[dict[str, JsonValue]] = {"training_file": own_training, field: tenants.owner_file_id}
        denied: Final = _create_job(gateway, tenants.outsider_key, tenants.model, files)
        assert denied.status_code == 403, denied.text
        assert _upstream_creates(gateway, tenants.handle, "/fine_tuning/jobs") == 0


def test_fine_tuning_create_from_own_managed_files_still_dispatches(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tenants: Final = _tenants(scenario, "fine-tune")
        validation: Final = _upload(scenario, tenants.teammate_key, tenants.model, "fine-tune")
        created: Final = _create_job(
            gateway,
            tenants.owner_key,
            tenants.model,
            {"training_file": tenants.owner_file_id, "validation_file": validation},
        )
        assert created.status_code == 200, created.text
        assert _upstream_creates(gateway, tenants.handle, "/fine_tuning/jobs") == 1
