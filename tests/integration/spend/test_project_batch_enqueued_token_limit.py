import json
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from integration._support.process import OwnedState, owned_proxy
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

ROWS_PER_BATCH: Final = 3
WORDS_PER_ROW: Final = 200
ONE_BATCH_BUT_NOT_TWO: Final = 2000

CUSTOM_AUTH_DOUBLE: Final = """
import json
import os

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    if api_key == os.environ["LITELLM_MASTER_KEY"]:
        return UserAPIKeyAuth(api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN)
    callers = json.loads(os.environ["INTEGRATION_CUSTOM_AUTH_CALLERS"])
    if api_key not in callers:
        raise Exception("unknown caller")
    return UserAPIKeyAuth(api_key=api_key, **callers[api_key])
"""


def _custom_auth_config(directory: Path) -> Path:
    (directory / "project_custom_auth.py").write_text(CUSTOM_AUTH_DOUBLE)
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = directory / "project-custom-auth.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "general_settings": {
                    **object_value(base["general_settings"]),
                    "custom_auth": "project_custom_auth.user_api_key_auth",
                    "custom_auth_run_common_checks": True,
                },
            }
        )
    )
    return path


def _input_lines(model_name: str) -> str:
    return "".join(
        json.dumps(
            {
                "custom_id": f"r{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": model_name, "messages": [{"role": "user", "content": "enqueued " * WORDS_PER_ROW}]},
            },
            separators=(",", ":"),
        )
        + "\n"
        for index in range(ROWS_PER_BATCH)
    )


def _batch(status: str) -> dict[str, JsonValue]:
    return {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": status,
        "output_file_id": None,
        "error_file_id": None,
        "created_at": 1,
        "expires_at": 1,
        "request_counts": {"total": ROWS_PER_BATCH, "completed": 0, "failed": 0},
        "metadata": None,
    }


def _batch_provider(scenario: Scenario) -> str:
    model_name: Final = f"integration-{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        f"project-batch-limit-{uuid.uuid4().hex[:12]}",
        RoutedResponse(
            content_type="application/x-routed",
            routes={
                "POST /files": JsonResponse(
                    content_type="application/json",
                    body={
                        "id": "file-in-$REQUEST_ID",
                        "object": "file",
                        "purpose": "batch",
                        "bytes": 100,
                        "created_at": 1,
                        "filename": "in.jsonl",
                        "status": "processed",
                    },
                ),
                "GET /files/file-in-$REQUEST_ID/content": TextResponse(
                    content_type="application/jsonl", body=_input_lines(model_name)
                ),
                "POST /batches": JsonResponse(content_type="application/json", body=_batch("validating")),
                "GET /batches/batch-$REQUEST_ID": JsonResponse(
                    content_type="application/json", body=_batch("cancelled")
                ),
            },
        ),
    )
    scenario.cleanups.callback(delete_scenario, handle)
    scenario.model(name=model_name, api_base=handle.api_base())
    return model_name


def _submit(candidate: Gateway, model_name: str, caller: str) -> httpx.Response:
    uploaded: Final = candidate.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": model_name},
        {"file": ("in.jsonl", _input_lines(model_name).encode(), "application/jsonl")},
        key=caller,
    )
    assert uploaded.status_code == 200, uploaded.text
    return candidate.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model_name,
        },
        key=caller,
    )


@pytest.mark.timeout(480)
def test_custom_auth_callers_share_the_project_per_model_enqueued_limit_until_a_batch_ends(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario:
        limited: Final = _batch_provider(scenario)
        unlimited: Final = _batch_provider(scenario)
        team: Final = scenario.team()
        project: Final = scenario.project(
            team, metadata={"model_batch_enqueued_token_limit": {limited: ONE_BATCH_BUT_NOT_TWO}}
        )
        first_caller: Final = "sk-custom-" + uuid.uuid4().hex
        rotated_caller: Final = "sk-custom-" + uuid.uuid4().hex
        callers: Final = json.dumps(
            {
                first_caller: {"user_id": "project-batch-caller", "team_id": team, "project_id": project},
                rotated_caller: {"user_id": "project-batch-caller", "team_id": team, "project_id": project},
            }
        )
        with owned_proxy(
            gateway,
            tmp_path,
            {"INTEGRATION_CUSTOM_AUTH_CALLERS": callers},
            config=_custom_auth_config(tmp_path),
            state=OwnedState.SHARED_WITH_WORKER,
        ) as candidate:
            first: Final = _submit(candidate, limited, first_caller)
            assert first.status_code == 200, first.text

            refused: Final = _submit(candidate, limited, rotated_caller)
            assert refused.status_code == 429, refused.text
            assert f"project_model: {project}:{limited}" in refused.text, refused.text

            other_model: Final = _submit(candidate, unlimited, rotated_caller)
            assert other_model.status_code == 200, other_model.text

            first_batch: Final = string_value(JSON_OBJECT.validate_json(first.content)["id"])
            ended: Final = candidate.request("GET", f"/v1/batches/{first_batch}", key=first_caller)
            assert ended.status_code == 200, ended.text
            assert JSON_OBJECT.validate_json(ended.content)["status"] == "cancelled", ended.text

            after_refund: Final = _submit(candidate, limited, rotated_caller)
            assert after_refund.status_code == 200, after_refund.text
