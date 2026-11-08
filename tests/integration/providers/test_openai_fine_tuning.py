import json
from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_FILE_ID: Final = "file-integration"
_JOB_ID: Final = "ftjob-integration"


def _job(status: str) -> dict[str, object]:
    return {
        "id": _JOB_ID,
        "object": "fine_tuning.job",
        "created_at": 1721764800,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": "auto", "batch_size": "auto", "learning_rate_multiplier": "auto"},
        "model": "gpt-4.1-mini-2025-04-14",
        "organization_id": "org-integration",
        "result_files": [],
        "seed": 42,
        "status": status,
        "trained_tokens": None,
        "training_file": _FILE_ID,
        "validation_file": None,
        "estimated_finish": None,
        "integrations": [],
    }


def _file(deleted: bool = False) -> dict[str, object]:
    if deleted:
        return {"id": _FILE_ID, "object": "file", "deleted": True}
    return {
        "id": _FILE_ID,
        "object": "file",
        "bytes": 1024,
        "created_at": 1721764800,
        "filename": "openai_fine_tuning.jsonl",
        "purpose": "fine-tune",
        "status": "processed",
        "status_details": None,
    }


def _fine_tuning_response(request: Request) -> Reply:
    routes: Final = {
        ("POST", "/v1/files"): _file(),
        ("POST", "/v1/fine_tuning/jobs"): _job("validating_files"),
        ("GET", "/v1/fine_tuning/jobs"): {"object": "list", "data": [_job("validating_files")], "has_more": False},
        ("POST", f"/v1/fine_tuning/jobs/{_JOB_ID}/cancel"): _job("cancelled"),
        ("DELETE", f"/v1/files/{_FILE_ID}"): _file(deleted=True),
    }
    body: Final = routes.get((request.method, request.target.split("?", 1)[0]))
    if body is None:
        return Reply(status=404, body=b'{"error":{"message":"unexpected fine-tuning request"}}')
    return Reply(body=json.dumps(body).encode())


def test_openai_fine_tuning_routes_forward_requests(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = Path(__file__).with_name("fine_tuning_proxy_config.yaml")
    with wire_server(_fine_tuning_response) as wire, owned_proxy(
        gateway, tmp_path, {"FINE_TUNING_UPSTREAM_URL": f"{wire.url}/v1"}, config=config
    ) as proxy:
        file_contents: Final = Path(__file__).with_name("openai_fine_tuning.jsonl").read_bytes()
        file_response: Final = proxy.client.post(
            "/v1/files",
            data={"purpose": "fine-tune"},
            files={"file": ("openai_fine_tuning.jsonl", file_contents, "application/jsonl")},
            headers={
                "Authorization": f"Bearer {proxy.key}",
                "custom-llm-provider": "openai",
            },
        )
        assert file_response.status_code == 200, file_response.text
        assert file_response.json()["id"] == _FILE_ID

        job_response: Final = proxy.request(
            "POST",
            "/v1/fine_tuning/jobs",
            {"model": "gpt-4.1-mini-2025-04-14", "training_file": _FILE_ID, "custom_llm_provider": "openai"},
        )
        assert job_response.status_code == 200, job_response.text
        assert job_response.json()["id"] == _JOB_ID

        list_response: Final = proxy.request(
            "GET",
            "/v1/fine_tuning/jobs",
            params={"custom_llm_provider": "openai"},
        )
        assert list_response.status_code == 200, list_response.text
        assert [job["id"] for job in list_response.json()["data"]] == [_JOB_ID]

        cancel_response: Final = proxy.request(
            "POST",
            f"/v1/fine_tuning/jobs/{_JOB_ID}/cancel",
            {"custom_llm_provider": "openai"},
        )
        assert cancel_response.status_code == 200, cancel_response.text
        assert cancel_response.json()["status"] == "cancelled"

        delete_response: Final = proxy.request(
            "DELETE",
            f"/v1/files/{_FILE_ID}",
            headers={"custom-llm-provider": "openai"},
        )
        assert delete_response.status_code == 200, delete_response.text
        assert delete_response.json() == {"id": _FILE_ID, "object": "file", "deleted": True}

        requests: Final = tuple(request for request in wire.drain() if request.target != "/v1/models")

    assert [(request.method, request.target.split("?", 1)[0]) for request in requests] == [
        ("POST", "/v1/files"),
        ("POST", "/v1/fine_tuning/jobs"),
        ("GET", "/v1/fine_tuning/jobs"),
        ("POST", f"/v1/fine_tuning/jobs/{_JOB_ID}/cancel"),
        ("DELETE", f"/v1/files/{_FILE_ID}"),
    ]
    assert b'name="purpose"\r\n\r\nfine-tune' in requests[0].body
    assert file_contents in requests[0].body
    assert json.loads(requests[1].body) == {
        "model": "gpt-4.1-mini-2025-04-14",
        "training_file": _FILE_ID,
        "hyperparameters": {},
    }
    assert {request.headers["authorization"] for request in requests} == {"Bearer synthetic-fine-tuning-key"}
