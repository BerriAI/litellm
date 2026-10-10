import json
from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_FILE_ID: Final = "file-integration"
_JOB_ID: Final = "ftjob-integration"


def _fine_tuning_response(request: Request) -> Reply:
    if (request.method, request.target) == ("GET", "/v1/models"):
        return Reply(body=b'{"object":"list","data":[]}')
    if (request.method, request.target) == ("POST", "/v1/files"):
        return Reply(
            body=json.dumps(
                {"id": _FILE_ID, "object": "file", "filename": "openai_fine_tuning.jsonl", "purpose": "fine-tune"}
            ).encode()
        )
    if (request.method, request.target) == ("POST", "/v1/fine_tuning/jobs"):
        return Reply(
            body=json.dumps(
                {
                    "id": _JOB_ID,
                    "object": "fine_tuning.job",
                    "model": "gpt-4.1-mini-2025-04-14",
                    "status": "created",
                    "training_file": _FILE_ID,
                }
            ).encode()
        )
    if (request.method, request.target) == ("GET", "/v1/fine_tuning/jobs"):
        return Reply(body=json.dumps({"object": "list", "data": [{"id": _JOB_ID}], "has_more": False}).encode())
    if (request.method, request.target) == ("POST", f"/v1/fine_tuning/jobs/{_JOB_ID}/cancel"):
        return Reply(
            body=json.dumps(
                {
                    "id": _JOB_ID,
                    "object": "fine_tuning.job",
                    "model": "gpt-4.1-mini-2025-04-14",
                    "status": "cancelled",
                    "training_file": _FILE_ID,
                }
            ).encode()
        )
    if (request.method, request.target) == ("DELETE", f"/v1/files/{_FILE_ID}"):
        return Reply(body=json.dumps({"id": _FILE_ID, "object": "file", "deleted": True}).encode())
    return Reply(status=404, body=b'{"error":"unexpected fine-tuning request"}')


def test_openai_fine_tuning_routes_forward_requests(gateway: Gateway) -> None:
    with wire_server(_fine_tuning_response) as wire, gateway.scenario() as scenario:
        scenario.model(
            model="gpt-4.1-mini-2025-04-14",
            custom_llm_provider="openai",
            api_base=f"{wire.url}/v1",
            api_key="fine-tuning-integration-key",
        )
        file_contents: Final = Path(__file__).with_name("openai_fine_tuning.jsonl").read_bytes()
        file_response: Final = gateway.client.post(
            "/v1/files",
            data={"purpose": "fine-tune"},
            files={"file": ("openai_fine_tuning.jsonl", file_contents, "application/jsonl")},
            headers={
                "Authorization": f"Bearer {gateway.key}",
                "custom-llm-provider": "openai",
            },
        )
        assert file_response.status_code == 200, file_response.text
        assert file_response.json()["id"] == _FILE_ID

        job_response: Final = gateway.request(
            "POST",
            "/v1/fine_tuning/jobs",
            {"model": "gpt-4.1-mini-2025-04-14", "training_file": _FILE_ID},
            headers={"custom-llm-provider": "openai"},
        )
        assert job_response.status_code == 200, job_response.text
        assert job_response.json()["id"] == _JOB_ID

        list_response: Final = gateway.request(
            "GET",
            "/v1/fine_tuning/jobs",
            headers={"custom-llm-provider": "openai"},
        )
        assert list_response.status_code == 200, list_response.text
        assert list_response.json()["data"] == [{"id": _JOB_ID}]

        cancel_response: Final = gateway.request(
            "POST",
            f"/v1/fine_tuning/jobs/{_JOB_ID}/cancel",
            headers={"custom-llm-provider": "openai"},
        )
        assert cancel_response.status_code == 200, cancel_response.text
        assert cancel_response.json()["status"] == "cancelled"

        delete_response: Final = gateway.request(
            "DELETE",
            f"/v1/files/{_FILE_ID}",
            headers={"custom-llm-provider": "openai"},
        )
        assert delete_response.status_code == 200, delete_response.text
        assert delete_response.json() == {"id": _FILE_ID, "object": "file", "deleted": True}

        requests: Final = wire.drain()
        upload: Final = next(request for request in requests if request.target == "/v1/files")
        assert b"purpose" in upload.body
        assert b"fine-tune" in upload.body
        assert file_contents in upload.body
        create_job: Final = next(request for request in requests if request.target == "/v1/fine_tuning/jobs")
        assert json.loads(create_job.body) == {
            "model": "gpt-4.1-mini-2025-04-14",
            "training_file": _FILE_ID,
        }
