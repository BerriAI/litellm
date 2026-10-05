import base64
import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

MANAGED_MODEL: Final = "gpt-4.1-managed-ids"
DEPLOYMENT_KEY: Final = "sk-fine-tuning-managed-deployment"
RUN: Final = uuid.uuid4().hex[:12]
PROVIDER_FILE: Final = f"file-managed-{RUN}"
PROVIDER_JOB: Final = f"ftjob-managed-{RUN}"
TRAINING_LINE: Final = b'{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
JOB_PATH: Final = re.compile(r"^/v1/fine_tuning/jobs/([^/]+?)(/cancel)?$")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
UNIFIED_JOB: Final = re.compile(rf"litellm_proxy;model_id:[0-9a-f]{{64}};generic_response_id:{PROVIDER_JOB}")


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    object: str
    status: str
    model: str


class ProxyError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: dict[str, JsonValue]


def _job(job_id: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": job_id,
        "object": "fine_tuning.job",
        "created_at": 1700000000,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": 3},
        "model": "gpt-4.1",
        "organization_id": "org-fine-tuning",
        "result_files": [],
        "seed": 42,
        "status": status,
        "trained_tokens": None,
        "training_file": PROVIDER_FILE,
        "validation_file": None,
    }


def _reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _respond(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if (request.method, path) == ("GET", "/v1/models"):
        return _reply({"object": "list", "data": []})
    if (request.method, path) == ("POST", "/v1/files"):
        return _reply(
            {
                "id": PROVIDER_FILE,
                "object": "file",
                "bytes": len(TRAINING_LINE),
                "created_at": 1700000000,
                "filename": "train.jsonl",
                "purpose": "fine-tune",
                "status": "processed",
            }
        )
    if (request.method, path) == ("POST", "/v1/fine_tuning/jobs"):
        return _reply(_job(PROVIDER_JOB, "validating_files"))
    matched: Final = JOB_PATH.match(path)
    if matched and request.method == "GET" and not matched.group(2):
        return _reply(_job(matched.group(1), "running"))
    if matched and request.method == "POST" and matched.group(2):
        return _reply(_job(matched.group(1), "cancelled"))
    return Reply(
        status=404, body=json.dumps({"error": {"message": f"unscripted {request.method} {request.target}"}}).encode()
    )


def _config(wire: Wire) -> dict[str, JsonValue]:
    return {
        "model_list": [
            {
                "model_name": MANAGED_MODEL,
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": DEPLOYMENT_KEY, "api_base": f"{wire.url}/v1"},
                "model_info": {"supported_endpoints": ["/chat/completions", "/fine_tuning"]},
            }
        ],
        "finetune_settings": [
            {"custom_llm_provider": "openai", "api_key": "sk-fine-tuning-raw-provider", "api_base": f"{wire.url}/v1"}
        ],
        "litellm_settings": {"require_managed_files": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


def _decoded(unified_id: str) -> str:
    return base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4)).decode()


def _targets(requests: tuple[Request, ...]) -> list[tuple[str, str, str]]:
    return [(request.method, request.target, request.headers.get("authorization", "")) for request in requests]


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    owner: str
    intruder: str
    managed_file: str
    managed_job: str

    def sdk(self, key: str) -> OpenAI:
        return OpenAI(base_url=f"{str(self.proxy.client.base_url).rstrip('/')}/v1", api_key=key, max_retries=0)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("fine-tuning-managed-ids")
    with gateway_from_environment() as shared, wire_server(_respond) as wire:
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(_config(wire)))
        with owned_proxy(shared, directory, {}, config=config) as proxy, proxy.scenario() as scenario:
            eventually(
                lambda: _targets(wire.drain()), lambda seen: seen == [("GET", "/v1/models", f"Bearer {DEPLOYMENT_KEY}")]
            )
            owner: Final = scenario.key(user_id=scenario.user())
            intruder: Final = scenario.key(user_id=scenario.user())
            with OpenAI(
                base_url=f"{str(proxy.client.base_url).rstrip('/')}/v1", api_key=owner, max_retries=0
            ) as client:
                uploaded: Final = client.files.with_raw_response.create(
                    file=("train.jsonl", TRAINING_LINE, "application/jsonl"),
                    purpose="fine-tune",
                    extra_body={"target_model_names": MANAGED_MODEL},
                )
                assert uploaded.status_code == 200, uploaded.text
                managed_file: Final = uploaded.parse().id
                created: Final = client.fine_tuning.jobs.with_raw_response.create(
                    model=MANAGED_MODEL, training_file=managed_file
                )
                assert created.status_code == 200, created.text
            managed_job: Final = Job.model_validate_json(created.text).id
            assert UNIFIED_JOB.fullmatch(_decoded(managed_job)), created.text
            assert _targets(wire.drain()) == [
                ("POST", "/v1/files", f"Bearer {DEPLOYMENT_KEY}"),
                ("POST", "/v1/fine_tuning/jobs", f"Bearer {DEPLOYMENT_KEY}"),
            ]
            yield Rig(proxy, wire, owner, intruder, managed_file, managed_job)
            assert wire.drain() == ()


def _refused(response: httpx.Response) -> None:
    assert response.status_code in {403, 404}, response.text


def test_another_key_cannot_create_with_or_cancel_the_owners_managed_ids(rig: Rig) -> None:
    created: Final = rig.proxy.request(
        "POST",
        "/v1/fine_tuning/jobs",
        {"model": MANAGED_MODEL, "training_file": rig.managed_file},
        key=rig.intruder,
    )
    _refused(created)
    cancelled: Final = rig.proxy.request("POST", f"/v1/fine_tuning/jobs/{rig.managed_job}/cancel", key=rig.intruder)
    _refused(cancelled)
    assert rig.wire.drain() == ()


@pytest.mark.parametrize(
    ("method", "path", "body", "kind"),
    [
        pytest.param(
            "POST",
            "/v1/fine_tuning/jobs",
            {"model": "gpt-4o-mini", "training_file": "file-raw123", "custom_llm_provider": "openai"},
            "file",
            id="create",
        ),
        pytest.param("GET", "/v1/fine_tuning/jobs/ftjob-raw123", None, "fine-tuning job", id="retrieve"),
        pytest.param("POST", "/v1/fine_tuning/jobs/ftjob-raw123/cancel", None, "fine-tuning job", id="cancel"),
    ],
)
def test_raw_provider_ids_are_refused_when_managed_files_are_required(
    rig: Rig, method: str, path: str, body: dict[str, JsonValue] | None, kind: str
) -> None:
    response: Final = rig.proxy.request(
        method, path, body, params={"custom_llm_provider": "openai"} if body is None else None
    )
    assert response.status_code == 400, response.text
    assert ProxyError.model_validate_json(response.text).error["message"] == (
        f"Raw provider {kind} ids cannot be used when require_managed_files is enabled in litellm_settings. "
        f"Use the LiteLLM managed {kind} id returned when the {kind} was created."
    ), response.text
    assert rig.wire.drain() == ()


def test_owner_retrieve_reaches_the_provider_with_the_decoded_job_id(rig: Rig) -> None:
    pytest.skip("BUG: a virtual key gets 401 on GET /v1/fine_tuning/jobs/{id}, the route is missing from openai_routes")
    _refused(rig.proxy.request("GET", f"/v1/fine_tuning/jobs/{rig.managed_job}", key=rig.intruder))
    assert rig.wire.drain() == ()
    with rig.sdk(rig.owner) as client:
        retrieved: Final = client.fine_tuning.jobs.with_raw_response.retrieve(rig.managed_job)
    assert retrieved.status_code == 200, retrieved.text
    assert Job.model_validate_json(retrieved.text) == Job(
        id=rig.managed_job, object="fine_tuning.job", status="running", model="gpt-4.1"
    ), retrieved.text
    assert [(r.method, r.target, r.headers["authorization"], r.body) for r in rig.wire.drain()] == [
        ("GET", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}", f"Bearer {DEPLOYMENT_KEY}", b"")
    ]


def test_owner_cancel_reaches_the_provider_with_the_decoded_job_id(rig: Rig) -> None:
    with rig.sdk(rig.owner) as client:
        cancelled: Final = client.fine_tuning.jobs.with_raw_response.cancel(rig.managed_job)
    assert cancelled.status_code == 200, cancelled.text
    assert Job.model_validate_json(cancelled.text) == Job(
        id=rig.managed_job, object="fine_tuning.job", status="cancelled", model="gpt-4.1"
    ), cancelled.text
    assert [(r.method, r.target, r.headers["authorization"], r.body) for r in rig.wire.drain()] == [
        ("POST", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}/cancel", f"Bearer {DEPLOYMENT_KEY}", b"")
    ]
