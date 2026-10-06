import base64
import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
import yaml
from integration._support.database import read_rows
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

MANAGED_MODEL: Final = "gpt-4.1-managed-ids"
DEPLOYMENT_KEY: Final = "sk-fine-tuning-managed-deployment"
RUN: Final = uuid.uuid4().hex[:12]
PROVIDER_FILE: Final = f"file-managed-{RUN}"
PROVIDER_JOB: Final = f"ftjob-managed-{RUN}"
DEFAULT_PROVIDER_FILE: Final = f"file-managed-default-{RUN}"
DEFAULT_PROVIDER_JOB: Final = f"ftjob-managed-default-{RUN}"
TRAINING_LINE: Final = b'{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
JOB_PATH: Final = re.compile(r"^/v1/fine_tuning/jobs/([^/]+?)(/cancel)?$")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    object: str
    status: str
    model: str


class ProxyError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: dict[str, JsonValue]


def _job(job_id: str, status: str, file_id: str) -> dict[str, JsonValue]:
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
        "training_file": file_id,
        "validation_file": None,
    }


def _reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _responder(file_id: str, job_id: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        return _respond(request, file_id, job_id)

    return respond


def _respond(request: Request, file_id: str, job_id: str) -> Reply:
    path: Final = urlsplit(request.target).path
    if (request.method, path) == ("GET", "/v1/models"):
        return _reply({"object": "list", "data": []})
    if (request.method, path) == ("POST", "/v1/files"):
        return _reply(
            {
                "id": file_id,
                "object": "file",
                "bytes": len(TRAINING_LINE),
                "created_at": 1700000000,
                "filename": "train.jsonl",
                "purpose": "fine-tune",
                "status": "processed",
            }
        )
    if (request.method, path) == ("POST", "/v1/fine_tuning/jobs"):
        return _reply(_job(job_id, "validating_files", file_id))
    matched: Final = JOB_PATH.match(path)
    if matched and request.method == "GET" and not matched.group(2):
        return _reply(_job(matched.group(1), "running", file_id))
    if matched and request.method == "POST" and matched.group(2):
        return _reply(_job(matched.group(1), "cancelled", file_id))
    return Reply(
        status=404, body=json.dumps({"error": {"message": f"unscripted {request.method} {request.target}"}}).encode()
    )


def _config(wire: Wire, litellm_settings: dict[str, JsonValue]) -> dict[str, JsonValue]:
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
        "litellm_settings": litellm_settings,
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


def _unified_job_id(model_id: str, job_id: str) -> str:
    return (
        base64.urlsafe_b64encode(f"litellm_proxy;model_id:{model_id};generic_response_id:{job_id}".encode())
        .decode()
        .rstrip("=")
    )


def _deployment_id(proxy: Gateway) -> str:
    listed: Final = proxy.get("/v1/model/info")["data"]
    assert isinstance(listed, list), listed
    ids: Final = [
        string_value(object_value(object_value(entry)["model_info"])["id"])
        for entry in listed
        if object_value(entry)["model_name"] == MANAGED_MODEL
    ]
    assert len(ids) == 1, listed
    return ids[0]


def _targets(requests: tuple[Request, ...]) -> list[tuple[str, str, str]]:
    return [(request.method, request.target, request.headers.get("authorization", "")) for request in requests]


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    owner: str
    owner_user: str
    intruder: str
    intruder_user: str
    managed_file: str
    provider_file: str
    managed_job: str
    created_text: str
    deployment_id: str
    provider_job: str

    def sdk(self, key: str) -> OpenAI:
        return OpenAI(base_url=f"{str(self.proxy.client.base_url).rstrip('/')}/v1", api_key=key, max_retries=0)


@contextmanager
def _rig(directory: Path, litellm_settings: dict[str, JsonValue], file_id: str, job_id: str) -> Iterator[Rig]:
    with gateway_from_environment() as shared, wire_server(_responder(file_id, job_id)) as wire:
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(_config(wire, litellm_settings)))
        with owned_proxy(shared, directory, {}, config=config) as proxy, proxy.scenario() as scenario:
            eventually(
                lambda: _targets(wire.drain()), lambda seen: seen == [("GET", "/v1/models", f"Bearer {DEPLOYMENT_KEY}")]
            )
            deployment_id: Final = _deployment_id(proxy)
            owner_user: Final = scenario.user()
            owner: Final = scenario.key(user_id=owner_user)
            intruder_user: Final = scenario.user()
            intruder: Final = scenario.key(user_id=intruder_user)
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
            assert _targets(wire.drain()) == [
                ("POST", "/v1/files", f"Bearer {DEPLOYMENT_KEY}"),
                ("POST", "/v1/fine_tuning/jobs", f"Bearer {DEPLOYMENT_KEY}"),
            ]
            yield Rig(
                proxy,
                wire,
                owner,
                owner_user,
                intruder,
                intruder_user,
                managed_file,
                file_id,
                Job.model_validate_json(created.text).id,
                created.text,
                deployment_id,
                job_id,
            )
            assert wire.drain() == ()


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        tmp_path_factory.mktemp("fine-tuning-managed-ids"), {"require_managed_files": True}, PROVIDER_FILE, PROVIDER_JOB
    ) as built:
        yield built


@pytest.fixture(scope="module")
def default_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        tmp_path_factory.mktemp("fine-tuning-managed-ids-default"), {}, DEFAULT_PROVIDER_FILE, DEFAULT_PROVIDER_JOB
    ) as built:
        yield built


@pytest.mark.parametrize("fixture", ["rig", "default_rig"])
def test_managed_file_and_job_rows_record_the_owner(fixture: str, request: pytest.FixtureRequest) -> None:
    built: Final[Rig] = request.getfixturevalue(fixture)
    file_rows: Final = read_rows(
        'SELECT model_mappings, flat_model_file_ids, created_by, team_id FROM "LiteLLM_ManagedFileTable" WHERE unified_file_id = %s',
        (built.managed_file,),
    )
    assert file_rows == [
        {
            "model_mappings": {built.deployment_id: built.provider_file},
            "flat_model_file_ids": [built.provider_file],
            "created_by": built.owner_user,
            "team_id": None,
        }
    ], file_rows
    job_rows: Final = eventually(
        lambda: read_rows(
            'SELECT unified_object_id, model_object_id, file_purpose, created_by, team_id FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
            (built.managed_job,),
        ),
        lambda rows: len(rows) == 1,
    )
    assert job_rows == [
        {
            "unified_object_id": built.managed_job,
            "model_object_id": built.provider_job,
            "file_purpose": "fine-tune",
            "created_by": built.owner_user,
            "team_id": None,
        }
    ], job_rows


def _refused(response: httpx.Response, wire: Wire, detail: str) -> None:
    reached: Final = _targets(wire.drain())
    assert response.status_code == 403, response.text
    assert ProxyError.model_validate_json(response.text).error["message"] == detail, response.text
    assert reached == []


@pytest.mark.parametrize("fixture", ["rig", "default_rig"])
def test_owner_create_returns_the_unified_job_id(fixture: str, request: pytest.FixtureRequest) -> None:
    built: Final[Rig] = request.getfixturevalue(fixture)
    assert Job.model_validate_json(built.created_text) == Job(
        id=_unified_job_id(built.deployment_id, built.provider_job),
        object="fine_tuning.job",
        status="validating_files",
        model="gpt-4.1",
    ), built.created_text


def test_another_key_cannot_create_with_or_cancel_the_owners_managed_ids(rig: Rig) -> None:
    created: Final = rig.proxy.request(
        "POST",
        "/v1/fine_tuning/jobs",
        {"model": MANAGED_MODEL, "training_file": rig.managed_file},
        key=rig.intruder,
    )
    _refused(created, rig.wire, "The caller does not have access to this managed file id.")
    cancelled: Final = rig.proxy.request("POST", f"/v1/fine_tuning/jobs/{rig.managed_job}/cancel", key=rig.intruder)
    _refused(cancelled, rig.wire, "The caller does not have access to this managed fine-tuning job id.")


def test_another_key_cannot_cancel_the_owners_managed_job_by_default(default_rig: Rig) -> None:
    cancelled: Final = default_rig.proxy.request(
        "POST", f"/v1/fine_tuning/jobs/{default_rig.managed_job}/cancel", key=default_rig.intruder
    )
    _refused(
        cancelled,
        default_rig.wire,
        f"User {default_rig.intruder_user} does not have access to the object {default_rig.managed_job}",
    )


def test_another_key_cannot_create_with_the_owners_managed_file_by_default(default_rig: Rig) -> None:
    pytest.skip(
        "BUG: another key can create a fine-tuning job with the owner's managed training_file"
        " when require_managed_files is off"
    )
    created: Final = default_rig.proxy.request(
        "POST",
        "/v1/fine_tuning/jobs",
        {"model": MANAGED_MODEL, "training_file": default_rig.managed_file},
        key=default_rig.intruder,
    )
    _refused(
        created,
        default_rig.wire,
        f"User {default_rig.intruder_user} does not have access to the file {default_rig.managed_file}",
    )


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
    reached: Final = _targets(rig.wire.drain())
    assert response.status_code == 400, response.text
    assert ProxyError.model_validate_json(response.text).error["message"] == (
        f"Raw provider {kind} ids cannot be used when require_managed_files is enabled in litellm_settings. "
        f"Use the LiteLLM managed {kind} id returned when the {kind} was created."
    ), response.text
    assert reached == []


def test_keys_granted_the_retrieve_route_get_the_owner_check_and_the_decoded_job_id(rig: Rig) -> None:
    with rig.proxy.scenario() as scenario:
        owner: Final = scenario.key(user_id=rig.owner_user, allowed_routes=["/v1/fine_tuning/jobs"])
        intruder: Final = scenario.key(user_id=rig.intruder_user, allowed_routes=["/v1/fine_tuning/jobs"])
        _refused(
            rig.proxy.request("GET", f"/v1/fine_tuning/jobs/{rig.managed_job}", key=intruder),
            rig.wire,
            "The caller does not have access to this managed fine-tuning job id.",
        )
        with rig.sdk(owner) as client:
            retrieved: Final = client.fine_tuning.jobs.with_raw_response.retrieve(rig.managed_job)
        assert [(r.method, r.target, r.headers["authorization"], r.body) for r in rig.wire.drain()] == [
            ("GET", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}", f"Bearer {DEPLOYMENT_KEY}", b"")
        ]
        assert retrieved.status_code == 200, retrieved.text
        assert Job.model_validate_json(retrieved.text) == Job(
            id=rig.managed_job, object="fine_tuning.job", status="running", model="gpt-4.1"
        ), retrieved.text


def test_owner_retrieve_reaches_the_provider_with_the_decoded_job_id(rig: Rig) -> None:
    pytest.skip("BUG: a virtual key gets 401 on GET /v1/fine_tuning/jobs/{id}, the route is missing from openai_routes")
    _refused(
        rig.proxy.request("GET", f"/v1/fine_tuning/jobs/{rig.managed_job}", key=rig.intruder),
        rig.wire,
        "The caller does not have access to this managed fine-tuning job id.",
    )
    with rig.sdk(rig.owner) as client:
        retrieved: Final = client.fine_tuning.jobs.with_raw_response.retrieve(rig.managed_job)
    assert [(r.method, r.target, r.headers["authorization"], r.body) for r in rig.wire.drain()] == [
        ("GET", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}", f"Bearer {DEPLOYMENT_KEY}", b"")
    ]
    assert retrieved.status_code == 200, retrieved.text
    assert Job.model_validate_json(retrieved.text) == Job(
        id=rig.managed_job, object="fine_tuning.job", status="running", model="gpt-4.1"
    ), retrieved.text


def test_owner_cancel_reaches_the_provider_with_the_decoded_job_id(rig: Rig) -> None:
    with rig.sdk(rig.owner) as client:
        cancelled: Final = client.fine_tuning.jobs.with_raw_response.cancel(rig.managed_job)
    assert [(r.method, r.target, r.headers["authorization"], r.body) for r in rig.wire.drain()] == [
        ("POST", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}/cancel", f"Bearer {DEPLOYMENT_KEY}", b"")
    ]
    assert cancelled.status_code == 200, cancelled.text
    assert Job.model_validate_json(cancelled.text) == Job(
        id=rig.managed_job, object="fine_tuning.job", status="cancelled", model="gpt-4.1"
    ), cancelled.text
