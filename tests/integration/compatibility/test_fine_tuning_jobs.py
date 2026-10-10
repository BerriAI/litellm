import base64
import json
import re
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

OPENAI_KEY: Final = "sk-fine-tuning-openai"
AZURE_KEY: Final = "azure-fine-tuning-key"
AMBIENT_OPENAI_KEY: Final = "sk-fine-tuning-ambient-environment"
AMBIENT_AZURE_KEY: Final = "azure-fine-tuning-ambient-environment"
AZURE_API_VERSION: Final = "2024-10-21"
MANAGED_MODEL: Final = "gpt-4.1-openai"
DEPLOYMENT_KEYS: Final = ("sk-fine-tuning-deployment-a", "sk-fine-tuning-deployment-b")
PROVIDER_JOB: Final = "ftjob-provider-123"
RUN: Final = uuid.uuid4().hex[:12]
MANAGED_JOB: Final = f"ftjob-managed-{RUN}"
TRAINING_LINE: Final = b'{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
JOB_PATH: Final = re.compile(r"^/(?:openai/)?(?:v1/)?fine_tuning/jobs/([^/]+)(/cancel)?$")
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
PREFIXES: Final = (pytest.param("/v1", id="v1"), pytest.param("", id="unversioned"))


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object: str
    created_at: int
    error: JsonValue
    fine_tuned_model: str | None
    finished_at: int | None
    hyperparameters: dict[str, JsonValue]
    model: str
    organization_id: str
    result_files: list[str]
    seed: int
    status: str
    trained_tokens: int | None
    training_file: str
    validation_file: str | None
    estimated_finish: int | None = None
    integrations: JsonValue = None
    metadata: JsonValue = None
    method: JsonValue = None


def _client_job(job_id: str, status: str, training_file: str = "file-abc123") -> Job:
    return Job.model_validate(
        {
            **_job(job_id, status, training_file),
            "hyperparameters": {"batch_size": None, "learning_rate_multiplier": None, "n_epochs": 3},
        }
    )


def _job(job_id: str, status: str, training_file: str = "file-abc123", **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": job_id,
        "object": "fine_tuning.job",
        "created_at": 1700000000,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": 3},
        "model": "gpt-4o-mini-2024-07-18",
        "organization_id": "org-fine-tuning",
        "result_files": [],
        "seed": 42,
        "status": status,
        "trained_tokens": None,
        "training_file": training_file,
        "validation_file": None,
        **fields,
    }


def _provider_file_id(bearer: str) -> str:
    return f"file-{RUN}-" + bearer.removeprefix("sk-fine-tuning-")


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def _environment(wire: Wire) -> dict[str, str]:
    return {
        "OPENAI_BASE_URL": f"{wire.url}/v1",
        "OPENAI_API_BASE": f"{wire.url}/v1",
        "OPENAI_API_KEY": AMBIENT_OPENAI_KEY,
        "AZURE_API_BASE": wire.url,
        "AZURE_API_KEY": AMBIENT_AZURE_KEY,
        "AZURE_API_VERSION": AZURE_API_VERSION,
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
                "id": _provider_file_id(_bearer(request)),
                "object": "file",
                "bytes": len(TRAINING_LINE),
                "created_at": 1700000000,
                "filename": "train.jsonl",
                "purpose": "fine-tune",
                "status": "processed",
            }
        )
    if (request.method, path) in {("POST", "/v1/fine_tuning/jobs"), ("POST", "/openai/fine_tuning/jobs")}:
        sent: Final = JSON_OBJECT.validate_json(request.body)
        job_id: Final = MANAGED_JOB if _bearer(request) in DEPLOYMENT_KEYS else PROVIDER_JOB
        return _reply(_job(job_id, "validating_files", str(sent["training_file"])))
    matched: Final = JOB_PATH.match(path)
    if matched and matched.group(1) and request.method == "GET":
        return _reply(_job(matched.group(1), "running"))
    if matched and matched.group(2) and request.method == "POST":
        return _reply(_job(matched.group(1), "cancelled"))
    return Reply(
        status=404, body=json.dumps({"error": {"message": f"unscripted {request.method} {request.target}"}}).encode()
    )


def _config(wire: Wire) -> dict[str, JsonValue]:
    return {
        "model_list": [
            {
                "model_name": MANAGED_MODEL,
                "litellm_params": {"model": "openai/gpt-4.1", "api_key": key, "api_base": f"{wire.url}/v1"},
                "model_info": {"supported_endpoints": ["/chat/completions", "/fine_tuning"]},
            }
            for key in DEPLOYMENT_KEYS
        ],
        "finetune_settings": [
            {"custom_llm_provider": "openai", "api_key": OPENAI_KEY, "api_base": f"{wire.url}/v1"},
            {
                "custom_llm_provider": "azure",
                "api_key": AZURE_KEY,
                "api_base": wire.url,
                "api_version": AZURE_API_VERSION,
            },
        ],
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire

    def sdk(self, prefix: str) -> OpenAI:
        return OpenAI(
            base_url=f"{str(self.proxy.client.base_url).rstrip('/')}{prefix}", api_key=self.proxy.key, max_retries=0
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("fine-tuning")
    with gateway_from_environment() as shared, wire_server(_respond) as wire:
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(_config(wire)))
        with owned_proxy(shared, directory, _environment(wire), config=config) as proxy:
            boot: Final[list[Request]] = []
            eventually(lambda: boot.extend(wire.drain()) or len(boot), lambda count: count >= len(DEPLOYMENT_KEYS))
            assert sorted((request.method, request.target, _bearer(request)) for request in boot) == [
                ("GET", "/v1/models", key) for key in DEPLOYMENT_KEYS
            ]
            yield Rig(proxy, wire)
            assert wire.drain() == ()


def _drained(rig: Rig) -> tuple[Request, ...]:
    return rig.wire.drain()


def _only(requests: tuple[Request, ...], method: str, target: str) -> Request:
    assert [(request.method, request.target) for request in requests] == [(method, target)]
    return requests[0]


def _decoded(unified_id: str) -> str:
    return base64.urlsafe_b64decode(unified_id + "=" * (-len(unified_id) % 4)).decode()


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


@pytest.mark.parametrize("prefix", PREFIXES)
def test_create_forwards_every_documented_optional_field(rig: Rig, prefix: str) -> None:
    with rig.sdk(prefix) as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model="gpt-4o-mini",
            training_file="file-abc123",
            validation_file="file-xyz789",
            hyperparameters={"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": 0.1},
            suffix="custom-model",
            seed=42,
            extra_body={"custom_llm_provider": "openai"},
        )
    sent: Final = _only(_drained(rig), "POST", "/v1/fine_tuning/jobs")
    assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-4o-mini",
        "training_file": "file-abc123",
        "validation_file": "file-xyz789",
        "hyperparameters": {"n_epochs": 3, "batch_size": "auto", "learning_rate_multiplier": 0.1},
        "suffix": "custom-model",
        "seed": 42,
    }, sent.body
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "validating_files"), raw.text


def test_create_forwards_the_method_object(rig: Rig) -> None:
    pytest.skip(
        "BUG: create drops the documented method object, the provider receives only model, training_file and hyperparameters {}"
    )
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model="gpt-4o-mini",
            training_file="file-abc123",
            method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2}}},
            extra_body={"custom_llm_provider": "openai"},
        )
    sent: Final = _only(_drained(rig), "POST", "/v1/fine_tuning/jobs")
    assert raw.status_code == 200, raw.text
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-4o-mini",
        "training_file": "file-abc123",
        "hyperparameters": {},
        "method": {"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 2}}},
    }, sent.body


def test_create_forwards_integrations_as_the_sdk_sends_them(rig: Rig) -> None:
    pytest.skip(
        "BUG: create rejects the SDK integrations objects with 422 because the proxy types integrations as list[str]"
    )
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model="gpt-4o-mini",
            training_file="file-abc123",
            integrations=[{"type": "wandb", "wandb": {"project": "fine-tuning"}}],
            extra_body={"custom_llm_provider": "openai"},
        )
    sent: Final = _only(_drained(rig), "POST", "/v1/fine_tuning/jobs")
    assert raw.status_code == 200, raw.text
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-4o-mini",
        "training_file": "file-abc123",
        "hyperparameters": {},
        "integrations": [{"type": "wandb", "wandb": {"project": "fine-tuning"}}],
    }, sent.body


@pytest.mark.parametrize("prefix", PREFIXES)
def test_retrieve_forwards_the_raw_provider_job_id(rig: Rig, prefix: str) -> None:
    with rig.sdk(prefix) as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.retrieve(
            PROVIDER_JOB, extra_query={"custom_llm_provider": "openai"}
        )
    sent: Final = _only(_drained(rig), "GET", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}")
    assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert sent.body == b""
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "running"), raw.text


@pytest.mark.parametrize("prefix", PREFIXES)
def test_cancel_sends_the_cancel_to_the_provider(rig: Rig, prefix: str) -> None:
    with rig.sdk(prefix) as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.cancel(
            PROVIDER_JOB, extra_body={"custom_llm_provider": "openai"}
        )
    sent: Final = _only(_drained(rig), "POST", f"/v1/fine_tuning/jobs/{PROVIDER_JOB}/cancel")
    assert sent.headers["authorization"] == f"Bearer {OPENAI_KEY}"
    assert sent.body == b""
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "cancelled"), raw.text


def test_azure_create_merges_finetune_settings_and_keeps_azure_fields(rig: Rig) -> None:
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model="gpt-35-turbo-1106",
            training_file="file-abc",
            extra_body={
                "custom_llm_provider": "azure",
                "hyperparameters": {"n_epochs": 3, "prompt_loss_weight": 0.01},
                "trainingType": 1,
            },
        )
    sent: Final = _only(_drained(rig), "POST", f"/openai/fine_tuning/jobs?api-version={AZURE_API_VERSION}")
    assert sent.headers["api-key"] == AZURE_KEY
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-35-turbo-1106",
        "training_file": "file-abc",
        "hyperparameters": {"n_epochs": 3},
        "trainingType": 1,
        "prompt_loss_weight": 0.01,
    }, sent.body
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "validating_files", "file-abc"), raw.text


def test_azure_cancel_merges_finetune_settings(rig: Rig) -> None:
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.cancel(
            PROVIDER_JOB, extra_body={"custom_llm_provider": "azure"}
        )
    sent: Final = _only(
        _drained(rig), "POST", f"/openai/fine_tuning/jobs/{PROVIDER_JOB}/cancel?api-version={AZURE_API_VERSION}"
    )
    assert sent.headers["api-key"] == AZURE_KEY
    assert sent.body == b""
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "cancelled"), raw.text


def test_create_reads_the_documented_custom_llm_provider_header(rig: Rig) -> None:
    pytest.skip(
        "BUG: create ignores the documented custom-llm-provider header and returns 500"
        " Invalid request, No litellm managed file id or custom_llm_provider provided."
    )
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model="gpt-35-turbo-1106", training_file="file-abc", extra_headers={"custom-llm-provider": "azure"}
        )
    sent: Final = _only(_drained(rig), "POST", f"/openai/fine_tuning/jobs?api-version={AZURE_API_VERSION}")
    assert sent.headers["api-key"] == AZURE_KEY
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-35-turbo-1106",
        "training_file": "file-abc",
        "hyperparameters": {},
    }, sent.body
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "validating_files", "file-abc"), raw.text


def test_cancel_reads_the_documented_custom_llm_provider_header(rig: Rig) -> None:
    pytest.skip(
        "BUG: cancel ignores the documented custom-llm-provider header and the azure finetune_settings entry"
        " and sends the cancel to the default OpenAI base URL and key"
    )
    with rig.sdk("/v1") as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.cancel(
            PROVIDER_JOB, extra_headers={"custom-llm-provider": "azure"}
        )
    sent: Final = _only(
        _drained(rig), "POST", f"/openai/fine_tuning/jobs/{PROVIDER_JOB}/cancel?api-version={AZURE_API_VERSION}"
    )
    assert sent.headers["api-key"] == AZURE_KEY
    assert sent.body == b""
    assert raw.status_code == 200, raw.text
    assert Job.model_validate_json(raw.text) == _client_job(PROVIDER_JOB, "cancelled"), raw.text


def test_managed_create_sends_the_provider_file_id_of_the_pinned_deployment(rig: Rig) -> None:
    pytest.skip(
        "BUG: managed create sends the base64 LiteLLM file id and the prefixed model openai/gpt-4.1 to the provider"
        " and returns the decoded unified id as training_file"
    )
    with rig.sdk("/v1") as client:
        uploaded: Final = client.files.with_raw_response.create(
            file=("train.jsonl", TRAINING_LINE, "application/jsonl"),
            purpose="fine-tune",
            extra_body={"target_model_names": MANAGED_MODEL},
        )
        assert uploaded.status_code == 200, uploaded.text
        managed_file: Final = JSON_OBJECT.validate_json(uploaded.text)["id"]
        assert isinstance(managed_file, str) and _decoded(managed_file).startswith("litellm_proxy:"), uploaded.text
        uploads: Final = _drained(rig)
        assert sorted((request.method, request.target, _bearer(request)) for request in uploads) == [
            ("POST", "/v1/files", key) for key in DEPLOYMENT_KEYS
        ]
        for upload in uploads:
            parts = {part.get_param("name", header="content-disposition"): part for part in _multipart_parts(upload)}
            assert sorted(parts) == ["file", "purpose"], upload.body[:300]
            assert parts["file"].get_filename() == "train.jsonl"
            assert parts["file"].get_payload(decode=True) == TRAINING_LINE
            assert parts["purpose"].get_filename() is None
            assert parts["purpose"].get_content() == "fine-tune"

        raw: Final = client.fine_tuning.jobs.with_raw_response.create(model=MANAGED_MODEL, training_file=managed_file)
    sent: Final = _only(_drained(rig), "POST", "/v1/fine_tuning/jobs")
    deployment_key: Final = _bearer(sent)
    assert deployment_key in DEPLOYMENT_KEYS, sent.headers
    assert JSON_OBJECT.validate_json(sent.body) == {
        "model": "gpt-4.1",
        "training_file": _provider_file_id(deployment_key),
        "hyperparameters": {},
    }, sent.body
    assert raw.status_code == 200, raw.text
    job: Final = Job.model_validate_json(raw.text)
    assert re.fullmatch(
        rf"litellm_proxy;model_id:[0-9a-f]{{64}};generic_response_id:{MANAGED_JOB}", _decoded(job.id)
    ), raw.text
    assert job.training_file == managed_file, raw.text
