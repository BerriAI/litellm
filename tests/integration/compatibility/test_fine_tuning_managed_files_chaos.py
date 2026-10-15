"""Managed fine-tuning creates on a two-worker proxy whose model group has two deployments: a concurrent burst from
sync and async clients, traffic pinned to each worker in turn, and a worker killed and replaced, all send every
deployment its own provider file ids and hand the caller back its own managed ids. A deployment added to the group
after the upload holds no copy of the files, so it never receives the create even when the router prefers it. A model
group the files were never uploaded for still gets the create and the provider's own 400, as before routing checked
which deployments hold the files."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support.client import eventually, gateway_from_environment
from integration._support.process import OwnedProxy, WorkerHealthcheck, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, BadRequestError, OpenAI
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(480)

MODEL: Final = "gpt-4.1-chaos"
DEPLOYMENT_KEYS: Final = ("sk-chaos-deployment-a", "sk-chaos-deployment-b")
LATE_DEPLOYMENT_KEY: Final = "sk-chaos-deployment-late"
OTHER_MODEL: Final = "gpt-4.1-chaos-other"
OTHER_DEPLOYMENT_KEY: Final = "sk-chaos-deployment-other"
RUN: Final = uuid.uuid4().hex[:12]
TRAINING_LINE: Final = b'{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
BURST: Final = 20
CREATE: Final = ("POST", "/v1/fine_tuning/jobs")
WORKER_BOOT_DISCOVERY: Final = ("GET", "/v1/models")


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    wire: Wire
    training_file: str
    validation_file: str

    def base_url(self) -> str:
        return f"{str(self.owned.gateway.client.base_url).rstrip('/')}/v1"

    def sdk(self) -> OpenAI:
        return OpenAI(base_url=self.base_url(), api_key=self.owned.gateway.key, max_retries=0)

    def async_sdk(self) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=self.base_url(), api_key=self.owned.gateway.key, max_retries=0)


def _bearer(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def _provider_file_id(bearer: str, purpose: str) -> str:
    return f"file-{RUN}-{bearer.removeprefix('sk-chaos-')}-{purpose}"


def _uploaded_purpose(request: Request) -> str:
    return "validate" if b'filename="validate.jsonl"' in request.body else "train"


def _invalid_training_file(training_file: str) -> Reply:
    error: Final = {
        "message": f"invalid training_file: {training_file}",
        "type": "invalid_request_error",
        "param": "training_file",
        "code": None,
    }
    return Reply(status=400, body=json.dumps({"error": error}).encode())


def _respond(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if (request.method, path) == ("POST", "/v1/files"):
        return Reply(
            body=json.dumps(
                {
                    "id": _provider_file_id(_bearer(request), _uploaded_purpose(request)),
                    "object": "file",
                    "bytes": len(TRAINING_LINE),
                    "created_at": 1700000000,
                    "filename": f"{_uploaded_purpose(request)}.jsonl",
                    "purpose": "fine-tune",
                    "status": "processed",
                }
            ).encode()
        )
    if (request.method, path) == ("POST", "/v1/fine_tuning/jobs"):
        sent: Final = JSON_OBJECT.validate_json(request.body)
        if not str(sent["training_file"]).startswith(f"file-{RUN}-"):
            return _invalid_training_file(str(sent["training_file"]))
        return Reply(
            body=json.dumps(
                {
                    "id": f"ftjob-{uuid.uuid4().hex}",
                    "object": "fine_tuning.job",
                    "created_at": 1700000000,
                    "error": None,
                    "fine_tuned_model": None,
                    "finished_at": None,
                    "hyperparameters": {"n_epochs": 3},
                    "model": sent["model"],
                    "organization_id": "org-chaos",
                    "result_files": [],
                    "seed": 42,
                    "status": "validating_files",
                    "trained_tokens": None,
                    "training_file": sent["training_file"],
                    "validation_file": sent.get("validation_file"),
                }
            ).encode()
        )
    return Reply(body=json.dumps({"object": "list", "data": []}).encode())


def _deployment(wire: Wire, key: str, routing: Mapping[str, JsonValue], model: str = MODEL) -> dict[str, JsonValue]:
    return {
        "model_name": model,
        "litellm_params": {"model": "openai/gpt-4.1", "api_key": key, "api_base": f"{wire.url}/v1", **routing},
        "model_info": {"supported_endpoints": ["/chat/completions", "/fine_tuning"]},
    }


def _config(wire: Wire, late_preferred_deployment: bool = False) -> dict[str, JsonValue]:
    late: Final = (_deployment(wire, LATE_DEPLOYMENT_KEY, {"order": 1}),) if late_preferred_deployment else ()
    return {
        "model_list": [
            *(_deployment(wire, key, {}) for key in DEPLOYMENT_KEYS),
            *late,
            _deployment(wire, OTHER_DEPLOYMENT_KEY, {}, OTHER_MODEL),
        ],
        "router_settings": {"num_retries": 0, "disable_cooldowns": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "database_url": "os.environ/DATABASE_URL"},
    }


def _config_file(directory: Path, wire: Wire, late_preferred_deployment: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config: Final = directory / "config.yaml"
    config.write_text(yaml.safe_dump(_config(wire, late_preferred_deployment)))
    return config


def _upload(client: OpenAI, purpose: str) -> str:
    uploaded: Final = client.files.with_raw_response.create(
        file=(f"{purpose}.jsonl", TRAINING_LINE, "application/jsonl"),
        purpose="fine-tune",
        extra_body={"target_model_names": MODEL},
    )
    assert uploaded.status_code == 200, uploaded.text
    return str(JSON_OBJECT.validate_json(uploaded.text)["id"])


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("fine-tuning-chaos")
    with gateway_from_environment() as shared, wire_server(_respond) as wire:
        config: Final = _config_file(directory, wire)
        with owned_proxy_process(
            shared, directory, {}, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT
        ) as owned:
            eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=120)
            unloaded: Final = Rig(owned, wire, "", "")
            with unloaded.sdk() as client:
                training_file: Final = _upload(client, "train")
                validation_file: Final = _upload(client, "validate")
            wire.drain()
            yield Rig(owned, wire, training_file, validation_file)


def _create(rig: Rig) -> tuple[int, Mapping[str, JsonValue]]:
    with rig.sdk() as client:
        raw: Final = client.fine_tuning.jobs.with_raw_response.create(
            model=MODEL, training_file=rig.training_file, validation_file=rig.validation_file
        )
    return raw.status_code, JSON_OBJECT.validate_json(raw.text)


async def _async_creates(rig: Rig, count: int) -> tuple[tuple[int, Mapping[str, JsonValue]], ...]:
    async with rig.async_sdk() as client:
        raws: Final = await asyncio.gather(
            *(
                client.fine_tuning.jobs.with_raw_response.create(
                    model=MODEL, training_file=rig.training_file, validation_file=rig.validation_file
                )
                for _ in range(count)
            )
        )
    return tuple((raw.status_code, JSON_OBJECT.validate_json(raw.text)) for raw in raws)


def _sent_creates(rig: Rig, expected: int) -> tuple[Request, ...]:
    collected: Final[list[Request]] = []

    def creates() -> tuple[Request, ...]:
        collected.extend(rig.wire.drain())
        return tuple(request for request in collected if (request.method, request.target) == CREATE)

    sent: Final = eventually(creates, lambda found: len(found) >= expected, seconds=30)
    assert len(sent) == expected, [(request.method, request.target) for request in collected]
    assert {(request.method, request.target) for request in collected} <= {CREATE, WORKER_BOOT_DISCOVERY}, collected
    return sent


def _assert_each_deployment_got_its_own_ids(creates: tuple[Request, ...]) -> None:
    for request in creates:
        bearer = _bearer(request)
        assert bearer in DEPLOYMENT_KEYS, request.headers
        assert JSON_OBJECT.validate_json(request.body) == {
            "model": "gpt-4.1",
            "training_file": _provider_file_id(bearer, "train"),
            "validation_file": _provider_file_id(bearer, "validate"),
            "hyperparameters": {},
        }, request.body


def _assert_caller_gets_its_own_ids(rig: Rig, outcomes: tuple[tuple[int, Mapping[str, JsonValue]], ...]) -> None:
    received: Final = [(status, body.get("training_file"), body.get("validation_file")) for status, body in outcomes]
    assert received == [(200, rig.training_file, rig.validation_file)] * len(outcomes), outcomes


def test_concurrent_sync_and_async_burst_sends_each_deployment_its_own_file_ids(rig: Rig) -> None:
    with ThreadPoolExecutor(max_workers=BURST // 2) as pool:
        synced: Final = tuple(pool.map(_create, (rig,) * (BURST // 2)))
    awaited: Final = asyncio.run(_async_creates(rig, BURST // 2))
    creates: Final = _sent_creates(rig, BURST)
    _assert_each_deployment_got_its_own_ids(creates)
    assert {_bearer(request) for request in creates} == set(DEPLOYMENT_KEYS)
    _assert_caller_gets_its_own_ids(rig, synced + awaited)


def test_each_worker_alone_sends_the_provider_file_ids(rig: Rig) -> None:
    for paused in sorted(rig.owned.worker_pids()):
        with rig.owned.suspended(paused):
            outcomes = tuple(_create(rig) for _ in range(4))
        _assert_caller_gets_its_own_ids(rig, outcomes)
        _assert_each_deployment_got_its_own_ids(_sent_creates(rig, 4))


def test_a_replaced_worker_still_sends_the_provider_file_ids(rig: Rig) -> None:
    for victim in sorted(rig.owned.worker_pids()):
        rig.owned.kill_worker(victim)
    outcomes: Final = tuple(_create(rig) for _ in range(6))
    _assert_caller_gets_its_own_ids(rig, outcomes)
    _assert_each_deployment_got_its_own_ids(_sent_creates(rig, 6))


def test_a_deployment_added_after_the_upload_never_receives_the_create(tmp_path: Path) -> None:
    with gateway_from_environment() as shared, wire_server(_respond) as wire:
        before: Final = tmp_path / "before"
        with owned_proxy_process(shared, before, {}, config=_config_file(before, wire)) as uploader:
            with Rig(uploader, wire, "", "").sdk() as client:
                training_file: Final = _upload(client, "train")
                validation_file: Final = _upload(client, "validate")
        after: Final = tmp_path / "after"
        with owned_proxy_process(
            shared, after, {}, config=_config_file(after, wire, late_preferred_deployment=True), workers=2
        ) as owned:
            eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=120)
            wire.drain()
            rig: Final = Rig(owned, wire, training_file, validation_file)
            outcomes: Final = tuple(_create(rig) for _ in range(6))
            creates: Final = _sent_creates(rig, 6)
    _assert_caller_gets_its_own_ids(rig, outcomes)
    _assert_each_deployment_got_its_own_ids(creates)


def test_a_model_group_without_a_copy_of_the_files_still_gets_the_providers_400(rig: Rig) -> None:
    with rig.sdk() as client, pytest.raises(BadRequestError) as rejected:
        client.fine_tuning.jobs.create(model=OTHER_MODEL, training_file=rig.training_file)
    creates: Final = _sent_creates(rig, 1)
    assert rejected.value.status_code == 400, rejected.value.response.text
    assert [_bearer(request) for request in creates] == [OTHER_DEPLOYMENT_KEY], creates
