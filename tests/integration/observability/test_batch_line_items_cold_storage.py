"""Finished batches write their lines to cold storage.

An owned two-worker proxy lists ``s3_v2`` and ``gcs_bucket`` under ``batch_line_item_callbacks``. Once the batch-cost
poller logs a finished batch, each sink gets one NDJSON object for the batch holding one record per line. The proxy runs
on its own database so the shared rig's pollers never claim these batches first
"""

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.forward_proxy import tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, TextResponse
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(420)

S3_BUCKET: Final = "line-items-bucket"
S3_PREFIX: Final = "cold"
GCS_BUCKET: Final = "line-items-gcs"
COMPLETED_AT: Final = 1791494130
LINE_DATE: Final = datetime.fromtimestamp(COMPLETED_AT, tz=timezone.utc).strftime("%Y-%m-%d")
PROMPT_TOKENS: Final = 100
COMPLETION_TOKENS: Final = 50
POLLER_SECONDS: Final = 150


@dataclass(slots=True)
class ObjectSink:
    """Stores every accepted object body by name and answers 503 to any upload whose body holds ``reject``."""

    reject: bytes = b"\x00never"
    lock: threading.Lock = field(default_factory=threading.Lock)
    stored: dict[str, bytes] = field(default_factory=dict)  # mutable-ok: written by the sink's request threads
    attempts: dict[str, int] = field(default_factory=dict)  # mutable-ok: written by the sink's request threads

    def accept(self, name: str, body: bytes) -> bool:
        with self.lock:
            self.attempts[name] = self.attempts.get(name, 0) + 1
            if self.reject in body:
                return False
            self.stored[name] = body
            return True

    def objects(self) -> Mapping[str, bytes]:
        with self.lock:
            return MappingProxyType(dict(self.stored))

    def tried(self, name: str) -> int:
        with self.lock:
            return self.attempts.get(name, 0)


def s3_peer(sink: ObjectSink) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "PUT":
            return Reply(status=404)
        name: Final = unquote(urlsplit(request.target).path).removeprefix(f"/{S3_BUCKET}/")
        if sink.accept(name, request.body):
            return Reply()
        return Reply(status=503, body=b"<Error><Code>SlowDown</Code></Error>", content_type="application/xml")

    return respond


def gcs_peer(sink: ObjectSink) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "POST" and path == "/_oauth/token":
            return Reply(body=b'{"access_token":"scripted-token","expires_in":3600,"token_type":"Bearer"}')
        if request.method == "POST" and path == f"/upload/storage/v1/b/{GCS_BUCKET}/o":
            name: Final = parse_qs(urlsplit(request.target).query)["name"][0]
            if sink.accept(name, request.body):
                return Reply(body=json.dumps({"name": name, "bucket": GCS_BUCKET}).encode())
            return Reply(status=503, body=b'{"error":"unavailable"}')
        return Reply(status=404, body=b'{"error":"not scripted"}')

    return respond


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    database_url: str
    s3: ObjectSink
    gcs: ObjectSink


def _config(directory: Path, s3_url: str) -> Path:
    shipped: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration: Final = {
        **shipped,
        "litellm_settings": {
            **shipped["litellm_settings"],
            "callbacks": ["s3_v2"],
            "batch_line_item_callbacks": ["s3_v2", "gcs_bucket"],
            "s3_callback_params": {
                "s3_bucket_name": S3_BUCKET,
                "s3_region_name": "us-east-1",
                "s3_endpoint_url": s3_url,
                "s3_path": S3_PREFIX,
                "s3_aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "s3_aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "s3_flush_interval": 1,
            },
        },
    }
    path: Final = directory / "batch-line-items.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("batch-line-items")
    certificate, key = write_self_signed_cert(directory, names=("storage.googleapis.com",))
    s3: Final = ObjectSink(reject=b"sink-down-")
    gcs: Final = ObjectSink()
    with (
        gateway_from_environment() as environment,
        scratch_database("batch_line_items") as database_url,
        wire_server(s3_peer(s3)) as s3_wire,
        wire_server(gcs_peer(gcs)) as token_wire,
        wire_server(gcs_peer(gcs), tls=server_context(certificate, key)) as gcs_wire,
        tunnelling_forward_proxy(int(urlsplit(gcs_wire.url).port or 0)) as tunnel,
    ):
        service_account: Final = directory / "service-account.json"
        service_account.write_text(service_account_json("line-items-project", token_wire.url))
        with owned_proxy(
            environment,
            directory,
            {
                "DATABASE_URL": database_url,
                "GCS_BUCKET_NAME": GCS_BUCKET,
                "GCS_PATH_SERVICE_ACCOUNT": str(service_account),
                "GCS_FLUSH_INTERVAL": "1",
                "HTTPS_PROXY": tunnel.url,
                "https_proxy": tunnel.url,
                "SSL_CERT_FILE": str(certificate),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            },
            config=_config(directory, s3_wire.url),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as candidate:
            yield Rig(candidate, database_url, s3, gcs)


def _input_line(custom_id: str, text: str) -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]},
    }


def _success_line(custom_id: str) -> dict[str, JsonValue]:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"req-{custom_id}",
            "body": {
                "id": f"chatcmpl-{custom_id}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            },
        },
        "error": None,
    }


def _error_line(custom_id: str) -> dict[str, JsonValue]:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 400,
            "request_id": f"req-{custom_id}",
            "body": {"error": {"message": "max_tokens is too large", "type": "invalid_request_error"}},
        },
        "error": None,
    }


def _jsonl(lines: tuple[dict[str, JsonValue], ...]) -> str:
    return "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)


def _file(identity: str, purpose: str) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": identity,
            "object": "file",
            "purpose": purpose,
            "bytes": 100,
            "created_at": 1,
            "filename": f"{identity}.jsonl",
            "status": "processed",
        },
    )


@dataclass(frozen=True, slots=True)
class Lines:
    inputs: tuple[dict[str, JsonValue], ...]
    outputs: tuple[dict[str, JsonValue], ...]
    errors: tuple[dict[str, JsonValue], ...]


def _routes(lines: Lines, *, finished: bool = True) -> RoutedResponse:
    has_output: Final = bool(lines.outputs)
    completed: Final[dict[str, JsonValue]] = {
        "id": "batch-$REQUEST_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "errors": None,
        "input_file_id": "file-in-$REQUEST_ID",
        "completion_window": "24h",
        "status": "completed",
        "output_file_id": "file-out-$REQUEST_ID" if has_output else None,
        "error_file_id": "file-err-$REQUEST_ID" if lines.errors else None,
        "created_at": COMPLETED_AT - 60,
        "in_progress_at": COMPLETED_AT - 30,
        "completed_at": COMPLETED_AT,
        "expires_at": COMPLETED_AT + 86400,
        "request_counts": {
            "total": len(lines.inputs),
            "completed": len(lines.outputs),
            "failed": len(lines.errors),
        },
        "metadata": None,
    }
    return RoutedResponse(
        content_type="application/x-routed",
        routes={
            "POST /files": _file("file-in-$REQUEST_ID", "batch"),
            "POST /batches": JsonResponse(
                content_type="application/json",
                body={**completed, "status": "validating", "output_file_id": None, "error_file_id": None},
            ),
            "GET /batches/batch-$REQUEST_ID": JsonResponse(
                content_type="application/json",
                body=completed
                if finished
                else {**completed, "status": "in_progress", "output_file_id": None, "error_file_id": None},
            ),
            "GET /files/file-in-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body=_jsonl(lines.inputs)
            ),
            "GET /files/file-out-$REQUEST_ID": _file("file-out-$REQUEST_ID", "batch_output"),
            "GET /files/file-out-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body=_jsonl(lines.outputs)
            ),
            "GET /files/file-err-$REQUEST_ID": _file("file-err-$REQUEST_ID", "batch_output"),
            "GET /files/file-err-$REQUEST_ID/content": TextResponse(
                content_type="application/jsonl", body=_jsonl(lines.errors)
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class SubmittedBatch:
    batch_id: str
    key: str
    handle: ScenarioHandle


def _submit(rig: Rig, lines: Lines, key_fields: Mapping[str, JsonValue], *, finished: bool = True) -> SubmittedBatch:
    handle: Final = register_scenario(f"batch-line-items-{uuid.uuid4().hex}", _routes(lines, finished=finished))
    model: Final = string_value(
        rig.gateway.post(
            "/model/new",
            {
                "model_name": f"line-items-{uuid.uuid4().hex[:12]}",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-scripted-provider",
                    "api_base": handle.api_base(),
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            },
        )["model_name"]
    )
    key: Final = string_value(rig.gateway.post("/key/generate", {"models": [model], **key_fields})["key"])
    uploaded: Final = rig.gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "target_model_names": model},
        {"file": ("in.jsonl", _jsonl(lines.inputs).encode(), "application/jsonl")},
        key=key,
    )
    assert uploaded.is_success, uploaded.text
    created: Final = rig.gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": string_value(JSON_OBJECT.validate_json(uploaded.content)["id"]),
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )
    assert created.is_success, created.text
    return SubmittedBatch(string_value(JSON_OBJECT.validate_json(created.content)["id"]), key, handle)


def _quiet_logging() -> dict[str, JsonValue]:
    return {
        "logging": [
            {
                "callback_name": "s3_v2",
                "callback_type": "success",
                "callback_vars": {"turn_off_message_logging": "true"},
            }
        ]
    }


def _object_name(prefix: str, batch_id: str) -> str:
    return f"{prefix}batch_line_items/{LINE_DATE}/{batch_id.replace('/', '_')}.jsonl"


def _records(body: bytes) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(line) for line in body.splitlines() if line.strip())


def _written(sink: ObjectSink, name: str) -> tuple[dict[str, JsonValue], ...]:
    return _records(eventually(lambda: sink.objects().get(name), lambda body: body is not None, POLLER_SECONDS) or b"")


def _batch_spend(rig: Rig, key: str) -> float:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
            (sha256(key.encode()).hexdigest(), "aretrieve_batch"),
            database_url=rig.database_url,
        ),
        lambda found: len(found) == 1,
        POLLER_SECONDS,
    )
    spend: Final = rows[0]["spend"]
    assert isinstance(spend, float), rows
    return spend


def _by_custom_id(records: tuple[dict[str, JsonValue], ...]) -> Mapping[JsonValue, dict[str, JsonValue]]:
    keyed: Final = {record["custom_id"]: record for record in records}
    assert len(keyed) == len(records), records
    return MappingProxyType(keyed)


def test_a_finished_batch_writes_every_line_with_its_request_and_result_to_s3_and_gcs(rig: Rig) -> None:
    lines: Final = Lines(
        inputs=(_input_line("ok-1", "first question"), _input_line("bad-1", "second question")),
        outputs=(_success_line("ok-1"),),
        errors=(_error_line("bad-1"),),
    )
    batch: Final = _submit(rig, lines, {})
    try:
        s3_records: Final = _written(rig.s3, _object_name(f"{S3_PREFIX}/", batch.batch_id))
        gcs_body: Final = eventually(
            lambda: rig.gcs.objects().get(_object_name("", batch.batch_id)), lambda body: body is not None
        )
        spend: Final = _batch_spend(rig, batch.key)
    finally:
        delete_scenario(batch.handle)

    assert gcs_body is not None
    assert _records(gcs_body) == s3_records
    records: Final = _by_custom_id(s3_records)
    assert set(records) == {"ok-1", "bad-1"}
    hashed_key: Final = sha256(batch.key.encode()).hexdigest()
    succeeded: Final = records["ok-1"]
    assert succeeded["batch_id"] == batch.batch_id
    assert succeeded["status"] == "success"
    assert succeeded["request"] == lines.inputs[0]
    assert succeeded["response"] == lines.outputs[0]
    assert succeeded["usage"] == {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    assert succeeded["model"] == "gpt-4o-mini"
    assert succeeded["custom_llm_provider"] == "openai"
    assert isinstance(succeeded["metadata"], dict)
    assert succeeded["metadata"]["user_api_key_hash"] == hashed_key
    failed: Final = records["bad-1"]
    assert failed["status"] == "failure"
    assert failed["request"] == lines.inputs[1]
    assert failed["response"] == lines.errors[0]
    assert failed["usage"] is None
    assert failed["response_cost"] == 0
    assert isinstance(succeeded["response_cost"], float) and succeeded["response_cost"] > 0
    assert succeeded["response_cost"] == pytest.approx(spend)
    assert rig.s3.tried(_object_name(f"{S3_PREFIX}/", batch.batch_id)) == 1


def test_a_key_that_turned_off_message_logging_gets_its_lines_without_request_or_response(rig: Rig) -> None:
    lines: Final = Lines(
        inputs=(_input_line("quiet-1", "a private question"),), outputs=(_success_line("quiet-1"),), errors=()
    )
    batch: Final = _submit(rig, lines, {"metadata": _quiet_logging()})
    name: Final = _object_name(f"{S3_PREFIX}/", batch.batch_id)
    try:
        records: Final = _written(rig.s3, name)
    finally:
        delete_scenario(batch.handle)

    assert len(records) == 1
    assert records[0]["custom_id"] == "quiet-1"
    assert records[0]["status"] == "success"
    assert records[0]["request"] is None
    assert records[0]["response"] is None
    assert records[0]["usage"] == {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
    }
    assert b"a private question" not in rig.s3.objects()[name]


def test_a_team_that_turned_off_message_logging_gets_its_members_lines_without_request_or_response(rig: Rig) -> None:
    team_id: Final = string_value(
        rig.gateway.post("/team/new", {"team_alias": f"quiet-{uuid.uuid4().hex[:8]}", "metadata": _quiet_logging()})[
            "team_id"
        ]
    )
    lines: Final = Lines(
        inputs=(_input_line("team-1", "a team secret"),), outputs=(_success_line("team-1"),), errors=()
    )
    batch: Final = _submit(rig, lines, {"team_id": team_id})
    name: Final = _object_name(f"{S3_PREFIX}/", batch.batch_id)
    try:
        records: Final = _written(rig.s3, name)
    finally:
        delete_scenario(batch.handle)

    assert [(record["custom_id"], record["request"], record["response"]) for record in records] == [
        ("team-1", None, None)
    ]
    assert isinstance(records[0]["metadata"], dict)
    assert records[0]["metadata"]["user_api_key_team_id"] == team_id
    assert b"a team secret" not in rig.s3.objects()[name]


def test_a_batch_whose_creating_key_was_deleted_gets_its_lines_without_request_or_response(rig: Rig) -> None:
    lines: Final = Lines(
        inputs=(_input_line("orphan-1", "asked before the key was deleted"),),
        outputs=(_success_line("orphan-1"),),
        errors=(),
    )
    batch: Final = _submit(rig, lines, {}, finished=False)
    name: Final = _object_name(f"{S3_PREFIX}/", batch.batch_id)
    try:
        rig.gateway.post("/key/delete", {"keys": [batch.key]})
        register_scenario(batch.handle.scenario_id, _routes(lines))
        records: Final = _written(rig.s3, name)
        spend: Final = _batch_spend(rig, batch.key)
    finally:
        delete_scenario(batch.handle)

    assert [(record["custom_id"], record["status"], record["request"], record["response"]) for record in records] == [
        ("orphan-1", "success", None, None)
    ]
    assert records[0]["response_cost"] == pytest.approx(spend)
    assert b"asked before the key was deleted" not in rig.s3.objects()[name]


def test_a_batch_where_every_line_failed_still_writes_its_error_lines(rig: Rig) -> None:
    lines: Final = Lines(
        inputs=(_input_line("bad-a", "first"), _input_line("bad-b", "second")),
        outputs=(),
        errors=(_error_line("bad-a"), _error_line("bad-b")),
    )
    batch: Final = _submit(rig, lines, {})
    try:
        records: Final = _by_custom_id(_written(rig.s3, _object_name(f"{S3_PREFIX}/", batch.batch_id)))
    finally:
        delete_scenario(batch.handle)

    assert set(records) == {"bad-a", "bad-b"}
    assert records["bad-a"]["status"] == "failure"
    assert records["bad-a"]["request"] == lines.inputs[0]
    assert records["bad-b"]["response"] == lines.errors[1]
    assert all(record["response_cost"] == 0 for record in records.values())


def test_a_failing_bucket_still_bills_the_batch_and_the_other_sink_keeps_its_lines(rig: Rig) -> None:
    custom_id: Final = f"sink-down-{uuid.uuid4().hex[:8]}"
    lines: Final = Lines(
        inputs=(_input_line(custom_id, "billed anyway"),), outputs=(_success_line(custom_id),), errors=()
    )
    batch: Final = _submit(rig, lines, {})
    try:
        gcs_body: Final = eventually(
            lambda: rig.gcs.objects().get(_object_name("", batch.batch_id)),
            lambda body: body is not None,
            POLLER_SECONDS,
        )
        spend: Final = _batch_spend(rig, batch.key)
        processed: Final = eventually(
            lambda: read_rows(
                'SELECT batch_processed FROM "LiteLLM_ManagedObjectTable" WHERE unified_object_id = %s',
                (batch.batch_id,),
                database_url=rig.database_url,
            ),
            lambda rows: rows == [{"batch_processed": True}],
        )
    finally:
        delete_scenario(batch.handle)

    assert processed == [{"batch_processed": True}]
    assert gcs_body is not None
    assert [record["custom_id"] for record in _records(gcs_body)] == [custom_id]
    assert spend > 0
    s3_name: Final = _object_name(f"{S3_PREFIX}/", batch.batch_id)
    assert rig.s3.tried(s3_name) >= 1
    assert s3_name not in rig.s3.objects()
