import json
import re
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from _s3_v2_support import RecordingS3Sink, collect_payloads
from _s3_v2_support import s3_config as _recording_s3_config
from integration._support.client import Gateway, JsonValue, eventually
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

BUCKET: Final = "integration-bucket"
PREFIX: Final = "integration-logs"
REQUESTS: Final = 64
PUT_DELAY_SECONDS: Final = 0.5


@dataclass(slots=True)
class S3Sink:
    """Accepts every PUT after a fixed delay and records the peak number of PUTs in flight."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: int = 0
    peak: int = 0

    def respond(self, request: Request) -> Reply:
        assert request.method == "PUT", request.method
        assert request.target.startswith(f"/{BUCKET}/{PREFIX}/"), request.target
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        time.sleep(PUT_DELAY_SECONDS)
        with self.lock:
            self.in_flight -= 1
        return Reply()


def _chat_reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(status=404)
    text: Final = json.loads(request.body)["messages"][0]["content"]
    return Reply(
        body=json.dumps(
            {
                "id": text,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _s3_config(path: Path, sink_url: str, extra: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update(
        {
            "callbacks": ["s3_v2"],
            "s3_callback_params": {
                "s3_bucket_name": BUCKET,
                "s3_region_name": "us-east-1",
                "s3_endpoint_url": sink_url,
                "s3_path": PREFIX,
                "s3_aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "s3_aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                **extra,
            },
        }
    )
    target: Final = path / "s3_v2.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def _burst(candidate: Gateway, model: str, key: str, marker: str) -> frozenset[str]:
    ids: Final = tuple(f"{marker}-{index}" for index in range(REQUESTS))

    def request(identity: str) -> str:
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": identity}], "cache": {"no-cache": True}},
            key=key,
        )
        assert response.status_code == 200, response.text
        return response.json()["id"]

    with ThreadPoolExecutor(max_workers=32) as pool:
        returned: Final = frozenset(pool.map(request, ids))
    assert returned == frozenset(ids)
    return returned


def _collect(bucket: Wire, count_lines: bool, expected: int) -> tuple[Request, ...]:
    puts: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier PUTs

    def delivered() -> int:
        puts.extend(bucket.drain())
        return sum(len(put.body.splitlines()) if count_lines else 1 for put in puts)

    eventually(delivered, lambda total: total >= expected, seconds=30)
    return tuple(puts)


PER_REQUEST_KEY: Final = re.compile(rf"^/{BUCKET}/{PREFIX}/\d{{4}}-\d{{2}}-\d{{2}}/.+\.json$")
BATCH_KEY: Final = re.compile(
    rf"^/{BUCKET}/{PREFIX}/\d{{4}}-\d{{2}}-\d{{2}}/batch_\d{{2}}-\d{{2}}-\d{{2}}_[0-9a-f]{{32}}\.jsonl$"
)


@pytest.mark.covers("other.observability.s3_v2.flush_bounds_concurrent_puts_to_default_and_keeps_every_log")
def test_s3_v2_flush_bounds_concurrent_puts_to_the_default_of_sixteen(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3fan" + uuid.uuid4().hex[:8]
    sink: Final = S3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(candidate, model, key, marker)
            puts: Final = _collect(bucket, count_lines=False, expected=REQUESTS)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert sink.peak <= 16, f"peak concurrent PUTs {sink.peak} exceeded the default bound for {REQUESTS} queued logs"
    assert all(PER_REQUEST_KEY.match(put.target) for put in puts), [put.target for put in puts]
    assert frozenset(json.loads(put.body)["id"] for put in puts) == ids
    assert len({put.target for put in puts}) == REQUESTS


@pytest.mark.covers("other.observability.s3_v2.configured_bound_and_env_backed_false_keeps_per_request_objects")
def test_s3_v2_honors_configured_bound_and_env_backed_false_batch_flag(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3cap" + uuid.uuid4().hex[:8]
    sink: Final = S3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(
            tmp_path,
            bucket.url,
            {"s3_max_concurrent_uploads": 4, "s3_batch_file_upload": "os.environ/INTEGRATION_S3_BATCH_FILE_UPLOAD"},
        )
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3", "INTEGRATION_S3_BATCH_FILE_UPLOAD": "false"},
                config=config,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(candidate, model, key, marker)
            puts: Final = _collect(bucket, count_lines=False, expected=REQUESTS)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert sink.peak <= 4, f"peak concurrent PUTs {sink.peak} exceeded s3_max_concurrent_uploads=4"
    assert all(PER_REQUEST_KEY.match(put.target) for put in puts), [put.target for put in puts]
    assert frozenset(json.loads(put.body)["id"] for put in puts) == ids


@pytest.mark.covers("other.observability.s3_v2.batch_file_upload_writes_one_ndjson_object_per_flush")
def test_s3_v2_batch_file_upload_writes_one_jsonl_object_per_flush(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3jsonl" + uuid.uuid4().hex[:8]
    sink: Final = S3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": True})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(candidate, model, key, marker)
            puts: Final = _collect(bucket, count_lines=True, expected=REQUESTS)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert len(puts) <= 2, f"{len(puts)} PUTs for {REQUESTS} logs; batch mode must write one object per flush"
    assert all(BATCH_KEY.match(put.target) for put in puts), [put.target for put in puts]
    assert all(put.headers["content-type"] == "application/x-ndjson" for put in puts), [put.headers for put in puts]
    lines: Final = tuple(line for put in puts for line in put.body.decode().splitlines())
    assert frozenset(json.loads(line)["id"] for line in lines) == ids
    assert len(lines) == REQUESTS


@pytest.mark.covers("other.observability.s3_v2.batch_file_upload_keeps_team_prefix_in_object_key")
def test_s3_v2_batch_file_upload_keeps_team_alias_prefix(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3team" + uuid.uuid4().hex[:8]
    team_alias: Final = f"alpha-{uuid.uuid4().hex[:8]}"
    team_batch_key: Final = re.compile(
        rf"^/{BUCKET}/{PREFIX}/{team_alias}/\d{{4}}-\d{{2}}-\d{{2}}/batch_\d{{2}}-\d{{2}}-\d{{2}}_[0-9a-f]{{32}}\.jsonl$"
    )
    sink: Final = S3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": True, "s3_use_team_prefix": True})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            team: Final = scenario.team(team_alias=team_alias, models=[model])
            key: Final = scenario.key(team_id=team, models=[model])
            ids: Final = _burst(candidate, model, key, marker)
            puts: Final = _collect(bucket, count_lines=True, expected=REQUESTS)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert len(puts) >= 1
    assert all(team_batch_key.match(put.target) for put in puts), [put.target for put in puts]
    lines: Final = tuple(line for put in puts for line in put.body.decode().splitlines())
    assert frozenset(json.loads(line)["id"] for line in lines) == ids
    assert len(lines) == REQUESTS


@pytest.mark.covers("other.observability.s3_v2.upstream_failure_events_land_alongside_successes")
def test_s3_v2_upstream_failure_events_land_alongside_successes(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3fail" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()

    def provider(request: Request) -> Reply:
        text: Final = json.loads(request.body)["messages"][0]["content"]
        if text.endswith("-fail"):
            return Reply(
                status=401,
                body=b'{"error": {"message": "synthetic upstream rejection", "code": "synthetic_401"}}',
            )
        return _chat_reply(request)

    with wire_server(provider) as upstream, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=upstream.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            success_ids: Final = tuple(f"{marker}-{index}" for index in range(8))
            failure_ids: Final = tuple(f"{marker}-{index}-fail" for index in range(4))

            def send(identity: str) -> httpx.Response:
                return candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": identity}], "cache": {"no-cache": True}},
                    key=key,
                )

            with ThreadPoolExecutor(max_workers=12) as pool:
                responses: Final = tuple(pool.map(send, (*success_ids, *failure_ids)))
            ok: Final = responses[:8]
            rejected: Final = responses[8:]
            assert all(response.status_code == 200 for response in ok), [r.text for r in ok]
            assert tuple(response.json()["id"] for response in ok) == success_ids
            for response in rejected:
                assert response.status_code in (400, 401), response.status_code
                assert "synthetic upstream rejection" in response.text, response.text
            failure_call_ids: Final = frozenset(response.headers["x-litellm-call-id"] for response in rejected)
            payloads: Final = collect_payloads(sink, len(success_ids) + len(failure_ids))
    assert len(upstream.drain()) == len(success_ids) + len(failure_ids)
    delivered: Final = frozenset(payload["id"] for payload in payloads if payload["status"] == "success")
    assert delivered == frozenset(success_ids)
    failures: Final = tuple(payload for payload in payloads if payload["status"] == "failure")
    assert len(failures) == len(failure_ids)
    assert frozenset(payload["litellm_call_id"] for payload in failures) == failure_call_ids
    assert all("synthetic upstream rejection" in json.dumps(payload["error_information"]) for payload in failures)


@pytest.mark.covers("other.observability.s3_v2.invalid_or_empty_bound_falls_back_to_sixteen")
@pytest.mark.parametrize(
    ("bad", "warns"),
    [
        pytest.param("abc", True, id="non_integer"),
        pytest.param(0, True, id="below_one"),
        pytest.param("", False, id="empty"),
    ],
)
def test_s3_v2_invalid_or_empty_bound_falls_back_to_sixteen(
    gateway: Gateway, tmp_path: Path, bad: JsonValue, warns: bool
) -> None:
    marker: Final = "s3bound" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_max_concurrent_uploads": bad})
        with (
            owned_proxy_process(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(owned.gateway, model, key, marker)
            payloads: Final = collect_payloads(sink, REQUESTS)
            if warns:
                eventually(
                    lambda: owned.log.read_text(),
                    lambda text: "s3_max_concurrent_uploads" in text,
                    seconds=15,
                )
            else:
                assert "s3_max_concurrent_uploads" not in owned.log.read_text()
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert sink.peak <= 16, f"peak concurrent PUTs {sink.peak} exceeded the fallback bound"
    assert frozenset(payload["id"] for payload in payloads) == ids


@pytest.mark.covers("other.observability.s3_v2.sink_rejection_requeues_and_delivers_every_id_once")
def test_s3_v2_sink_rejection_requeues_and_delivers_every_id_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3deny" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink(fail_status=503, delay_seconds=0.2)
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy_process(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            sink.fail_until = time.time() + 10
            ids: Final = _burst(owned.gateway, model, key, marker)
            payloads: Final = collect_payloads(sink, REQUESTS, seconds=90)
            eventually(
                lambda: owned.log.read_text(),
                lambda text: "S3BatchUploadError" in text,
                seconds=15,
            )
            readiness: Final = owned.gateway.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert len(sink.objects()) == REQUESTS
    assert frozenset(payload["id"] for payload in payloads) == ids


@dataclass(slots=True)
class RejectingS3Sink:
    """Answers every PUT whose body carries `reject_marker` with `reject_status`, accepts the rest,
    and counts the rejected attempts so a test can see whether the proxy keeps re-sending them."""

    reject_marker: str
    reject_status: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    rejected_attempts: int = 0
    store: dict[str, bytes] = field(default_factory=dict)  # mutable-ok: later PUTs must be visible to earlier polls

    def respond(self, request: Request) -> Reply:
        assert request.method == "PUT", request.method
        with self.lock:
            if self.reject_marker.encode() in request.body:
                self.rejected_attempts += 1
                return Reply(status=self.reject_status, body=b"<Error><Code>AccessDenied</Code></Error>")
            self.store[request.target] = request.body
        return Reply()

    def landed_ids(self) -> frozenset[str]:
        with self.lock:
            bodies: Final = tuple(self.store.values())
        return frozenset(json.loads(line)["id"] for body in bodies for line in body.splitlines())


def _send(candidate: Gateway, model: str, key: str, identity: str) -> None:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": identity}], "cache": {"no-cache": True}},
        key=key,
    )
    assert response.status_code == 200, response.text


def _send_and_wait_until_landed(candidate: Gateway, model: str, key: str, sink: RejectingS3Sink, identity: str) -> None:
    _send(candidate, model, key, identity)
    eventually(sink.landed_ids, lambda landed: identity in landed, seconds=60)


def test_s3_v2_access_denied_object_is_put_once_and_never_requeued(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3denied" + uuid.uuid4().hex[:8]
    sink: Final = RejectingS3Sink(reject_marker=f"{marker}-denied", reject_status=403)
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": False})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            _send(candidate, model, key, f"{marker}-denied")
            _send_and_wait_until_landed(candidate, model, key, sink, f"{marker}-first-flush")
            _send_and_wait_until_landed(candidate, model, key, sink, f"{marker}-second-flush")
            _send_and_wait_until_landed(candidate, model, key, sink, f"{marker}-third-flush")
            readiness: Final = candidate.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
    assert sum(1 for r in provider.drain() if r.method == "POST") == 4
    assert sink.rejected_attempts == 1, (
        f"a 403 object was PUT {sink.rejected_attempts} times across three flushes; it must be attempted once and dropped"
    )


def test_s3_v2_persistently_unavailable_object_stops_being_retried_after_the_flush_budget(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3doomed" + uuid.uuid4().hex[:8]
    sink: Final = RejectingS3Sink(reject_marker=f"{marker}-doomed", reject_status=503)
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": False})
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "2", "DEFAULT_S3_MAX_FLUSH_ATTEMPTS": "2"},
                config=config,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            _send(owned.gateway, model, key, f"{marker}-doomed")
            eventually(
                lambda: owned.log.read_text(),
                lambda text: "dropped after 2 flush attempts" in text,
                seconds=60,
            )
            exhausted: Final = sink.rejected_attempts
            _send_and_wait_until_landed(owned.gateway, model, key, sink, f"{marker}-one-flush-later")
            _send_and_wait_until_landed(owned.gateway, model, key, sink, f"{marker}-two-flushes-later")
    assert sum(1 for r in provider.drain() if r.method == "POST") == 3
    assert 2 <= exhausted <= 2 * 3, f"{exhausted} PUTs for a budget of two flushes with at most three attempts each"
    assert sink.rejected_attempts == exhausted, (
        f"a 503 object kept being PUT after its flush budget: {exhausted} -> {sink.rejected_attempts}"
    )


@pytest.mark.covers("other.observability.s3_v2.batch_retry_resends_identical_key_and_body")
def test_s3_v2_batch_retry_resends_identical_key_and_body(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3retry" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink(fail_status=500, delay_seconds=0.2)
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": True})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            sink.fail_until = time.time() + 8
            ids: Final = _burst(candidate, model, key, marker)
            payloads: Final = collect_payloads(sink, REQUESTS, seconds=90)
    puts: Final = bucket.drain()
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    by_target: Final = {}
    for put in puts:
        by_target.setdefault(put.target, set()).add(put.body)  # mutable-ok: grouping attempts seen so far per target
    assert all(len(bodies) == 1 for bodies in by_target.values()), "a retried batch PUT changed key or body"
    assert max(sum(1 for put in puts if put.target == target) for target in by_target) >= 2, "no retried PUT observed"
    assert frozenset(payload["id"] for payload in payloads) == ids
    assert len(payloads) == REQUESTS


@pytest.mark.covers("other.observability.s3_v2.unknown_model_rejection_keeps_other_requests_logging")
def test_s3_v2_unknown_model_rejection_keeps_other_requests_logging(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3ghost" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ghost: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": f"ghost-{uuid.uuid4().hex}", "messages": [{"role": "user", "content": "hi"}]},
                key=key,
            )
            assert ghost.status_code in (400, 403, 404), ghost.text
            ids: Final = _burst(candidate, model, key, marker)
            eventually(
                lambda: frozenset(payload["id"] for payload in sink.payloads()),
                lambda landed: ids <= landed,
                seconds=90,
            )
            payloads: Final = sink.payloads()
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert ids <= frozenset(payload["id"] for payload in payloads)
    extras: Final = tuple(payload for payload in payloads if payload["id"] not in ids)
    assert all(payload["status"] == "failure" for payload in extras), extras


@pytest.mark.covers("other.observability.s3_v2.batch_flag_ignored_when_s3_v2_is_cold_storage_logger")
def test_s3_v2_batch_flag_ignored_when_s3_v2_is_cold_storage_logger(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3cold" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _recording_s3_config(
            tmp_path,
            bucket.url,
            {"s3_batch_file_upload": True},
            {"cold_storage_custom_logger": "s3_v2"},
        )
        with (
            owned_proxy_process(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
                key=key,
            )
            assert response.status_code == 200, response.text
            request_id: Final = str(response.json()["id"])
            payloads: Final = collect_payloads(sink, 1)
            assert all(PER_REQUEST_KEY.match(target) for target in sink.objects()), list(sink.objects())
            eventually(
                lambda: owned.log.read_text(),
                lambda text: "s3_batch_file_upload is ignored because s3_v2 is the cold storage logger" in text,
                seconds=15,
            )
            spend: Final = eventually(
                lambda: owned.gateway.request("GET", f"/spend/logs/ui/{request_id}"),
                lambda reply: reply.status_code == 200 and bool((reply.json() or {}).get("messages")),
                seconds=60,
            )
            assert spend.status_code == 200, spend.text
            body: Final = spend.json()
            assert body["messages"], spend.text
            assert body["response"], spend.text
    assert payloads[0]["id"] == request_id


@pytest.mark.covers("other.observability.s3_v2.identical_requests_land_distinct_objects")
def test_s3_v2_identical_requests_land_distinct_objects(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3same" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])

            def send(_: int) -> str:
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
                    key=key,
                )
                assert response.status_code == 200, response.text
                return str(response.json()["id"])

            with ThreadPoolExecutor(max_workers=16) as pool:
                returned: Final = frozenset(pool.map(send, range(16)))
            payloads: Final = collect_payloads(sink, 16)
    assert sum(1 for r in provider.drain() if r.method == "POST") == 16
    assert returned == {marker}, "the upstream echo keeps the same id for identical requests"
    assert len(sink.objects()) == 16, "identical requests must still land as distinct objects"
    assert all(payload["id"] == marker for payload in payloads)


@pytest.mark.covers("other.observability.s3_v2.two_workers_bound_and_deliver_every_id")
def test_s3_v2_two_workers_bound_and_deliver_every_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3work" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy(
                gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config, workers=2
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(candidate, model, key, marker)
            payloads: Final = collect_payloads(sink, REQUESTS)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert sink.peak <= 32, f"peak concurrent PUTs {sink.peak} exceeded two workers at the default bound"
    assert len(sink.objects()) == REQUESTS
    assert frozenset(payload["id"] for payload in payloads) == ids


@pytest.mark.covers("other.observability.s3_v2.slow_sink_never_duplicates_or_stalls_readiness")
def test_s3_v2_slow_sink_never_duplicates_or_stalls_readiness(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3slow" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink(delay_seconds=1.5)
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {"s3_batch_file_upload": True})
        with (
            owned_proxy(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "1"}, config=config) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            ids: Final = _burst(candidate, model, key, marker)

            def delivered() -> int:
                readiness: Final = candidate.client.get("/health/readiness")
                assert readiness.status_code == 200, readiness.text
                return sum(len(body.splitlines()) for body in sink.objects().values())

            eventually(delivered, lambda total: total >= REQUESTS, seconds=90)
            payloads: Final = sink.payloads()
    puts: Final = bucket.drain()
    targets: Final = tuple(put.target for put in puts)
    assert sum(1 for r in provider.drain() if r.method == "POST") == REQUESTS
    assert len(set(targets)) == len(targets), "the same object was PUT more than once"
    assert frozenset(payload["id"] for payload in payloads) == ids
    assert len(payloads) == REQUESTS


@pytest.mark.covers("other.observability.s3_v2.worker_kill_mid_burst_keeps_surviving_deliveries")
def test_s3_v2_worker_kill_mid_burst_keeps_surviving_deliveries(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3kill" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        with (
            owned_proxy_process(
                gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config, workers=2
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            sent: Final = tuple(f"{marker}-{index}" for index in range(REQUESTS))

            def send(identity: str) -> tuple[str, bool]:
                try:
                    response: Final = owned.gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": identity}],
                            "cache": {"no-cache": True},
                        },
                        key=key,
                    )
                except Exception:
                    return identity, False
                return identity, response.status_code == 200

            with ThreadPoolExecutor(max_workers=32) as pool:
                futures: Final = tuple(pool.submit(send, identity) for identity in sent)
                time.sleep(0.5)
                children: Final = tuple(
                    process for process in group_members(owned.process.pid) if process.pid != owned.process.pid
                )
                assert children, "no worker children found to kill"
                children[0].kill()
                results: Final = tuple(future.result() for future in futures)
            survivors: Final = frozenset(identity for identity, ok in results if ok)
            assert survivors, "no request survived the worker kill"
            readiness: Final = owned.gateway.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            payloads: Final = collect_payloads(sink, len(survivors), seconds=90)
    landed: Final = frozenset(payload["id"] for payload in payloads)
    assert survivors <= landed, "an id whose response succeeded never landed"
    assert landed <= frozenset(sent), "an id that was never sent landed"


@pytest.mark.covers("other.observability.s3_v2.sigterm_mid_burst_loses_only_inflight_without_duplicates")
def test_s3_v2_sigterm_mid_burst_loses_only_inflight_without_duplicates(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3term" + uuid.uuid4().hex[:8]
    sink: Final = RecordingS3Sink()
    with wire_server(_chat_reply) as provider, wire_server(sink.respond) as bucket:
        config: Final = _s3_config(tmp_path, bucket.url, {})
        owned: Final = owned_proxy_process(gateway, tmp_path, {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "3"}, config=config)
        candidate_owned: Final = owned.__enter__()
        try:
            created: Final = candidate_owned.gateway.post(
                "/model/new",
                {
                    "model_name": f"integration-{marker}",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "synthetic-provider-key",
                        "api_base": provider.url + "/v1",
                    },
                    "model_info": {},
                },
            )
            model: Final = str(created["model_name"])
            key: Final = str(candidate_owned.gateway.post("/key/generate", {"models": [model]})["key"])
            sent: Final = tuple(f"{marker}-{index}" for index in range(REQUESTS))

            def send(identity: str) -> tuple[str, bool]:
                try:
                    response: Final = candidate_owned.gateway.request(
                        "POST",
                        "/v1/chat/completions",
                        {
                            "model": model,
                            "messages": [{"role": "user", "content": identity}],
                            "cache": {"no-cache": True},
                        },
                        key=key,
                    )
                except Exception:
                    return identity, False
                return identity, response.status_code == 200

            with ThreadPoolExecutor(max_workers=32) as pool:
                futures: Final = tuple(pool.submit(send, identity) for identity in sent)
                time.sleep(0.5)
                candidate_owned.process.terminate()
                results: Final = tuple(future.result() for future in futures)
            candidate_owned.process.wait(timeout=30)
        finally:
            owned.__exit__(None, None, None)
    answered: Final = frozenset(identity for identity, ok in results if ok)
    landed: Final = frozenset(payload["id"] for payload in sink.payloads())
    assert landed <= answered, (
        "a delivered object has no matching answered request; lost in-flight ids are expected, extras are not"
    )
    targets: Final = tuple(sink.objects())
    assert len(set(targets)) == len(targets), "the same object was PUT more than once"
