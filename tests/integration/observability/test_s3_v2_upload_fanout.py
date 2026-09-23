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

import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually
from integration._support.process import owned_proxy
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
    assert len(provider.drain()) == REQUESTS
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
    assert len(provider.drain()) == REQUESTS
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
    assert len(provider.drain()) == REQUESTS
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
    assert len(provider.drain()) == REQUESTS
    assert len(puts) >= 1
    assert all(team_batch_key.match(put.target) for put in puts), [put.target for put in puts]
    lines: Final = tuple(line for put in puts for line in put.body.decode().splitlines())
    assert frozenset(json.loads(line)["id"] for line in lines) == ids
    assert len(lines) == REQUESTS
