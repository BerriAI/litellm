import json
import re
import threading
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final
from urllib.parse import quote, unquote

import httpx
import openai
import psutil
import pytest
import yaml
from _s3_v2_support import (
    BUCKET,
    PREFIX,
    SURFACES,
    RecordingS3Sink,
    call_surface,
    collect_payloads,
    matched_ids,
    mixed_burst,
    s3_config,
    surface_reply,
)
from integration._support.client import Gateway, JsonValue, Scenario, eventually, object_value
from integration._support.database import read_rows, scratch_database
from integration._support.database_relay import database_relay
from integration._support.process import OwnedProxy, group_members, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

FLUSH: Final = {"DEFAULT_S3_FLUSH_INTERVAL_SECONDS": "1"}
HOUR: Final = {"s3_partition_granularity": "hour"}
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
WARNING: Final = "s3 logging: s3_partition_granularity="
SINK_CREDENTIALS: Final = {
    "s3_bucket_name": BUCKET,
    "s3_region_name": "us-east-1",
    "s3_path": PREFIX,
    "s3_aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "s3_aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}


@dataclass(slots=True)
class CountingUpstream:
    """Scripted provider that answers every surface and fails any prompt ending in -fail with a 401."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    prompts: list[str] = field(default_factory=list)  # mutable-ok: appended per upstream request under lock

    def respond(self, request: Request) -> Reply:
        if request.method != "POST" or not request.body:
            return Reply(status=404)
        body: Final = json.loads(request.body)
        prompt: Final = str(body["input"] if "input" in body else body["messages"][0]["content"])
        with self.lock:
            self.prompts.append(prompt)
        if prompt.endswith("-fail"):
            return Reply(status=401, body=b'{"error": {"message": "synthetic upstream rejection", "code": "401"}}')
        return surface_reply(request)

    def received(self) -> tuple[str, ...]:
        with self.lock:
            return tuple(self.prompts)


def _prompt(payload: Mapping[str, JsonValue]) -> str:
    messages: Final = payload["messages"]
    if isinstance(messages, str):
        return messages
    assert isinstance(messages, list) and len(messages) == 1, payload
    first: Final = messages[0]
    return first if isinstance(first, str) else str(object_value(first)["content"])


def _start(payload: Mapping[str, JsonValue]) -> datetime:
    return datetime.fromtimestamp(float(str(payload["startTime"])))


def _folder(payload: Mapping[str, JsonValue], granularity: str, prefix: str = "") -> str:
    start: Final = _start(payload)
    hour: Final = f"{start:%H}/" if granularity == "hour" else ""
    return f"/{BUCKET}/{PREFIX}/{prefix}{start:%Y-%m-%d}/{hour}"


def _object_pattern(payload: Mapping[str, JsonValue], granularity: str, prefix: str = "") -> re.Pattern[str]:
    return re.compile(
        re.escape(_folder(payload, granularity, prefix)) + rf"time-{_start(payload):%H-%M-%S}-\d{{6}}_[^/]+\.json"
    )


def _outside_layout(objects: Mapping[str, bytes], granularity: str, prefix: str = "") -> tuple[str, ...]:
    return tuple(
        target
        for target, body in objects.items()
        if not _object_pattern(object_value(json.loads(body)), granularity, prefix).fullmatch(unquote(target))
    )


def _batches_outside_layout(objects: Mapping[str, bytes], granularity: str) -> tuple[str, ...]:
    def folders(body: bytes) -> frozenset[str]:
        return frozenset(_folder(object_value(json.loads(line)), granularity) for line in body.splitlines())

    return tuple(
        target
        for target, body in objects.items()
        if len(folders(body)) != 1
        or not re.fullmatch(
            re.escape(next(iter(folders(body)))) + r"batch_\d{2}-\d{2}-\d{2}_[0-9a-f]{32}\.jsonl", unquote(target)
        )
    )


@contextmanager
def _s3_proxy(
    gateway: Gateway,
    tmp_path: Path,
    sink_url: str,
    extra: Mapping[str, JsonValue],
    settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    workers: int = 2,
    models: tuple[Mapping[str, JsonValue], ...] = (),
) -> Iterator[OwnedProxy]:
    config: Final = s3_config(tmp_path, sink_url, extra, settings)
    if models:
        declared: Final = yaml.safe_load(config.read_text())
        config.write_text(yaml.safe_dump({**declared, "model_list": [*declared["model_list"], *models]}))
    with owned_proxy_process(
        gateway, tmp_path, {**FLUSH, **(environment or {})}, config=config, workers=workers
    ) as owned:
        yield owned


def _models(scenario: Scenario, provider_url: str, **key_fields: JsonValue) -> tuple[str, str, str]:
    openai_model: Final = scenario.model(api_base=provider_url + "/v1", api_key="synthetic-provider-key")
    anthropic_model: Final = scenario.model(
        model=ANTHROPIC_MODEL, api_base=provider_url, api_key="synthetic-provider-key"
    )
    return openai_model, anthropic_model, scenario.key(models=[openai_model, anthropic_model], **key_fields)


def _config_model(name: str, model: str, api_base: str) -> Mapping[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": "synthetic-provider-key"},
    }


def _sdk_chats(candidate: Gateway, model: str, key: str, prompts: tuple[str, ...]) -> tuple[str, ...]:
    client: Final = openai.OpenAI(base_url=f"{str(candidate.client.base_url).rstrip('/')}/v1", api_key=key)

    def send(prompt: str) -> str:
        reply: Final = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], extra_body={"cache": {"no-cache": True}}
        )
        assert reply.choices[0].finish_reason == "stop", reply.model_dump_json()
        return reply.id

    with ThreadPoolExecutor(max_workers=16) as pool:
        return tuple(pool.map(send, prompts))


def _surface_prompts(marker: str, per_surface: int) -> frozenset[str]:
    return frozenset(f"{marker}-{surface}-{index}" for surface in SURFACES for index in range(per_surface))


def _cold_storage_key(request_id: str, database_url: str | None = None) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,), database_url=database_url
        ),
        lambda values: len(values) == 1,
        seconds=60,
    )
    metadata: Final = rows[0]["metadata"]
    return str(object_value(json.loads(metadata) if isinstance(metadata, str) else metadata)["cold_storage_object_key"])


def _update_environment(candidate: Gateway, values: Mapping[str, JsonValue]) -> None:
    candidate.post(
        "/config/update",
        {"environment_variables": dict(values), "litellm_settings": {"success_callback": ["s3_v2"]}},
    )


def _keys_on_fresh_connections(candidate: Gateway, aliases: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    def generate(alias: str) -> tuple[str, str]:
        with httpx.Client(base_url=candidate.client.base_url, timeout=30, trust_env=False) as fresh:
            response: Final = fresh.post(
                "/key/generate",
                json={"key_alias": alias},
                headers={"Authorization": f"Bearer {candidate.key}", "Connection": "close"},
            )
        assert response.status_code == 200, response.text
        return str(response.json()["key"]), str(response.json()["token_id"])

    with ThreadPoolExecutor(max_workers=len(aliases)) as pool:
        return tuple(pool.map(generate, aliases))


def _created_key_hashes(sink: RecordingS3Sink, audit_prefix: str) -> frozenset[str]:
    created: Final = (
        object_value(json.loads(body)) for target, body in sink.objects().items() if target.startswith(audit_prefix)
    )
    return frozenset(
        str(audit["object_id"])
        for audit in created
        if audit["action"] == "created" and audit["table_name"] == "LiteLLM_VerificationToken"
    )


def test_s3_v2_hour_granularity_files_every_surface_under_its_hour_folder(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hour" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, anthropic_model, key = _models(scenario, provider.url)
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=2)
        payloads: Final = collect_payloads(sink, len(answered))
        objects: Final = sink.objects()
        log: Final = owned.log.read_text()
    sent: Final = _surface_prompts(marker, 2)
    assert len(answered) == len(sent) and len(payloads) == len(sent), payloads
    assert matched_ids(payloads, answered) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(upstream.received()) == sorted(sent)
    assert len(objects) == len(sent)
    assert sorted(_prompt(payload) for payload in payloads) == sorted(sent)
    assert all(payload["status"] == "success" for payload in payloads), payloads
    assert _outside_layout(objects, "hour") == (), "every object must sit in YYYY-MM-DD/HH/ of its start time"
    assert WARNING not in log


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({}, id="missing"),
        pytest.param({"s3_partition_granularity": "day"}, id="day"),
        pytest.param({"s3_partition_granularity": ""}, id="empty"),
        pytest.param({"s3_partition_granularity": None}, id="null"),
    ],
)
def test_s3_v2_missing_day_empty_or_null_granularity_keeps_the_daily_layout(
    gateway: Gateway, tmp_path: Path, extra: Mapping[str, JsonValue]
) -> None:
    marker: Final = "s3day" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, extra) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, anthropic_model, key = _models(scenario, provider.url)
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=1)
        payloads: Final = collect_payloads(sink, len(answered))
        objects: Final = sink.objects()
        log: Final = owned.log.read_text()
    sent: Final = _surface_prompts(marker, 1)
    assert len(answered) == len(sent) and len(payloads) == len(sent), payloads
    assert matched_ids(payloads, answered) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(upstream.received()) == sorted(sent)
    assert sorted(_prompt(payload) for payload in payloads) == sorted(sent)
    assert len(objects) == len(sent)
    assert _outside_layout(objects, "day") == ()
    assert WARNING not in log


@pytest.mark.parametrize(
    ("extra", "environment", "shown"),
    [
        pytest.param({"s3_partition_granularity": "hourly"}, {}, "'hourly'", id="unknown_word"),
        pytest.param({"s3_partition_granularity": "HOUR"}, {}, "'HOUR'", id="wrong_case"),
        pytest.param({"s3_partition_granularity": 1}, {}, "1", id="integer"),
        pytest.param({"s3_partition_granularity": ["hour"]}, {}, "['hour']", id="list"),
        pytest.param({"s3_partition_granularity": "h" * 5120}, {}, "'[base64_data truncated: 3.8KB]'", id="five_kb"),
        pytest.param({}, {"S3_PARTITION_GRANULARITY": "weekly"}, "'weekly'", id="env_unknown_word"),
    ],
)
def test_s3_v2_unrecognized_granularity_warns_once_per_worker_and_keeps_the_daily_layout(
    gateway: Gateway, tmp_path: Path, extra: Mapping[str, JsonValue], environment: Mapping[str, str], shown: str
) -> None:
    marker: Final = "s3bad" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, extra, environment=environment) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, anthropic_model, key = _models(scenario, provider.url)
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=2)
        payloads: Final = collect_payloads(sink, len(answered))
        objects: Final = sink.objects()
        warning: Final = f"{WARNING}{shown} is not one of day, hour, using day"
        log: Final = eventually(owned.log.read_text, lambda text: warning in text, seconds=15)
    sent: Final = _surface_prompts(marker, 2)
    assert len(answered) == len(sent) and len(payloads) == len(sent), payloads
    assert matched_ids(payloads, answered) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(upstream.received()) == sorted(sent)
    assert sorted(_prompt(payload) for payload in payloads) == sorted(sent)
    assert _outside_layout(objects, "day") == ()
    assert 1 <= log.count(warning) <= 2, "the warning is memoized per distinct value in each of the two workers"


@pytest.mark.parametrize(
    ("extra", "environment", "granularity"),
    [
        pytest.param({}, {"S3_PARTITION_GRANULARITY": "hour"}, "hour", id="env_hour_applies"),
        pytest.param({"s3_partition_granularity": "day"}, {"S3_PARTITION_GRANULARITY": "hour"}, "day", id="yaml_wins"),
    ],
)
def test_s3_v2_env_granularity_applies_only_when_callback_params_leave_it_unset(
    gateway: Gateway, tmp_path: Path, extra: Mapping[str, JsonValue], environment: Mapping[str, str], granularity: str
) -> None:
    marker: Final = "s3env" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, extra, environment=environment) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(8))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        payloads: Final = collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
    assert returned == prompts
    assert sorted(upstream.received()) == sorted(prompts)
    assert frozenset(str(payload["id"]) for payload in payloads) == frozenset(prompts)
    assert _outside_layout(objects, granularity) == ()


def test_s3_v2_hour_batch_files_group_lines_under_the_hour_folder(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hbat" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, {**HOUR, "s3_batch_file_upload": True}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, anthropic_model, key = _models(scenario, provider.url)
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=4)
        payloads: Final = collect_payloads(sink, len(answered))
        objects: Final = sink.objects()
    sent: Final = _surface_prompts(marker, 4)
    assert len(answered) == len(sent) and len(payloads) == len(sent), payloads
    assert matched_ids(payloads, answered) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(upstream.received()) == sorted(sent)
    assert sorted(_prompt(payload) for payload in payloads) == sorted(sent)
    assert _batches_outside_layout(objects, "hour") == ()


def test_s3_v2_hour_folder_sits_below_the_team_and_key_prefix(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hpre" + uuid.uuid4().hex[:8]
    team_alias: Final = f"alpha-{uuid.uuid4().hex[:8]}"
    key_alias: Final = f"beta-{uuid.uuid4().hex[:8]}"
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    extra: Final = {**HOUR, "s3_use_team_prefix": True, "s3_use_key_prefix": True}
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, extra) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
        team: Final = scenario.team(team_alias=team_alias, models=[openai_model])
        key: Final = scenario.key(team_id=team, key_alias=key_alias, models=[openai_model])
        prompts: Final = tuple(f"{marker}-{index}" for index in range(6))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        payloads: Final = collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
    assert returned == prompts
    assert sorted(upstream.received()) == sorted(prompts)
    assert frozenset(str(payload["id"]) for payload in payloads) == frozenset(prompts)
    assert _outside_layout(objects, "hour", f"{team_alias}/{key_alias}/") == ()


def _payload_values(payloads: tuple[dict[str, JsonValue], ...], status: str, field: str) -> frozenset[str]:
    return frozenset(str(payload[field]) for payload in payloads if payload["status"] == status)


def test_s3_v2_hour_failure_and_rejected_requests_keep_the_hour_layout(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hfail" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)

        def send(prompt: str, model: str = openai_model, caller: str = key) -> httpx.Response:
            return owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": prompt}], "cache": {"no-cache": True}},
                key=caller,
            )

        successes: Final = tuple(f"{marker}-{index}" for index in range(4))
        failures: Final = tuple(f"{marker}-{index}-fail" for index in range(3))
        with ThreadPoolExecutor(max_workers=8) as pool:
            responses: Final = tuple(pool.map(send, (*successes, *failures)))
        ghost: Final = send(f"{marker}-ghost", model=f"ghost-{uuid.uuid4().hex}")
        unauthenticated: Final = send(f"{marker}-anon", caller="sk-not-a-real-key")
        after: Final = send(f"{marker}-after")
        rejected_call_ids: Final = frozenset(response.headers["x-litellm-call-id"] for response in responses[4:])
        payloads: Final = eventually(
            sink.payloads,
            lambda stored: (
                _payload_values(stored, "success", "id") >= frozenset((*successes, f"{marker}-after"))
                and _payload_values(stored, "failure", "litellm_call_id") >= rejected_call_ids
            ),
            seconds=60,
        )
        objects: Final = sink.objects()
    assert [response.status_code for response in responses[:4]] == [200] * 4, [r.text for r in responses]
    assert tuple(response.json()["id"] for response in responses[:4]) == successes
    assert all(response.status_code == 401 for response in responses[4:]), [r.text for r in responses[4:]]
    assert all("synthetic upstream rejection" in response.text for response in responses[4:])
    assert ghost.status_code == 403 and "key_model_access_denied" in ghost.text, ghost.text
    assert unauthenticated.status_code == 401 and "error" in unauthenticated.json(), unauthenticated.text
    assert after.status_code == 200 and after.json()["id"] == f"{marker}-after", after.text
    assert sorted(upstream.received()) == sorted((*successes, *failures, f"{marker}-after"))
    assert _payload_values(payloads, "success", "id") == frozenset((*successes, f"{marker}-after"))
    assert _payload_values(payloads, "failure", "litellm_call_id") >= rejected_call_ids
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_cache_hit_twins_land_one_object_each_under_the_hour_folder(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3hcache" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, anthropic_model, key = _models(scenario, provider.url)
        first: Final = tuple(
            call_surface(owned.gateway, surface, openai_model, anthropic_model, key, f"{marker}-{surface}", False)
            for surface in ("chat", "responses")
        )
        eventually(lambda: len(sink.objects()), lambda count: count >= 2, seconds=30)
        repeated: Final = tuple(
            call_surface(owned.gateway, surface, openai_model, anthropic_model, key, f"{marker}-{surface}", False)
            for surface in ("chat", "responses")
        )
        payloads: Final = collect_payloads(sink, 4)
        objects: Final = sink.objects()
    assert first[0][0] == f"{marker}-chat" and repeated[0][0] == first[0][0]
    assert matched_ids(payloads, first + repeated) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(_prompt(payload) for payload in payloads) == sorted((f"{marker}-chat", f"{marker}-responses") * 2)
    assert sorted(upstream.received()) == sorted((f"{marker}-chat", f"{marker}-responses"))
    assert len(objects) == 4, list(objects)
    assert sum(1 for payload in payloads if payload["cache_hit"] is True) == 2
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_cold_storage_key_names_the_uploaded_object_and_reads_back(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hcold" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR, {"cold_storage_custom_logger": "s3_v2"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = (f"{marker}-kept", f"{marker}-missing")
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
        keys: Final = {prompt: _cold_storage_key(prompt) for prompt in prompts}
        with sink.lock:
            sink.store.pop(f"/{BUCKET}/{quote(keys[prompts[1]], safe='/')}")
        kept: Final = eventually(
            lambda: owned.gateway.request("GET", f"/spend/logs/ui/{prompts[0]}"),
            lambda reply: reply.status_code == 200 and bool((reply.json() or {}).get("messages")),
            seconds=30,
        )
        missing: Final = owned.gateway.request("GET", f"/spend/logs/ui/{prompts[1]}")
    assert returned == prompts
    assert sorted(upstream.received()) == sorted(prompts)
    assert frozenset(f"/{BUCKET}/{quote(key, safe='/')}" for key in keys.values()) == frozenset(objects)
    assert _outside_layout(objects, "hour") == ()
    assert kept.json()["messages"] == [{"role": "user", "content": prompts[0]}], kept.text
    assert prompts[0] in json.dumps(kept.json()["response"]), kept.text
    assert missing.status_code == 200, missing.text
    assert prompts[1] not in json.dumps(missing.json()["response"]), missing.text


def test_s3_v2_hour_layout_holds_when_another_logger_owns_cold_storage(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hgcs" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    lock: Final = threading.Lock()
    puts: Final[dict[str, bytes]] = {}  # mutable-ok: filled per PUT by the bucket thread under lock

    def bucket_reply(request: Request) -> Reply:
        assert request.method == "PUT", request.method
        with lock:
            puts[unquote(request.target)] = request.body
        return Reply(status=200)

    def uploaded() -> Mapping[str, bytes]:
        with lock:
            return dict(puts)

    with (
        wire_server(upstream.respond) as provider,
        wire_server(bucket_reply) as bucket,
        _s3_proxy(
            gateway, tmp_path, bucket.url, {**HOUR, "s3_path": ""}, {"cold_storage_custom_logger": "gcs_bucket"}
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(3))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        objects: Final = eventually(uploaded, lambda values: len(values) >= len(prompts), seconds=60)
        cold_keys: Final = tuple(_cold_storage_key(prompt) for prompt in prompts)
    hour_object: Final = re.compile(rf"/{BUCKET}/\d{{4}}-\d{{2}}-\d{{2}}/(\d{{2}})/time-(\d{{2}})-[^/]+\.json")
    matches: Final = tuple(hour_object.fullmatch(target) for target in objects)
    assert returned == prompts
    assert sorted(str(object_value(json.loads(body))["id"]) for body in objects.values()) == sorted(prompts)
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}/time-[^/]+\.json", cold_key) for cold_key in cold_keys), cold_keys
    assert all(match is not None and match.group(1) == match.group(2) for match in matches), sorted(objects)


def test_s3_v2_hour_cold_storage_rebuilds_previous_response_id_history_from_the_hour_object(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3hsess" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    histories: Final[list[str]] = []  # mutable-ok: appended per upstream request by the scripted provider thread
    reads: Final[list[str]] = []  # mutable-ok: appended per sink GET by the recording sink thread
    sink: Final = RecordingS3Sink(delay_seconds=0.05)

    def provider_reply(request: Request) -> Reply:
        histories.append(request.body.decode())
        return upstream.respond(request)

    def bucket_reply(request: Request) -> Reply:
        if request.method == "GET":
            reads.append(unquote(request.target))
        return sink.respond(request)

    with (
        wire_server(provider_reply) as provider,
        wire_server(bucket_reply) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR, {"cold_storage_custom_logger": "s3_v2"}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        _, anthropic_model, key = _models(scenario, provider.url)
        first: Final = owned.gateway.request(
            "POST", "/v1/responses", {"model": anthropic_model, "input": f"{marker}-first"}, key=key
        )
        assert first.status_code == 200, first.text
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (anthropic_model,)
            ),
            lambda values: len(values) == 1,
            seconds=60,
        )
        metadata: Final = rows[0]["metadata"]
        cold_key: Final = str(
            object_value(json.loads(metadata) if isinstance(metadata, str) else metadata)["cold_storage_object_key"]
        )
        eventually(sink.objects, lambda objects: f"/{BUCKET}/{quote(cold_key, safe='/')}" in objects, seconds=30)
        second: Final = owned.gateway.request(
            "POST",
            "/v1/responses",
            {"model": anthropic_model, "input": f"{marker}-second", "previous_response_id": first.json()["id"]},
            key=key,
        )
        objects: Final = sink.objects()
    assert second.status_code == 200, second.text
    assert second.json()["id"] != first.json()["id"], second.text
    assert re.fullmatch(rf"{re.escape(PREFIX)}/\d{{4}}-\d{{2}}-\d{{2}}/\d{{2}}/time-[^/]+\.json", cold_key), cold_key
    assert _outside_layout(objects, "hour") == ()
    assert f"/{BUCKET}/{cold_key}" in reads, reads
    assert len(histories) == 2, histories
    assert f"{marker}-first" in histories[0] and f"{marker}-second" not in histories[0], histories[0]
    assert f"{marker}-first" in histories[1] and f"{marker}-second" in histories[1], histories[1]


def test_s3_v2_audit_logs_follow_the_audit_params_granularity_not_the_request_logs(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3haudit" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with wire_server(upstream.respond) as provider, wire_server(sink.respond) as bucket:
        settings: Final = {
            "store_audit_logs": True,
            "audit_log_callbacks": ["s3_v2"],
            "s3_audit_callback_params": {**SINK_CREDENTIALS, "s3_endpoint_url": bucket.url, **HOUR},
        }
        with (
            _s3_proxy(gateway, tmp_path, bucket.url, {}, settings) as owned,
            owned.gateway.scenario() as scenario,
        ):
            openai_model, _, key = _models(scenario, provider.url, key_alias=marker)
            returned: Final = _sdk_chats(owned.gateway, openai_model, key, (marker,))
            aliases: Final = tuple(f"{marker}-fresh{index}" for index in range(16))
            fresh_keys: Final = _keys_on_fresh_connections(owned.gateway, aliases)
            audit_prefix: Final = f"/{BUCKET}/{PREFIX}/audit_logs/"
            eventually(
                lambda: _created_key_hashes(sink, audit_prefix),
                lambda created: frozenset(token for _, token in fresh_keys) <= created,
                seconds=30,
            )
            owned.gateway.post("/key/delete", {"keys": [key for key, _ in fresh_keys]})
            collect_payloads(sink, 2)
            objects: Final = sink.objects()
    audits: Final = {
        target: object_value(json.loads(body)) for target, body in objects.items() if target.startswith(audit_prefix)
    }
    requests: Final = {target: body for target, body in objects.items() if not target.startswith(audit_prefix)}
    assert returned == (marker,)
    assert upstream.received() == (marker,)
    assert _outside_layout(requests, "day") == ()
    created: Final = tuple(audit for audit in audits.values() if audit["action"] == "created")
    assert "LiteLLM_VerificationToken" in frozenset(str(audit["table_name"]) for audit in created), audits
    for target, audit in audits.items():
        located: Final = re.fullmatch(
            re.escape(audit_prefix)
            + rf"(\d{{4}}-\d{{2}}-\d{{2}})/(\d{{2}})/(\d{{2}})-\d{{2}}-\d{{2}}_{re.escape(str(audit['id']))}\.json",
            unquote(target),
        )
        assert located and located[2] == located[3], (target, audit["updated_at"])
        folder: Final = datetime.fromisoformat(f"{located[1]}T{located[2]}:00:00+00:00")
        updated: Final = datetime.fromisoformat(str(audit["updated_at"]))
        assert timedelta(0) < folder + timedelta(hours=1) - updated <= timedelta(hours=1, minutes=1), (
            target,
            audit["updated_at"],
        )


@pytest.mark.parametrize("level", ["key", "team"])
def test_s3_v2_key_and_team_logging_callback_vars_cannot_change_the_proxy_hour_layout(
    gateway: Gateway, tmp_path: Path, level: str
) -> None:
    marker: Final = f"s3h{level}vars" + uuid.uuid4().hex[:8]
    logging: Final[list[JsonValue]] = [
        {"callback_name": "s3_v2", "callback_type": "success", "callback_vars": {"s3_partition_granularity": "day"}}
    ]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, _ = _models(scenario, provider.url)
        key: Final = (
            scenario.key(models=[openai_model], metadata={"logging": logging})
            if level == "key"
            else scenario.key(models=[openai_model], team_id=scenario.team(metadata={"logging": logging}))
        )
        prompts: Final = tuple(f"{marker}-{index}" for index in range(8))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
    assert returned == prompts
    assert sorted(upstream.received()) == sorted(prompts)
    assert sorted(str(object_value(json.loads(body))["id"]) for body in objects.values()) == sorted(prompts), (
        f"{level}-level s3_v2 logging must land exactly one object per request"
    )
    assert _outside_layout(objects, "hour") == (), f"{level}-level callback_vars must not change the proxy granularity"


def test_s3_v2_admin_ui_granularity_update_moves_live_traffic_on_both_workers(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hui" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        scratch_database() as database_url,
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, {}, environment={"DATABASE_URL": database_url}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        before: Final = _sdk_chats(owned.gateway, openai_model, key, (f"{marker}-before",))
        eventually(lambda: len(sink.objects()), lambda count: count >= 1, seconds=30)
        listed: Final = owned.gateway.get("/get/config/callbacks")
        _update_environment(owned.gateway, {"callback": "s3_v2", "s3_partition_granularity": "hour"})
        probe_round: Final = iter(range(1000))

        def probe() -> Mapping[str, bytes]:
            round_id: Final = next(probe_round)
            prompts: Final = tuple(f"{marker}-probe{round_id}-{index}" for index in range(8))
            _sdk_chats(owned.gateway, openai_model, key, prompts)
            eventually(
                lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
                lambda landed: frozenset(prompts) <= landed,
                seconds=20,
            )
            return {target: body for target, body in sink.objects().items() if f"-probe{round_id}-" in target}

        eventually(probe, lambda probed: len(probed) == 8 and _outside_layout(probed, "hour") == (), seconds=60)
        prompts: Final = tuple(f"{marker}-after-{index}" for index in range(16))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        eventually(
            lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
            lambda landed: frozenset(prompts) <= landed,
            seconds=30,
        )
        after: Final = {target: body for target, body in sink.objects().items() if f"{marker}-after-" in target}
        before_objects: Final = {
            target: body for target, body in sink.objects().items() if f"{marker}-before" in target
        }
        readback: Final = owned.gateway.get("/get/config/callbacks")
    s3_rows: Final = tuple(row for row in listed["callbacks"] if object_value(row)["name"] in ("s3", "s3_v2"))
    assert s3_rows and all(
        "S3_PARTITION_GRANULARITY" in object_value(object_value(row)["variables"]) for row in s3_rows
    ), listed
    after_rows: Final = tuple(row for row in readback["callbacks"] if object_value(row)["name"] in ("s3", "s3_v2"))
    assert all(
        object_value(object_value(row)["variables"])["S3_PARTITION_GRANULARITY"] == "hour" for row in after_rows
    ), readback
    assert before == (f"{marker}-before",)
    assert returned == prompts
    assert _outside_layout(before_objects, "day") == ()
    assert len(after) == len(prompts)
    assert _outside_layout(after, "hour") == ()


def test_s3_v2_granularity_toggles_mid_burst_keep_every_cold_storage_key_on_its_object(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3htog" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        scratch_database() as database_url,
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(
            gateway,
            tmp_path,
            bucket.url,
            {},
            {"cold_storage_custom_logger": "s3_v2"},
            environment={"DATABASE_URL": database_url},
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(32))
        with ThreadPoolExecutor(max_workers=1) as burst:
            pending: Final = burst.submit(_sdk_chats, owned.gateway, openai_model, key, prompts)
            for value in ("hour", "day", "hour", "day", "hour", "day"):
                _update_environment(owned.gateway, {"s3_partition_granularity": value})
            returned: Final = pending.result()
        collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
        keys: Final = {prompt: _cold_storage_key(prompt, database_url) for prompt in prompts}
    assert returned == prompts
    assert sorted(upstream.received()) == sorted(prompts)
    assert len(objects) == len(prompts)
    assert frozenset(f"/{BUCKET}/{quote(key, safe='/')}" for key in keys.values()) == frozenset(objects), (
        "every spend log cold_storage_object_key must name the object the logger uploaded"
    )
    assert all(
        _object_pattern(object_value(json.loads(body)), "hour").fullmatch(unquote(target))
        or _object_pattern(object_value(json.loads(body)), "day").fullmatch(unquote(target))
        for target, body in objects.items()
    )


def test_s3_v2_in_flight_request_keeps_its_cold_storage_key_on_its_object_across_owner_and_granularity_switches(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3hflight" + uuid.uuid4().hex[:8]
    held_prompt: Final = f"{marker}-held"
    upstream: Final = CountingUpstream()
    arrived: Final = threading.Event()
    release: Final = threading.Event()

    def held(request: Request) -> Reply:
        if held_prompt.encode() in request.body:
            arrived.set()
            assert release.wait(90), "held request was never released"
        return upstream.respond(request)

    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        scratch_database() as database_url,
        wire_server(held) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(
            gateway,
            tmp_path,
            bucket.url,
            {},
            {"cold_storage_custom_logger": "s3_v2"},
            environment={"DATABASE_URL": database_url},
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        with ThreadPoolExecutor(max_workers=1) as flight:
            pending: Final = flight.submit(_sdk_chats, owned.gateway, openai_model, key, (held_prompt,))
            assert arrived.wait(60), "held request never reached the upstream"
            owner_switch: Final = owned.gateway.request(
                "POST", "/config/update", {"litellm_settings": {"cold_storage_custom_logger": "gcs_bucket"}}
            )
            _update_environment(owned.gateway, HOUR)
            probe_round: Final = iter(range(1000))

            def probe() -> Mapping[str, bytes]:
                round_id: Final = next(probe_round)
                prompts: Final = tuple(f"{marker}-probe{round_id}-{index}" for index in range(8))
                _sdk_chats(owned.gateway, openai_model, key, prompts)
                eventually(
                    lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
                    lambda landed: frozenset(prompts) <= landed,
                    seconds=20,
                )
                return {target: body for target, body in sink.objects().items() if f"-probe{round_id}-" in target}

            eventually(probe, lambda probed: len(probed) == 8 and _outside_layout(probed, "hour") == (), seconds=60)
            release.set()
            returned: Final = pending.result()
        eventually(
            lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
            lambda landed: held_prompt in landed,
            seconds=30,
        )
        held_objects: Final = {target: body for target, body in sink.objects().items() if held_prompt in target}
        cold_key: Final = _cold_storage_key(held_prompt, database_url)
    assert owner_switch.status_code == 400, owner_switch.text
    assert "cold_storage_custom_logger" in owner_switch.text and "config file" in owner_switch.text, owner_switch.text
    assert returned == (held_prompt,)
    assert upstream.received().count(held_prompt) == 1
    assert frozenset(held_objects) == frozenset({f"/{BUCKET}/{quote(cold_key, safe='/')}"}), (
        "the in-flight request's cold_storage_object_key must name the one object the logger uploaded",
        cold_key,
        tuple(held_objects),
    )
    assert _outside_layout(held_objects, "hour") == ()


def test_s3_v2_cold_storage_owner_saved_through_config_update_is_not_applied_to_a_running_proxy(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3howner" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink()
    with (
        scratch_database() as database_url,
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR, environment={"DATABASE_URL": database_url}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        saved: Final = owned.gateway.request(
            "POST", "/config/update", {"litellm_settings": {"cold_storage_custom_logger": "s3_v2"}}
        )
        prompts: Final = tuple(f"{marker}-{index}" for index in range(8))
        answered: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        landed: Final = collect_payloads(sink, len(prompts))
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
                (list(answered),),
                database_url=database_url,
            ),
            lambda values: len(values) == len(prompts),
            seconds=60,
        )
        objects: Final = sink.objects()
        stored: Final = read_rows(
            'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
            ("litellm_settings",),
            database_url=database_url,
        )
    cold_keys: Final = {
        str(row["request_id"]): object_value(
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        ).get("cold_storage_object_key")
        for row in rows
    }
    assert saved.status_code == 200, saved.text
    assert [
        object_value(json.loads(row["param_value"]) if isinstance(row["param_value"], str) else row["param_value"]).get(
            "cold_storage_custom_logger"
        )
        for row in stored
    ] == ["s3_v2"], "the owner switch must be persisted, so the unchanged live keys are not a rejected write"
    assert sorted(upstream.received()) == sorted(prompts)
    assert sorted(_prompt(payload) for payload in landed) == sorted(prompts)
    assert cold_keys == dict.fromkeys(answered), "a DB-saved cold storage owner must not change a live request"
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_postgres_outage_mid_mixed_burst_lands_every_id_exactly_once_and_recovers(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3hpg" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    sent: Final = _surface_prompts(marker, 5)
    openai_model: Final = f"{marker}openai"
    anthropic_model: Final = f"{marker}anthropic"
    with (
        scratch_database() as database_url,
        database_relay(database_url, f"{marker}-".encode()) as (relay, relayed_url),
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(
            gateway,
            tmp_path,
            bucket.url,
            HOUR,
            {"cold_storage_custom_logger": "s3_v2"},
            environment={"DATABASE_URL": relayed_url},
            models=(
                _config_model(openai_model, "openai/gpt-4o-mini", provider.url + "/v1"),
                _config_model(anthropic_model, ANTHROPIC_MODEL, provider.url),
            ),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        key: Final = scenario.key(models=[openai_model, anthropic_model])
        warm: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, f"{marker}warm", per_surface=2)
        eventually(
            lambda: frozenset(_prompt(payload) for payload in sink.payloads()),
            lambda landed: _surface_prompts(f"{marker}warm", 2) <= landed,
            seconds=60,
        )
        relay.arm()
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=5)
        assert relay.tripped.wait(90), "no spend log write reached the database during the burst"
        eventually(lambda: relay.refused, lambda count: count >= 1, seconds=30)
        assert relay.reconnected.wait(60), "the proxy never reconnected to the database after the outage"
        burst_payloads: Final = eventually(
            lambda: tuple(payload for payload in sink.payloads() if _prompt(payload) in sent),
            lambda landed: frozenset(_prompt(payload) for payload in landed) == frozenset(sent),
            seconds=60,
        )
        recovered_prompt: Final = f"{marker}-recovered"
        recovered: Final = _sdk_chats(owned.gateway, openai_model, key, (recovered_prompt,))
        recovered_key: Final = _cold_storage_key(recovered_prompt, database_url)
        eventually(
            lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
            lambda landed: recovered_prompt in landed,
            seconds=30,
        )
        objects: Final = sink.objects()
        uploads: Final = sink.attempts
    burst: Final = burst_payloads
    assert len(warm) == len(_surface_prompts(f"{marker}warm", 2))
    assert len(answered) == len(sent) == 30
    assert sorted(prompt for prompt in upstream.received() if prompt.startswith(f"{marker}-")) == sorted(
        (*sent, recovered_prompt)
    )
    assert matched_ids(burst, answered) == frozenset(str(payload["id"]) for payload in burst)
    assert sorted(_prompt(payload) for payload in burst) == sorted(sent), "every burst id lands exactly once"
    assert uploads == len(objects), "no object is uploaded twice"
    assert _outside_layout(objects, "hour") == ()
    assert recovered == (recovered_prompt,)
    assert f"/{BUCKET}/{quote(recovered_key, safe='/')}" in objects, "cold key written after recovery names its object"


def test_legacy_s3_callback_ignores_hour_granularity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3v1hour" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR, {"callbacks": [], "success_callback": ["s3"]}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(3))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        payloads: Final = collect_payloads(sink, len(prompts))
        objects: Final = sink.objects()
    assert returned == prompts
    assert frozenset(str(payload["id"]) for payload in payloads) == frozenset(prompts)
    assert _outside_layout(objects, "day") == (), "legacy s3 keeps the daily layout, the setting is s3_v2 only"


def test_s3_v2_hour_sink_outage_mid_mixed_burst_lands_every_id_exactly_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hout" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05, fail_until=float("inf"), fail_status=503)
    openai_model: Final = f"{marker}openai"
    anthropic_model: Final = f"{marker}anthropic"
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(
            gateway,
            tmp_path,
            bucket.url,
            HOUR,
            models=(
                _config_model(openai_model, "openai/gpt-4o-mini", provider.url + "/v1"),
                _config_model(anthropic_model, ANTHROPIC_MODEL, provider.url),
            ),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        key: Final = scenario.key(models=[openai_model, anthropic_model])
        answered: Final = mixed_burst(owned.gateway, openai_model, anthropic_model, key, marker, per_surface=6)
        eventually(lambda: sink.attempts, lambda attempts: attempts >= 1, seconds=30)
        during: Final = owned.gateway.client.get("/health/readiness")
        rejected: Final = sink.attempts
        sink.fail_until = 0.0
        payloads: Final = collect_payloads(sink, len(answered), seconds=60)
        objects: Final = sink.objects()
    sent: Final = _surface_prompts(marker, 6)
    assert len(answered) == len(sent) and len(payloads) == len(sent), payloads
    assert matched_ids(payloads, answered) == frozenset(str(payload["id"]) for payload in payloads)
    assert sorted(upstream.received()) == sorted(sent)
    assert during.status_code == 200, during.text
    assert rejected >= 1 and sink.attempts > len(objects)
    assert sorted(_prompt(payload) for payload in payloads) == sorted(sent), "every burst id lands exactly once"
    assert len(objects) == len(sent)
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_coded_403_retries_reuse_the_same_hour_key(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3h403" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05, fail_attempts=10, fail_status=403, fail_code="AccessDenied")
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(16))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)
        payloads: Final = collect_payloads(sink, len(prompts), seconds=60)
        objects: Final = sink.objects()
        attempted: Final = dict(sink.attempt_counts)
    assert returned == prompts
    assert sorted(str(payload["id"]) for payload in payloads) == sorted(prompts)
    assert frozenset(attempted) == frozenset(objects), "a retried upload must reuse the key of its first attempt"
    assert sum(attempted.values()) == len(objects) + 10
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_slow_sink_batches_never_duplicate_an_upload(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hslow" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=1.5)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, {**HOUR, "s3_batch_file_upload": True}) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        prompts: Final = tuple(f"{marker}-{index}" for index in range(32))
        returned: Final = _sdk_chats(owned.gateway, openai_model, key, prompts)

        def delivered() -> int:
            readiness: Final = owned.gateway.client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            return sum(len(body.splitlines()) for body in sink.objects().values())

        eventually(delivered, lambda total: total >= len(prompts), seconds=60)
        payloads: Final = sink.payloads()
        objects: Final = sink.objects()
    targets: Final = tuple(put.target for put in bucket.drain())
    assert returned == prompts
    assert len(set(targets)) == len(targets), "the same batch object was PUT more than once"
    assert sorted(str(payload["id"]) for payload in payloads) == sorted(prompts)
    assert _batches_outside_layout(objects, "hour") == ()


def _worker_processes(owned: OwnedProxy) -> tuple[int, ...]:
    return tuple(
        process.pid
        for process in group_members(owned.process.pid)
        if process.pid != owned.process.pid and "spawn_main" in " ".join(process.cmdline())
    )


def test_s3_v2_hour_worker_kill_mid_burst_keeps_the_other_worker_logging(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "s3hkill" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with (
        wire_server(upstream.respond) as provider,
        wire_server(sink.respond) as bucket,
        _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as owned,
        owned.gateway.scenario() as scenario,
    ):
        openai_model, _, key = _models(scenario, provider.url)
        workers: Final = _worker_processes(owned)
        sent: Final = tuple(f"{marker}-{index}" for index in range(40))

        def send(prompt: str) -> tuple[str, bool]:
            try:
                response: Final = owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": openai_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "cache": {"no-cache": True},
                    },
                    key=key,
                )
            except httpx.HTTPError:
                return prompt, False
            return prompt, response.status_code == 200 and response.json()["id"] == prompt

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures: Final = tuple(pool.submit(send, prompt) for prompt in sent)
            eventually(lambda: len(upstream.received()), lambda count: count >= 8, seconds=30)
            psutil.Process(workers[0]).kill()
            results: Final = tuple(future.result() for future in futures)
        later: Final = tuple(f"{marker}-later-{index}" for index in range(8))
        later_results: Final = tuple(send(prompt) for prompt in later)
        eventually(
            lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
            lambda landed: frozenset(later) <= landed,
            seconds=45,
        )
        payloads: Final = sink.payloads()
        objects: Final = sink.objects()
    assert len(workers) == 2, workers
    assert all(ok for _, ok in later_results), "the surviving worker must keep serving after the kill"
    landed: Final = tuple(str(payload["id"]) for payload in payloads)
    assert frozenset(landed) <= frozenset((*sent, *later)), "only ids this test sent may land"
    assert len(results) == len(sent), results
    assert len(landed) == len(set(landed)), "no id may land twice"
    assert _outside_layout(objects, "hour") == ()


def test_s3_v2_hour_proxy_restart_mid_burst_keeps_the_layout_without_duplicates(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "s3hterm" + uuid.uuid4().hex[:8]
    upstream: Final = CountingUpstream()
    sink: Final = RecordingS3Sink(delay_seconds=0.05)
    with wire_server(upstream.respond) as provider, wire_server(sink.respond) as bucket:
        model_name: Final = f"integration-{marker}"

        def register(candidate: Gateway) -> str:
            return str(
                candidate.post(
                    "/model/new",
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_key": "synthetic-provider-key",
                            "api_base": provider.url + "/v1",
                        },
                        "model_info": {},
                    },
                )["model_info"]["id"]
            )

        def send(candidate: Gateway, key: str, prompt: str) -> tuple[str, bool]:
            try:
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "cache": {"no-cache": True},
                    },
                    key=key,
                )
            except httpx.HTTPError:
                return prompt, False
            return prompt, response.status_code == 200

        sent: Final = tuple(f"{marker}-{index}" for index in range(40))
        with _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as first:
            model_id: Final = register(first.gateway)
            first_key: Final = str(first.gateway.post("/key/generate", {"models": [model_name]})["key"])
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures: Final = tuple(pool.submit(send, first.gateway, first_key, prompt) for prompt in sent)
                eventually(lambda: len(upstream.received()), lambda count: count >= 8, seconds=30)
                first.process.terminate()
                results: Final = tuple(future.result() for future in futures)
            first.process.wait(timeout=30)
        answered: Final = frozenset(prompt for prompt, ok in results if ok)
        landed_before_restart: Final = frozenset(str(payload["id"]) for payload in sink.payloads())
        with _s3_proxy(gateway, tmp_path, bucket.url, HOUR) as second:
            restarted: Final = tuple(f"{marker}-restart-{index}" for index in range(8))
            second_key: Final = second.gateway.post("/key/generate", {"models": [model_name]})["key"]
            restart_results: Final = tuple(send(second.gateway, str(second_key), prompt) for prompt in restarted)
            eventually(
                lambda: frozenset(str(payload["id"]) for payload in sink.payloads()),
                lambda landed: frozenset(restarted) <= landed,
                seconds=30,
            )
            second.gateway.post("/model/delete", {"id": model_id})
        payloads: Final = sink.payloads()
        objects: Final = sink.objects()
    assert all(ok for _, ok in restart_results)
    assert landed_before_restart <= answered, "a delivered object has no answered request"
    landed: Final = tuple(str(payload["id"]) for payload in payloads)
    assert len(landed) == len(set(landed)), "no id may land twice across the restart"
    assert frozenset(restarted) <= frozenset(landed)
    targets: Final = tuple(put.target for put in bucket.drain())
    assert len(set(targets)) == len(targets)
    assert _outside_layout(objects, "hour") == ()
