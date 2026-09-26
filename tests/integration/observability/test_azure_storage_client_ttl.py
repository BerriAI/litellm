import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Final

from _azure_storage_support import (
    SINK_HOSTS,
    RecordingDataLakeSink,
    azure_storage_config,
    azure_storage_environment,
    collect_files,
)
from _s3_v2_support import SURFACES, call_surface, matched_ids, surface_reply
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, wire_server

WORKERS: Final = 2
FLUSH_SECONDS: Final = "1"


def _chat_completion(candidate: Gateway, model: str, key: str, marker: str) -> tuple[str, str | None]:
    response: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}},
        key=key,
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"]), response.headers.get("x-litellm-call-id")


def _marker_of(request: Request) -> str | None:
    if request.method != "POST" or not request.body:
        return None
    body: Final = json.loads(request.body)
    messages: Final = body.get("messages")
    if isinstance(messages, list) and messages:
        content: Final = messages[0].get("content") if isinstance(messages[0], dict) else None
        if isinstance(content, str):
            return content
    input_value: Final = body.get("input")
    return input_value if isinstance(input_value, str) else None


def upstream_rejecting_fail_markers(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        marker: Final = _marker_of(request)
        if marker is not None and marker.startswith("fail-"):
            return Reply(status=status, body=json.dumps({"error": {"message": f"upstream rejected {marker}"}}).encode())
        return surface_reply(request)

    return respond


def _spend_row_visible(response_id: str) -> None:
    eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
        lambda rows: len(rows) == 1,
        seconds=60,
    )


def test_every_surface_lands_once_and_the_client_is_reused_across_uploads(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            answered: Final = tuple(
                call_surface(candidate, surface, openai_model, anthropic_model, key, f"{marker}-{surface}-{index}")
                for index in range(3)
                for surface in SURFACES
            )
            payloads: Final = collect_files(sink, len(answered))
            assert len(matched_ids(payloads, answered)) == len(answered), tuple(sink.stored())
            assert sink.duplicated() == (), sink.duplicated()
            assert store.connections() <= 2 * WORKERS, (
                f"{store.connections()} sink connections for {len(answered)} uploads"
            )
        assert provider.drain()


def test_success_callback_mode_uploads_success_and_skips_failure(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(upstream_rejecting_fail_markers(500)) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path, callback_setting="success_callback")
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            first_id, _ = _chat_completion(candidate, model, key, f"{marker}-a")
            failed: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"fail-{marker}-b"}]},
                key=key,
            )
            assert failed.status_code >= 500 and f"fail-{marker}-b" in failed.text, failed.text
            third_id, _ = _chat_completion(candidate, model, key, f"{marker}-c")
            collect_files(sink, 2)
            landed: Final = frozenset(str(payload["id"]) for payload in sink.payloads().values())
            assert landed == frozenset({first_id, third_id}), tuple(sink.stored())
            assert all(f"fail-{marker}-b".encode() not in body for body in sink.stored().values())


def test_failure_callback_mode_uploads_only_failures(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(upstream_rejecting_fail_markers(500)) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path, callback_setting="failure_callback")
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            _chat_completion(candidate, model, key, f"{marker}-a")
            failed: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"fail-{marker}-b"}]},
                key=key,
            )
            assert failed.status_code >= 500 and f"fail-{marker}-b" in failed.text, failed.text
            collect_files(sink, 1)
            bodies: Final = tuple(sink.stored().values())
            assert len(bodies) == 1 and f"fail-{marker}-b".encode() in bodies[0], tuple(sink.stored())
            assert f"{marker}-a".encode() not in bodies[0]


def _sink_rejection_keeps_the_caller_and_proxy_healthy(gateway: Gateway, tmp_path: Path, status: int) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink(fail_status=status)
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            _chat_completion(candidate, model, key, f"{marker}-a")
            eventually(sink.attempts, lambda count: count >= 1, seconds=30)
            assert not sink.stored(), tuple(sink.stored())
            other_key: Final = scenario.key(models=[model])
            _chat_completion(candidate, model, other_key, f"{marker}-other")
            readiness: Final = candidate.request("GET", "/health/readiness")
            assert readiness.status_code == 200, readiness.text
            sink.fail_status = 0
            third_id, _ = _chat_completion(candidate, model, key, f"{marker}-c")
            eventually(
                lambda: tuple(sink.payloads().values()),
                lambda stored: third_id in {str(payload["id"]) for payload in stored},
                seconds=60,
            )
            bodies: Final = tuple(sink.stored().values())
            assert all(f"{marker}-a".encode() not in body for body in bodies), f"{marker}-a should be lost, not retried"


def test_sink_403_keeps_the_caller_and_proxy_healthy(gateway: Gateway, tmp_path: Path) -> None:
    _sink_rejection_keeps_the_caller_and_proxy_healthy(gateway, tmp_path, 403)


def test_sink_404_keeps_the_caller_and_proxy_healthy(gateway: Gateway, tmp_path: Path) -> None:
    _sink_rejection_keeps_the_caller_and_proxy_healthy(gateway, tmp_path, 404)


def test_upstream_401_reaches_the_caller_and_lands_as_a_failure_payload(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(upstream_rejecting_fail_markers(401)) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            failed: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": f"fail-{marker}"}]},
                key=key,
            )
            assert failed.status_code == 401 and f"fail-{marker}" in failed.text, failed.text
            payloads: Final = collect_files(sink, 1)
            assert len(payloads) == 1 and f"fail-{marker}".encode() in next(iter(sink.stored().values()))
            assert payloads[0]["status"] == "failure", payloads[0]


def test_unknown_model_lands_as_a_failure_payload(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            rejected: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": "does-not-exist", "messages": [{"role": "user", "content": f"{marker}-unknown"}]},
                key=key,
            )
            assert 400 <= rejected.status_code < 500 and "does-not-exist" in rejected.text, rejected.text
            success_id, _ = _chat_completion(candidate, model, key, f"{marker}-ok")
            payloads: Final = collect_files(sink, 2)
            successful: Final = tuple(payload for payload in payloads if str(payload["id"]) == success_id)
            failures: Final = tuple(payload for payload in payloads if payload["status"] == "failure")
            assert len(successful) == 1 and len(failures) == 1, tuple(sink.stored())


def test_missing_file_system_setting_fails_the_callback_init_and_keeps_the_proxy_serving(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            name: value
            for name, value in {
                **azure_storage_environment(store.url, cert),
                "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
            }.items()
            if name != "AZURE_STORAGE_FILE_SYSTEM"
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(
                gateway,
                tmp_path,
                environment,
                config=config,
                remove_environment=("AZURE_STORAGE_FILE_SYSTEM",),
                workers=WORKERS,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            response_id, _ = _chat_completion(candidate, model, key, f"{marker}-ok")
            _spend_row_visible(response_id)
            assert store.connections() == 0, f"{store.connections()} sink connections without a configured sink"


def test_repeated_identical_requests_each_land_exactly_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            first_id, _ = _chat_completion(candidate, model, key, f"{marker}-a")
            second_id, _ = _chat_completion(candidate, model, key, f"{marker}-b")
            payloads: Final = collect_files(sink, 2)
            landed: Final = frozenset(str(payload["id"]) for payload in payloads)
            assert landed == frozenset({first_id, second_id}), tuple(sink.stored())
            assert sink.duplicated() == (), sink.duplicated()
            received: Final = tuple(_marker_of(request) for request in provider.drain())
            assert received.count(f"{marker}-a") == 1 and received.count(f"{marker}-b") == 1, received


def test_disabled_callback_opens_no_sink_connection(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply) as provider,
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = {
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        with (
            owned_proxy(gateway, tmp_path, environment, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            key: Final = scenario.key(models=[model])
            response_id, _ = _chat_completion(candidate, model, key, f"{marker}-ok")
            _spend_row_visible(response_id)
            assert store.connections() == 0, f"{store.connections()} sink connections with the callback disabled"


def test_files_upload_to_azure_storage_sibling_path_is_unchanged(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink()
    cert, key = write_self_signed_cert(tmp_path, SINK_HOSTS)
    with (
        wire_server(surface_reply),
        wire_server(sink.respond, tls=server_context(cert, key), keep_alive=True) as store,
    ):
        environment: Final = azure_storage_environment(store.url, cert)
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy(gateway, tmp_path, environment, config=config, workers=WORKERS) as candidate,
            candidate.scenario() as scenario,
        ):
            key: Final = scenario.key()
            content: Final = f'{{"marker": "{marker}"}}\n'.encode()
            uploaded: Final = candidate.request_multipart(
                "/v1/files",
                {"purpose": "user_data", "target_storage": "azure_storage"},
                {"file": ("batch.jsonl", content, "application/jsonl")},
                key=key,
            )
            assert uploaded.status_code == 200, uploaded.text
            assert uploaded.json()["id"].startswith("file-"), uploaded.text
            eventually(
                lambda: any(content in body for body in sink.stored().values()),
                lambda found: found,
                seconds=30,
            )
