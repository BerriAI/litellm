import os
import signal
import uuid
from pathlib import Path
from typing import Final

import httpx
from _azure_storage_support import (
    SINK_HOSTS,
    RecordingDataLakeSink,
    azure_storage_config,
    azure_storage_environment,
    collect_files,
)
from _s3_v2_support import matched_ids, mixed_burst, surface_reply
from integration._support.client import Gateway, JsonValue, eventually
from integration._support.process import group_members, owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import wire_server

WORKERS: Final = 2
FLUSH_SECONDS: Final = "1"


def _readiness_ok(candidate: Gateway) -> bool:
    try:
        return candidate.request("GET", "/health/readiness").status_code == 200
    except httpx.TransportError:
        return False


def _present_count(payloads: tuple[dict[str, JsonValue], ...], answered: tuple[tuple[str, str | None], ...]) -> int:
    response_ids: Final = frozenset(response_id for response_id, _ in answered)
    call_ids: Final = frozenset(call_id for _, call_id in answered if call_id is not None)
    return sum(1 for payload in payloads if payload["id"] in response_ids or payload["litellm_call_id"] in call_ids)


def test_sink_outage_mid_burst_loses_only_the_outage_window_and_recovers_exactly_once(
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
            **azure_storage_environment(store.url, cert),
            "DEFAULT_FLUSH_INTERVAL_SECONDS": FLUSH_SECONDS,
        }
        config: Final = azure_storage_config(tmp_path)
        with (
            owned_proxy_process(gateway, tmp_path, environment, config=config, workers=WORKERS) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            first: Final = mixed_burst(candidate, openai_model, anthropic_model, key, f"{marker}-first", per_surface=2)
            collect_files(sink, len(first))
            attempts_before_outage: Final = sink.attempts()
            sink.fail_status = 503
            outage: Final = mixed_burst(
                candidate, openai_model, anthropic_model, key, f"{marker}-outage", per_surface=1
            )
            eventually(sink.attempts, lambda count: count > attempts_before_outage, seconds=30)
            readiness: Final = candidate.request("GET", "/health/readiness")
            assert readiness.status_code == 200, readiness.text
            sink.fail_status = 0
            tail: Final = mixed_burst(candidate, openai_model, anthropic_model, key, f"{marker}-tail", per_surface=1)
            answered: Final = first + outage + tail
            payloads: Final = eventually(
                lambda: tuple(sink.payloads().values()),
                lambda stored: _present_count(stored, tail) == len(tail),
                seconds=60,
            )
            landed: Final = matched_ids(payloads, answered)
            assert sink.duplicated() == (), sink.duplicated()
            assert len(sink.stored()) == len(landed), f"{len(sink.stored())} files for {len(landed)} matched ids"
            assert len(landed) >= len(first) + len(tail), (
                f"lost {len(answered) - len(landed)} of {len(answered)} payloads, "
                f"expected at most the {len(outage)} sent during the outage"
            )
            assert len(answered) - len(landed) <= len(outage)


def test_slow_sink_lands_every_id_once_without_deadlock(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"azure-{uuid.uuid4().hex[:8]}"
    sink: Final = RecordingDataLakeSink(delay_seconds=0.3)
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
            owned_proxy_process(gateway, tmp_path, environment, config=config, workers=WORKERS) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            answered: Final = mixed_burst(candidate, openai_model, anthropic_model, key, marker, per_surface=6)
            payloads: Final = collect_files(sink, len(answered), seconds=70)
            assert len(matched_ids(payloads, answered)) == len(answered), tuple(sink.stored())
            assert len(sink.stored()) == len(answered)
            assert sink.duplicated() == (), sink.duplicated()
            assert sink.peak >= 1
            assert store.connections() <= 2 * WORKERS, (
                f"{store.connections()} sink connections for {len(answered)} uploads"
            )


def test_killing_one_worker_keeps_the_other_serving_and_uploading(gateway: Gateway, tmp_path: Path) -> None:
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
            owned_proxy_process(gateway, tmp_path, environment, config=config, workers=WORKERS) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            openai_model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            anthropic_model: Final = scenario.model(
                model="anthropic/claude-sonnet-4-5-20250929", api_base=provider.url, api_key="synthetic-provider-key"
            )
            key: Final = scenario.key(models=[openai_model, anthropic_model])
            first: Final = mixed_burst(candidate, openai_model, anthropic_model, key, f"{marker}-first", per_surface=2)
            collect_files(sink, len(first))
            workers: Final = tuple(
                process for process in group_members(owned.process.pid) if process.pid != owned.process.pid
            )
            assert workers, "no uvicorn workers in the owned proxy process group"
            os.kill(workers[0].pid, signal.SIGKILL)
            eventually(lambda: _readiness_ok(candidate), lambda ok: ok, seconds=30)
            rest: Final = mixed_burst(candidate, openai_model, anthropic_model, key, f"{marker}-rest", per_surface=4)
            payloads: Final = eventually(
                lambda: tuple(sink.payloads().values()),
                lambda stored: _present_count(stored, rest) == len(rest),
                seconds=60,
            )
            members_after: Final = eventually(
                lambda: len(group_members(owned.process.pid)),
                lambda count: count >= 1 + WORKERS,
                seconds=30,
                return_last_on_timeout=True,
            )
            landed: Final = matched_ids(payloads, first + rest)
            assert sink.duplicated() == (), sink.duplicated()
            assert len(landed) >= len(rest), f"only {len(landed)} payloads landed for {len(rest)} post-kill requests"
            assert _present_count(payloads, rest) == len(rest), (
                f"lost {len(rest) - _present_count(payloads, rest)} post-kill payloads; "
                f"process group holds {members_after - 1} workers after the kill"
            )
