"""Graceful shutdown of the proxy: what SIGTERM does to work in flight, to accounting and to the exit status.

An orchestrator stops a pod by sending SIGTERM and, after `terminationGracePeriodSeconds`, SIGKILL. The proxy
promises (litellm/proxy/shutdown/graceful_shutdown_manager.py, litellm/proxy/proxy_server.py lifespan teardown
around line 2085, litellm/proxy/proxy_cli.py:437 `timeout_graceful_shutdown`):

* SIGTERM marks the worker as shutting down at once and logs `graceful_shutdown_started in_flight_requests=N`.
* The listener closes, so no new connection is accepted, but a request or stream already being served is
  allowed to finish, for at most `GRACEFUL_SHUTDOWN_TIMEOUT` seconds (default 30).
* Only after that does the lifespan teardown run: spend counters, the spend log queue and the daily tables are
  flushed to Postgres even when the batch writer interval has not elapsed, the worker deregisters its
  heartbeat row, and `Shutting down LiteLLM Proxy Server` is logged.
* The process then exits 0.

When the HTTP front door and the process model move to Rust, every one of these is an operator contract:
rolling deploys depend on in-flight calls finishing, finance depends on the final flush, and the orchestrator
reads the exit status. Callback and tag spend shutdown flushes are covered in `spend/test_shutdown_flush.py`;
the Rust preview gateway's own SIGTERM rules are in `test_dec1_gwnative_streaming_and_shutdown.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration.configuration.dec1_deploy_support import (
    EXPECTED_SPEND,
    REPLY_TEXT,
    Provider,
    await_rows,
    chat,
    deploy_config,
    digest,
    heartbeat_rows,
    held_chat,
    log_text,
    provider,
    response_id,
    streamed_text,
    terminate,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(600)

STARTED: Final = "graceful_shutdown_started in_flight_requests="
COMPLETE: Final = "graceful_shutdown_complete"
SHUTTING_DOWN_LOG: Final = "Shutting down LiteLLM Proxy Server"
NEVER_BATCH: Final[dict[str, JsonValue]] = {"proxy_batch_write_at": 3600}


@contextmanager
def _owned(
    gateway: Gateway,
    tmp_path: Path,
    upstream: Provider,
    *,
    environment: dict[str, str] | None = None,
    general_settings: dict[str, JsonValue] | None = None,
    workers: int = 1,
) -> Iterator[OwnedProxy]:
    config: Final = write_config(tmp_path, deploy_config(upstream.api_base, general_settings=general_settings))
    with owned_proxy_process(
        gateway,
        tmp_path / "proxy",
        {"LITELLM_LOG": "INFO", **(environment or {})},
        config=config,
        workers=workers,
    ) as owned:
        yield owned


def _await_log(owned: OwnedProxy, needle: str, seconds: float = 30) -> str:
    return eventually(lambda: log_text(owned), lambda text: needle in text, seconds=seconds)


def _refuses_new_connections(port_url: str) -> bool:
    try:
        with httpx.Client(base_url=port_url, timeout=2, trust_env=False) as fresh:
            fresh.get("/health/liveliness")
    except httpx.TransportError:
        return True
    return False


def _base_url(owned: OwnedProxy) -> str:
    return str(owned.gateway.client.base_url)


def _outcome(future: Future[object]) -> object:
    try:
        return future.result(timeout=120)
    except httpx.TransportError as error:
        return error


def test_sigterm_on_an_idle_proxy_exits_zero(gateway: Gateway, tmp_path: Path) -> None:
    """A ready proxy with no traffic gets SIGTERM. The process must exit with status 0 within the
    graceful window: an orchestrator treats any other status as a crash and may page someone."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        assert terminate(owned) == 0, log_text(owned)[-3000:]


def test_sigterm_logs_the_start_of_the_graceful_shutdown_with_the_in_flight_count(
    gateway: Gateway, tmp_path: Path
) -> None:
    """With `LITELLM_LOG=INFO` a ready proxy holds one chat call at the provider and gets SIGTERM. Its log must
    gain `graceful_shutdown_started in_flight_requests=1` before the call finishes, which is the line operators
    grep to tell a drain from a crash (graceful_shutdown_manager.py:93).
    UNVERIFIED: that the in-flight middleware counts exactly the one held call and no probe."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        pending: Final = held_chat(owned.gateway, upstream)
        owned.begin_shutdown()
        text: Final = _await_log(owned, STARTED)
        assert f"{STARTED}1" in text, text[-3000:]
        upstream.release.set()
        _outcome(pending)
        owned.process.wait(timeout=90)


def test_sigterm_lets_a_held_unary_call_finish_with_its_full_answer(gateway: Gateway, tmp_path: Path) -> None:
    """One chat call is held at the provider and the proxy gets SIGTERM. The process must still be running
    after the shutdown started, and once the provider answers the caller must get 200 with the provider's
    content `dec1 deploy reply`, then the process exits 0. A rewrite that cancels in-flight calls on SIGTERM
    turns every rolling deploy into client errors."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        pending: Final = held_chat(owned.gateway, upstream)
        owned.begin_shutdown()
        _await_log(owned, STARTED)
        assert owned.process.poll() is None, "the proxy exited while a request was still in flight"
        upstream.release.set()
        answered: Final = _outcome(pending)
        assert isinstance(answered, httpx.Response) and answered.status_code == 200, answered
        assert answered.json()["choices"][0]["message"]["content"] == REPLY_TEXT
        assert owned.process.wait(timeout=90) == 0


def test_sigterm_lets_an_open_stream_finish_with_its_done_frame(gateway: Gateway, tmp_path: Path) -> None:
    """A streamed chat call has received its first chunk and the provider holds the rest; the proxy gets
    SIGTERM. After release the caller must read status 200, the full text `dec1 deploy reply` and a final
    `data: [DONE]` frame, and the process exits 0. A stream cut at shutdown looks to the client like a model
    that stopped mid-sentence."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        pending: Final = held_chat(owned.gateway, upstream, stream=True)
        owned.begin_shutdown()
        _await_log(owned, STARTED)
        upstream.release.set()
        status, body = pending.result(timeout=120)  # type: ignore[misc]
        assert status == 200, body
        assert streamed_text(body) == REPLY_TEXT, body
        assert body.rstrip().endswith("data: [DONE]"), body[-500:]
        assert owned.process.wait(timeout=90) == 0


def test_no_new_connection_is_accepted_while_a_held_call_drains(gateway: Gateway, tmp_path: Path) -> None:
    """While one chat call is held and the proxy is draining after SIGTERM, a brand new TCP connection to the
    proxy port must be refused (or reset) instead of served: the pod has left the load balancer and new work
    must go to another replica. The held call still finishes with 200 once released."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        pending: Final = held_chat(owned.gateway, upstream)
        owned.begin_shutdown()
        _await_log(owned, STARTED)
        eventually(lambda: _refuses_new_connections(_base_url(owned)), bool, seconds=15)
        upstream.release.set()
        answered: Final = _outcome(pending)
        assert isinstance(answered, httpx.Response) and answered.status_code == 200, answered
        owned.process.wait(timeout=90)


def test_a_call_held_past_the_graceful_timeout_does_not_keep_the_process_alive(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`GRACEFUL_SHUTDOWN_TIMEOUT=3`, and a chat call is held at the provider forever. After SIGTERM the
    process must exit within 60 seconds, well before the call would finish, and the held caller must not get
    a 200: the timeout bounds the drain so the pod dies before SIGKILL.
    UNVERIFIED: the exit status after a timed out drain is not asserted, only that the process ends."""
    with (
        provider() as upstream,
        _owned(gateway, tmp_path, upstream, environment={"GRACEFUL_SHUTDOWN_TIMEOUT": "3"}) as owned,
    ):
        pending: Final = held_chat(owned.gateway, upstream)
        owned.begin_shutdown()
        owned.process.wait(timeout=60)
        cut: Final = _outcome(pending)
        assert not (isinstance(cut, httpx.Response) and cut.status_code == 200), cut


def test_a_malformed_graceful_timeout_is_reported_and_the_default_is_used(gateway: Gateway, tmp_path: Path) -> None:
    """`GRACEFUL_SHUTDOWN_TIMEOUT=soon`. The proxy must still boot and serve, and its log must contain
    `GRACEFUL_SHUTDOWN_TIMEOUT='soon' is not a number; using default 30.0s` (graceful_shutdown_manager.py:71):
    a typo in a manifest must not stop the pod, but it must be visible."""
    with (
        provider() as upstream,
        _owned(gateway, tmp_path, upstream, environment={"GRACEFUL_SHUTDOWN_TIMEOUT": "soon"}) as owned,
    ):
        assert chat(owned.gateway).status_code == 200
        assert "GRACEFUL_SHUTDOWN_TIMEOUT='soon' is not a number; using default 30.0s" in log_text(owned)


def test_shutdown_logs_the_completion_of_the_drain_and_the_teardown(gateway: Gateway, tmp_path: Path) -> None:
    """With `LITELLM_LOG=INFO` a proxy that served one call gets SIGTERM and exits. Its log must contain
    `graceful_shutdown_complete` and, after it, `Shutting down LiteLLM Proxy Server`: the drain finished
    before the dependencies were torn down (proxy_server.py:2091 then :2121)."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        assert chat(owned.gateway).status_code == 200
        assert terminate(owned) == 0
        text: Final = log_text(owned)
        assert COMPLETE in text and SHUTTING_DOWN_LOG in text, text[-3000:]
        assert text.index(COMPLETE) < text.rindex(SHUTTING_DOWN_LOG), text[-3000:]


def test_key_spend_is_written_at_shutdown_even_before_the_batch_writer_runs(gateway: Gateway, tmp_path: Path) -> None:
    """`proxy_batch_write_at: 3600`, so the periodic spend writer never runs during the test. A virtual key
    makes one chat call priced at 0.001 per input and 0.002 per output token (5 and 3 tokens), then the
    proxy gets SIGTERM. Once it has exited, `LiteLLM_VerificationToken.spend` for that key must be 0.011:
    spend held in memory is flushed by the teardown (`flush_spend_counters_on_shutdown`, proxy_server.py:1398)."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, general_settings=NEVER_BATCH) as owned:
        key: Final = str(owned.gateway.post("/key/generate", {})["key"])
        assert chat(owned.gateway, key=key).status_code == 200
        assert terminate(owned) == 0
        rows: Final = owned.gateway.rows(
            'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(key),)
        )
        assert len(rows) == 1 and float(str(rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows


def test_team_spend_is_written_at_shutdown_even_before_the_batch_writer_runs(gateway: Gateway, tmp_path: Path) -> None:
    """Same as the key case, for a key that belongs to a team: after SIGTERM and exit, `LiteLLM_TeamTable.spend`
    for the team must be 0.011. Team budgets are enforced from this column after the next boot."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, general_settings=NEVER_BATCH) as owned:
        team: Final = str(owned.gateway.post("/team/new", {"team_alias": "dec1-deploy-team"})["team_id"])
        key: Final = str(owned.gateway.post("/key/generate", {"team_id": team})["key"])
        assert chat(owned.gateway, key=key).status_code == 200
        assert terminate(owned) == 0
        rows: Final = owned.gateway.rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
        assert len(rows) == 1 and float(str(rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows


def test_the_spend_log_row_of_the_last_request_lands_at_shutdown(gateway: Gateway, tmp_path: Path) -> None:
    """`proxy_batch_write_at: 3600`. One chat call is answered, then the proxy gets SIGTERM. After exit
    `LiteLLM_SpendLogs` must hold exactly one row whose `request_id` is the response `id`, with `spend` 0.011
    (`_flush_spend_logs_queue_on_shutdown`, proxy_server.py:1409)."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, general_settings=NEVER_BATCH) as owned:
        answered: Final = chat(owned.gateway)
        assert answered.status_code == 200, answered.text
        assert terminate(owned) == 0
        rows: Final = owned.gateway.rows(
            'SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (response_id(answered),)
        )
        assert len(rows) == 1 and float(str(rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows


def test_a_call_finished_during_the_drain_is_still_billed(gateway: Gateway, tmp_path: Path) -> None:
    """`proxy_batch_write_at: 3600`. A key's chat call is held, SIGTERM arrives, then the provider answers.
    After exit the key's `spend` must be 0.011: the final flush runs after the drain, so work that completed
    during the drain is not lost."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, general_settings=NEVER_BATCH) as owned:
        key: Final = str(owned.gateway.post("/key/generate", {})["key"])
        pending: Final = held_chat(replace(owned.gateway, key=key), upstream)
        owned.begin_shutdown()
        _await_log(owned, STARTED)
        upstream.release.set()
        answered: Final = _outcome(pending)
        assert isinstance(answered, httpx.Response) and answered.status_code == 200, answered
        assert owned.process.wait(timeout=90) == 0
        rows: Final = owned.gateway.rows(
            'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (digest(key),)
        )
        assert float(str(rows[0]["spend"])) == pytest.approx(EXPECTED_SPEND), rows


def test_a_graceful_shutdown_removes_the_workers_heartbeat_row(gateway: Gateway, tmp_path: Path) -> None:
    """A single worker proxy registers one row in `LiteLLM_ProxyWorkerHeartbeat` at boot. After SIGTERM and
    exit the table must be empty: the worker deletes its own row (`ProxyWorkerHeartbeat.deregister`,
    litellm/proxy/db/proxy_worker_heartbeat.py:85), so the live worker census drops at once instead of after
    the 180 second liveness window."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 1)
        assert terminate(owned) == 0
        assert heartbeat_rows(owned.gateway) == []


def test_a_killed_proxy_leaves_its_heartbeat_row_for_the_liveness_window(gateway: Gateway, tmp_path: Path) -> None:
    """A single worker proxy is killed with SIGKILL, so no teardown runs. Its heartbeat row must still be in
    `LiteLLM_ProxyWorkerHeartbeat`: only the 180 second liveness window ages it out of the census. This pins
    that deregistration is a shutdown step, not something a crash can be relied on to do."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream) as owned:
        rows: Final = await_rows(lambda: heartbeat_rows(owned.gateway), 1)
        owned.kill_supervisor()
        assert heartbeat_rows(owned.gateway) == rows


def test_sigterm_stops_every_worker_of_a_two_worker_proxy_and_exits_zero(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with `--num_workers 2` gets SIGTERM. The root must exit 0 and neither worker process may
    outlive it: a leftover worker keeps the port and the database connections of a pod that is gone."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, workers=2) as owned:
        workers: Final = owned.worker_pids()
        assert len(workers) == 2, workers
        assert terminate(owned) == 0
        eventually(lambda: [pid for pid in workers if psutil.pid_exists(pid)], lambda alive: alive == [], seconds=30)


def test_both_workers_deregister_their_heartbeat_rows_on_sigterm(gateway: Gateway, tmp_path: Path) -> None:
    """A two worker proxy has two heartbeat rows. After SIGTERM and exit `LiteLLM_ProxyWorkerHeartbeat` must
    be empty: every worker runs its own teardown, not only the first one to stop."""
    with provider() as upstream, _owned(gateway, tmp_path, upstream, workers=2) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 2)
        assert terminate(owned) == 0
        assert heartbeat_rows(owned.gateway) == []
