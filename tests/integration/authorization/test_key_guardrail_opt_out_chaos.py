import json
import os
import signal
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import psutil

from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy, owned_proxy_process
from integration.authorization._guardrail_opt_out import (
    chat,
    guardrail_config,
    non_admin_caller,
    stored_metadata,
    upstream_hits,
    upstream_observations,
)


class _GuardrailSink:
    """Test-owned guardrail endpoint that can be stopped and restarted on the same port."""

    def __init__(self, *, delay_seconds: float = 0.0, action: str = "BLOCKED") -> None:
        self.received: SimpleQueue[bytes] = SimpleQueue()
        self._delay: Final = delay_seconds
        self._action: Final = action
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: Final = self._claim_port()
        self.start()

    def _claim_port(self) -> int:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self) -> None:
        received = self.received
        delay = self._delay
        action = self._action

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
                received.put(body)
                if delay:
                    time.sleep(delay)
                payload: Final = json.dumps({"action": action, "blocked_reason": "synthetic policy denial"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server(("127.0.0.1", self._port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05})
        self._thread.start()

    def stop(self) -> None:
        assert self._server is not None and self._thread is not None
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=6)
        assert not self._thread.is_alive()
        self._server = None

    def drain(self) -> tuple[bytes, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))

    def __enter__(self) -> "_GuardrailSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self.stop()


def test_concurrent_flag_writes_split_expected_outcomes(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {}) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
        caller: Final = non_admin_caller(scenario, member, team, model)
        alias: Final = "audit-concurrent-" + uuid.uuid4().hex

        bodies: Final = [
            {"team_id": team, "models": [model], "key_alias": f"{alias}-{index}", "disable_global_guardrails": flag}
            for index in range(20)
            for flag in (True, False)
        ]
        with ThreadPoolExecutor(max_workers=20) as pool:
            responses: Final = tuple(
                pool.map(lambda body: candidate.request("POST", "/key/generate", body, key=caller), bodies)
            )
        created_aliases: Final = [
            row["key_alias"]
            for row in read_rows(
                'SELECT key_alias FROM "LiteLLM_VerificationToken" WHERE key_alias LIKE %s', (f"{alias}-%",)
            )
        ]
        for response in responses:
            if response.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(response.json()["key"]))
        flagged: Final = tuple(
            response for response, body in zip(responses, bodies) if body["disable_global_guardrails"] is True
        )
        flagless: Final = tuple(
            response for response, body in zip(responses, bodies) if body["disable_global_guardrails"] is False
        )
        assert sorted(response.status_code for response in flagged) == [403] * 20, [
            response.text for response in flagged
        ]
        assert sorted(response.status_code for response in flagless) == [200] * 20, [
            response.text for response in flagless
        ]
        assert len(created_aliases) == 20, created_aliases
        for entry in created_aliases:
            stored: Final = read_rows('SELECT metadata FROM "LiteLLM_VerificationToken" WHERE key_alias = %s', (entry,))
            assert stored[0]["metadata"].get("disable_global_guardrails") is not True, entry


def test_revoked_exemption_denies_later_non_admin_resave(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy(gateway, tmp_path, {}) as candidate, candidate.scenario() as scenario:
        model: Final = scenario.model()
        member: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
        caller: Final = non_admin_caller(scenario, member, team, model)
        exempt: Final = scenario.key(team_id=team, models=[model], disable_global_guardrails=True)
        assert stored_metadata(exempt)["disable_global_guardrails"] is True

        candidate.post("/key/update", {"key": exempt, "disable_global_guardrails": False})
        assert stored_metadata(exempt)["disable_global_guardrails"] is False

        resave: Final = candidate.request(
            "POST",
            "/key/update",
            {
                "key": exempt,
                "key_alias": "audit-revoked-" + uuid.uuid4().hex,
                "metadata": {"disable_global_guardrails": True},
            },
            key=caller,
        )
        assert resave.status_code == 403, resave.text
        assert "disable_global_guardrails" in resave.text, resave.text
        assert stored_metadata(exempt)["disable_global_guardrails"] is False


def test_revoked_exemption_blocks_chats_on_both_workers(gateway: Gateway, tmp_path: Path) -> None:
    with _GuardrailSink() as sink:
        config: Final = guardrail_config(sink.url, tmp_path / "revoke-workers.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as first:
            with owned_proxy(gateway, tmp_path, {}, config=config) as second:
                with first.scenario() as scenario:
                    model: Final = scenario.model()
                    exempt: Final = scenario.key(models=[model], disable_global_guardrails=True)
                    for worker in (first, second):
                        served: Final = chat(worker, model, exempt, "audit-both-" + uuid.uuid4().hex)
                        assert served.status_code == 200, served.text
                    first.post("/key/update", {"key": exempt, "disable_global_guardrails": False})
                    for worker in (first, second):
                        denied: Final = eventually(
                            lambda w=worker: chat(w, model, exempt, "audit-both-" + uuid.uuid4().hex),
                            lambda response: response.status_code == 400 and "synthetic policy denial" in response.text,
                            seconds=70,
                        )
                        assert denied.status_code == 400, denied.text


def test_exempt_burst_survives_guardrail_sink_outage(gateway: Gateway, tmp_path: Path) -> None:
    with _GuardrailSink() as sink:
        config: Final = guardrail_config(sink.url, tmp_path / "sink-outage.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            exempt: Final = scenario.key(models=[model], disable_global_guardrails=True)
            plain: Final = scenario.key(models=[model])

            warm: Final = chat(candidate, model, plain, "warm-" + uuid.uuid4().hex)
            assert warm.status_code == 400 and "synthetic policy denial" in warm.text, warm.text
            assert sink.drain() != ()

            def burst(keys: tuple[str, ...], tag: str) -> tuple[httpx.Response, ...]:
                with ThreadPoolExecutor(max_workers=15) as pool:
                    return tuple(
                        pool.map(
                            lambda pair: chat(candidate, model, pair[1], f"{tag}-{pair[0]}-{uuid.uuid4().hex}"),
                            enumerate(keys * 10),
                        )
                    )

            outage_keys: Final = (exempt, plain)
            with ThreadPoolExecutor(max_workers=2) as pool:
                bursts: Final = pool.submit(burst, outage_keys, "outage")
                eventually(
                    lambda: sink.received.qsize(),
                    lambda count: count >= 2,
                    seconds=30,
                )
                sink.stop()
                outage_responses: Final = bursts.result(timeout=90)
            exempt_outage: Final = [response for index, response in enumerate(outage_responses) if index % 2 == 0]
            non_exempt_outage: Final = [response for index, response in enumerate(outage_responses) if index % 2 == 1]
            assert all(response.status_code == 200 for response in exempt_outage), [
                response.status_code for response in exempt_outage
            ]
            outage_statuses: Final = {response.status_code for response in non_exempt_outage}
            assert outage_statuses <= {400, 500}, outage_statuses
            assert all(
                "synthetic policy denial" in response.text or response.status_code == 500
                for response in non_exempt_outage
            ), [response.text for response in non_exempt_outage if response.status_code not in {400, 500}]
            assert all(upstream_hits(gateway, f"outage-{index}-") == 0 for index in range(1, 20, 2)), (
                upstream_observations(gateway)
            )

            sink.start()
            recovered: Final = chat(candidate, model, plain, "recovered-" + uuid.uuid4().hex)
            assert recovered.status_code == 400 and "synthetic policy denial" in recovered.text, recovered.text


def test_flag_denial_survives_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {}, workers=2) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            member: Final = scenario.user(user_role="internal_user")
            team: Final = scenario.team(models=[model], members_with_roles=[{"role": "admin", "user_id": member}])
            caller: Final = non_admin_caller(scenario, member, team, model)
            alias: Final = "audit-kill-" + uuid.uuid4().hex

            workers: Final = eventually(
                lambda: psutil.Process(owned.process.pid).children(recursive=True),
                lambda children: len(children) >= 2,
                seconds=30,
            )
            victim: Final = workers[0]
            os.kill(victim.pid, signal.SIGKILL)

            probe: Final = eventually(
                lambda: candidate.request(
                    "POST",
                    "/key/generate",
                    {"team_id": team, "models": [model], "key_alias": f"{alias}-probe"},
                    key=caller,
                ),
                lambda response: response.status_code in (200, 403),
                seconds=30,
            )
            if probe.status_code == 200:
                scenario.cleanups.callback(scenario.delete_key, string_value(probe.json()["key"]))
            for index in range(10):
                denied: Final = candidate.request(
                    "POST",
                    "/key/generate",
                    {
                        "team_id": team,
                        "models": [model],
                        "key_alias": f"{alias}-{index}",
                        "disable_global_guardrails": True,
                    },
                    key=caller,
                )
                if denied.status_code == 200:
                    scenario.cleanups.callback(scenario.delete_key, string_value(denied.json()["key"]))
                assert denied.status_code == 403, denied.text
                assert "disable_global_guardrails" in denied.text, denied.text
            assert (
                read_rows(
                    'SELECT token FROM "LiteLLM_VerificationToken" WHERE key_alias LIKE %s AND metadata::text LIKE %s',
                    (f"{alias}-%", '%"disable_global_guardrails": true%'),
                )
                == []
            )


def test_exempt_chats_do_not_wait_on_slow_guardrail_sink(gateway: Gateway, tmp_path: Path) -> None:
    sink_delay: Final = 10.0
    with _GuardrailSink(delay_seconds=sink_delay) as sink:
        config: Final = guardrail_config(sink.url, tmp_path / "slow-sink.yaml")
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model()
            exempt: Final = scenario.key(models=[model], disable_global_guardrails=True)

            started: Final = time.monotonic()
            with ThreadPoolExecutor(max_workers=10) as pool:
                responses: Final = tuple(
                    pool.map(
                        lambda index: chat(candidate, model, exempt, f"slow-sink-{index}-{uuid.uuid4().hex}"),
                        range(10),
                    )
                )
            elapsed: Final = time.monotonic() - started
            assert all(response.status_code == 200 for response in responses), [
                (response.status_code, response.text) for response in responses
            ]
            assert elapsed < sink_delay, f"exempt chats waited on the guardrail sink: {elapsed}s"
            assert sink.drain() == ()
