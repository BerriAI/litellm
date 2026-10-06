import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import scratch_database
from integration._support.database_relay import database_relay, held_statement_relay
from integration._support.process import owned_proxy

_READINESS_PROBE_QUERY: Final = b"SELECT 1"


def _health_routing_config(directory: Path, general_settings: dict[str, object]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {**config.get("general_settings", {}), **general_settings}
    path: Final = directory / "health-routing.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _plaintext_database_url(database_url: str) -> str:
    parts: Final = urlsplit(database_url)
    parameters: Final = tuple(
        (name, value) for name, value in parse_qsl(parts.query, keep_blank_values=True) if name != "sslmode"
    ) + (("sslmode", "disable"),)
    return urlunsplit(parts._replace(query=urlencode(parameters)))


def _assert_liveness_and_readiness(gateway: Gateway, *, status_code: int, body: dict[str, str]) -> None:
    for path in ("/health/liveliness", "/health/liveness"):
        response: Final = gateway.request("GET", path)
        assert response.status_code == status_code, response.text
        assert response.json() == ("I'm alive!" if status_code == 200 else body), response.text
    readiness: Final = gateway.request("GET", "/health/readiness")
    assert readiness.status_code == status_code, readiness.text
    assert readiness.json() == body, readiness.text


@pytest.mark.timeout(180)
def test_public_readiness_reports_database_outage_and_allow_unavailable_setting(
    gateway: Gateway, tmp_path: Path
) -> None:
    with scratch_database() as database_url:
        with database_relay(database_url, _READINESS_PROBE_QUERY) as (relay, relayed_url):
            with owned_proxy(
                gateway,
                tmp_path,
                {
                    "DATABASE_URL": _plaintext_database_url(relayed_url),
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                },
                workers=1,
            ) as candidate:
                ready_before_outage: Final = candidate.request("GET", "/health/readiness")
                assert ready_before_outage.status_code == 200, ready_before_outage.text
                assert ready_before_outage.json() == {"status": "healthy", "db": "connected"}, ready_before_outage.text
                relay.arm()
                disconnected_after_outage: Final = eventually(
                    lambda: candidate.request("GET", "/health/readiness"),
                    lambda response: (
                        relay.tripped.is_set() and response.json() == {"status": "healthy", "db": "disconnected"}
                    ),
                    seconds=45,
                    return_last_on_timeout=True,
                )
                assert relay.tripped.is_set(), "readiness database probe did not reach the relay"
                assert disconnected_after_outage.status_code == 503, disconnected_after_outage.text
                assert disconnected_after_outage.json() == {
                    "status": "healthy",
                    "db": "disconnected",
                }, disconnected_after_outage.text

        allow_unavailable_config: Final = _health_routing_config(
            tmp_path,
            {"allow_requests_on_db_unavailable": True},
        )
        with database_relay(database_url, _READINESS_PROBE_QUERY) as (relay, relayed_url):
            with owned_proxy(
                gateway,
                tmp_path,
                {
                    "DATABASE_URL": _plaintext_database_url(relayed_url),
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                },
                config=allow_unavailable_config,
                workers=1,
            ) as candidate:
                ready_before_outage_with_allow: Final = candidate.request("GET", "/health/readiness")
                assert ready_before_outage_with_allow.status_code == 200, ready_before_outage_with_allow.text
                assert ready_before_outage_with_allow.json() == {"status": "healthy", "db": "connected"}, (
                    ready_before_outage_with_allow.text
                )
                relay.arm()
                disconnected_with_allow: Final = eventually(
                    lambda: candidate.request("GET", "/health/readiness"),
                    lambda response: (
                        relay.tripped.is_set() and response.json() == {"status": "healthy", "db": "disconnected"}
                    ),
                    seconds=45,
                    return_last_on_timeout=True,
                )
                assert relay.tripped.is_set(), "allow-unavailable readiness probe did not reach the relay"
                assert disconnected_with_allow.status_code == 200, disconnected_with_allow.text
                assert disconnected_with_allow.json() == {
                    "status": "healthy",
                    "db": "disconnected",
                }, disconnected_with_allow.text


@pytest.mark.timeout(180)
def test_public_readiness_reports_stalled_database_lookups(gateway: Gateway, tmp_path: Path) -> None:
    with scratch_database() as database_url:
        for allow_unavailable in (False, True):
            unknown_key = "sk-" + uuid.uuid4().hex
            lookup_hash = hashlib.sha256(unknown_key.encode()).hexdigest().encode()
            config = (
                _health_routing_config(tmp_path, {"allow_requests_on_db_unavailable": True})
                if allow_unavailable
                else None
            )
            with held_statement_relay(database_url, lookup_hash) as (relay, relayed_url):
                overrides = {
                    "DATABASE_URL": _plaintext_database_url(relayed_url),
                    "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
                    "PROXY_DB_LOOKUP_DEADLINE_SECONDS": "1",
                    "PROXY_DB_LOOKUP_STALL_WINDOW_SECONDS": "3",
                }
                with owned_proxy(
                    gateway,
                    tmp_path,
                    overrides,
                    config=config,
                    workers=1,
                ) as candidate:
                    expected_request_status = 400 if allow_unavailable else 503
                    unknown_model_message = (
                        "/chat/completions: Invalid model name passed in model=unconfigured-stalled-readiness-model. "
                        "Call `/v1/models` to view available models for your key."
                    )
                    expected_request_body = (
                        {
                            "error": {
                                "message": unknown_model_message,
                                "type": "invalid_request_error",
                                "param": None,
                                "code": "400",
                                "provider_specific_fields": {"error": unknown_model_message},
                            }
                        }
                        if allow_unavailable
                        else {
                            "error": {
                                "message": (
                                    "Service Unavailable, the authentication database is temporarily unreachable. "
                                    "Please retry shortly."
                                ),
                                "type": "no_db_connection",
                                "param": "None",
                                "code": "503",
                            }
                        }
                    )
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        pending = executor.submit(
                            candidate.request,
                            "POST",
                            "/v1/chat/completions",
                            {
                                "model": "unconfigured-stalled-readiness-model",
                                "messages": [{"role": "user", "content": "stalled lookup control"}],
                            },
                            key=unknown_key,
                        )
                        assert relay.held.wait(30), "unknown-key lookup did not reach the held statement relay"
                        expected_status = 200 if allow_unavailable else 503
                        stalled = eventually(
                            lambda: candidate.request("GET", "/health/readiness"),
                            lambda response, expected_status=expected_status: (
                                response.status_code == expected_status
                                and response.json() == {"status": "healthy", "db": "stalled"}
                            ),
                            seconds=5,
                            return_last_on_timeout=True,
                        )
                        assert stalled.status_code == expected_status, stalled.text
                        assert stalled.json() == {"status": "healthy", "db": "stalled"}, stalled.text

                        relay.release()
                        recovered = eventually(
                            lambda: candidate.request("GET", "/health/readiness"),
                            lambda response: (
                                response.status_code == 200
                                and response.json() == {"status": "healthy", "db": "connected"}
                            ),
                            seconds=10,
                            return_last_on_timeout=True,
                        )
                        assert recovered.status_code == 200, recovered.text
                        assert recovered.json() == {"status": "healthy", "db": "connected"}, recovered.text

                        rejected = pending.result(timeout=60)
                        assert (rejected.status_code, JSON_OBJECT.validate_json(rejected.content)) == (
                            expected_request_status,
                            expected_request_body,
                        ), rejected.text


def test_drain_endpoint_requires_token_and_changes_health_state_only_after_authorization(
    gateway: Gateway, tmp_path: Path
) -> None:
    disabled: Final = gateway.request("GET", "/health/drain")
    assert disabled.status_code == 404, disabled.text
    assert disabled.json() == {"detail": "Not Found"}, disabled.text

    token: Final = "synthetic-health-drain-token"
    config: Final = _health_routing_config(
        tmp_path,
        {
            "enable_drain_endpoint": True,
            "drain_endpoint_token": token,
        },
    )
    with owned_proxy(gateway, tmp_path, {}, config=config, workers=1) as candidate:
        _assert_liveness_and_readiness(
            candidate,
            status_code=200,
            body={"status": "healthy", "db": "connected"},
        )

        missing: Final = candidate.request("GET", "/health/drain")
        assert missing.status_code == 401, missing.text
        assert missing.json() == {"detail": "Invalid or missing X-Drain-Token"}, missing.text
        wrong: Final = candidate.request(
            "GET",
            "/health/drain",
            headers={"X-Drain-Token": "synthetic-wrong-token"},
        )
        assert wrong.status_code == 401, wrong.text
        assert wrong.json() == {"detail": "Invalid or missing X-Drain-Token"}, wrong.text
        _assert_liveness_and_readiness(
            candidate,
            status_code=200,
            body={"status": "healthy", "db": "connected"},
        )

        drained: Final = candidate.request(
            "GET",
            "/health/drain",
            headers={"X-Drain-Token": token},
        )
        assert drained.status_code == 200, drained.text
        assert drained.json() == {"status": "drained", "drained_requests": 0}, drained.text
        for path in ("/health/liveliness", "/health/liveness", "/health/readiness"):
            response: Final = candidate.request("GET", path)
            assert response.status_code == 503, response.text
            assert response.json() == {"status": "shutting_down"}, response.text
