import asyncio
import json
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Final

import pytest
from pydantic import SecretStr

from litellm.integrations.postgres_olap import (
    PostgresOlapConfig,
    PostgresOlapLogger,
    Row,
)

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
PAYLOAD: Final = {
    "id": "request-1",
    "status": "success",
    "model": "example",
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5,
    "response_cost": 0.01,
    "messages": [{"role": "user", "content": "private prompt"}],
    "response": {
        "choices": [
            {
                "message": {"content": "private response"},
                "logprobs": {"content": [{"token": "private response", "bytes": [112, 114, 105, 118, 97, 116, 101]}]},
            }
        ]
    },
    "error_str": "provider rejected input_value='private prompt'",
    "model_parameters": {
        "query": "private prompt",
        "documents": ["private response"],
        "prediction": {"content": "private response"},
    },
    "error_information": {
        "error_code": "429",
        "error_message": "invalid messages: private prompt",
        "traceback": "provider exception: private response",
    },
    "metadata": {
        "prompt_management_metadata": {"prompt_variables": {"question": "private prompt"}},
        "mcp_tool_call_metadata": {"arguments": {"input": "private prompt"}},
        "vector_store_request_metadata": {"query": "private prompt", "results": ["private response"]},
    },
}


class Destination:
    def __init__(self, failures: int = 0) -> None:
        self.rows: list[Row] = []
        self.attempts: list[tuple[Row, ...]] = []
        self.failures = failures

    def __call__(self, config: PostgresOlapConfig, rows: Sequence[Row]) -> None:
        self.attempts.append(tuple(rows))
        if len(self.attempts) <= self.failures:
            raise ConnectionError("sensitive rejected row")
        self.rows.extend(rows)


def config(**kwargs: object) -> PostgresOlapConfig:
    return PostgresOlapConfig.model_validate({"dsn": "dbname=test", "flush_interval": 0.001, **kwargs})


@pytest.mark.parametrize(
    "method", ["log_success_event", "log_failure_event", "async_log_success_event", "async_log_failure_event"]
)
def test_callbacks_deliver_redacted_log_without_mutating_input(method: str) -> None:
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(), destination)
    try:
        status: Final = "failure" if "failure" in method else "success"
        callback: Final = getattr(logger, method)
        result: Final = callback({"standard_logging_object": {**PAYLOAD, "status": status}}, None, NOW, NOW)
        if method.startswith("async"):
            asyncio.run(result)
        assert logger.flush()
        assert len(destination.rows) == 1
        row: Final = destination.rows[0]
        assert row[1:10] == ("request-1", status, "example", NOW, NOW, 3, 2, 5, 0.01)
        assert "private prompt" not in str(row[-1])
        assert "private response" not in str(row[-1])
        assert json.loads(str(row[-1]))["error_information"] == {"error_code": "429"}
        assert "metadata" not in json.loads(str(row[-1]))
        assert "private response" in json.dumps(PAYLOAD["response"])
        assert PAYLOAD["error_information"]["error_message"] == "invalid messages: private prompt"
        assert logger.statistics == {"accepted": 1, "delivered": 1, "dropped": 0, "pending": 0}
    finally:
        logger.close()


def test_opt_in_content_and_excluded_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "standard_logging_payload_excluded_fields", ["model"])
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(turn_off_message_logging=False), destination)
    try:
        logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert logger.flush()
        payload: Final = json.loads(str(destination.rows[0][-1]))
        assert payload == {key: value for key, value in PAYLOAD.items() if key != "model"}
        assert destination.rows[0][3] == ""
    finally:
        logger.close()


def test_retry_preserves_event_identity_and_reports_exhaustion(caplog: pytest.LogCaptureFixture) -> None:
    destination: Final = Destination(failures=1)
    logger: Final = PostgresOlapLogger(config(retries=1), destination)
    try:
        logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert logger.flush()
        assert destination.attempts[0] == destination.attempts[1] == tuple(destination.rows)
        assert "sensitive rejected row" not in caplog.text
    finally:
        logger.close()
    failing: Final = PostgresOlapLogger(config(retries=0), Destination(failures=1))
    try:
        failing.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert failing.flush()
        assert failing.statistics == {"accepted": 1, "delivered": 0, "dropped": 1, "pending": 0}
    finally:
        failing.close()


def test_queue_budget_includes_inflight_batch_and_rejects_after_close() -> None:
    started: Final = threading.Event()
    release: Final = threading.Event()

    def blocked(config: PostgresOlapConfig, rows: Sequence[Row]) -> None:
        started.set()
        assert release.wait(5)

    logger: Final = PostgresOlapLogger(config(batch_size=1, max_queue_bytes=800, retries=0), blocked)
    try:
        logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert started.wait(5)
        assert not logger.flush(0)
        for _ in range(10):
            logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert logger.statistics["dropped"] > 0
        assert logger.statistics["accepted"] < 11
    finally:
        release.set()
        logger.close()
    logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
    assert logger.statistics["pending"] == 0
    assert logger.statistics["accepted"] == logger.statistics["delivered"]


def test_oversized_and_invalid_payloads_do_not_reach_destination() -> None:
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(max_event_bytes=10), destination)
    try:
        logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        logger.log_failure_event({"standard_logging_object": "invalid"}, None, NOW, NOW)
        assert logger.flush()
        assert destination.rows == []
        assert logger.statistics == {"accepted": 0, "delivered": 0, "dropped": 2, "pending": 0}
    finally:
        logger.close()


def test_secret_dsn_is_not_in_config_repr() -> None:
    settings: Final = PostgresOlapConfig(dsn=SecretStr("postgresql://user:private-password@host/db"))
    assert "private-password" not in repr(settings)


def test_proxy_shutdown_drains_partial_batch_and_rejects_new_events() -> None:
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(flush_interval=100), destination)
    logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
    asyncio.run(logger.async_close_proxy())
    logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
    assert len(destination.rows) == 1
    assert logger.statistics == {"accepted": 1, "delivered": 1, "dropped": 1, "pending": 0}


@pytest.mark.parametrize("scope", ["global", "request"])
def test_redaction_overrides_content_opt_in(scope: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "turn_off_message_logging", scope == "global")
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(turn_off_message_logging=False), destination)
    try:
        logger.log_success_event(
            {
                "standard_logging_object": PAYLOAD,
                **(
                    {"standard_callback_dynamic_params": {"turn_off_message_logging": True}}
                    if scope == "request"
                    else {}
                ),
            },
            None,
            NOW,
            NOW,
        )
        assert logger.flush()
        assert len(destination.rows) == 1
        assert "private prompt" not in str(destination.rows[0][-1])
        assert "private response" not in str(destination.rows[0][-1])
    finally:
        logger.close()


def test_explicit_flush_drains_tail_without_waiting_for_periodic_interval() -> None:
    destination: Final = Destination()
    logger: Final = PostgresOlapLogger(config(batch_size=2, flush_interval=100), destination)
    try:
        for _ in range(3):
            logger.log_success_event({"standard_logging_object": PAYLOAD}, None, NOW, NOW)
        assert logger.flush(5)
        assert len(destination.rows) == 3
        assert all(len(batch) <= 2 for batch in destination.attempts)
    finally:
        logger.close()
