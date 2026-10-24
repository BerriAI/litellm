"""Tests for the Asqav local-first audit-log callback.

No network and no provider calls: every test writes to a tmp_path log.

The chain checks re-derive record hashes with hashlib and json directly, so they
also prove an auditor can verify the log without litellm installed.
"""

import asyncio
import hashlib
import json
import os
import stat
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.integrations.asqav import AsqavLogger
from litellm.integrations.asqav.asqav import GENESIS_HASH
from litellm.litellm_core_utils import litellm_logging as logging_module
from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry
from litellm.types.utils import Choices, Message, ModelResponse, Usage
from tests._wait_helpers import wait_until

START: Final = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END: Final = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)


def _kwargs(content: str = "hello", call_id: str = "call-1") -> dict[str, object]:
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "litellm_call_id": call_id,
    }


def _response(content: str = "world") -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-test",
        model="gpt-4o",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content=content, role="assistant"))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _logger(path: Path, redact_content: bool = True) -> AsqavLogger:
    return AsqavLogger(log_path=str(path), redact_content=redact_content)


def _append(logger: AsqavLogger, count: int, prefix: str = "msg") -> None:
    for i in range(count):
        logger.log_success_event(_kwargs(content=f"{prefix}-{i}"), _response(content=f"resp-{i}"), START, END)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records), encoding="utf-8")


def _stdlib_hash(record: Mapping[str, object]) -> str:
    """The record hash, recomputed the way an auditor would with only the standard library."""
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_first_record_starts_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _logger(path).log_success_event(_kwargs(), _response(), START, END)

    [record] = _records(path)
    assert record["seq"] == 0
    assert record["prev_hash"] == GENESIS_HASH
    assert record["status"] == "success"
    assert record["record_hash"] == _stdlib_hash(record)


def test_records_link_and_hash_reproduces_with_stdlib(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    _append(logger, 10)

    records = _records(path)
    assert [r["seq"] for r in records] == list(range(10))
    assert records[0]["prev_hash"] == GENESIS_HASH
    for previous, current in zip(records, records[1:]):
        assert current["prev_hash"] == previous["record_hash"]
    assert all(r["record_hash"] == _stdlib_hash(r) for r in records)
    assert logger.verify_chain() == (True, "ok")


def test_record_carries_usage_latency_and_call_metadata(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _logger(path).log_success_event(_kwargs(call_id="call-abc"), _response(), START, END)

    [record] = _records(path)
    assert record["call_id"] == "call-abc"
    assert record["model"] == "gpt-4o"
    assert record["latency_ms"] == 1000
    assert (record["prompt_tokens"], record["completion_tokens"], record["total_tokens"]) == (10, 5, 15)
    assert record["finish_reason"] == "stop"


def test_verify_detects_an_edited_field(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    _append(logger, 5)
    records = _records(path)
    _rewrite(path, [*records[:2], {**records[2], "model": "tampered-model"}, *records[3:]])

    ok, reason = logger.verify_chain()

    assert ok is False
    assert reason.startswith("line 3: hash mismatch")


def test_verify_detects_a_rehashed_prev_hash_edit(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    _append(logger, 5)
    records = _records(path)
    forged = {**records[3], "prev_hash": "a" * 64}
    _rewrite(path, [*records[:3], {**forged, "record_hash": _stdlib_hash(forged)}, *records[4:]])

    ok, reason = logger.verify_chain()

    assert ok is False
    assert reason.startswith("line 4: prev_hash chain break")


def test_verify_detects_a_deleted_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    _append(logger, 5)
    records = _records(path)
    _rewrite(path, [*records[:2], *records[3:]])

    ok, reason = logger.verify_chain()

    assert ok is False
    assert reason.startswith("line 3: prev_hash chain break")


def test_verify_reports_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"

    assert _logger(tmp_path / "audit.jsonl").verify_chain(str(missing)) == (False, f"log file not found: {missing}")


def test_chain_and_seq_resume_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _append(_logger(path), 5, prefix="first")
    last_before_restart = _records(path)[-1]

    restarted = _logger(path)
    _append(restarted, 1, prefix="second")

    records = _records(path)
    assert len(records) == 6
    assert records[5]["seq"] == 5
    assert records[5]["prev_hash"] == last_before_restart["record_hash"]
    assert restarted.verify_chain() == (True, "ok")


def test_restart_resumes_when_last_record_exceeds_tail_window(tmp_path: Path) -> None:
    """A last record bigger than the 4 KB first read window still resumes the chain."""
    path = tmp_path / "audit.jsonl"
    big_kwargs = {**_kwargs(content="big"), "metadata": {"blob": "x" * 10_000}}
    _logger(path).log_success_event(big_kwargs, _response(), START, END)
    assert os.path.getsize(path) > 4096

    restarted = _logger(path)
    _append(restarted, 2)

    records = _records(path)
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert restarted.verify_chain() == (True, "ok")


def test_failure_event_is_logged(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _logger(path).log_failure_event(_kwargs(), None, START, END)

    [record] = _records(path)
    assert record["status"] == "failure"
    assert record["prompt_tokens"] is None


def test_malformed_response_still_writes_a_record(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _logger(path).log_success_event(_kwargs(), object(), START, END)

    [record] = _records(path)
    assert record["status"] == "success"
    assert record["response_content_digest"] is None


def test_failed_write_does_not_raise_or_advance_the_chain(tmp_path: Path) -> None:
    blocker = tmp_path / "logs"
    blocker.write_text("", encoding="utf-8")
    path = blocker / "audit.jsonl"
    logger = _logger(path)

    logger.log_success_event(_kwargs(call_id="call-dropped"), _response(), START, END)
    blocker.unlink()
    logger.log_success_event(_kwargs(call_id="call-kept"), _response(), START, END)

    [record] = _records(path)
    assert record["call_id"] == "call-kept"
    assert record["seq"] == 0
    assert record["prev_hash"] == GENESIS_HASH


def test_content_is_digested_by_default(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    _logger(path).log_success_event(_kwargs(content="my private prompt"), _response("my private answer"), START, END)

    raw = path.read_text(encoding="utf-8")
    [record] = _records(path)
    assert "my private prompt" not in raw
    assert "my private answer" not in raw
    expected_digest = hashlib.sha256(
        json.dumps([{"role": "user", "content": "my private prompt"}], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert record["messages_digest"] == expected_digest
    assert isinstance(record["response_content_digest"], str)


def test_redact_content_false_stores_plaintext(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path, redact_content=False)
    logger.log_success_event(_kwargs(content="visible prompt"), _response("visible answer"), START, END)

    [record] = _records(path)
    assert record["messages"] == [{"role": "user", "content": "visible prompt"}]
    assert record["response_content"] == "visible answer"
    assert logger.verify_chain() == (True, "ok")


def test_env_vars_configure_path_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "from-env.jsonl"
    monkeypatch.setenv("ASQAV_LOG_PATH", str(path))
    monkeypatch.setenv("ASQAV_REDACT_CONTENT", "false")

    logger = AsqavLogger()
    logger.log_success_event(_kwargs(content="env prompt"), _response(), START, END)

    assert logger.log_path == str(path)
    assert _records(path)[0]["messages"] == [{"role": "user", "content": "env prompt"}]


def test_proxy_identity_is_kept_and_secrets_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    kwargs = {
        **_kwargs(),
        "litellm_params": {
            "metadata": {
                "user_api_key_user_id": "user-abc",
                "user_api_key_team_id": "team-xyz",
                "user_api_key_org_id": "org-123",
                "user_api_key_alias": "my-key",
                "user_api_key": "hashed-key-value",
                "Authorization": "placeholder-auth-header",
                "requester_ip_address": "10.0.0.1",
            }
        },
    }
    _logger(path).log_success_event(kwargs, _response(), START, END)

    metadata = _records(path)[0]["metadata"]
    assert metadata == {
        "user_api_key_user_id": "user-abc",
        "user_api_key_team_id": "team-xyz",
        "user_api_key_org_id": "org-123",
        "user_api_key_alias": "my-key",
    }


def test_non_json_metadata_is_stored_as_text_and_verifies(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    kwargs = {**_kwargs(), "metadata": {"requested_at": START}}
    logger.log_success_event(kwargs, _response(), START, END)

    assert _records(path)[0]["metadata"] == {"requested_at": str(START)}
    assert logger.verify_chain() == (True, "ok")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_log_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)

    _logger(path).log_success_event(_kwargs(), _response(), START, END)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_concurrent_callbacks_keep_disk_order_equal_to_chain_order(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)
    threads_count, per_thread = 8, 25
    barrier = threading.Barrier(threads_count)

    def worker(thread_id: int) -> None:
        barrier.wait()
        _append(logger, per_thread, prefix=f"t{thread_id}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [r["seq"] for r in _records(path)] == list(range(threads_count * per_thread))
    assert logger.verify_chain() == (True, "ok")


def test_async_hooks_write_records(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = _logger(path)

    async def run() -> None:
        await logger.async_log_success_event(_kwargs(content="async-ok"), _response(), START, END)
        await logger.async_log_failure_event(_kwargs(content="async-fail"), None, START, END)

    asyncio.run(run())

    assert [r["status"] for r in _records(path)] == ["success", "failure"]
    assert logger.verify_chain() == (True, "ok")


def test_asqav_is_a_registered_callback_name() -> None:
    assert "asqav" in litellm._known_custom_logger_compatible_callbacks
    assert CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE["asqav"] is AsqavLogger


def test_callbacks_asqav_builds_one_logger_and_reuses_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`callbacks=["asqav"]` goes through litellm_logging, which must hand back one instance."""
    monkeypatch.setenv("ASQAV_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])

    assert logging_module.get_custom_logger_compatible_class("asqav") is None

    first = logging_module._init_custom_logger_compatible_class(
        logging_integration="asqav", internal_usage_cache=None, llm_router=None, custom_logger_init_args={}
    )
    second = logging_module._init_custom_logger_compatible_class(
        logging_integration="asqav", internal_usage_cache=None, llm_router=None, custom_logger_init_args={}
    )

    assert isinstance(first, AsqavLogger)
    assert second is first
    assert first.log_path == str(tmp_path / "audit.jsonl")
    assert logging_module.get_custom_logger_compatible_class("asqav") is first


def test_completion_with_callbacks_asqav_appends_a_verifiable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real litellm.completion call (mock_response, no network) lands one chained record."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ASQAV_LOG_PATH", str(path))
    monkeypatch.setattr(logging_module, "_in_memory_loggers", [])
    monkeypatch.setattr(litellm, "callbacks", ["asqav"])

    litellm.completion(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}], mock_response="hello from the mock"
    )
    wait_until(lambda: path.exists() and bool(_records(path)), message="asqav record written")

    [record] = _records(path)
    assert record["status"] == "success"
    assert record["model"] == "gpt-4o"
    assert record["record_hash"] == _stdlib_hash(record)
    assert "hello from the mock" not in path.read_text(encoding="utf-8")
