"""Asqav local-first audit-log callback for LiteLLM.

Each LLM call appends one record to a local JSONL file.  Every record carries
a SHA-256 chain hash over its own canonical fields plus the previous record's
hash, giving a tamper-evident sequence that can be verified entirely offline
with stdlib tools.

Design goals (matching the on-device ask from litellm#25329):
- Zero runtime dependencies beyond Python stdlib + litellm itself.
- Never breaks an LLM call: every code path is wrapped fail-soft.
- Does not log message content by default; logs content digests so
  auditors can prove a payload was present without reconstructing it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, BinaryIO, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

__all__ = ["AsqavLogger"]

# Sentinel used as the genesis prev_hash (no predecessor).
_GENESIS_HASH = "0" * 64

# Default log path; can be overridden via ASQAV_LOG_PATH.
_DEFAULT_LOG_PATH = os.path.join(os.path.expanduser("~"), ".litellm_asqav_audit.jsonl")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    """Stable canonical serialisation for hashing.

    We sort keys and use separators=(',', ':') so the byte sequence is
    deterministic across Python versions and platforms.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_tail(fh: BinaryIO, size: int) -> bytes:
    """Read backwards from the end of fh until the buffer contains the entire
    last line, doubling the window each pass so records of any length survive
    a restart."""
    chunk_size = 4096
    while True:
        read_size = min(chunk_size, size)
        fh.seek(size - read_size)
        tail = fh.read(read_size)
        if read_size == size or b"\n" in tail.rstrip(b"\n"):
            return tail
        chunk_size *= 2


def _content_digest(value: Any) -> Optional[str]:
    """Return a SHA-256 hex digest of a content value, or None if empty."""
    if value is None:
        return None
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_hex(raw)


def _extract_loggable(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    status: str,
) -> dict[str, Any]:
    """Pull metadata + digests out of a callback invocation.

    Message content and response text are never stored in the clear; only their
    SHA-256 digests appear in the log so callers can prove a payload existed
    without reconstructing it.
    """
    model: str = kwargs.get("model", "")
    messages: Any = kwargs.get("messages")

    # Root metadata (user-supplied tags, etc.)
    metadata: Any = dict(kwargs.get("metadata") or kwargs.get("litellm_metadata") or {})

    # Merge proxy identity fields from litellm_params.metadata.  Sensitive
    # header/key values are filtered so raw auth tokens never reach the log.
    _SENSITIVE_KEYS = frozenset(
        {
            "user_api_key",
            "Authorization",
            "authorization",
            "token",
            "api_key",
        }
    )
    _PROXY_IDENTITY_KEYS = frozenset(
        {
            "user_api_key_user_id",
            "user_api_key_team_id",
            "user_api_key_org_id",
            "user_api_key_alias",
            "user_id",
            "team_id",
            "org_id",
        }
    )
    try:
        lp_meta: Any = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        for k, v in lp_meta.items():
            if k in _SENSITIVE_KEYS:
                continue
            # Always include explicit proxy identity keys; skip other
            # litellm_params.metadata keys to avoid unexpected bleed.
            if k in _PROXY_IDENTITY_KEYS:
                metadata.setdefault(k, v)
    except Exception:
        pass

    # Timing
    latency_ms: Optional[int] = None
    try:
        if start_time is not None and end_time is not None:
            latency_ms = int((end_time - start_time).total_seconds() * 1000)
    except Exception:
        pass

    # Usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    provider_request_id: Optional[str] = None
    try:
        if hasattr(response_obj, "usage") and response_obj.usage:
            prompt_tokens = response_obj.usage.prompt_tokens
            completion_tokens = response_obj.usage.completion_tokens
            total_tokens = response_obj.usage.total_tokens
        if hasattr(response_obj, "choices") and response_obj.choices:
            finish_reason = response_obj.choices[0].finish_reason
        if hasattr(response_obj, "_hidden_params"):
            provider_request_id = response_obj._hidden_params.get(
                "x-request-id"
            ) or response_obj._hidden_params.get("cf-ray")
    except Exception:
        pass

    # Content digests (not content itself)
    messages_digest: Optional[str] = _content_digest(messages)

    response_content_digest: Optional[str] = None
    try:
        if hasattr(response_obj, "choices") and response_obj.choices:
            content = response_obj.choices[0].message.content
            response_content_digest = _content_digest(content)
    except Exception:
        pass

    # Standard logging payload may carry call_id / litellm_call_id
    call_id: Optional[str] = None
    try:
        slp: Any = kwargs.get("standard_logging_object")
        if slp and isinstance(slp, dict):
            call_id = slp.get("id") or slp.get("litellm_call_id")
    except Exception:
        pass
    if not call_id:
        call_id = kwargs.get("litellm_call_id") or kwargs.get(
            "id", str(int(time.time() * 1e6))
        )

    return {
        "call_id": call_id,
        "model": model,
        "status": status,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": finish_reason,
        "provider_request_id": provider_request_id,
        "messages_digest": messages_digest,
        "response_content_digest": response_content_digest,
        "metadata": {k: v for k, v in (metadata or {}).items() if isinstance(k, str)},
    }


class AsqavLogger(CustomLogger):
    """Tamper-evident local-first audit-log callback for LiteLLM.

    Configuration (all via environment variables):

    ASQAV_LOG_PATH
        Path to the JSONL audit log.  Defaults to ~/.litellm_asqav_audit.jsonl.

    ASQAV_REDACT_CONTENT
        Set to "false" to store message/response content in the clear instead
        of as SHA-256 digests.  Defaults to "true" (digest only).

    Multi-worker limitation: this logger is designed for a single audit writer
    per log file.  The threading.Lock serializes concurrent threads within one
    process; it does NOT serialize across OS processes.  In a multi-worker
    proxy deployment, run a single dedicated audit-writer process, or use a
    shared filesystem with OS-level exclusive locking (fcntl.flock) via a
    custom wrapper.  Without this, multiple workers will produce records with
    duplicate seq/prev_hash values and verify_chain will report a break.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        redact_content: bool = True,
    ) -> None:
        super().__init__()

        self._log_path: str = log_path or os.environ.get(
            "ASQAV_LOG_PATH", _DEFAULT_LOG_PATH
        )
        self._redact_content: bool = (
            os.environ.get("ASQAV_REDACT_CONTENT", "true").lower() != "false"
            if log_path is None
            else redact_content
        )

        self._lock: threading.Lock = threading.Lock()
        self._call_count: int = 0
        self._prev_hash: str = _GENESIS_HASH

        # Load chain state from an existing log file so we chain correctly
        # across process restarts.
        self._load_chain_tail()

    def __repr__(self) -> str:
        return (
            f"AsqavLogger(log_path={self._log_path!r},"
            f" redact_content={self._redact_content})"
        )

    # ------------------------------------------------------------------
    # Chain state persistence
    # ------------------------------------------------------------------

    def _load_chain_tail(self) -> None:
        """Read the last line of an existing log file to resume the chain."""
        try:
            if not os.path.exists(self._log_path):
                return
            with open(self._log_path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if size == 0:
                    return
                tail = _read_tail(fh, size)
            lines = [ln for ln in tail.split(b"\n") if ln.strip()]
            if not lines:
                return
            last_record = json.loads(lines[-1].decode("utf-8"))
            self._prev_hash = last_record.get("record_hash", _GENESIS_HASH)
            self._call_count = last_record.get("seq", -1) + 1
        except Exception:
            verbose_logger.debug(
                f"[AsqavLogger] Could not load chain tail: {traceback.format_exc()}"
            )

    # ------------------------------------------------------------------
    # Core record append
    # ------------------------------------------------------------------

    def _build_and_append(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        status: str,
    ) -> None:
        """Build one audit record and append it to the JSONL log.

        seq/prev_hash assignment and the file write happen under _lock so the
        on-disk order always matches the chain order.
        """
        try:
            loggable = _extract_loggable(
                kwargs, response_obj, start_time, end_time, status
            )

            if not self._redact_content:
                # Store content in the clear when the operator explicitly opts in.
                loggable["messages"] = kwargs.get("messages")
                try:
                    if hasattr(response_obj, "choices") and response_obj.choices:
                        loggable["response_content"] = response_obj.choices[
                            0
                        ].message.content
                except Exception:
                    pass

            # The file write happens under the same lock that assigns seq and
            # prev_hash, so records always land on disk in chain order even
            # when callbacks fire concurrently.  Chain state only advances
            # after a successful write; a failed write drops the record and
            # the chain continues from the last record actually on disk.
            with self._lock:
                seq = self._call_count

                # The fields that enter the hash are fixed and canonical so that
                # an auditor can reproduce the digest from the log alone.
                hashable: dict[str, Any] = {
                    "seq": seq,
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "prev_hash": self._prev_hash,
                    **loggable,
                }
                record_hash = _sha256_hex(_canonical_bytes(hashable))

                if not self._write_record({**hashable, "record_hash": record_hash}):
                    return

                self._prev_hash = record_hash
                self._call_count += 1

        except Exception:
            verbose_logger.debug(
                f"[AsqavLogger] Unhandled error in _build_and_append: {traceback.format_exc()}"
            )

    def _write_record(self, record: dict[str, Any]) -> bool:
        """Append one record to the log file. Returns False if the write failed."""
        try:
            parent = os.path.dirname(self._log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Tighten permissions on an existing file before appending so a
            # file created by a previous run with a permissive umask is locked
            # down.  Create new files via os.open with 0o600 to skip umask.
            if os.path.exists(self._log_path):
                os.chmod(self._log_path, 0o600)
            fd = os.open(
                self._log_path,
                os.O_CREAT | os.O_WRONLY | os.O_APPEND,
                0o600,
            )
            try:
                with os.fdopen(fd, "a", encoding="utf-8", closefd=True) as fh:
                    fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            except Exception:
                # fdopen owns fd; if it raises before returning the context
                # manager the fd may still be open - close defensively.
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            return True
        except Exception:
            verbose_logger.warning(
                f"[AsqavLogger] Failed to write audit record: {traceback.format_exc()}"
            )
            return False

    # ------------------------------------------------------------------
    # CustomLogger hooks
    # ------------------------------------------------------------------

    def log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._build_and_append(kwargs, response_obj, start_time, end_time, "success")

    def log_failure_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._build_and_append(kwargs, response_obj, start_time, end_time, "failure")

    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        await asyncio.to_thread(
            self._build_and_append,
            kwargs,
            response_obj,
            start_time,
            end_time,
            "success",
        )

    async def async_log_failure_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        await asyncio.to_thread(
            self._build_and_append,
            kwargs,
            response_obj,
            start_time,
            end_time,
            "failure",
        )

    # ------------------------------------------------------------------
    # Chain verification (utility; not called on the hot path)
    # ------------------------------------------------------------------

    def verify_chain(self, log_path: Optional[str] = None) -> tuple[bool, str]:
        """Verify the integrity of the audit log at log_path.

        Returns (True, "ok") when every record's hash matches its content and
        its prev_hash matches the previous record's hash.  Returns
        (False, reason) on the first violation found.

        This method is intentionally a pure stdlib utility so auditors can
        paste it anywhere.
        """
        path = log_path or self._log_path
        try:
            prev_hash = _GENESIS_HASH
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    stored_hash = record.get("record_hash", "")
                    # Recompute hash over all fields except record_hash itself.
                    hashable = {k: v for k, v in record.items() if k != "record_hash"}
                    computed_hash = _sha256_hex(_canonical_bytes(hashable))

                    if computed_hash != stored_hash:
                        return (
                            False,
                            f"line {lineno}: hash mismatch"
                            f" (stored={stored_hash[:12]},"
                            f" computed={computed_hash[:12]})",
                        )

                    rec_prev = record.get("prev_hash", _GENESIS_HASH)
                    if rec_prev != prev_hash:
                        return (
                            False,
                            f"line {lineno}: prev_hash chain break"
                            f" (expected={prev_hash[:12]},"
                            f" got={rec_prev[:12]})",
                        )

                    prev_hash = stored_hash

            return True, "ok"
        except FileNotFoundError:
            return False, f"log file not found: {path}"
        except Exception as exc:
            return False, f"verification error: {exc}"
