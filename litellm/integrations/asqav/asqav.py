"""Asqav local-first audit-log callback for LiteLLM.

Each LLM call appends one record to a local JSONL file. Every record carries a
SHA-256 hash over its own canonical fields plus the previous record's hash, so
the file forms a tamper-evident chain that can be checked offline with the
standard library alone.

- No dependencies beyond litellm itself, and no network calls.
- Fail-soft: an error while building or writing a record is logged and never
  reaches the LLM call.
- Message and response content are stored as SHA-256 digests by default, so an
  auditor can prove a payload existed without the log holding it.
"""

import asyncio
import hashlib
import itertools
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import BinaryIO, Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

__all__ = ("AsqavLogger",)

# prev_hash of the first record in a chain (it has no predecessor).
GENESIS_HASH: Final = "0" * 64

# Overridable with ASQAV_LOG_PATH or the log_path argument.
DEFAULT_LOG_PATH: Final = os.path.join(os.path.expanduser("~"), ".litellm_asqav_audit.jsonl")

_TAIL_CHUNK_BYTES: Final = 4096

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_MAPPING: Final = TypeAdapter(Mapping[object, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_SEQUENCE: Final = TypeAdapter(Sequence[object], config=ConfigDict(hide_input_in_errors=True))

# Header and key values that must never reach the log.
_SENSITIVE_KEYS: Final = frozenset({"user_api_key", "Authorization", "authorization", "token", "api_key"})

# Proxy identity fields copied from litellm_params.metadata so each record says who made the call.
_PROXY_IDENTITY_KEYS: Final = frozenset(
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


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(record: Mapping[str, object]) -> bytes:
    """Deterministic serialisation for hashing: sorted keys, no whitespace.

    Values JSON cannot encode are stored as their ``str()`` form, so the bytes
    written to disk and the bytes hashed are the same after a JSON round trip.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _content_digest(value: object) -> str | None:
    """SHA-256 hex digest of a content value, or None when there is no content."""
    if value is None:
        return None
    return _sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _read_tail(fh: BinaryIO, size: int) -> bytes:
    """Read back from the end of fh, doubling the window until it holds the whole last line."""
    for shift in itertools.count():
        read_size = min(_TAIL_CHUNK_BYTES << shift, size)
        fh.seek(size - read_size)
        tail = fh.read(read_size)
        if read_size == size or b"\n" in tail.rstrip(b"\n"):
            return tail
    return b""


def _str_keyed(value: object) -> Mapping[str, object]:
    """The string-keyed entries of value when it is a mapping, else an empty mapping."""
    if not isinstance(value, Mapping):
        return {}
    entries: Final = _OBJECT_MAPPING.validate_python(value)
    return {key: item for key, item in entries.items() if isinstance(key, str)}


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _first_choice(response_obj: object) -> object:
    choices: Final[object] = getattr(response_obj, "choices", None)
    if isinstance(choices, list | tuple) and choices:
        return _OBJECT_SEQUENCE.validate_python(choices)[0]
    return None


def _response_content(response_obj: object) -> object:
    message: Final[object] = getattr(_first_choice(response_obj), "message", None)
    content: Final[object] = getattr(message, "content", None)
    return content


def _latency_ms(start_time: object, end_time: object) -> int | None:
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        return int((end_time - start_time).total_seconds() * 1000)
    return None


def _proxy_identity(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """Identity fields from litellm_params.metadata; raw keys and auth headers are never copied."""
    litellm_params: Final = _str_keyed(kwargs.get("litellm_params"))
    proxy_metadata: Final = _str_keyed(litellm_params.get("metadata"))
    return {
        key: value
        for key, value in proxy_metadata.items()
        if key in _PROXY_IDENTITY_KEYS and key not in _SENSITIVE_KEYS
    }


def _call_id(kwargs: Mapping[str, object]) -> str:
    standard_logging_object: Final = _str_keyed(kwargs.get("standard_logging_object"))
    candidates: Final = (
        standard_logging_object.get("id"),
        standard_logging_object.get("litellm_call_id"),
        kwargs.get("litellm_call_id"),
        kwargs.get("id"),
    )
    found: Final = next((str(candidate) for candidate in candidates if candidate), None)
    return found if found is not None else str(int(time.time() * 1e6))


def _extract_loggable(
    kwargs: Mapping[str, object],
    response_obj: object,
    start_time: object,
    end_time: object,
    status: str,
) -> Mapping[str, object]:
    """Metadata and content digests for one callback invocation.

    Message content and response text are never stored in the clear here; only
    their SHA-256 digests appear, so a caller can prove a payload existed
    without the log reconstructing it.
    """
    root_metadata: Final = _str_keyed(kwargs.get("metadata")) or _str_keyed(kwargs.get("litellm_metadata"))
    usage: Final[object] = getattr(response_obj, "usage", None)
    hidden_params: Final = _str_keyed(getattr(response_obj, "_hidden_params", None))
    first_choice: Final = _first_choice(response_obj)
    model: Final = kwargs.get("model")

    return {
        "call_id": _call_id(kwargs),
        "model": model if isinstance(model, str) else "",
        "status": status,
        "latency_ms": _latency_ms(start_time, end_time),
        "prompt_tokens": _int_or_none(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": _int_or_none(getattr(usage, "completion_tokens", None)),
        "total_tokens": _int_or_none(getattr(usage, "total_tokens", None)),
        "finish_reason": _str_or_none(getattr(first_choice, "finish_reason", None)),
        "provider_request_id": _str_or_none(hidden_params.get("x-request-id"))
        or _str_or_none(hidden_params.get("cf-ray")),
        "messages_digest": _content_digest(kwargs.get("messages")),
        "response_content_digest": _content_digest(_response_content(response_obj)),
        # Root metadata wins over the proxy identity copy, matching the user's own tags.
        "metadata": {**_proxy_identity(kwargs), **root_metadata},
    }


def _private_opener(path: str, flags: int) -> int:
    """Create the log owner-only (0o600) whatever the process umask."""
    return os.open(path, flags, 0o600)


def _redact_from_env() -> bool:
    return os.environ.get("ASQAV_REDACT_CONTENT", "true").lower() != "false"


class AsqavLogger(CustomLogger):
    """Tamper-evident local-first audit-log callback for LiteLLM.

    Enable with ``litellm.callbacks = ["asqav"]`` or ``callbacks: ["asqav"]``
    under ``litellm_settings`` in the proxy config.

    Configuration (environment variables, overridden by constructor arguments):

    ASQAV_LOG_PATH
        Path to the JSONL audit log. Defaults to ~/.litellm_asqav_audit.jsonl.

    ASQAV_REDACT_CONTENT
        Set to "false" to store message and response content in the clear
        instead of as SHA-256 digests. Defaults to "true" (digests only).

    Multi-worker limitation: this logger expects a single audit writer per log
    file. Its lock serialises threads within one process, not separate OS
    processes. In a multi-worker proxy, give each worker its own
    ASQAV_LOG_PATH or run one dedicated writer; otherwise records from several
    workers share seq and prev_hash values and verify_chain reports a break.
    """

    def __init__(
        self,
        log_path: str | None = None,
        redact_content: bool | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]  # CustomLogger.__init__ declares untyped **kwargs
        self._log_path: Final[str] = log_path or os.environ.get("ASQAV_LOG_PATH") or DEFAULT_LOG_PATH
        self._redact_content: Final[bool] = _redact_from_env() if redact_content is None else redact_content
        self._lock: Final = threading.Lock()
        self._call_count: int = 0
        self._prev_hash: str = GENESIS_HASH
        # Resume from an existing log so the chain continues across restarts.
        self._load_chain_tail()

    def __repr__(self) -> str:
        return f"AsqavLogger(log_path={self._log_path!r}, redact_content={self._redact_content})"

    @property
    def log_path(self) -> str:
        return self._log_path

    def _load_chain_tail(self) -> None:
        """Read the last record of an existing log to resume seq and prev_hash."""
        try:
            if not os.path.exists(self._log_path):
                return
            with open(self._log_path, "rb") as fh:
                size: Final = fh.seek(0, os.SEEK_END)
                if size == 0:
                    return
                tail: Final = _read_tail(fh, size)
            lines: Final = [line for line in tail.split(b"\n") if line.strip()]
            if not lines:
                return
            last_record: Final = _JSON_OBJECT.validate_python(json.loads(lines[-1].decode("utf-8")))
            record_hash: Final = last_record.get("record_hash")
            seq: Final = last_record.get("seq")
            self._prev_hash = record_hash if isinstance(record_hash, str) else GENESIS_HASH
            self._call_count = seq + 1 if isinstance(seq, int) else 0
        except (OSError, ValueError, ValidationError):
            verbose_logger.debug("[AsqavLogger] Could not load chain tail", exc_info=True)

    def _build_and_append(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
        status: str,
    ) -> None:
        """Build one audit record and append it to the JSONL log.

        seq and prev_hash are assigned and the file is written under one lock,
        so the on-disk order always matches the chain order. Chain state only
        advances after a successful write: a failed write drops the record and
        the chain continues from the last record on disk.
        """
        try:
            loggable: Final = _extract_loggable(kwargs, response_obj, start_time, end_time, status)
            clear_content: Final[Mapping[str, object]] = (
                {}
                if self._redact_content
                else {"messages": kwargs.get("messages"), "response_content": _response_content(response_obj)}
            )
            with self._lock:
                # Every field enters the hash, so an auditor can reproduce it from the log alone.
                hashable: Final[Mapping[str, object]] = {
                    "seq": self._call_count,
                    "ts": datetime.now(tz=timezone.utc).isoformat(),
                    "prev_hash": self._prev_hash,
                    **loggable,
                    **clear_content,
                }
                record_hash: Final = _sha256_hex(_canonical_bytes(hashable))
                if not self._write_record({**hashable, "record_hash": record_hash}):
                    return
                self._prev_hash = record_hash
                self._call_count += 1
        except Exception:  # noqa: BLE001  # an audit-log failure must never break the LLM call
            verbose_logger.debug("[AsqavLogger] Unhandled error while building a record", exc_info=True)

    def _write_record(self, record: Mapping[str, object]) -> bool:
        """Append one record to the log file. Returns False if the write failed."""
        try:
            parent: Final = os.path.dirname(self._log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Lock down a file an earlier run created under a permissive umask; new
            # files are created 0o600 by _private_opener.
            if os.path.exists(self._log_path):
                os.chmod(self._log_path, 0o600)
            with open(self._log_path, "a", encoding="utf-8", opener=_private_opener) as fh:
                fh.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        except (OSError, TypeError, ValueError):
            verbose_logger.warning("[AsqavLogger] Failed to write audit record", exc_info=True)
            return False
        else:
            return True

    def log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._build_and_append(kwargs, response_obj, start_time, end_time, "success")

    def log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._build_and_append(kwargs, response_obj, start_time, end_time, "failure")

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await asyncio.to_thread(self._build_and_append, kwargs, response_obj, start_time, end_time, "success")

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await asyncio.to_thread(self._build_and_append, kwargs, response_obj, start_time, end_time, "failure")

    def verify_chain(self, log_path: str | None = None) -> tuple[bool, str]:
        """Check the audit log at log_path (default: this logger's log).

        Returns (True, "ok") when every record's hash matches its content and its
        prev_hash matches the previous record's hash, else (False, reason) for
        the first violation found. Not called on the request path.
        """
        path: Final = log_path or self._log_path
        try:
            with open(path, encoding="utf-8") as fh:
                return _verify_lines(fh)
        except FileNotFoundError:
            return False, f"log file not found: {path}"
        except (OSError, ValueError, ValidationError) as exc:
            return False, f"verification error: {exc}"


def _verify_lines(lines: Iterable[str]) -> tuple[bool, str]:
    prev_hash = GENESIS_HASH  # rebind-ok: walks the chain one record at a time
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        record = _JSON_OBJECT.validate_python(json.loads(line))
        stored_hash = record.get("record_hash")
        computed_hash = _sha256_hex(_canonical_bytes({k: v for k, v in record.items() if k != "record_hash"}))
        if computed_hash != stored_hash:
            return (
                False,
                f"line {lineno}: hash mismatch (stored={str(stored_hash)[:12]}, computed={computed_hash[:12]})",
            )
        record_prev = record.get("prev_hash", GENESIS_HASH)
        if record_prev != prev_hash:
            return (
                False,
                f"line {lineno}: prev_hash chain break (expected={prev_hash[:12]}, got={str(record_prev)[:12]})",
            )
        prev_hash = computed_hash
    return True, "ok"
