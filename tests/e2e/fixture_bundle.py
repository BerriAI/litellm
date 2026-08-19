"""On-disk fixture bundle format for record/replay e2e runs (LIT-5729).

A bundle is a directory: one ``manifest.json`` (record timestamp + harness
version + format version) plus one subdirectory per test, holding one JSON file
per transport interaction in call order. Bundles older than
``MAX_BUNDLE_AGE`` hard-fail replay at collection time (see conftest), so a
green replay run can never certify against fixtures that have drifted more than
a week from the live proxy.

This module owns the format only. The transports that produce and consume it
live in fixture_transport.py; canonical request matching, streaming chunk
fidelity, and provider-scoping are follow-ups (LIT-5741/5742/5745) and are
deliberately absent here, which is why every interaction file stores the full
redacted request even though replay today matches by call order.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, Field, JsonValue, TypeAdapter

from e2e_http import (
    BinaryStream,
    NetworkError,
    ProbeResult,
    RateLimitedError,
    Result,
    StreamingResponse,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
)

BUNDLE_FORMAT_VERSION: Final = 1
MAX_BUNDLE_AGE: Final = timedelta(days=7)
MANIFEST_FILENAME: Final = "manifest.json"

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class Manifest(BaseModel):
    format_version: int
    recorded_at: datetime
    harness_version: str


class RecordedRequest(BaseModel):
    """The request as the transport saw it, auth header values redacted.

    Replay today only matches ``method`` (the transport verb, not the HTTP verb)
    and ``path`` in call order; the rest is stored so LIT-5741 can move to
    content-based match keys without re-recording. File uploads store a content
    digest instead of the bytes."""

    method: str
    path: str
    headers: dict[str, str]
    params: dict[str, str] = {}
    body: JsonValue | None = None
    form: dict[str, str] | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    file_bytes: int | None = None


class RecordedResult(BaseModel):
    """A ``Result[R]`` flattened for disk. ``data`` holds the success payload as
    raw JSON; replay re-validates it against the ``response_type`` the caller
    passes, exactly like a live response body."""

    shape: Literal["result"] = "result"
    kind: Literal["success", "network", "unauthorized", "rate_limited", "validation", "unknown"]
    status_code: int | None = None
    data: JsonValue | None = None
    message: str | None = None
    body: str | None = None
    retry_after_seconds: int | None = None


class RecordedStreaming(BaseModel):
    shape: Literal["streaming"] = "streaming"
    payload: StreamingResponse


class RecordedBinary(BaseModel):
    shape: Literal["binary"] = "binary"
    payload: BinaryStream


class RecordedProbe(BaseModel):
    shape: Literal["probe"] = "probe"
    payload: ProbeResult


type RecordedResponse = RecordedResult | RecordedStreaming | RecordedBinary | RecordedProbe


class Interaction(BaseModel):
    request: RecordedRequest
    response: Annotated[
        RecordedResult | RecordedStreaming | RecordedBinary | RecordedProbe,
        Field(discriminator="shape"),
    ]


def to_json_value(model: BaseModel) -> JsonValue:
    return _JSON.validate_json(model.model_dump_json(by_alias=True))


def from_result[R: BaseModel](result: Result[R]) -> RecordedResult:
    match result:
        case Success(status_code=status_code, data=data):
            return RecordedResult(kind="success", status_code=status_code, data=to_json_value(data))
        case NetworkError(message=message):
            return RecordedResult(kind="network", message=message)
        case UnauthorizedError():
            return RecordedResult(kind="unauthorized")
        case RateLimitedError(retry_after_seconds=retry_after_seconds, body=body):
            return RecordedResult(kind="rate_limited", retry_after_seconds=retry_after_seconds, body=body)
        case ValidationError(message=message):
            return RecordedResult(kind="validation", message=message)
        case UnknownApiError(status_code=status_code, body=body):
            return RecordedResult(kind="unknown", status_code=status_code, body=body)


def to_result[R: BaseModel](recorded: RecordedResult, response_type: type[R]) -> Result[R]:
    match recorded.kind:
        case "success":
            return Success(
                status_code=recorded.status_code or 200,
                data=response_type.model_validate(recorded.data),
            )
        case "network":
            return NetworkError(message=recorded.message or "")
        case "unauthorized":
            return UnauthorizedError()
        case "rate_limited":
            return RateLimitedError(
                retry_after_seconds=recorded.retry_after_seconds, body=recorded.body or ""
            )
        case "validation":
            return ValidationError(message=recorded.message or "")
        case "unknown":
            return UnknownApiError(status_code=recorded.status_code or 0, body=recorded.body or "")


def slugify(raw: str, *, limit: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-")
    return clean[:limit].rstrip("-")


def slug_for_test(test_key: str) -> str:
    """Directory name for one test's interactions: a readable tail plus a short
    digest of the full node id, so same-named methods in different classes or
    files never collide."""
    digest = hashlib.sha1(test_key.encode()).hexdigest()[:8]
    tail = slugify(test_key.rsplit("::", 1)[-1])
    return f"{tail}-{digest}" if tail else digest


def interaction_filename(ordinal: int, request: RecordedRequest) -> str:
    path_part = slugify(request.path, limit=40) or "root"
    return f"{ordinal:04d}-{request.method}-{path_part}.json"


def harness_version() -> str:
    try:
        proc = subprocess.run(
            ("git", "rev-parse", "--short", "HEAD"),
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


@dataclass(slots=True)
class BundleRecorder:
    """Appends interaction files under ``root``, one subdirectory per test, with
    a per-test ordinal that fixes replay order. ``prepare_bundle`` is the only
    constructor: it guarantees the directory started empty with a fresh
    manifest, so record mode never reads (or merges into) an existing bundle."""

    root: Path
    _ordinals: dict[str, int] = field(default_factory=dict)

    def record(self, *, test_key: str, request: RecordedRequest, response: RecordedResponse) -> None:
        slug = slug_for_test(test_key)
        ordinal = self._ordinals.get(slug, 0)
        self._ordinals[slug] = ordinal + 1
        directory = self.root / slug
        directory.mkdir(parents=True, exist_ok=True)
        interaction = Interaction(request=request, response=response)
        target = directory / interaction_filename(ordinal, request)
        target.write_text(interaction.model_dump_json(indent=2), encoding="utf-8")


@dataclass(frozen=True, slots=True)
class UnsafeBundleDir:
    path: Path
    reason: str


def prepare_bundle(root: Path) -> BundleRecorder | UnsafeBundleDir:
    """Start a fresh bundle at ``root`` for record mode: wipe whatever bundle is
    there and write a new manifest. Refuses to wipe a directory that is neither
    empty nor a bundle (no manifest.json), so a mistyped E2E_FIXTURE_DIR can
    never delete unrelated files."""
    if root.exists():
        if not root.is_dir():
            return UnsafeBundleDir(path=root, reason="exists and is not a directory")
        entries = tuple(root.iterdir())
        if entries and not (root / MANIFEST_FILENAME).is_file():
            return UnsafeBundleDir(
                path=root,
                reason=f"is not empty and has no {MANIFEST_FILENAME}; refusing to wipe a non-bundle directory",
            )
        shutil.rmtree(root)
    root.mkdir(parents=True)
    manifest = Manifest(
        format_version=BUNDLE_FORMAT_VERSION,
        recorded_at=datetime.now(timezone.utc),
        harness_version=harness_version(),
    )
    (root / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return BundleRecorder(root=root)


@dataclass(frozen=True, slots=True)
class FreshBundle:
    manifest: Manifest


@dataclass(frozen=True, slots=True)
class StaleBundle:
    recorded_at: datetime
    age: timedelta
    limit: timedelta


@dataclass(frozen=True, slots=True)
class UnreadableBundle:
    reason: str


type BundleFreshness = FreshBundle | StaleBundle | UnreadableBundle


def _read_manifest(root: Path) -> Manifest | UnreadableBundle:
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return UnreadableBundle(reason=f"no {MANIFEST_FILENAME} found (record one with E2E_FIXTURE_MODE=record)")
    try:
        return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return UnreadableBundle(reason=f"{MANIFEST_FILENAME} is invalid: {exc}")


def check_freshness(root: Path, *, now: datetime) -> BundleFreshness:
    manifest = _read_manifest(root)
    if isinstance(manifest, UnreadableBundle):
        return manifest
    if manifest.format_version != BUNDLE_FORMAT_VERSION:
        return UnreadableBundle(
            reason=f"format_version {manifest.format_version} != supported {BUNDLE_FORMAT_VERSION}"
        )
    recorded_at = (
        manifest.recorded_at
        if manifest.recorded_at.tzinfo is not None
        else manifest.recorded_at.replace(tzinfo=timezone.utc)
    )
    age = now - recorded_at
    if age > MAX_BUNDLE_AGE:
        return StaleBundle(recorded_at=recorded_at, age=age, limit=MAX_BUNDLE_AGE)
    return FreshBundle(manifest=manifest)


def format_age(age: timedelta) -> str:
    total_hours = int(age.total_seconds()) // 3600
    return f"{total_hours // 24}d{total_hours % 24}h"


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    manifest: Manifest
    interactions: dict[str, tuple[Interaction, ...]]


def load_bundle(root: Path) -> LoadedBundle | UnreadableBundle:
    manifest = _read_manifest(root)
    if isinstance(manifest, UnreadableBundle):
        return manifest
    interactions = {
        directory.name: tuple(
            Interaction.model_validate_json(file.read_text(encoding="utf-8"))
            for file in sorted(directory.glob("*.json"))
        )
        for directory in sorted(root.iterdir())
        if directory.is_dir()
    }
    return LoadedBundle(manifest=manifest, interactions=interactions)
