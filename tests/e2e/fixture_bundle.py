"""On-disk fixture bundle format for record/replay e2e runs (LIT-5729/LIT-5745).

A bundle is a directory: one ``manifest.json`` (record timestamp + harness
version + format version) plus one subdirectory per test, holding one JSON file
per provider-bound interaction in call order. Bundles older than
``MAX_BUNDLE_AGE`` hard-fail replay at collection time (see conftest), so a
green replay run can never certify against fixtures that have drifted more than
a week from the live providers. Bump ``BUNDLE_FORMAT_VERSION`` whenever a change
moves recorded keys or changes the stored shape: a bundle recorded under the old
rules then fails naming both versions instead of quietly missing on every call.

This module owns the format only. The provider-edge server that produces and
consumes it lives in provider_edge.py (LIT-5745) and the canonical match keys it
computes live in fixture_canonical.py (LIT-5741). Every interaction file stores
the full redacted request because replay matches on its canonicalized content,
and a response in one of two shapes, told apart by their ``kind`` tag: an
ordinary ``RecordedHttpResponse`` holding one base64 body, or, for a response the
provider streamed, a ``RecordedStreamedResponse`` holding its transfer chunks in
order so replay reproduces the same split points (LIT-5742).
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

from pydantic import BaseModel, Field, JsonValue

BUNDLE_FORMAT_VERSION: Final = 4
MAX_BUNDLE_AGE: Final = timedelta(days=7)
MANIFEST_FILENAME: Final = "manifest.json"


class Manifest(BaseModel):
    format_version: int
    recorded_at: datetime
    harness_version: str


class RecordedRequest(BaseModel):
    """The provider-bound request as the edge saw it, headers empty (SDK
    telemetry headers vary run to run and auth material never touches disk).

    Replay matches on the canonical content key fixture_canonical.py computes
    over ``method``, ``path`` (the edge path including the provider mount,
    query string excluded), and the canonicalized headers, params, body, form,
    and file identity. Non-JSON bodies store a canonicalized content digest
    instead of the bytes.

    ``file_name`` is a JSON list of the uploaded parts' ``[field, filename,
    content-type]`` triples rather than a flat label, so a separator inside a
    filename cannot impersonate a field boundary. ``file_bytes`` is recorded for
    a reader's benefit and stays out of the key: the canonicalizer absorbs
    timestamp and id drift inside an uploaded file, and that drift moves the
    byte count."""

    method: str
    path: str
    headers: dict[str, str]
    params: dict[str, str] = {}
    body: JsonValue | None = None
    form: dict[str, str] | None = None
    file_name: str | None = None
    file_sha256: str | None = None
    file_bytes: int | None = None


class RecordedHttpResponse(BaseModel):
    """The provider's raw HTTP response: status, headers minus hop-by-hop and
    volatile entries (see provider_edge.py), and the body as base64 so binary
    payloads survive JSON."""

    kind: Literal["http"] = "http"
    status_code: int
    headers: dict[str, str]
    body_b64: str


class RecordedStreamedResponse(BaseModel):
    """A response the provider streamed, kept chunk by chunk instead of buffered.

    ``chunks_b64`` holds one entry per upstream transfer chunk, in order, so replay
    reproduces the split points the provider chose rather than one coalesced body.
    ``truncated`` is None for a stream that reached its terminator and otherwise
    says why it did not, prefixed by which side ended it (``upstream:`` for a
    provider that hung up mid-stream, ``downstream:`` for a proxy that stopped
    reading). Replay behaves the same for any truncation, delivering the recorded
    chunks and then closing; the reason is there for whoever reads the bundle."""

    kind: Literal["streamed"] = "streamed"
    status_code: int
    headers: dict[str, str]
    chunks_b64: list[str]
    truncated: str | None = None


type RecordedResponse = Annotated[
    RecordedHttpResponse | RecordedStreamedResponse, Field(discriminator="kind")
]


class Interaction(BaseModel):
    request: RecordedRequest
    response: RecordedResponse


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


def _supported_manifest(root: Path) -> Manifest | UnreadableBundle:
    """The manifest, refused when it was written under a different format version.
    A bundle is atomic (record wipes and rewrites the whole directory and never
    merges), so a foreign version is a hard reject rather than a partial read."""
    manifest = _read_manifest(root)
    if isinstance(manifest, UnreadableBundle):
        return manifest
    if manifest.format_version != BUNDLE_FORMAT_VERSION:
        return UnreadableBundle(
            reason=(
                f"format_version {manifest.format_version} != supported {BUNDLE_FORMAT_VERSION}; "
                "re-record with E2E_FIXTURE_MODE=record"
            )
        )
    return manifest


def check_freshness(root: Path, *, now: datetime) -> BundleFreshness:
    manifest = _supported_manifest(root)
    if isinstance(manifest, UnreadableBundle):
        return manifest
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
    manifest = _supported_manifest(root)
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
