"""Record/replay transports behind the same ``Transport`` protocol (LIT-5729).

``RecordingTransport`` decorates the live transport: every call passes through
unchanged and its request/response pair is appended to the fixture bundle.
``ReplayTransport`` implements the protocol from a recorded bundle alone: no
HTTP, no proxy, no provider spend. Because both fulfil ``Transport``, no test
or client changes shape; ``build_proxy_client`` picks the transport from
``E2E_FIXTURE_MODE`` (live | record | replay, default live).

Replay matches each call by test node id and call order, verifying transport
verb + path and failing hard on any drift (``ReplayMiss``). Canonical
content-based match keys are LIT-5741; streaming chunk fidelity is LIT-5742;
scoping record/replay to provider-bound traffic is LIT-5745.
"""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, assert_never

from pydantic import BaseModel

from e2e_http import AuthHeaders, BinaryStream, ProbeResult, Result, StreamingResponse
from fixture_bundle import (
    BundleRecorder,
    FreshBundle,
    Interaction,
    LoadedBundle,
    RecordedBinary,
    RecordedProbe,
    RecordedRequest,
    RecordedResponse,
    RecordedResult,
    RecordedStreaming,
    StaleBundle,
    UnreadableBundle,
    UnsafeBundleDir,
    check_freshness,
    format_age,
    from_result,
    load_bundle,
    prepare_bundle,
    slug_for_test,
    to_json_value,
    to_result,
)
from transport import Transport

type FixtureMode = Literal["live", "record", "replay"]

FIXTURE_MODES: Final[tuple[FixtureMode, ...]] = ("live", "record", "replay")

SESSION_TEST_KEY: Final = "session"

REDACTED_HEADER_NAMES: Final[frozenset[str]] = frozenset({"authorization", "x-litellm-api-key"})
REDACTED_VALUE: Final = "<redacted>"


@dataclass(frozen=True, slots=True)
class InvalidFixtureMode:
    value: str


def parse_fixture_mode(raw: str) -> FixtureMode | InvalidFixtureMode:
    normalized = raw.strip().lower() or "live"
    match normalized:
        case "live" | "record" | "replay":
            return normalized
        case _:
            return InvalidFixtureMode(value=raw)


def current_test_key() -> str:
    """The pytest node id of the running test, from the PYTEST_CURRENT_TEST env
    var pytest maintains (``<nodeid> (setup|call|teardown)``); ``session`` for
    calls outside any test (e.g. session-finish cleanup)."""
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    if not raw:
        return SESSION_TEST_KEY
    return raw.rsplit(" (", 1)[0]


class ReplayMiss(AssertionError):
    """Replay had no recorded interaction for a call the suite made. The test
    drifted from the bundle (or the bundle from the suite): re-record."""


_marker_ordinals: Final[dict[str, int]] = {}


def deterministic_marker() -> str:
    """Stable stand-in for uuid-based unique markers in record and replay modes:
    the Nth marker of a test is a pure function of the test's node id and N, so a
    replay run regenerates exactly the model names, prompts, and tags the record
    run sent and every recorded poll response still satisfies its predicate."""
    test_key = current_test_key()
    ordinal = _marker_ordinals.get(test_key, 0)
    _marker_ordinals[test_key] = ordinal + 1
    return hashlib.sha1(f"{test_key}#{ordinal}".encode()).hexdigest()[:12]


def _dump_flat(model: BaseModel | None) -> dict[str, str]:
    if model is None:
        return {}
    dumped: dict[str, object] = model.model_dump(by_alias=True, exclude_none=True)
    return {key: str(value) for key, value in dumped.items()}


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {
        name: REDACTED_VALUE if name.lower() in REDACTED_HEADER_NAMES else value
        for name, value in headers.items()
    }


def recorded_request(
    method: str,
    path: str,
    *,
    headers: BaseModel,
    body: BaseModel | None = None,
    params: BaseModel | None = None,
    form: BaseModel | None = None,
    file_name: str | None = None,
    file_content: bytes | None = None,
) -> RecordedRequest:
    return RecordedRequest(
        method=method,
        path=path,
        headers=_redact(_dump_flat(headers)),
        params=_dump_flat(params),
        body=None if body is None else to_json_value(body),
        form=None if form is None else _dump_flat(form),
        file_name=file_name,
        file_sha256=None if file_content is None else hashlib.sha256(file_content).hexdigest(),
        file_bytes=None if file_content is None else len(file_content),
    )


@dataclass(frozen=True, slots=True)
class RecordingTransport:
    """Decorator over the live transport: forwards every call and appends the
    interaction to the bundle, so a green live run leaves behind exactly the
    traffic replay needs."""

    inner: Transport
    recorder: BundleRecorder

    def _record(self, request: RecordedRequest, response: RecordedResponse) -> None:
        self.recorder.record(test_key=current_test_key(), request=request, response=response)

    def bearer(self, key: str) -> AuthHeaders:
        return self.inner.bearer(key)

    @property
    def master(self) -> AuthHeaders:
        return self.inner.master

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        result = self.inner.post(path, headers=headers, json=json, response_type=response_type)
        self._record(recorded_request("post", path, headers=headers, body=json), from_result(result))
        return result

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
        timeout: float | None = None,
    ) -> Result[R]:
        result = self.inner.get(
            path, headers=headers, params=params, response_type=response_type, timeout=timeout
        )
        self._record(recorded_request("get", path, headers=headers, params=params), from_result(result))
        return result

    def delete[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        response_type: type[R],
        params: BaseModel | None = None,
    ) -> Result[R]:
        result = self.inner.delete(
            path, headers=headers, json=json, response_type=response_type, params=params
        )
        self._record(
            recorded_request("delete", path, headers=headers, body=json, params=params),
            from_result(result),
        )
        return result

    def patch[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        result = self.inner.patch(path, headers=headers, json=json, response_type=response_type)
        self._record(recorded_request("patch", path, headers=headers, body=json), from_result(result))
        return result

    def put[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        result = self.inner.put(path, headers=headers, json=json, response_type=response_type)
        self._record(recorded_request("put", path, headers=headers, body=json), from_result(result))
        return result

    def stream(self, path: str, *, headers: BaseModel, json: BaseModel) -> StreamingResponse:
        response = self.inner.stream(path, headers=headers, json=json)
        self._record(
            recorded_request("stream", path, headers=headers, body=json),
            RecordedStreaming(payload=response),
        )
        return response

    def stream_binary(
        self, path: str, *, headers: BaseModel, json: BaseModel, chunk_size: int = 8192
    ) -> BinaryStream:
        response = self.inner.stream_binary(path, headers=headers, json=json, chunk_size=chunk_size)
        self._record(
            recorded_request("stream_binary", path, headers=headers, body=json),
            RecordedBinary(payload=response),
        )
        return response

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        response = self.inner.send(path, headers=headers, json=json, params=params, stream=stream)
        self._record(
            recorded_request("send", path, headers=headers, body=json, params=params),
            RecordedStreaming(payload=response),
        )
        return response

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        response = self.inner.probe(path, params=params)
        self._record(
            recorded_request("probe", path, headers=self.master, params=params),
            RecordedProbe(payload=response),
        )
        return response

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: BaseModel,
        filename: str,
        content: bytes,
        file_content_type: str = "application/jsonl",
        file_field: str = "file",
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        result = self.inner.upload(
            path,
            headers=headers,
            form=form,
            filename=filename,
            content=content,
            file_content_type=file_content_type,
            file_field=file_field,
            params=params,
            response_type=response_type,
        )
        self._record(
            recorded_request(
                "upload",
                path,
                headers=headers,
                params=params,
                form=form,
                file_name=filename,
                file_content=content,
            ),
            from_result(result),
        )
        return result

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        response = self.inner.download(path, headers=headers)
        self._record(
            recorded_request("download", path, headers=headers),
            RecordedStreaming(payload=response),
        )
        return response


@dataclass(slots=True)
class ReplaySource:
    """One shared cursor set over a loaded bundle, so every client built in the
    session consumes the same recorded sequence per test."""

    bundle: LoadedBundle
    _cursors: dict[str, int] = field(default_factory=dict)

    def next_interaction(self, method: str, path: str) -> Interaction:
        test_key = current_test_key()
        slug = slug_for_test(test_key)
        recorded = self.bundle.interactions.get(slug, ())
        index = self._cursors.get(slug, 0)
        if index >= len(recorded):
            raise ReplayMiss(
                f"replay exhausted for {test_key}: call #{index + 1} ({method} {path}) has no recorded "
                f"interaction ({len(recorded)} recorded under {slug}); re-record with E2E_FIXTURE_MODE=record"
            )
        interaction = recorded[index]
        if interaction.request.method != method or interaction.request.path != path:
            raise ReplayMiss(
                f"replay mismatch for {test_key} at call #{index + 1}: recorded "
                f"{interaction.request.method} {interaction.request.path}, test made {method} {path}; "
                "re-record with E2E_FIXTURE_MODE=record"
            )
        self._cursors[slug] = index + 1
        return interaction


def _expect_result(interaction: Interaction) -> RecordedResult:
    match interaction.response:
        case RecordedResult() as recorded:
            return recorded
        case RecordedStreaming() | RecordedBinary() | RecordedProbe():
            raise ReplayMiss(
                f"recorded {interaction.request.method} {interaction.request.path} is not a typed result"
            )


def _expect_streaming(interaction: Interaction) -> StreamingResponse:
    match interaction.response:
        case RecordedStreaming(payload=payload):
            return payload
        case RecordedResult() | RecordedBinary() | RecordedProbe():
            raise ReplayMiss(
                f"recorded {interaction.request.method} {interaction.request.path} is not a streaming response"
            )


@dataclass(frozen=True, slots=True)
class ReplayTransport:
    """A ``Transport`` served entirely from a recorded bundle: never opens a
    connection, so a replay run cannot bill a provider."""

    source: ReplaySource
    master_key: str

    def bearer(self, key: str) -> AuthHeaders:
        return AuthHeaders(authorization=f"Bearer {key}")

    @property
    def master(self) -> AuthHeaders:
        return self.bearer(self.master_key)

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("post", path)), response_type)

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
        timeout: float | None = None,
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("get", path)), response_type)

    def delete[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        response_type: type[R],
        params: BaseModel | None = None,
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("delete", path)), response_type)

    def patch[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("patch", path)), response_type)

    def put[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("put", path)), response_type)

    def stream(self, path: str, *, headers: BaseModel, json: BaseModel) -> StreamingResponse:
        return _expect_streaming(self.source.next_interaction("stream", path))

    def stream_binary(
        self, path: str, *, headers: BaseModel, json: BaseModel, chunk_size: int = 8192
    ) -> BinaryStream:
        interaction = self.source.next_interaction("stream_binary", path)
        match interaction.response:
            case RecordedBinary(payload=payload):
                return payload
            case RecordedResult() | RecordedStreaming() | RecordedProbe():
                raise ReplayMiss(
                    f"recorded stream_binary {interaction.request.path} is not a binary stream"
                )

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        return _expect_streaming(self.source.next_interaction("send", path))

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        interaction = self.source.next_interaction("probe", path)
        match interaction.response:
            case RecordedProbe(payload=payload):
                return payload
            case RecordedResult() | RecordedStreaming() | RecordedBinary():
                raise ReplayMiss(f"recorded probe {interaction.request.path} is not a probe result")

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: BaseModel,
        filename: str,
        content: bytes,
        file_content_type: str = "application/jsonl",
        file_field: str = "file",
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        return to_result(_expect_result(self.source.next_interaction("upload", path)), response_type)

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        return _expect_streaming(self.source.next_interaction("download", path))


@functools.lru_cache(maxsize=8)
def _shared_recorder(root: Path) -> BundleRecorder:
    prepared = prepare_bundle(root)
    if isinstance(prepared, UnsafeBundleDir):
        raise ValueError(f"E2E_FIXTURE_DIR {prepared.path} {prepared.reason}")
    return prepared


@functools.lru_cache(maxsize=8)
def _shared_replay_source(root: Path) -> ReplaySource:
    loaded = load_bundle(root)
    if isinstance(loaded, UnreadableBundle):
        raise ValueError(f"cannot replay from {root}: {loaded.reason}")
    return ReplaySource(bundle=loaded)


def select_transport(
    live: Transport, *, mode_raw: str, bundle_dir: Path, master_key: str
) -> Transport:
    """The one seam every client build goes through: wraps (record), replaces
    (replay), or passes through (live) the transport per E2E_FIXTURE_MODE. The
    recorder and replay cursors are process-wide singletons per bundle dir, so
    every client in a session shares one bundle and one recorded sequence."""
    mode = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode(value=value):
            raise ValueError(f"E2E_FIXTURE_MODE={value!r} is not one of {', '.join(FIXTURE_MODES)}")
        case "live":
            return live
        case "record":
            return RecordingTransport(inner=live, recorder=_shared_recorder(bundle_dir))
        case "replay":
            return ReplayTransport(source=_shared_replay_source(bundle_dir), master_key=master_key)
        case _:
            assert_never(mode)


def fixture_mode_collection_error(mode_raw: str, bundle_dir: Path, *, now: datetime) -> str | None:
    """Session-abort reason for a fixture-mode setup that can never work, or None.
    Called at collection time (conftest pytest_sessionstart) so a stale or missing
    bundle fails the whole run up front, naming the bundle age, instead of failing
    every test individually."""
    mode = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode(value=value):
            return f"E2E_FIXTURE_MODE={value!r} is not one of {', '.join(FIXTURE_MODES)}"
        case "live" | "record":
            return None
        case "replay":
            freshness = check_freshness(bundle_dir, now=now)
            match freshness:
                case FreshBundle():
                    return None
                case StaleBundle(recorded_at=recorded_at, age=age, limit=limit):
                    return (
                        f"fixture bundle at {bundle_dir} is stale: recorded {recorded_at.isoformat()}, "
                        f"age {format_age(age)} exceeds the {limit.days}-day limit; "
                        "re-record with E2E_FIXTURE_MODE=record"
                    )
                case UnreadableBundle(reason=reason):
                    return f"E2E_FIXTURE_MODE=replay cannot use bundle at {bundle_dir}: {reason}"
                case _:
                    assert_never(freshness)
        case _:
            assert_never(mode)


def fixture_report_lines(mode_raw: str, bundle_dir: Path, *, now: datetime) -> list[str]:
    """pytest report-header lines; empty in live mode so an unset
    E2E_FIXTURE_MODE keeps today's output byte-identical."""
    mode = parse_fixture_mode(mode_raw)
    match mode:
        case InvalidFixtureMode() | "live":
            return []
        case "record":
            return [f"e2e fixture mode: record -> {bundle_dir}"]
        case "replay":
            freshness = check_freshness(bundle_dir, now=now)
            match freshness:
                case FreshBundle(manifest=manifest):
                    return [
                        f"e2e fixture mode: replay <- {bundle_dir} "
                        f"(recorded {manifest.recorded_at.isoformat()}, harness {manifest.harness_version})"
                    ]
                case StaleBundle() | UnreadableBundle():
                    return [f"e2e fixture mode: replay <- {bundle_dir}"]
                case _:
                    assert_never(freshness)
        case _:
            assert_never(mode)
