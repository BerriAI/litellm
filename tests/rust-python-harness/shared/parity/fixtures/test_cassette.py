from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import httpx
import pytest
from vcr import VCR
from vcr.request import Request

from ..fixture_models import ParityCase, SdkInputBase
from .cassette import deserialize_cassette
from .recording import RecordedInteraction
from .store import load_fixture, save_fixture
from ..recorded_http import (
    HttpHeader,
    RecordedHttpResponse,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)
from ..replay import replay_server

_URI: Final = "http://parity-provider.invalid/operation?api-version=1"


class _Input(SdkInputBase):
    model: str = "fixture-model"


@pytest.mark.parametrize("body", (b'{"text":"caf\xc3\xa9"}', b"\x00\xff\x80", b""))
def test_cassette_replays_repeated_requests_with_vcr_and_preserves_bytes(tmp_path: Path, body: bytes) -> None:
    sdk_input: Final = _Input()
    responses: Final = tuple(
        RecordedHttpResponse.from_bytes(
            status,
            (HttpHeader(name="content-type", value="application/octet-stream"),),
            body,
        )
        for status in (200, 429)
    )
    case: Final = ParityCase[_Input](litellm_input=sdk_input, provider_responses=responses)
    interactions: Final = tuple(
        RecordedInteraction(Request("POST", _URI, b"\xffrequest", {}), response) for response in responses
    )
    timestamp: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)
    path: Final = save_fixture(tmp_path, sdk_input, case, interactions, recorded_at=timestamp)

    assert load_fixture(tmp_path, sdk_input, ParityCase[_Input]) == case
    assert deserialize_cassette(path.read_text()).recorded_at == timestamp
    with VCR().use_cassette(str(path), record_mode="none", match_on=("method", "uri", "body")) as cassette:
        for status in (200, 429):
            replayed: Final = httpx.post(_URI, content=b"\xffrequest")
            assert replayed.status_code == status
            assert replayed.content == body
        assert cassette.all_played


def test_stream_cassette_preserves_chunk_boundaries_through_local_replay(tmp_path: Path) -> None:
    sdk_input: Final = _Input()
    chunks: Final = (b"data: caf\xc3", b"\xa9\n\n", b"data: [DONE]\n\n")
    response: Final = RecordedHttpStreamResponse(
        kind="http_stream",
        status_code=200,
        headers=(HttpHeader(name="content-type", value="text/event-stream"),),
        chunks=tuple(RecordedStreamChunk.from_bytes(chunk) for chunk in chunks),
    )
    case: Final = ParityCase[_Input](litellm_input=sdk_input, provider_responses=(response,))
    path: Final = save_fixture(
        tmp_path, sdk_input, case, (RecordedInteraction(Request("POST", _URI, b"{}", {}), response),)
    )
    loaded: Final = load_fixture(tmp_path, sdk_input, ParityCase[_Input])
    assert loaded == case
    with replay_server() as server:
        server.enqueue_response(loaded.provider_responses[0])
        with httpx.stream("POST", f"{server.url}/operation", content=b"{}") as replayed:
            assert tuple(replayed.iter_raw()) == chunks
        server.take_requests(1)
    path.write_text(path.read_text().replace("- 10\n", "- 999\n"))
    with pytest.raises(ValueError, match="invalid parity cassette"):
        load_fixture(tmp_path, sdk_input, ParityCase[_Input])


def test_cassette_preserves_duplicate_response_headers(tmp_path: Path) -> None:
    sdk_input: Final = _Input()
    response: Final[RecordedResponse] = RecordedHttpResponse.from_bytes(
        200,
        (HttpHeader(name="x-test", value="first"), HttpHeader(name="x-test", value="second")),
        b"{}",
    )
    case: Final = ParityCase[_Input](litellm_input=sdk_input, provider_responses=(response,))
    save_fixture(tmp_path, sdk_input, case, (RecordedInteraction(Request("POST", _URI, b"", {}), response),))

    assert load_fixture(tmp_path, sdk_input, ParityCase[_Input]) == case
