from typing import Final

import pytest

from tests.test_litellm_rust.ocr.data import OCR_RESPONSE
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.fixture(params=[False, True], ids=["python", "rust"])
def ocr_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    enabled: Final = bool(request.param)
    monkeypatch.setenv("LITELLM_RUST", "1" if enabled else "0")
    return enabled
