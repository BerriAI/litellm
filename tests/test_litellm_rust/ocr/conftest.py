from collections.abc import Generator

import pytest

from litellm.rust_bridge.catalog import Route
from tests.test_litellm_rust.ocr.data import OCR_RESPONSE
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.routing import native_route


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=OCR_RESPONSE)
    return recording_server


@pytest.fixture(autouse=True)
def native() -> Generator[None]:
    with native_route(Route.OCR):
        yield
