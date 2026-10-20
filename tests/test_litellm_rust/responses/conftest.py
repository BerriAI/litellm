import pytest

from tests.test_litellm_rust.responses.data import RESPONSES_RESPONSE
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=RESPONSES_RESPONSE)
    return recording_server
