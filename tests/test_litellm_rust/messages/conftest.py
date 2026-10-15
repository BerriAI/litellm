import pytest

from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server
