from functools import partial

import pytest

from tests.test_litellm_rust.chat_completions.data import call
from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE
from tests.test_litellm_rust.support.caching import Call
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.serve(ResponseSpec(body=MESSAGES_RESPONSE))
    return recording_server


@pytest.fixture
def native_call(server: RecordingServer) -> Call:
    return partial(call, server)
