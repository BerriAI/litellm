from functools import partial

import pytest

import litellm
from tests.test_litellm_rust.messages.data import MESSAGES_RESPONSE, REQUEST, call
from tests.test_litellm_rust.support.caching import Call, public_call
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec


@pytest.fixture
def server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.serve(ResponseSpec(body=MESSAGES_RESPONSE))
    return recording_server


@pytest.fixture
def python_call(server: RecordingServer) -> Call:
    return public_call(litellm.anthropic.messages.acreate, REQUEST, server)


@pytest.fixture
def native_call(server: RecordingServer) -> Call:
    return partial(call, server)
