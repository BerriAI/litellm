from collections.abc import Generator

import pytest

from tests.test_litellm_rust.support.recording_server import RecordingServer, recording_service


@pytest.fixture
def reducto_server() -> Generator[RecordingServer]:
    with recording_service() as server:
        yield server
