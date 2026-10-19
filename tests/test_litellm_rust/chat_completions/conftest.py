from collections.abc import Generator

import pytest

from litellm.rust_bridge.catalog import Route
from tests._support.provider import SharedProvider
from tests.test_litellm_rust.support.routing import native_route


@pytest.fixture
def server(recording_server: SharedProvider) -> SharedProvider:
    return recording_server


@pytest.fixture(autouse=True)
def native() -> Generator[None]:
    with native_route(Route.CHAT_COMPLETIONS):
        yield
