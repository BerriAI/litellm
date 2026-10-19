from collections.abc import Generator

import pytest

from tests._support.provider import SharedProvider, shared_provider


@pytest.fixture
def reducto_server(request: pytest.FixtureRequest) -> Generator[SharedProvider]:
    with shared_provider() as server, server.checked(request.node.nodeid):
        yield server
