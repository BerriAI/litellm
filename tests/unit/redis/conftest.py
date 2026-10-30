import fakeredis
import pytest


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw(server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server)
