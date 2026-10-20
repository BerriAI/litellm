from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def fixture_planted_prisma_mock():
    with patch("litellm.proxy.proxy_server.prisma_client", MagicMock()):
        yield


def test_monkeypatch_over_fixture_patched_prisma_client(
    fixture_planted_prisma_mock, monkeypatch
):
    """
    Mirrors the flake in test_team_endpoints.py: an autouse fixture patches
    prisma_client, the test monkeypatches the same global, and monkeypatch
    records the fixture's MagicMock as the value to restore. Its undo runs
    after every other finalizer, so without hook-level isolation the mock
    leaks and every later no-database test on the worker fails awaiting it.
    """
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", AsyncMock())
    assert isinstance(proxy_server.prisma_client, AsyncMock)


def test_prisma_client_did_not_leak_from_previous_test():
    import litellm.proxy.proxy_server as proxy_server

    assert not isinstance(proxy_server.prisma_client, MagicMock)
