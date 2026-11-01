import asyncio
from unittest.mock import MagicMock, patch


def test_initialize_shared_aiohttp_session_sets_enable_cleanup_closed_when_needed(
    monkeypatch,
):
    from litellm.proxy import proxy_server as proxy_server_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(proxy_server_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", True)

    with patch("aiohttp.TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with patch("aiohttp.ClientSession", return_value=session_mock):
            asyncio.run(proxy_server_module._initialize_shared_aiohttp_session())

    assert mock_tcp_connector.call_args.kwargs["enable_cleanup_closed"] is True


def test_initialize_shared_aiohttp_session_omits_enable_cleanup_closed_when_not_needed(
    monkeypatch,
):
    from litellm.proxy import proxy_server as proxy_server_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(proxy_server_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", False)

    with patch("aiohttp.TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with patch("aiohttp.ClientSession", return_value=session_mock):
            asyncio.run(proxy_server_module._initialize_shared_aiohttp_session())

    assert "enable_cleanup_closed" not in mock_tcp_connector.call_args.kwargs


def test_initialize_shared_aiohttp_session_raises_header_size_limits():
    from litellm.constants import AIOHTTP_MAX_FIELD_SIZE, AIOHTTP_MAX_LINE_SIZE
    from litellm.proxy import proxy_server as proxy_server_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")

    with patch("aiohttp.TCPConnector", return_value=connector_mock):
        with patch("aiohttp.ClientSession", return_value=session_mock) as mock_session:
            asyncio.run(proxy_server_module._initialize_shared_aiohttp_session())

    assert mock_session.call_args.kwargs["max_field_size"] == AIOHTTP_MAX_FIELD_SIZE
    assert mock_session.call_args.kwargs["max_line_size"] == AIOHTTP_MAX_LINE_SIZE
    assert AIOHTTP_MAX_FIELD_SIZE > 8190
