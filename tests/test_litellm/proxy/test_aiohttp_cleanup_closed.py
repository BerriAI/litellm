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
