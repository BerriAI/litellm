from unittest.mock import MagicMock, patch


def test_create_aiohttp_transport_sets_enable_cleanup_closed_when_needed(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", True)

    with patch.object(http_handler_module, "TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with patch.object(http_handler_module, "ClientSession", return_value=session_mock):
            transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(shared_session=None)
            transport._get_valid_client_session()

    assert mock_tcp_connector.call_args.kwargs["enable_cleanup_closed"] is True


def test_create_aiohttp_transport_omits_enable_cleanup_closed_when_not_needed(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", False)

    with patch.object(http_handler_module, "TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with patch.object(http_handler_module, "ClientSession", return_value=session_mock):
            transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(shared_session=None)
            transport._get_valid_client_session()

    assert "enable_cleanup_closed" not in mock_tcp_connector.call_args.kwargs
