from unittest.mock import MagicMock, patch


def test_create_aiohttp_transport_sets_enable_cleanup_closed_when_needed(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", True)

    with patch.object(http_handler_module, "TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with (
            patch.object(http_handler_module, "ClientSession", return_value=session_mock),
            patch.object(http_handler_module, "DummyCookieJar", return_value=MagicMock(name="cookie_jar")),
        ):
            transport = http_handler_module.AsyncHTTPHandler.create_aiohttp_transport(shared_session=None)
            transport.get_valid_client_session()

    assert mock_tcp_connector.call_args.kwargs["enable_cleanup_closed"] is True


def test_create_aiohttp_transport_omits_enable_cleanup_closed_when_not_needed(
    monkeypatch,
):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_NEEDS_CLEANUP_CLOSED", False)

    with patch.object(http_handler_module, "TCPConnector", return_value=connector_mock) as mock_tcp_connector:
        with (
            patch.object(http_handler_module, "ClientSession", return_value=session_mock),
            patch.object(http_handler_module, "DummyCookieJar", return_value=MagicMock(name="cookie_jar")),
        ):
            transport = http_handler_module.AsyncHTTPHandler.create_aiohttp_transport(shared_session=None)
            transport.get_valid_client_session()

    assert "enable_cleanup_closed" not in mock_tcp_connector.call_args.kwargs


def test_create_aiohttp_transport_raises_header_size_limits():
    from litellm.constants import AIOHTTP_MAX_FIELD_SIZE, AIOHTTP_MAX_LINE_SIZE
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")

    with patch.object(http_handler_module, "TCPConnector", return_value=connector_mock):
        with (
            patch.object(http_handler_module, "ClientSession", return_value=session_mock) as mock_session,
            patch.object(http_handler_module, "DummyCookieJar", return_value=MagicMock(name="cookie_jar")),
        ):
            transport = http_handler_module.AsyncHTTPHandler.create_aiohttp_transport(shared_session=None)
            transport.get_valid_client_session()

    assert mock_session.call_args.kwargs["max_field_size"] == AIOHTTP_MAX_FIELD_SIZE
    assert mock_session.call_args.kwargs["max_line_size"] == AIOHTTP_MAX_LINE_SIZE
    assert AIOHTTP_MAX_FIELD_SIZE > 8190
    assert AIOHTTP_MAX_LINE_SIZE > 8190
