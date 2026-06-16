import socket
from unittest.mock import MagicMock, patch


def test_create_aiohttp_transport_sets_socket_factory_when_enabled(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPALIVE", True)
    monkeypatch.setattr(
        http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True
    )

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(
                shared_session=None
            )
            transport._get_valid_client_session()

    assert (
        mock_tcp_connector.call_args.kwargs["socket_factory"]
        is http_handler_module._tcp_keepalive_socket_factory
    )


def test_create_aiohttp_transport_omits_socket_factory_when_disabled(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPALIVE", False)
    monkeypatch.setattr(
        http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True
    )

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(
                shared_session=None
            )
            transport._get_valid_client_session()

    assert "socket_factory" not in mock_tcp_connector.call_args.kwargs


def test_create_aiohttp_transport_omits_socket_factory_when_unsupported(monkeypatch):
    """Older aiohttp without socket_factory must not receive the kwarg (would TypeError)."""
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPALIVE", True)
    monkeypatch.setattr(
        http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", False
    )

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(
                shared_session=None
            )
            transport._get_valid_client_session()

    assert "socket_factory" not in mock_tcp_connector.call_args.kwargs


def test_tcp_keepalive_socket_factory_sets_socket_options():
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    addr_info = socket.getaddrinfo(
        "127.0.0.1", 80, socket.AF_INET, socket.SOCK_STREAM
    )[0]
    sock = http_handler_module._tcp_keepalive_socket_factory(addr_info)
    try:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
        tcp_idle_option = getattr(
            socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None)
        )
        if tcp_idle_option is not None:
            assert (
                sock.getsockopt(socket.IPPROTO_TCP, tcp_idle_option)
                == http_handler_module.AIOHTTP_TCP_KEEPALIVE_IDLE
            )
        if hasattr(socket, "TCP_KEEPINTVL"):
            assert (
                sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)
                == http_handler_module.AIOHTTP_TCP_KEEPALIVE_INTVL
            )
        if hasattr(socket, "TCP_KEEPCNT"):
            assert (
                sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)
                == http_handler_module.AIOHTTP_TCP_KEEPALIVE_CNT
            )
    finally:
        sock.close()
