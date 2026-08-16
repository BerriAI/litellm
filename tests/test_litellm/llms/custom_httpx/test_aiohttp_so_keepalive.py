import socket
from unittest.mock import MagicMock, patch


def _invoke_connector_factory(http_handler_module):
    """
    Drive the lambda factory installed on the transport so TCPConnector is
    actually constructed. _create_aiohttp_transport returns a transport whose
    _client_factory is the lambda that builds (TCPConnector → ClientSession);
    invoking it directly avoids relying on _get_valid_client_session's internal
    branching to trigger connector construction.
    """
    transport = http_handler_module.AsyncHTTPHandler._create_aiohttp_transport(
        shared_session=None
    )
    transport._client_factory()
    return transport


def test_socket_factory_omitted_when_disabled(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    monkeypatch.setattr(http_handler_module, "AIOHTTP_SO_KEEPALIVE", False)
    monkeypatch.setattr(http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True)

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            _invoke_connector_factory(http_handler_module)

    assert mock_tcp_connector.call_count >= 1
    assert "socket_factory" not in mock_tcp_connector.call_args.kwargs


def test_socket_factory_attached_when_enabled(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    monkeypatch.setattr(http_handler_module, "AIOHTTP_SO_KEEPALIVE", True)
    monkeypatch.setattr(http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True)

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            _invoke_connector_factory(http_handler_module)

    assert mock_tcp_connector.call_count >= 1
    factory = mock_tcp_connector.call_args.kwargs.get("socket_factory")
    assert callable(factory)


def test_socket_factory_skipped_on_old_aiohttp(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    monkeypatch.setattr(http_handler_module, "AIOHTTP_SO_KEEPALIVE", True)
    monkeypatch.setattr(http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", False)

    connector_mock = MagicMock(name="connector")
    session_mock = MagicMock(name="session")

    with patch.object(
        http_handler_module, "TCPConnector", return_value=connector_mock
    ) as mock_tcp_connector:
        with patch.object(
            http_handler_module, "ClientSession", return_value=session_mock
        ):
            _invoke_connector_factory(http_handler_module)

    assert mock_tcp_connector.call_count >= 1
    assert "socket_factory" not in mock_tcp_connector.call_args.kwargs


def test_socket_factory_sets_keepalive_options(monkeypatch):
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    monkeypatch.setattr(http_handler_module, "AIOHTTP_SO_KEEPALIVE", True)
    monkeypatch.setattr(http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True)
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPIDLE", 45)
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPINTVL", 15)
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPCNT", 4)

    factory = http_handler_module._build_aiohttp_keepalive_socket_factory()
    assert factory is not None

    addr_info = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("", 0))

    fake_sock = MagicMock(spec=socket.socket)
    with patch("socket.socket", return_value=fake_sock) as sock_ctor:
        returned = factory(addr_info)

    sock_ctor.assert_called_once_with(
        family=socket.AF_INET, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
    )
    assert returned is fake_sock
    fake_sock.setblocking.assert_called_once_with(False)

    setsockopt_calls = {
        (call.args[0], call.args[1]): call.args[2]
        for call in fake_sock.setsockopt.call_args_list
    }
    assert setsockopt_calls[(socket.SOL_SOCKET, socket.SO_KEEPALIVE)] == 1

    if hasattr(socket, "TCP_KEEPIDLE"):
        assert setsockopt_calls[(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE)] == 45
    elif hasattr(socket, "TCP_KEEPALIVE"):
        assert setsockopt_calls[(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE)] == 45
    if hasattr(socket, "TCP_KEEPINTVL"):
        assert setsockopt_calls[(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL)] == 15
    if hasattr(socket, "TCP_KEEPCNT"):
        assert setsockopt_calls[(socket.IPPROTO_TCP, socket.TCP_KEEPCNT)] == 4


def test_socket_factory_uses_tcp_keepalive_when_keepidle_unavailable(monkeypatch):
    """
    Cover the macOS/Darwin branch: when TCP_KEEPIDLE is missing but TCP_KEEPALIVE
    is present, the factory should fall back to TCP_KEEPALIVE for the idle timer.
    Linux CI runners always have TCP_KEEPIDLE, so we patch socket itself to
    simulate the BSD-derived environment.
    """
    from litellm.llms.custom_httpx import http_handler as http_handler_module

    monkeypatch.setattr(http_handler_module, "AIOHTTP_SO_KEEPALIVE", True)
    monkeypatch.setattr(http_handler_module, "_AIOHTTP_SUPPORTS_SOCKET_FACTORY", True)
    monkeypatch.setattr(http_handler_module, "AIOHTTP_TCP_KEEPIDLE", 60)

    factory = http_handler_module._build_aiohttp_keepalive_socket_factory()
    assert factory is not None

    fake_socket_module = MagicMock(spec=[])
    fake_socket_module.SOL_SOCKET = socket.SOL_SOCKET
    fake_socket_module.SO_KEEPALIVE = socket.SO_KEEPALIVE
    fake_socket_module.IPPROTO_TCP = socket.IPPROTO_TCP
    fake_socket_module.TCP_KEEPALIVE = getattr(socket, "TCP_KEEPALIVE", 0x10)
    fake_sock = MagicMock(spec=socket.socket)
    fake_socket_module.socket = MagicMock(return_value=fake_sock)

    addr_info = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("", 0))

    with patch.object(http_handler_module, "socket", fake_socket_module):
        factory(addr_info)

    setsockopt_calls = {
        (call.args[0], call.args[1]): call.args[2]
        for call in fake_sock.setsockopt.call_args_list
    }
    assert setsockopt_calls[(socket.SOL_SOCKET, socket.SO_KEEPALIVE)] == 1
    assert (
        setsockopt_calls[(socket.IPPROTO_TCP, fake_socket_module.TCP_KEEPALIVE)] == 60
    )
    assert (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", -1)) not in setsockopt_calls
