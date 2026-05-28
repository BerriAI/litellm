"""
Regression tests for LIT-3369.

Setting `ssl_verify: False` (or `SSL_VERIFY` env var) globally in
`litellm_settings` should disable SSL verification on the
``aiohttp_openai`` provider path, which uses
``BaseLLMAIOHTTPHandler`` rather than ``AsyncHTTPHandler.create_client``.

Pre-fix: ``BaseLLMAIOHTTPHandler._create_client_session_with_transport``
fell through to a bare ``aiohttp.ClientSession()`` with the default
verifying SSL context, so the global setting was silently ignored.

Post-fix: it lazily creates a transport via
``BaseLLMAIOHTTPHandler._get_or_create_transport``, which calls
``get_ssl_configuration(None)`` to honor ``litellm.ssl_verify``,
``SSL_VERIFY`` env, ``SSL_CERT_FILE``, etc.
"""
import asyncio
import ssl
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.custom_httpx.aiohttp_handler import BaseLLMAIOHTTPHandler
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


@pytest.fixture(autouse=True)
def _reset_global_ssl_verify():
    """Restore litellm.ssl_verify after each test."""
    original = litellm.ssl_verify
    yield
    litellm.ssl_verify = original


def _create_owned_transport(handler: BaseLLMAIOHTTPHandler) -> LiteLLMAiohttpTransport:
    """Drive the lazy-create path used at runtime by
    ``_create_client_session_with_transport``."""
    handler._get_or_create_transport()
    assert handler.transport is not None, "transport should be created"
    return handler.transport


class TestLit3369GlobalSSLVerifyHonored:
    def test_ssl_verify_false_propagates_to_transport(self):
        """`litellm.ssl_verify = False` -> transport's per-request override is False."""
        litellm.ssl_verify = False
        handler = BaseLLMAIOHTTPHandler()
        transport = _create_owned_transport(handler)
        assert transport._ssl_verify is False, (
            "Expected per-request ssl override to be False, "
            f"got {transport._ssl_verify!r}"
        )

    def test_ssl_verify_false_via_env_propagates_to_transport(self, monkeypatch):
        """`SSL_VERIFY` env var also disables verification on this path."""
        litellm.ssl_verify = True
        monkeypatch.setenv("SSL_VERIFY", "False")
        handler = BaseLLMAIOHTTPHandler()
        transport = _create_owned_transport(handler)
        assert transport._ssl_verify is False

    def test_ssl_verify_true_does_not_force_disable(self):
        """Default (verify=True) -> transport must NOT collapse to ssl=False.
        Ensures the fix does not silently weaken TLS verification for users
        who did not opt out."""
        litellm.ssl_verify = True
        handler = BaseLLMAIOHTTPHandler()
        transport = _create_owned_transport(handler)
        assert transport._ssl_verify is not False, (
            "Default ssl_verify=True must not collapse to ssl=False; "
            f"got {transport._ssl_verify!r}"
        )

    def test_lazy_transport_used_by_session_factory(self):
        """`_create_client_session_with_transport` must drive the transport
        path (which carries SSL config), not the bare ClientSession fallback."""
        litellm.ssl_verify = False
        handler = BaseLLMAIOHTTPHandler()

        sentinel_session = MagicMock(name="sentinel_session")

        def _install_mock(self):
            mock_transport = MagicMock(spec=LiteLLMAiohttpTransport)
            mock_transport._get_valid_client_session.return_value = sentinel_session
            self.transport = mock_transport
            self._owns_transport = True
            return mock_transport

        with patch.object(
            BaseLLMAIOHTTPHandler,
            "_get_or_create_transport",
            _install_mock,
        ):
            got = handler._create_client_session_with_transport()

        assert got is sentinel_session, (
            "Expected the transport's session factory to be used; "
            f"got {got!r}"
        )

    def test_externally_supplied_transport_is_not_overwritten(self):
        """If a transport was injected at construction time, the lazy
        create-on-demand path must not replace it."""
        injected = MagicMock(spec=LiteLLMAiohttpTransport)
        injected._get_valid_client_session.return_value = MagicMock(name="inj")
        handler = BaseLLMAIOHTTPHandler(transport=injected)
        assert handler._owns_transport is False
        before = handler.transport
        _ = handler._create_client_session_with_transport()
        assert handler.transport is before, "Injected transport must be preserved"

    def test_externally_supplied_connector_is_used(self):
        """If a connector was injected at construction time, the new code
        path must still use it (not lazily create a different transport)."""
        import aiohttp as _aiohttp

        mock_connector = MagicMock(spec=_aiohttp.BaseConnector)
        handler = BaseLLMAIOHTTPHandler(connector=mock_connector)
        assert handler._owns_connector is False

        with patch(
            "litellm.llms.custom_httpx.aiohttp_handler.aiohttp.ClientSession"
        ) as mock_cs:
            mock_session = MagicMock()
            mock_cs.return_value = mock_session
            session = handler._create_client_session_with_transport()

        mock_cs.assert_called_once_with(connector=mock_connector)
        assert session is mock_session

    def test_owned_transport_creation_failure_falls_back_to_default_session(self):
        """If transport creation raises, we still return a usable session
        (back-compat with the legacy `aiohttp.ClientSession()` fallback)."""
        litellm.ssl_verify = False

        async def _drive():
            handler = BaseLLMAIOHTTPHandler()
            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler._create_aiohttp_transport",
                side_effect=RuntimeError("simulated"),
            ):
                session = handler._create_client_session_with_transport()
            try:
                assert session is not None
                # transport creation bailed -> stays None
                assert handler.transport is None
            finally:
                if session is not None and not session.closed:
                    await session.close()

        asyncio.run(_drive())


class TestLit3369GetOrCreateTransportSSLResolution:
    """Direct unit tests on _get_or_create_transport's SSL resolution."""

    def test_get_or_create_transport_with_ssl_verify_false_yields_ssl_false_kwargs(self):
        """When the global is False, the transport's connector kwargs include ssl=False."""
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        litellm.ssl_verify = False
        kwargs = AsyncHTTPHandler._get_ssl_connector_kwargs(
            ssl_verify=False, ssl_context=None
        )
        assert kwargs.get("ssl") is False

    def test_get_ssl_configuration_returns_context_for_default_verify_true(self):
        """When the global is True (default), get_ssl_configuration yields an SSLContext."""
        from litellm.llms.custom_httpx.http_handler import get_ssl_configuration

        litellm.ssl_verify = True
        resolved = get_ssl_configuration(None)
        assert isinstance(resolved, ssl.SSLContext), (
            "Expected SSLContext for ssl_verify=True; got "
            f"{type(resolved).__name__}"
        )
