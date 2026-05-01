"""
Unit tests for the MCP Elicitation Handler.
Tests the elicitation/create handler that relays elicitation requests
from upstream MCP servers to downstream clients or declines them.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest


# ─────────────────────────────────────────────────────────────
# Helper factories
# ─────────────────────────────────────────────────────────────
def _make_form_params(message="Please provide info", schema=None):
    """Create a mock ElicitRequestFormParams."""
    from mcp.types import ElicitRequestFormParams

    params = MagicMock(spec=ElicitRequestFormParams)
    params.mode = "form"
    params.message = message
    params.requestedSchema = schema
    return params


def _make_url_params(message="Click the link", url="https://auth.example.com"):
    """Create a mock ElicitRequestURLParams."""
    from mcp.types import ElicitRequestURLParams

    params = MagicMock(spec=ElicitRequestURLParams)
    params.mode = "url"
    params.message = message
    params.url = url
    params.elicitationId = "elicit-123"
    return params


def _make_capabilities(form=True, url=True):
    """Create mock client capabilities with elicitation support."""
    caps = MagicMock()
    elicit = MagicMock()
    elicit.form = MagicMock() if form else None
    elicit.url = MagicMock() if url else None
    caps.elicitation = elicit
    return caps


def _make_capabilities_no_elicitation():
    """Create mock client capabilities without elicitation."""
    caps = MagicMock()
    caps.elicitation = None
    return caps


# ─────────────────────────────────────────────────────────────
# Tests: No downstream session (Tool Bridge mode)
# ─────────────────────────────────────────────────────────────
class TestElicitationNoDownstream:
    """Tests when no downstream client is available."""

    @pytest.mark.asyncio
    async def test_should_decline_when_no_downstream_session(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params()
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=None,
        )
        assert result.action == "decline"


# ─────────────────────────────────────────────────────────────
# Tests: Downstream session relay
# ─────────────────────────────────────────────────────────────
class TestElicitationRelay:
    """Tests for relaying elicitation to downstream clients."""

    @pytest.mark.asyncio
    async def test_should_relay_form_mode(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params(message="Enter your name")
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.action = "submit"
        mock_result.content = {"name": "Alice"}
        mock_session.elicit_form.return_value = mock_result
        caps = _make_capabilities(form=True, url=True)
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        mock_session.elicit_form.assert_called_once()
        assert result.action == "submit"

    @pytest.mark.asyncio
    async def test_should_relay_url_mode(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_url_params(
            message="Authenticate", url="https://oauth.example.com"
        )
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.action = "submit"
        mock_session.elicit_url.return_value = mock_result
        caps = _make_capabilities(form=True, url=True)
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        mock_session.elicit_url.assert_called_once()
        assert result.action == "submit"

    @pytest.mark.asyncio
    async def test_should_decline_when_client_lacks_elicitation(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params()
        mock_session = AsyncMock()
        caps = _make_capabilities_no_elicitation()
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        assert result.action == "decline"

    @pytest.mark.asyncio
    async def test_should_decline_when_client_lacks_url_mode(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_url_params()
        mock_session = AsyncMock()
        caps = _make_capabilities(form=True, url=False)
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        assert result.action == "decline"

    @pytest.mark.asyncio
    async def test_should_decline_when_client_lacks_form_mode(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params()
        mock_session = AsyncMock()
        caps = _make_capabilities(form=False, url=True)
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        assert result.action == "decline"

    @pytest.mark.asyncio
    async def test_should_decline_gracefully_on_relay_failure(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params()
        mock_session = AsyncMock()
        mock_session.elicit_form.side_effect = Exception("Connection lost")
        caps = _make_capabilities(form=True, url=True)
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=caps,
        )
        assert result.action == "decline"


# ─────────────────────────────────────────────────────────────
# Tests: Error handling
# ─────────────────────────────────────────────────────────────
class TestElicitationErrorHandling:
    """Tests for error handling in the elicitation handler."""

    @pytest.mark.asyncio
    async def test_should_relay_without_capability_check_when_caps_none(self):
        from litellm.proxy._experimental.mcp_server.elicitation_handler import (
            handle_elicitation_request,
        )

        params = _make_form_params()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.action = "submit"
        mock_session.elicit_form.return_value = mock_result
        # No capabilities provided — should still attempt relay
        result = await handle_elicitation_request(
            context=MagicMock(),
            params=params,
            downstream_session=mock_session,
            downstream_capabilities=None,
        )
        mock_session.elicit_form.assert_called_once()
        assert result.action == "submit"
