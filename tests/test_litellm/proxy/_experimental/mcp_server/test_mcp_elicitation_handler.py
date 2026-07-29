"""
Tests for the MCP elicitation handler.

Covers the gateway-mode relay logic (`elicitation/create` requests from an
upstream MCP server being forwarded to the connected downstream client) as
well as the decline paths used in tool-bridge mode or when the downstream
client lacks the requested elicitation capability.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ElicitResult,
    ErrorData,
)

from litellm.proxy._experimental.mcp_server import elicitation_handler
from litellm.proxy._experimental.mcp_server.elicitation_handler import (
    _relay_elicitation_to_downstream,
    handle_elicitation_request,
)


def _form_params(message: str = "fill the form") -> ElicitRequestFormParams:
    return ElicitRequestFormParams(
        mode="form",
        message=message,
        requestedSchema={"type": "object", "properties": {}},
    )


def _url_params(message: str = "please authorize") -> ElicitRequestURLParams:
    return ElicitRequestURLParams(
        mode="url",
        message=message,
        url="https://example.com/oauth",
        elicitationId="elc-1",
    )


def _caps(*, url=True, form=True) -> SimpleNamespace:
    elicit = SimpleNamespace(
        url=object() if url else None,
        form=object() if form else None,
    )
    return SimpleNamespace(elicitation=elicit)


class TestHandleElicitationRequest:
    async def test_should_decline_when_no_downstream_session(self):
        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_form_params(),
            downstream_session=None,
        )
        assert isinstance(result, ElicitResult)
        assert result.action == "decline"

    async def test_should_relay_to_downstream_when_session_present(self):
        accepted = ElicitResult(action="accept", content={"name": "ada"})
        session = SimpleNamespace(elicit_form=AsyncMock(return_value=accepted))

        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=None,
        )

        assert result is accepted
        session.elicit_form.assert_awaited_once()

    async def test_should_return_error_data_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(elicitation_handler, "MCP_ELICITATION_AVAILABLE", False)
        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_form_params(),
            downstream_session=SimpleNamespace(),
        )
        assert isinstance(result, ErrorData)
        assert "not available" in result.message

    async def test_should_return_error_data_on_unexpected_failure(self):
        class _ExplodingParams:
            mode = "form"

            @property
            def message(self):
                raise RuntimeError("boom")

        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_ExplodingParams(),
            downstream_session=None,
        )
        assert isinstance(result, ErrorData)
        assert "boom" in result.message


class TestRelayElicitationToDownstream:
    async def test_should_relay_form_mode(self):
        accepted = ElicitResult(action="accept", content={"name": "ada"})
        session = SimpleNamespace(elicit_form=AsyncMock(return_value=accepted))

        params = _form_params("collect name")
        result = await _relay_elicitation_to_downstream(
            params=params,
            downstream_session=session,
            downstream_capabilities=_caps(form=True),
        )

        assert result is accepted
        session.elicit_form.assert_awaited_once()
        _, kwargs = session.elicit_form.call_args
        assert kwargs["message"] == "collect name"
        assert kwargs["requestedSchema"] == params.requestedSchema

    async def test_should_relay_url_mode(self):
        accepted = ElicitResult(action="accept")
        session = SimpleNamespace(elicit_url=AsyncMock(return_value=accepted))

        result = await _relay_elicitation_to_downstream(
            params=_url_params(),
            downstream_session=session,
            downstream_capabilities=_caps(url=True),
        )

        assert result is accepted
        session.elicit_url.assert_awaited_once()
        _, kwargs = session.elicit_url.call_args
        assert kwargs["url"] == "https://example.com/oauth"
        assert kwargs["elicitation_id"] == "elc-1"

    async def test_should_use_generic_elicit_for_unknown_param_type(self):
        accepted = ElicitResult(action="accept")
        session = SimpleNamespace(elicit=AsyncMock(return_value=accepted))

        # A bare params object that is neither Form nor URL params triggers
        # the generic fallback path.
        params = SimpleNamespace(mode="form", message="hi", requestedSchema={})
        result = await _relay_elicitation_to_downstream(
            params=params,
            downstream_session=session,
            downstream_capabilities=None,
        )

        assert result is accepted
        session.elicit.assert_awaited_once()

    async def test_should_decline_when_elicitation_unsupported(self):
        session = SimpleNamespace(elicit_form=AsyncMock())
        caps = SimpleNamespace(elicitation=None)

        result = await _relay_elicitation_to_downstream(
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=caps,
        )

        assert isinstance(result, ElicitResult)
        assert result.action == "decline"
        session.elicit_form.assert_not_awaited()

    async def test_should_decline_url_mode_when_url_unsupported(self):
        session = SimpleNamespace(elicit_url=AsyncMock())

        result = await _relay_elicitation_to_downstream(
            params=_url_params(),
            downstream_session=session,
            downstream_capabilities=_caps(url=False, form=True),
        )

        assert isinstance(result, ElicitResult)
        assert result.action == "decline"
        session.elicit_url.assert_not_awaited()

    async def test_should_decline_form_mode_when_form_unsupported(self):
        session = SimpleNamespace(elicit_form=AsyncMock())

        result = await _relay_elicitation_to_downstream(
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=_caps(url=True, form=False),
        )

        assert isinstance(result, ElicitResult)
        assert result.action == "decline"
        session.elicit_form.assert_not_awaited()

    async def test_should_decline_when_downstream_relay_raises(self):
        session = SimpleNamespace(
            elicit_form=AsyncMock(side_effect=RuntimeError("transport closed"))
        )

        result = await _relay_elicitation_to_downstream(
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=_caps(form=True),
        )

        assert isinstance(result, ElicitResult)
        assert result.action == "decline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
