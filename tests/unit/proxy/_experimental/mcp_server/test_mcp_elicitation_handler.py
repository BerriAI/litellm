"""
Tests for the MCP elicitation handler.

Covers the gateway-mode relay logic (`elicitation/create` requests from an
upstream MCP server being forwarded to the connected downstream client) as
well as the decline paths used in tool-bridge mode or when the downstream
client lacks the requested elicitation capability.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp.types import (
    ClientCapabilities,
    ElicitationCapability,
    FormElicitationCapability,
    UrlElicitationCapability,
    REQUEST_TIMEOUT,
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
        requested_schema={"type": "object", "properties": {}},
    )


def _url_params(message: str = "please authorize") -> ElicitRequestURLParams:
    return ElicitRequestURLParams(
        mode="url",
        message=message,
        url="https://example.com/oauth",
        elicitation_id="elc-1",
    )


def _caps(*, url=True, form=True) -> ClientCapabilities:
    return ClientCapabilities(elicitation=ElicitationCapability(
        url=UrlElicitationCapability() if url else None,
        form=FormElicitationCapability() if form else None,
    ))


class TestHandleElicitationRequest:
    async def test_should_error_when_no_downstream_session(self):
        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_form_params(),
            downstream_session=None,
        )
        assert isinstance(result, ErrorData)

    async def test_should_relay_to_downstream_when_session_present(self):
        accepted = ElicitResult(action="accept", content={"name": "ada"})
        session = SimpleNamespace(elicit_form=AsyncMock(return_value=accepted))

        result = await handle_elicitation_request(
            context=SimpleNamespace(),
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=_caps(),
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
        assert kwargs["requested_schema"] == params.requested_schema

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

    async def test_should_reject_invalid_elicitation_parameters(self):
        session = SimpleNamespace(elicit_form=AsyncMock())
        result = await _relay_elicitation_to_downstream(
            params=SimpleNamespace(mode="form"), downstream_session=session, downstream_capabilities=_caps(),
        )
        assert isinstance(result, ErrorData)
        session.elicit_form.assert_not_awaited()

    async def test_should_error_when_elicitation_unsupported(self):
        session = SimpleNamespace(elicit_form=AsyncMock())
        caps = ClientCapabilities()

        result = await _relay_elicitation_to_downstream(
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=caps,
        )

        assert isinstance(result, ErrorData)
        session.elicit_form.assert_not_awaited()

    async def test_should_error_url_mode_when_url_unsupported(self):
        session = SimpleNamespace(elicit_url=AsyncMock())

        result = await _relay_elicitation_to_downstream(
            params=_url_params(),
            downstream_session=session,
            downstream_capabilities=_caps(url=False, form=True),
        )

        assert isinstance(result, ErrorData)
        session.elicit_url.assert_not_awaited()

    async def test_should_error_form_mode_when_form_unsupported(self):
        session = SimpleNamespace(elicit_form=AsyncMock())

        result = await _relay_elicitation_to_downstream(
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=_caps(url=True, form=False),
        )

        assert isinstance(result, ErrorData)
        session.elicit_form.assert_not_awaited()

    async def test_should_error_when_downstream_relay_raises(self):
        session = SimpleNamespace(
            elicit_form=AsyncMock(side_effect=RuntimeError("transport closed"))
        )

        result = await handle_elicitation_request(
            context=None,
            params=_form_params(),
            downstream_session=session,
            downstream_capabilities=_caps(form=True),
        )

        assert isinstance(result, ErrorData)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.asyncio
async def test_relay_failure_is_not_user_decline():
    session = SimpleNamespace(elicit_form=AsyncMock(side_effect=RuntimeError("private upstream credential")))
    result = await handle_elicitation_request(
        context=None, params=_form_params(), downstream_session=session, downstream_capabilities=_caps(),
    )
    assert isinstance(result, ErrorData), "A failed relay must not claim that the user declined"
    assert "private upstream credential" not in result.message


@pytest.mark.asyncio
async def test_unknown_client_capabilities_prevent_relay():
    session = SimpleNamespace(elicit_form=AsyncMock(return_value=ElicitResult(action="accept")))
    result = await handle_elicitation_request(
        context=None, params=_form_params(), downstream_session=session, downstream_capabilities=None,
    )
    assert isinstance(result, ErrorData), "Unknown capabilities must fail explicitly"
    session.elicit_form.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["form", "url"])
@pytest.mark.parametrize("action", ["accept", "decline", "cancel"])
async def test_relay_preserves_response_and_request_correlation(mode, action):
    response = ElicitResult(action=action)
    request = AsyncMock(return_value=response)
    session = SimpleNamespace(elicit_form=request, elicit_url=request)
    params = _form_params() if mode == "form" else _url_params()
    result = await handle_elicitation_request(
        context=SimpleNamespace(request_id="upstream-id"), params=params,
        downstream_session=session, downstream_capabilities=_caps(), related_request_id=0,
    )
    assert result is response
    assert request.await_args.kwargs["related_request_id"] == 0
    if mode == "url":
        assert request.await_args.kwargs["url"] == params.url
        assert request.await_args.kwargs["elicitation_id"] == params.elicitation_id
    else:
        assert request.await_args.kwargs["requested_schema"] == params.requested_schema


@pytest.mark.asyncio
async def test_expired_relay_deadline_returns_explicit_timeout():
    request = AsyncMock()
    result = await handle_elicitation_request(
        context=None, params=_form_params(), downstream_session=SimpleNamespace(elicit_form=request),
        downstream_capabilities=_caps(), timeout=0,
    )
    assert isinstance(result, ErrorData)
    assert result.code == REQUEST_TIMEOUT
    assert "timed out" in result.message
    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_relay_cancellation_releases_downstream_waiter():
    started = asyncio.Event()
    finished = asyncio.Event()

    async def wait_for_user(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    task = asyncio.create_task(handle_elicitation_request(
        context=None, params=_form_params(), downstream_session=SimpleNamespace(elicit_form=wait_for_user),
        downstream_capabilities=_caps(), related_request_id="tool-call",
    ))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_legacy_empty_elicitation_capability_supports_form():
    accepted = ElicitResult(action="accept")
    session = SimpleNamespace(elicit_form=AsyncMock(return_value=accepted))
    result = await handle_elicitation_request(
        context=None, params=_form_params(), downstream_session=session,
        downstream_capabilities=ClientCapabilities(elicitation=ElicitationCapability()),
        related_request_id="legacy-call",
    )
    assert result is accepted
