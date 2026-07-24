"""Shared fixtures for Alice WonderFence guardrail tests."""

import sys
from unittest.mock import AsyncMock, Mock

import pytest


def _install_sdk_stub(monkeypatch, client_factory=None):
    """Install a stub ``wonderfence_sdk`` module so the guardrail can import it."""
    sdk = Mock()
    client_pkg = Mock()
    models_pkg = Mock()

    factory = client_factory or (lambda **kwargs: Mock(close=AsyncMock()))
    client_pkg.WonderFenceV2Client = Mock(side_effect=factory)
    sdk.client = client_pkg

    models_pkg.AnalysisContext = Mock(return_value=Mock())
    sdk.models = models_pkg

    monkeypatch.setitem(sys.modules, "wonderfence_sdk", sdk)
    monkeypatch.setitem(sys.modules, "wonderfence_sdk.client", client_pkg)
    monkeypatch.setitem(sys.modules, "wonderfence_sdk.models", models_pkg)
    return sdk


def _make_guardrail(monkeypatch, **overrides):
    """Build a WonderFenceGuardrail with stubbed SDK and a mock V2 client."""
    from litellm.types.guardrails import GuardrailEventHooks

    mock_client = Mock()
    mock_client.evaluate_prompt = AsyncMock()
    mock_client.evaluate_response = AsyncMock()
    mock_client.close = AsyncMock()

    _install_sdk_stub(monkeypatch, client_factory=lambda **kwargs: mock_client)

    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
        WonderFenceGuardrail,
    )

    kwargs = dict(
        guardrail_name="wonderfence-test",
        api_key="default-api-key",
        event_hook=[
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ],
        default_on=True,
    )
    kwargs.update(overrides)
    guardrail = WonderFenceGuardrail(**kwargs)
    return guardrail, mock_client


def _request_data(**overrides):
    """Build a request-data dict.

    Default metadata pins ``alice_wonderfence_app_id`` on
    ``user_api_key_metadata`` (admin-controlled) so the request resolves
    cleanly under the safe-by-default precedence model. Tests that want to
    drive the value through request metadata must (a) construct a guardrail
    with ``allow_request_metadata_override=True`` and (b) pass the value via
    the ``metadata`` kwarg explicitly.
    """
    metadata = overrides.pop("metadata", None)
    if metadata is None:
        metadata = {"user_api_key_metadata": {"alice_wonderfence_app_id": "test-app"}}
    base = {"model": "gpt-4", "metadata": metadata}
    base.update(overrides)
    return base


def _make_logging_obj():
    """Build a real ``LiteLLMLoggingObj``.

    The post_call bridge stashes resolved credentials on a private attribute of
    this object, so tests must use the real class (not a Mock, whose attribute
    auto-creation would mask whether the attribute is genuinely settable and
    readable) to validate that the stash survives request -> response.
    """
    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=0,
        litellm_call_id="alice-wonderfence-test",
        function_id="alice-wonderfence-test",
    )


@pytest.fixture
def guardrail_and_client(monkeypatch):
    g, c = _make_guardrail(monkeypatch)
    # Pre-seed cache so apply_guardrail uses our mock without rebuilding.
    g._client_cache["default-api-key"] = c
    return g, c


@pytest.fixture
def install_sdk_stub(monkeypatch):
    """Expose ``_install_sdk_stub`` as a fixture for tests that need direct access."""

    def _factory(client_factory=None):
        return _install_sdk_stub(monkeypatch, client_factory=client_factory)

    return _factory


@pytest.fixture
def make_guardrail(monkeypatch):
    """Expose ``_make_guardrail`` as a fixture."""

    def _factory(**overrides):
        return _make_guardrail(monkeypatch, **overrides)

    return _factory


@pytest.fixture
def make_request_data():
    """Expose ``_request_data`` as a fixture."""
    return _request_data


@pytest.fixture
def make_logging_obj():
    """Expose ``_make_logging_obj`` as a fixture."""
    return _make_logging_obj
