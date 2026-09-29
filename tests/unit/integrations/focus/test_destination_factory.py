"""Tests for FocusDestinationFactory with vantage provider."""

from __future__ import annotations

import pytest

from litellm.integrations.focus.destinations.factory import FocusDestinationFactory
from litellm.integrations.focus.destinations.vantage_destination import (
    FocusVantageDestination,
)


def test_should_create_vantage_destination():
    dest = FocusDestinationFactory.create(
        provider="vantage",
        prefix="exports",
        config={
            "api_key": "test-key",
            "integration_token": "test-token",
        },
    )
    assert isinstance(dest, FocusVantageDestination)


def test_should_raise_when_vantage_missing_api_key():
    with pytest.raises(ValueError, match="VANTAGE_API_KEY"):
        FocusDestinationFactory.create(
            provider="vantage",
            prefix="exports",
            config={"integration_token": "tok"},
        )


def test_should_raise_when_vantage_missing_token():
    with pytest.raises(ValueError, match="VANTAGE_INTEGRATION_TOKEN"):
        FocusDestinationFactory.create(
            provider="vantage",
            prefix="exports",
            config={"api_key": "key"},
        )


def test_should_raise_for_unsupported_provider():
    with pytest.raises(NotImplementedError):
        FocusDestinationFactory.create(
            provider="unknown_provider",
            prefix="exports",
        )
