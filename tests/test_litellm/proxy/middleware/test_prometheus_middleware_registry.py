"""Tests for conditional Prometheus middleware registration."""

import pytest
from fastapi import FastAPI
from starlette.middleware import Middleware

from litellm.proxy.middleware import prometheus_middleware_registry as registry
from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy.middleware.prometheus_auth_middleware import (
    PrometheusAuthMiddleware,
)


@pytest.fixture(autouse=True)
def reset_registry():
    registry._PROMETHEUS_MIDDLEWARES_REGISTERED = False
    yield
    registry._PROMETHEUS_MIDDLEWARES_REGISTERED = False


@pytest.mark.parametrize(
    "litellm_settings,expected",
    [
        (None, False),
        ({}, False),
        ({"callbacks": ["langfuse"]}, False),
        ({"callbacks": "prometheus"}, True),
        ({"success_callback": ["prometheus"]}, True),
        ({"failure_callback": ["prometheus"]}, True),
        (
            {
                "callbacks": ["langfuse"],
                "success_callback": ["prometheus"],
            },
            True,
        ),
    ],
)
def test_prometheus_callbacks_enabled(litellm_settings, expected):
    assert registry.prometheus_callbacks_enabled(litellm_settings) is expected


def test_maybe_register_skips_without_prometheus_callback():
    app = FastAPI()

    registered = registry.maybe_register_prometheus_middlewares(
        app, litellm_settings={"callbacks": ["langfuse"]}
    )

    assert registered is False
    assert app.user_middleware == []


def test_maybe_register_adds_middleware_when_prometheus_enabled():
    app = FastAPI()

    registered = registry.maybe_register_prometheus_middlewares(
        app, litellm_settings={"callbacks": ["prometheus"]}
    )

    assert registered is True
    assert registry._PROMETHEUS_MIDDLEWARES_REGISTERED is True
    middleware_classes = [
        m.cls if isinstance(m, Middleware) else m for m in app.user_middleware
    ]
    assert PrometheusAuthMiddleware in middleware_classes
    assert InFlightRequestsMiddleware in middleware_classes


def test_maybe_register_is_idempotent():
    app = FastAPI()
    settings = {"callbacks": ["prometheus"]}

    assert registry.maybe_register_prometheus_middlewares(app, settings) is True
    assert registry.maybe_register_prometheus_middlewares(app, settings) is False
    assert len(app.user_middleware) == 2
