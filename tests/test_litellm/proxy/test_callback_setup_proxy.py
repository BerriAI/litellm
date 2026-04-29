"""
Tests for the CustomLogger.setup_proxy(app) hook.

Verifies that:
1. CustomLogger has a setup_proxy method (no-op default)
2. _invoke_callback_setup_proxy calls setup_proxy on all CustomLogger callbacks
3. Non-CustomLogger callbacks are skipped
4. Exceptions in setup_proxy are caught and logged (don't crash startup)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import _invoke_callback_setup_proxy


class TestCustomLoggerSetupProxy:
    """Tests for the setup_proxy method on CustomLogger."""

    def test_custom_logger_has_setup_proxy_method(self):
        """CustomLogger base class should have a setup_proxy method."""
        logger = CustomLogger()
        assert hasattr(logger, "setup_proxy")
        assert callable(logger.setup_proxy)

    def test_setup_proxy_default_is_noop(self):
        """Default setup_proxy should do nothing and not raise."""
        logger = CustomLogger()
        mock_app = MagicMock()
        logger.setup_proxy(mock_app)  # Should not raise


class TestInvokeCallbackSetupProxy:
    """Tests for _invoke_callback_setup_proxy function."""

    def test_calls_setup_proxy_on_custom_logger_callbacks(self):
        """setup_proxy should be called on each CustomLogger in litellm.callbacks."""

        class MyCallback(CustomLogger):
            def __init__(self):
                super().__init__()
                self.setup_called_with = None

            def setup_proxy(self, app):
                self.setup_called_with = app

        callback = MyCallback()
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [callback]
        try:
            _invoke_callback_setup_proxy(mock_app)
            assert callback.setup_called_with is mock_app
        finally:
            litellm.callbacks = original_callbacks

    def test_skips_non_custom_logger_callbacks(self):
        """String callbacks and non-CustomLogger objects should be skipped."""
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = ["langfuse", "sentry", 42]
        try:
            # Should not raise
            _invoke_callback_setup_proxy(mock_app)
        finally:
            litellm.callbacks = original_callbacks

    def test_exception_in_setup_proxy_is_caught(self):
        """If setup_proxy raises, it should be caught and logged, not crash."""

        class BadCallback(CustomLogger):
            def setup_proxy(self, app):
                raise RuntimeError("setup failed")

        bad_callback = BadCallback()
        good_callback = CustomLogger()
        mock_app = MagicMock()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [bad_callback, good_callback]
        try:
            # Should not raise even though bad_callback.setup_proxy raises
            _invoke_callback_setup_proxy(mock_app)
        finally:
            litellm.callbacks = original_callbacks

    def test_multiple_callbacks_all_called(self):
        """All CustomLogger callbacks should have setup_proxy called."""
        call_order = []

        class Callback1(CustomLogger):
            def setup_proxy(self, app):
                call_order.append("cb1")

        class Callback2(CustomLogger):
            def setup_proxy(self, app):
                call_order.append("cb2")

        mock_app = MagicMock()
        original_callbacks = litellm.callbacks
        litellm.callbacks = [Callback1(), "langfuse", Callback2()]
        try:
            _invoke_callback_setup_proxy(mock_app)
            assert call_order == ["cb1", "cb2"]
        finally:
            litellm.callbacks = original_callbacks


# ------------------------------------------------------------------ #
# E2E test: setup_proxy adds a real route accessible via HTTP
# ------------------------------------------------------------------ #


class TestSetupProxyE2E:
    """E2E tests verifying that setup_proxy can modify the FastAPI app."""

    def test_e2e_setup_proxy_adds_route(self):
        """
        A CustomLogger.setup_proxy(app) that adds a /custom-health route
        should be reachable via HTTP after _invoke_callback_setup_proxy.
        """

        class RouteAddingCallback(CustomLogger):
            def setup_proxy(self, app):
                @app.get("/custom-health")
                async def custom_health():
                    return JSONResponse({"status": "ok", "source": "callback"})

        app = FastAPI()
        callback = RouteAddingCallback()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [callback]
        try:
            _invoke_callback_setup_proxy(app)

            client = TestClient(app)
            response = client.get("/custom-health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok", "source": "callback"}
        finally:
            litellm.callbacks = original_callbacks

    def test_e2e_setup_proxy_adds_middleware(self):
        """
        A CustomLogger.setup_proxy(app) that adds middleware should affect
        all subsequent requests.
        """

        class HeaderMiddlewareCallback(CustomLogger):
            def setup_proxy(self, app):
                from starlette.middleware.base import BaseHTTPMiddleware

                class AddHeaderMiddleware(BaseHTTPMiddleware):
                    async def dispatch(self, request, call_next):
                        response = await call_next(request)
                        response.headers["X-Custom-Callback"] = "active"
                        return response

                app.add_middleware(AddHeaderMiddleware)

        app = FastAPI()

        @app.get("/test-endpoint")
        async def test_endpoint():
            return {"ok": True}

        callback = HeaderMiddlewareCallback()

        original_callbacks = litellm.callbacks
        litellm.callbacks = [callback]
        try:
            _invoke_callback_setup_proxy(app)

            client = TestClient(app)
            response = client.get("/test-endpoint")
            assert response.status_code == 200
            assert response.headers.get("X-Custom-Callback") == "active"
        finally:
            litellm.callbacks = original_callbacks
