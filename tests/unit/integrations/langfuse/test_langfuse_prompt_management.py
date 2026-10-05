import sys
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

# langfuse_client_init imports this lazily; cache it before any test mocks
# sys.modules["langfuse"], or a single-file run dies on the real import
import litellm.integrations.langfuse.langfuse_sdk  # noqa: F401
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    langfuse_client_init,
)


class TestLangfusePromptManagement:
    def setup_method(self):
        # Mock langfuse package to avoid triggering real import.
        # The real langfuse import fails on Python 3.14 due to pydantic v1 incompatibility.
        # This also prevents test-ordering issues when earlier tests remove sys.modules["langfuse"].
        self._mock_langfuse = MagicMock()
        self._mock_langfuse.version.__version__ = "3.0.0"
        self._langfuse_patcher = patch.dict("sys.modules", {"langfuse": self._mock_langfuse})
        self._langfuse_patcher.start()

    def teardown_method(self):
        self._langfuse_patcher.stop()

    def test_get_prompt_from_id(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with (
            patch.object(
                langfuse_prompt_management, "should_run_prompt_management"
            ) as mock_should_run_prompt_management,
            patch.object(langfuse_prompt_management, "_get_prompt_from_id") as mock_get_prompt_from_id,
        ):
            mock_should_run_prompt_management.return_value = True
            langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/langfuse-model",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                non_default_params={},
                prompt_id="test-chat-prompt",
                prompt_variables={},
                dynamic_callback_params={},
                prompt_version=4,
            )

            mock_get_prompt_from_id.assert_called_once()
            assert mock_get_prompt_from_id.call_args.kwargs["prompt_version"] == 4

    def test_log_failure_event_runs_async_logger(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch("litellm.integrations.langfuse.langfuse_prompt_management.run_async_function") as mock_run_async:
            kwargs = {"standard_callback_dynamic_params": {}}
            start_time, end_time = 1, 2

            langfuse_prompt_management.log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
            )

            mock_run_async.assert_called_once()
            assert mock_run_async.call_args[0][0] == langfuse_prompt_management.async_log_failure_event

    def test_langfuse_client_init_passes_dedicated_httpx_client(self):
        import httpx

        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        shared_client = _get_httpx_client().client
        built = MagicMock()
        with (
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.resolve_langfuse_credentials",
                return_value=("pk-1234", "sk-1234", "https://localhost"),
            ),
            patch(
                "litellm.integrations.langfuse.langfuse_sdk.build_langfuse_client", built
            ),  # test-quality-ok: the REST client is built where langfuse_client_init resolves it; the transport it gets is the behavior under test
            patch(
                "litellm.llms.custom_httpx.http_handler.get_ssl_configuration",
                return_value=False,
            ) as mock_get_ssl,
        ):
            langfuse_client_init(
                langfuse_public_key="pk-1234",
                langfuse_secret="sk-1234",
                langfuse_host="https://localhost",
            )

            built.assert_called_once()
            passed_client = built.call_args.kwargs["httpx_client"]
            assert isinstance(passed_client, httpx.Client)
            assert passed_client is not shared_client
            mock_get_ssl.assert_called_once()

        langfuse_client_init.cache_clear()


@pytest.mark.parametrize(
    ("env_value", "expected"),
    (("Production", "default"), ("production ", "production"), ("prod", "prod")),
)
def test_prompt_management_logger_exports_the_resolved_deployment_environment(monkeypatch, env_value, expected):
    from langfuse import LangfuseOtelSpanAttributes

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")
    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", env_value)
    langfuse_client_init.cache_clear()
    logger = LangfusePromptManagement()
    langfuse_client_init.cache_clear()
    assert logger.tracing.provider.resource.attributes[LangfuseOtelSpanAttributes.ENVIRONMENT] == expected


def test_langfuse_client_init_warns_that_upstream_langfuse_is_ignored(monkeypatch, caplog):
    """The YAML `callbacks: ["langfuse"]` path builds its client here, not through LangFuseLogger.__init__,
    so an operator who still sets UPSTREAM_LANGFUSE_* must get the same startup warning on this path."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_SECRET_KEY", "sk-upstream")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_HOST", "https://upstream.example")
    with caplog.at_level("WARNING", logger="LiteLLM"):
        langfuse_client_init.cache_clear()
        langfuse_client_init()
    langfuse_client_init.cache_clear()
    assert any("UPSTREAM_LANGFUSE_* is no longer supported" in record.getMessage() for record in caplog.records)


def test_langfuse_client_init_mock_mode_makes_no_network_calls(monkeypatch):
    """LANGFUSE_MOCK promises full execution without egress.

    The registry maps the "langfuse" callback to LangfusePromptManagement, so
    this logger is the one the standard proxy path emits observations through;
    they travel over litellm's own OTLP exporter, which the httpx mock cannot see.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import litellm

    received = []

    class _Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-pm-mock-egress")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-pm-mock-egress")
    langfuse_client_init.cache_clear()
    now: Final = datetime.now(timezone.utc)

    try:
        logger = LangfusePromptManagement()
        logged = logger.log_event_on_langfuse(
            kwargs={
                "litellm_call_id": "call-pm-mock-egress",
                "call_type": "completion",
                "litellm_params": {"metadata": {"trace_id": "a" * 32}},
                "messages": [{"role": "user", "content": "hi"}],
                "optional_params": {},
            },
            response_obj=litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}]),
            start_time=now,
            end_time=now,
        )
        logger.flush()
    finally:
        server.shutdown()
        langfuse_client_init.cache_clear()

    assert logged["trace_id"] == "a" * 32
    assert received == [], f"LANGFUSE_MOCK still sent spans to the configured host: {received}"


def test_langfuse_debug_reaches_the_export_channel_through_the_registered_callback(monkeypatch):
    """The registry maps ``langfuse`` to this class, whose constructor never runs ``LangFuseLogger.__init__``,
    so wiring ``LANGFUSE_DEBUG`` only there left the flag a no-op on the YAML callback path."""
    import logging

    from litellm.integrations.langfuse.langfuse_sdk import release_langfuse_tracing

    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-pm-debug-wire")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-pm-debug-wire")
    monkeypatch.setenv("LANGFUSE_DEBUG", "true")
    langfuse_client_init.cache_clear()
    langfuse_logger: Final = logging.getLogger("langfuse")
    level_before: Final = langfuse_logger.level
    langfuse_logger.setLevel(logging.WARNING)
    try:
        logger = LangfusePromptManagement()
        assert langfuse_logger.level == logging.DEBUG
        release_langfuse_tracing(logger.tracing, grace_seconds=0.0)
    finally:
        langfuse_logger.setLevel(level_before)
        langfuse_client_init.cache_clear()


@pytest.mark.asyncio
async def test_async_log_failure_event_records_trace_id_for_alerting(monkeypatch):
    from litellm.integrations.langfuse.langfuse_sdk import resolve_trace_id
    from litellm.litellm_core_utils.specialty_caches.service_trace_id_cache import in_memory_trace_id_cache

    monkeypatch.setenv("LANGFUSE_MOCK", "true")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-pm-trace-cache")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-pm-trace-cache")
    langfuse_client_init.cache_clear()
    call_id: Final = "call-trace-cache-1"
    now: Final = datetime.now(timezone.utc)
    kwargs: Final = {
        "litellm_call_id": call_id,
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "hi"}],
        "litellm_params": {"metadata": {"trace_id": "alert-trace-1"}},
        "optional_params": {},
        "standard_callback_dynamic_params": {},
        "exception": RuntimeError("provider down"),
    }

    try:
        await LangfusePromptManagement().async_log_failure_event(
            kwargs=kwargs, response_obj=None, start_time=now, end_time=now
        )
    finally:
        langfuse_client_init.cache_clear()

    assert in_memory_trace_id_cache.get_cache(litellm_call_id=call_id, service_name="langfuse") == resolve_trace_id(
        "alert-trace-1"
    )


def test_old_sdk_fails_with_the_upgrade_message_before_the_otel_module_is_imported(monkeypatch):
    """On a v2 install `langfuse_sdk` itself fails to import, so the version gate must run first."""
    import litellm.integrations.langfuse.langfuse_prompt_management as pm_module

    monkeypatch.setattr(pm_module, "installed_langfuse_version", lambda: "2.59.7")
    monkeypatch.setitem(sys.modules, "litellm.integrations.langfuse.langfuse_sdk", None)

    with pytest.raises(ImportError) as raised:
        LangfusePromptManagement(
            langfuse_public_key="pk-old", langfuse_secret="sk-old", langfuse_host="http://127.0.0.1:1"
        )

    assert "2.59.7" in str(raised.value)
    assert "langfuse_otel" in str(raised.value)


@pytest.mark.parametrize("raw", ["abc", "2.5"], ids=["text", "fraction"])
def test_prompt_cache_ttl_typo_is_named_instead_of_reported_as_not_installed(monkeypatch, raw):
    """The v4 SDK runs ``int()`` on this variable at import, and ``langfuse_client_init`` wraps any import
    failure as "Langfuse not installed", so the gate has to run before that import."""
    monkeypatch.setenv("LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS", raw)
    monkeypatch.setitem(sys.modules, "litellm.integrations.langfuse.langfuse_sdk", None)
    langfuse_client_init.cache_clear()

    with pytest.raises(ValueError, match="LANGFUSE_PROMPT_CACHE_DEFAULT_TTL_SECONDS") as raised:
        langfuse_client_init(langfuse_public_key="pk-ttl", langfuse_secret="sk-ttl", langfuse_host="http://127.0.0.1:1")

    assert "not installed" not in str(raised.value)
    assert repr(raw) in str(raised.value)
