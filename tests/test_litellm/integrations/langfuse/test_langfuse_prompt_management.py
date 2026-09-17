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

        mock_langfuse_class = MagicMock()
        with (
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.resolve_langfuse_credentials",
                return_value=("pk-1234", "sk-1234", "https://localhost"),
            ),
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseLogger._get_langfuse_flush_interval",
                return_value=1,
            ),
            patch(
                "litellm.integrations.langfuse.langfuse_sdk.Langfuse", mock_langfuse_class
            ),  # test-quality-ok: the ctor must be intercepted where build_langfuse_client resolves it; a real client spawns export threads
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

            mock_langfuse_class.assert_called_once()
            call_kwargs = mock_langfuse_class.call_args[1]
            assert "httpx_client" in call_kwargs
            passed_client = call_kwargs["httpx_client"]
            assert isinstance(passed_client, httpx.Client)
            assert passed_client is not shared_client
            mock_get_ssl.assert_called_once()

        langfuse_client_init.cache_clear()


class _RecordingLangfuseForEnv:
    last_environment: str | None = None

    def __init__(
        self, *, environment: str | None = None, **parameters: object
    ) -> None:  # kwargs-ok: records only environment out of whatever langfuse_client_init forwards
        type(self).last_environment = environment


@pytest.mark.parametrize(
    ("env_value", "expected"),
    (("Production", "default"), ("production ", "production"), ("prod", "prod")),
)
def test_langfuse_client_init_resolves_deployment_environment(monkeypatch, env_value, expected):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", env_value)
    monkeypatch.setattr(_RecordingLangfuseForEnv, "last_environment", None)
    with patch(
        "litellm.integrations.langfuse.langfuse_sdk.Langfuse", _RecordingLangfuseForEnv
    ):  # test-quality-ok: the ctor must be intercepted where build_langfuse_client resolves it; a real client spawns export threads
        langfuse_client_init.cache_clear()
        langfuse_client_init()
    langfuse_client_init.cache_clear()
    assert _RecordingLangfuseForEnv.last_environment == expected


def test_langfuse_client_init_warns_that_upstream_langfuse_is_ignored(monkeypatch, caplog):
    """The YAML `callbacks: ["langfuse"]` path builds its client here, not through LangFuseLogger.__init__,
    so an operator who still sets UPSTREAM_LANGFUSE_* must get the same startup warning on this path."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_SECRET_KEY", "sk-upstream")
    monkeypatch.setenv("UPSTREAM_LANGFUSE_HOST", "https://upstream.example")
    with (
        patch(
            "litellm.integrations.langfuse.langfuse_sdk.Langfuse", _RecordingLangfuseForEnv
        ),  # test-quality-ok: the ctor must be intercepted where build_langfuse_client resolves it; a real client spawns export threads
        caplog.at_level("WARNING", logger="LiteLLM"),
    ):
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
