from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache


def test_langfuse_e2e_sync(monkeypatch):
    """A sync completion must reach langfuse over the wire, not just build a span.

    v4 exports OTLP over ``requests`` rather than the v2 ingestion endpoint over
    httpx, so this stands up a real receiver and asserts langfuse posted to it.
    """
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import litellm
    from litellm import completion
    from litellm.integrations.langfuse.langfuse import LangFuseLogger
    from litellm.integrations.langfuse.langfuse_prompt_management import langfuse_client_init
    from litellm.litellm_core_utils import litellm_logging

    received_paths = []

    class _Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received_paths.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("LANGFUSE_HOST", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-e2e-sync")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-e2e-sync")
    monkeypatch.setattr(litellm, "success_callback", ["langfuse"])
    monkeypatch.setattr(litellm_logging, "langFuseLogger", None)
    monkeypatch.setattr(litellm_logging, "in_memory_dynamic_logger_cache", DynamicLoggingCache())
    monkeypatch.setattr(litellm_logging, "_in_memory_loggers", [])
    langfuse_client_init.cache_clear()

    try:
        completion(
            model="openai/my-fake-endpoint",
            messages=[{"role": "user", "content": "hello from litellm"}],
            stream=False,
            mock_response="Hello from litellm 2",
        )
        for logger in litellm.logging_callback_manager.get_all_callbacks():
            if isinstance(logger, LangFuseLogger):
                logger.flush()
        deadline = time.time() + 10
        while not received_paths and time.time() < deadline:
            time.sleep(0.1)
    finally:
        server.shutdown()

    assert received_paths, "langfuse exported nothing"
    assert all(path.endswith("/api/public/otel/v1/traces") for path in received_paths)
