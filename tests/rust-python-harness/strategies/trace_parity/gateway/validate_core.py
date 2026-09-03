"""Validate gateway endpoints that map to Rust core."""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from shared.tracing.tracer import ExecutionTracer, print_trace_table


RUST_IMPLEMENTATIONS = {
    "litellm.rust_bridge.chat_completions.chat_completions",
    "litellm.rust_bridge.chat_completions.achat_completions",
}


def validate_chat_completions():
    """Validate /chat/completions endpoint maps to Rust core."""
    from litellm.proxy.proxy_server import app

    client = TestClient(app)
    tracer = ExecutionTracer(target_modules=['litellm.proxy', 'litellm.main', 'litellm.rust_bridge'])
    tracer.trace.endpoint = "/chat/completions"

    tracer.start()
    try:
        response = client.post(
            "/chat/completions",
            json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer test-key"}
        )
    finally:
        trace = tracer.stop()

    print_trace_table(trace, RUST_IMPLEMENTATIONS)
    assert len(trace.calls) > 0


if __name__ == "__main__":
    validate_chat_completions()
