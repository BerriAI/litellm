"""Trace /chat/completions gateway endpoint - maps to litellm-rust/crates/ai-gateway/src/routes/chat.rs"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracer import ExecutionTracer, print_trace_table


RUST_FUNCTIONS = {
    "litellm.rust_bridge.chat_completions.chat_completions",
    "litellm.rust_bridge.chat_completions.achat_completions",
}


def test_chat_completions_endpoint():
    """Trace /chat/completions endpoint execution."""
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

    print_trace_table(trace, RUST_FUNCTIONS)
    assert len(trace.calls) > 0


if __name__ == "__main__":
    test_chat_completions_endpoint()
