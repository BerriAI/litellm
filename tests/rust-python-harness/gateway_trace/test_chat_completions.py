"""Gateway trace test for /chat/completions."""
import pytest
from fastapi.testclient import TestClient

try:
    from .tracer import ExecutionTracer, print_trace_table
except ImportError:
    from tracer import ExecutionTracer, print_trace_table


# Known Rust implementations
RUST_FUNCTIONS = {
    "litellm.rust_bridge.chat_completions.chat_completions",
    "litellm.rust_bridge.chat_completions.achat_completions",
}


def test_chat_completions_gateway():
    """Trace /chat/completions gateway execution."""
    from litellm.proxy.proxy_server import app

    client = TestClient(app)

    # Set up tracer
    tracer = ExecutionTracer(target_modules=[
        'litellm.proxy',
        'litellm.main',
        'litellm.rust_bridge'
    ])
    tracer.trace.endpoint = "/chat/completions"

    tracer.start()

    try:
        # Make actual request
        response = client.post(
            "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers={"Authorization": "Bearer test-key"}
        )

        # May fail auth, but that's ok - we traced the gateway flow
        assert response.status_code in (200, 401, 403)
    finally:
        trace = tracer.stop()

    # Print formatted trace table
    print_trace_table(trace, RUST_FUNCTIONS)

    # Verify we traced the gateway endpoint
    python_calls = {f"{c.module}.{c.function}" for c in trace.calls}
    assert any("chat_completion" in call for call in python_calls), \
        "Expected chat_completion endpoint to be called"


if __name__ == "__main__":
    test_chat_completions_gateway()
