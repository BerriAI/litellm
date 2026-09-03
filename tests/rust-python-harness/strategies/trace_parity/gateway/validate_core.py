"""Validate gateway /chat/completions endpoint parity."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rust_python_harness.strategies.trace_parity.gateway.profiler import FunctionTraceEvent, profile_gateway


def test_chat_completions_parity():
    """Validate /chat/completions Python vs Rust trace parity."""
    from litellm.proxy.proxy_server import app

    client = TestClient(app)
    request_body = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }

    # Profile Python execution
    litellm_root = Path(__file__).parent.parent.parent.parent.parent.parent / "litellm"
    with profile_gateway(source_root=litellm_root) as profiler:
        python_response = client.post(
            "/chat/completions",
            json=request_body,
            headers={"Authorization": "Bearer test-key"}
        )

    python_trace = tuple(profiler.events)

    # TODO: Get Rust trace from endpoint with trace=True
    # For now, just verify Python execution succeeded
    assert python_response.status_code == 200, f"Python endpoint failed: {python_response.text}"
    assert len(python_trace) > 0, "No Python functions traced"

    # Verify critical functions were called
    function_names = {event.function.split()[-1] for event in python_trace}
    assert "chat_completion" in function_names, f"chat_completion not found in trace: {function_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
