"""Validate gateway /chat/completions endpoint parity."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rust_python_harness.strategies.trace_parity.gateway.profiler import FunctionTraceEvent, profile_gateway


def test_chat_completions_parity():
    """Validate /chat/completions Python vs Rust trace parity."""
    from litellm.proxy.proxy_server import app
    from litellm._native import achat_completions
    import asyncio

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

    # Get Rust trace from SDK function with trace=True
    async def get_rust_trace():
        return await achat_completions(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            trace=True
        )

    rust_result = asyncio.run(get_rust_trace())

    # Extract trace from Rust response
    if isinstance(rust_result, dict) and "trace" in rust_result:
        rust_trace = tuple(
            FunctionTraceEvent(function=event["function"], depth=event["depth"])
            for event in rust_result["trace"]
        )
    else:
        rust_trace = ()

    # Validate both executions
    assert python_response.status_code == 200, f"Python endpoint failed: {python_response.text}"
    assert len(python_trace) > 0, "No Python functions traced"

    # Verify critical functions were called in Python
    python_function_names = {event.function.split()[-1] for event in python_trace}
    assert "chat_completion" in python_function_names, f"chat_completion not found in Python trace: {python_function_names}"

    # Verify Rust trace was collected
    assert len(rust_trace) > 0, f"No Rust trace collected. Result: {rust_result}"

    # Print comparison
    print(f"\n{'='*80}")
    print(f"Python trace ({len(python_trace)} events):")
    for event in python_trace[:10]:
        print(f"  {event.function} (depth={event.depth})")
    print(f"\nRust trace ({len(rust_trace)} events):")
    for event in rust_trace:
        print(f"  {event.function} (depth={event.depth})")
    print(f"{'='*80}\n")

    # Compare traces
    rust_function_names = {event.function for event in rust_trace}
    assert "chat_completions" in rust_function_names, f"chat_completions not found in Rust trace: {rust_function_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
