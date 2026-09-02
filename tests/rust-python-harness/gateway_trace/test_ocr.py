"""Gateway trace test for /ocr."""
import pytest
from fastapi.testclient import TestClient

try:
    from .tracer import ExecutionTracer, print_trace_table
except ImportError:
    from tracer import ExecutionTracer, print_trace_table


# Known Rust implementations
RUST_FUNCTIONS = {
    "litellm.rust_bridge.ocr.ocr",
    "litellm.rust_bridge.ocr.aocr",
}


def test_ocr_gateway():
    """Trace /ocr gateway execution."""
    from litellm.proxy.proxy_server import app

    client = TestClient(app)

    # Set up tracer
    tracer = ExecutionTracer(target_modules=[
        'litellm.proxy',
        'litellm.ocr',
        'litellm.rust_bridge'
    ])
    tracer.trace.endpoint = "/ocr"

    tracer.start()

    try:
        # Make actual request
        response = client.post(
            "/ocr",
            json={
                "model": "mistral-ocr",
                "document": {
                    "type": "document_url",
                    "document_url": "https://example.com/test.pdf"
                }
            },
            headers={"Authorization": "Bearer test-key"}
        )

        # May fail auth or validation
        assert response.status_code in (200, 400, 401, 403)
    finally:
        trace = tracer.stop()

    # Print formatted trace table
    print_trace_table(trace, RUST_FUNCTIONS, max_rows=30)

    # Verify we traced the OCR endpoint
    python_calls = {f"{c.module}.{c.function}" for c in trace.calls}
    assert any("ocr" in call.lower() for call in python_calls), \
        "Expected OCR endpoint to be called"


if __name__ == "__main__":
    test_ocr_gateway()
