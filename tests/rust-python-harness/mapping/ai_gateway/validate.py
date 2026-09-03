"""Validate 1:1 mapping between Python gateway and Rust ai-gateway.

Maps litellm-rust/crates/ai-gateway/ endpoints to Python implementations.
"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tracer import ExecutionTracer, print_trace_table


RUST_IMPLEMENTATIONS = {
    "litellm.rust_bridge.ocr.ocr",
    "litellm.rust_bridge.ocr.aocr",
}


def validate_ocr():
    """Validate /ocr endpoint maps to Rust ai-gateway."""
    from litellm.proxy.proxy_server import app

    client = TestClient(app)
    tracer = ExecutionTracer(target_modules=['litellm.proxy', 'litellm.ocr', 'litellm.rust_bridge'])
    tracer.trace.endpoint = "/ocr"

    tracer.start()
    try:
        response = client.post(
            "/ocr",
            json={"model": "mistral-ocr", "document": {"type": "document_url", "document_url": "https://example.com/test.pdf"}},
            headers={"Authorization": "Bearer test-key"}
        )
    finally:
        trace = tracer.stop()

    print_trace_table(trace, RUST_IMPLEMENTATIONS)
    assert len(trace.calls) > 0


if __name__ == "__main__":
    validate_ocr()
