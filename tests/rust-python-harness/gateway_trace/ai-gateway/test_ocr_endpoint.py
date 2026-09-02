"""Trace /ocr gateway endpoint - maps to litellm-rust/crates/ai-gateway/src/routes/ocr.rs"""
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent to path for tracer import
sys.path.insert(0, str(Path(__file__).parent.parent))
from tracer import ExecutionTracer, print_trace_table


RUST_FUNCTIONS = {
    "litellm.rust_bridge.ocr.ocr",
    "litellm.rust_bridge.ocr.aocr",
}


def test_ocr_endpoint():
    """Trace /ocr endpoint execution."""
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

    print_trace_table(trace, RUST_FUNCTIONS)
    assert len(trace.calls) > 0


if __name__ == "__main__":
    test_ocr_endpoint()
