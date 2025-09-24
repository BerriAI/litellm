"""
Purpose
- Prove local Ollama is reachable and returns a streaming/accumulated result.

Scope
- DOES: POST /api/generate and assert presence of 'model'/'response' keys
- DOES NOT: benchmark, specify exact tokens, or depend on a specific model string

Run
- OLLAMA_BASE=http://127.0.0.1:11434 OLLAMA_MODEL=qwen3:8b \
  pytest -q tests/ndsmoke/test_ollama_generate_ndsmoke.py::test_ollama_generate_optional
"""
import os, pytest
from tests.ndsmoke._util import parse_base, can_connect, post_json

@pytest.mark.timeout(20)
def test_ollama_generate_optional():
    host, port, base = parse_base("OLLAMA_BASE", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    if not can_connect(host, port):
        pytest.skip(f"Ollama not reachable on {host}:{port}")
    r = post_json(base + "/api/generate", {"model": model, "prompt": "hi"}, timeout=15.0)
    r.raise_for_status()
    data = r.json()
    # different ollama versions return different shapes; keep to minimal invariants
    assert "model" in data
    assert "response" in data or "done" in data