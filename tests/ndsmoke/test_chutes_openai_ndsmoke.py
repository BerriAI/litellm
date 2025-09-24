"""
Purpose
- Prove an OpenAI-compatible endpoint (e.g., Chutes) responds minimally.

Scope
- DOES: POST /chat/completions with tiny payload, assert 'choices' exists
- DOES NOT: validate model quality or token accounting

Run
- CHUTES_API_BASE=https://llm.chutes.ai/v1 CHUTES_API_KEY=sk-... \
  CHUTES_MODEL=deepseek-ai/DeepSeek-R1 \
  pytest -q tests/ndsmoke/test_chutes_openai_ndsmoke.py::test_chutes_chat_minimal_optional
"""
import os, pytest, httpx
from tests.ndsmoke._util import parse_base, can_connect

@pytest.mark.timeout(30)
def test_chutes_chat_minimal_optional():
    base = os.getenv("CHUTES_API_BASE", "").rstrip("/")
    key = os.getenv("CHUTES_API_KEY", "")
    model = os.getenv("CHUTES_MODEL", "deepseek-ai/DeepSeek-R1")

    if not base or not key:
        pytest.skip("CHUTES_API_BASE/CHUTES_API_KEY not set")

    host, port, _ = parse_base("CHUTES_API_BASE", base)
    # Don't hard fail if port parsing picks 80 for https; only skip if TCP blocked
    if not can_connect(host, port):
        pytest.skip(f"Chutes not reachable on {host}:{port}")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    r = httpx.post(base + "/chat/completions", headers=headers, json=payload, timeout=25.0)
    r.raise_for_status()
    data = r.json()
    assert "choices" in data and isinstance(data["choices"], list)