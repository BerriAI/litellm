"""
Purpose
- Live escalation: /agent/run escalates to a chutes/* model on the last step.

Scope
- DOES: post to agent API; assert metrics.escalated and used_model startswith 'chutes/'.
- DOES NOT: run by default; relies on reachable agent API and chutes credentials.

Run
- DOCKER_MINI_AGENT=1 MINI_AGENT_API_HOST=127.0.0.1 MINI_AGENT_API_PORT=8788 CHUTES_API_KEY=... \
  pytest tests/ndsmoke -k test_escalation_to_chutes_live_optional -q
"""

import os, httpx, pytest

@pytest.mark.ndsmoke
def test_escalation_to_chutes_live_optional():
    if not (os.getenv('CHUTES_API_KEY') or os.getenv('CHUTES_API_TOKEN')):
        pytest.skip('CHUTES_API_KEY not set; skipping live chutes escalation')
    host = os.getenv('MINI_AGENT_API_HOST','127.0.0.1')
    port = int(os.getenv('MINI_AGENT_API_PORT','8788'))
    url=f"http://{host}:{port}/agent/run"
    payload = {
        "messages": [{"role":"user","content":"Write a short function and print result; keep it minimal."}],
        "model": os.getenv('LITELLM_DEFAULT_CODE_MODEL','ollama/deepseek-coder:33b'),
        "tool_backend": "local",
        "use_tools": False,
        "max_iterations": 1,
        "max_total_seconds": 20,
        "escalate_on_budget_exceeded": True,
        "escalate_model": "chutes/deepseek-ai/DeepSeek-R1"
    }
    r = httpx.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    metrics = data.get('metrics', {})
    # We expect escalation on the last step
    assert metrics.get('escalated') is True
    assert str(metrics.get('used_model','')).startswith('chutes/')
