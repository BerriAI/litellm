"""
Purpose
- Optional live check: mini-agent produces a non-empty final answer with a local Ollama model and no tools.

Scope
- DOES: reach Ollama /api/tags; pick a small text model; assert final_answer is non-empty.
- DOES NOT: run by default; skips if Ollama is unreachable or no model is found.

Run
- `DOCKER_MINI_AGENT=1 pytest tests/smoke -k test_mini_agent_with_ollama_optional -q`
  (Set OLLAMA_HOST/PORT or LITELLM_DEFAULT_MODEL as needed.)
"""
import os
import socket
import asyncio
import pytest


def _can_connect(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
@pytest.mark.live_ollama
def test_mini_agent_with_ollama_optional():
    """Optional live validation: run the mini-agent against a local Ollama model.

    Skips cleanly when Ollama is not reachable or no suitable model is available.
    """
    host = os.getenv("OLLAMA_HOST", "127.0.0.1")
    port = int(os.getenv("OLLAMA_PORT", "11434"))
    if not _can_connect(host, port):
        pytest.skip(f"ollama not reachable on {host}:{port}")

    async def run():
        # Choose a local model, preferring a small text-only option.
        import httpx
        preferred = [
            "qwen3:8b",
            "qwen2.5:7b",
            "llama3.1:8b",
            "mistral:7b",
            "phi3:3.8b",
            "gemma2:9b",
        ]
        chosen = None
        try:
            tags = httpx.get(f"http://{host}:{port}/api/tags", timeout=1.0).json()
            names = {m.get("name") for m in (tags.get("models") or [])}
            for name in preferred:
                if name in names:
                    chosen = f"ollama/{name}"
                    break
        except Exception:
            pass
        if not chosen:
            env_model = os.getenv("LITELLM_DEFAULT_MODEL", "")
            if env_model.startswith("ollama/"):
                chosen = env_model
            else:
                pytest.skip("no suitable ollama text model found")

        from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
            AgentConfig,
            EchoMCP,
            arun_mcp_mini_agent,
        )

        messages = [
            {"role": "system", "content": "Reply directly and do not call tools."},
            {"role": "user", "content": "Say hello in one word."},
        ]
        cfg = AgentConfig(model=chosen, max_iterations=2, enable_repair=False, use_tools=False)
        out = await arun_mcp_mini_agent(messages, mcp=EchoMCP(), cfg=cfg)
        assert isinstance(out.final_answer, str) and len(out.final_answer.strip()) > 0

    asyncio.run(run())
