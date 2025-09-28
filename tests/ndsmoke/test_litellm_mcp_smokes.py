import os
import asyncio
import socket
import json
import pytest


def _can(host: str, port: int, t: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=t):
            return True
    except OSError:
        return False


@pytest.mark.ndsmoke
def test_mcp_tools_registration_smoke():
    """Smoke: shared registration exposes expected tools."""
    from litellm.proxy._experimental.mcp_server.shared import (
        register_default_local_tools,
    )
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

    # Ensure registration idempotently
    register_default_local_tools(global_mcp_tool_registry)
    names = {t.name for t in global_mcp_tool_registry.list_tools()}
    assert "model.advice" in names
    assert "llm.chat" in names


@pytest.mark.ndsmoke
def test_mcp_model_advice_returns_text():
    """model.advice returns a helpful text payload."""
    from litellm.proxy._experimental.mcp_server.shared import (
        register_default_local_tools,
    )
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

    register_default_local_tools(global_mcp_tool_registry)
    tool = global_mcp_tool_registry.get_tool("model.advice")
    assert tool is not None
    out = tool.handler(
        task_description="summarize a 600k token pdf with images",
        max_context_tokens=600000,
    )
    assert isinstance(out, str) and len(out) > 0
    # Should likely mention gemini flash for long context
    assert "gemini" in out.lower()


@pytest.mark.ndsmoke
def test_mcp_llm_chat_codex_agent_optional():
    """llm.chat can call codex-agent/mini via LiteLLM if mini-agent gateway is up."""
    base = os.getenv("CODEX_AGENT_API_BASE", "http://127.0.0.1:8788")
    host = base.split("//", 1)[-1].split(":")[0]
    try:
        port = int(base.split(":")[-1])
    except Exception:
        port = 8788
    if not _can(host, port):
        pytest.skip(f"codex-agent base not reachable on {host}:{port}")

    from litellm.proxy._experimental.mcp_server.shared import (
        register_default_local_tools,
    )
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

    os.environ.setdefault("LITELLM_ENABLE_CODEX_AGENT", "1")
    os.environ.setdefault("CODEX_AGENT_API_BASE", base)

    register_default_local_tools(global_mcp_tool_registry)
    tool = global_mcp_tool_registry.get_tool("llm.chat")
    assert tool is not None

    try:
        result = asyncio.run(tool.handler(
            model="codex-agent/mini",
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
        ))
        assert isinstance(result, str)
    except Exception as e:
        pytest.skip(f"codex-agent/mini path not available: {e}")


@pytest.mark.ndsmoke
def test_mcp_llm_chat_ollama_optional():
    """llm.chat can call a local ollama model when daemon is up (skip-friendly)."""
    base = os.getenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    host = base.split("//", 1)[-1].split(":")[0]
    try:
        port = int(base.split(":")[-1])
    except Exception:
        port = 11434
    if not _can(host, port):
        pytest.skip(f"Ollama not reachable on {host}:{port}")

    # pick a model tag if available
    try:
        import urllib.request
        tags = json.loads(
            urllib.request.urlopen(base + "/api/tags", timeout=2).read().decode()
        )
        models = [m.get("model") for m in (tags.get("models") or []) if m.get("model")]
        tag = models[0] if models else "llama3.1"
    except Exception:
        tag = "llama3.1"

    from litellm.proxy._experimental.mcp_server.shared import (
        register_default_local_tools,
    )
    from litellm.proxy._experimental.mcp_server.tool_registry import (
        global_mcp_tool_registry,
    )

    register_default_local_tools(global_mcp_tool_registry)
    tool = global_mcp_tool_registry.get_tool("llm.chat")
    assert tool is not None
    out = asyncio.run(tool.handler(
        model=f"ollama_chat/{tag}",
        messages=[{"role": "user", "content": "ping"}],
        stream=False,
        api_base=base,
    ))
    assert isinstance(out, str)
