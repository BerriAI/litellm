"""
Purpose
- Hermetic proof that mini-agent surfaces a non-empty final answer (codex path stubbed).

Scope
- DOES: stub litellm.acompletion to return deterministic content; assert final_answer is a string.
- DOES NOT: call real providers or codex binaries.

Run
- `pytest tests/smoke -k test_mini_agent_codex_optional -q`
"""
import os, pytest
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

@pytest.mark.asyncio
async def test_mini_agent_codex_optional(monkeypatch):

    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        EchoMCP,
        arun_mcp_mini_agent,
    )
    # Hermetic stub: bypass provider resolution by faking litellm.acompletion
    import litellm as _litellm
    async def _fake_acompletion(*args, **kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "codex says hi"}}
            ]
        }
    monkeypatch.setattr(_litellm, "acompletion", _fake_acompletion, raising=True)


    messages = [{"role": "user", "content": "Say hello."}]
    cfg = AgentConfig(model="codex-agent/gpt-5", max_iterations=1, enable_repair=False, use_tools=False)
    out = await arun_mcp_mini_agent(messages, mcp=EchoMCP(), cfg=cfg)
    assert isinstance(out.final_answer, str) and len(out.final_answer.strip()) > 0
