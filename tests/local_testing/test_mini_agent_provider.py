import asyncio
import os

import pytest

from litellm.utils import ModelResponse

os.environ.setdefault("LITELLM_ENABLE_MINI_AGENT", "1")

from litellm.llms.mini_agent import MiniAgentLLM  # noqa: E402


@pytest.mark.asyncio
async def test_mini_agent_completion(monkeypatch):
    provider = MiniAgentLLM()

    class DummyResult:
        def __init__(self, messages, cfg):
            self.final_answer = "analysis complete"
            self.iterations = []
            self.stopped_reason = "success"
            self.messages = messages
            self.used_model = cfg.model

    async def fake_run(messages, mcp, cfg):
        return DummyResult(messages, cfg)

    monkeypatch.setattr("litellm.llms.mini_agent.arun_mcp_mini_agent", fake_run, raising=False)

    response = provider.completion(
        model="mini-agent",
        messages=[{"role": "user", "content": "hi"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={"target_model": "deepseek-ai/DeepSeek-R1", "allowed_languages": ["python"]},
    )

    assert response.choices[0].message["content"] == "analysis complete"


@pytest.mark.asyncio
async def test_mini_agent_acompletion(monkeypatch):
    provider = MiniAgentLLM()

    class DummyResult:
        def __init__(self, messages, cfg):
            self.final_answer = "done"
            self.iterations = []
            self.stopped_reason = "success"
            self.messages = messages
            self.used_model = cfg.model

    async def fake_run(messages, mcp, cfg):
        return DummyResult(messages, cfg)

    monkeypatch.setattr("litellm.llms.mini_agent.arun_mcp_mini_agent", fake_run, raising=False)

    response = await provider.acompletion(
        model="mini-agent",
        messages=[{"role": "user", "content": "hi"}],
        api_base=None,
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=None,
        encoding=None,
        api_key=None,
        logging_obj=None,
        optional_params={"target_model": "deepseek-ai/DeepSeek-R1", "allowed_languages": ["python"]},
    )

    assert response.choices[0].message["content"] == "done"
