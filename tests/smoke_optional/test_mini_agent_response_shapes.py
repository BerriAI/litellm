import asyncio
import types


def _make_dict_resp(text: str):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class _MsgObj:
    def __init__(self, text: str):
        self.role = "assistant"
        self.content = text


class _ChoiceObj:
    def __init__(self, msg):
        self.message = msg


class _RespObj:
    def __init__(self, text: str):
        self.choices = [_ChoiceObj(_MsgObj(text))]


async def _fake_router_call_dict(**kwargs):
    return _make_dict_resp("dict-shape")


async def _fake_router_call_obj(**kwargs):
    return _RespObj("obj-shape")


def test_mini_agent_accepts_dict_and_object(monkeypatch):
    from litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent import (
        AgentConfig,
        LocalMCPInvoker,
        run_mcp_mini_agent,
    )
    import litellm.experimental_mcp_client.mini_agent.litellm_mcp_mini_agent as mod

    # 1) Dict-shaped response
    monkeypatch.setattr(mod, "arouter_call", _fake_router_call_dict)
    cfg = AgentConfig(model="noop", max_iterations=1)
    res = run_mcp_mini_agent([{"role": "user", "content": "hi"}], mcp=LocalMCPInvoker(), cfg=cfg)
    assert res.stopped_reason == "success"

    # 2) Object-shaped response
    monkeypatch.setattr(mod, "arouter_call", _fake_router_call_obj)
    cfg = AgentConfig(model="noop", max_iterations=1)
    res2 = run_mcp_mini_agent([{"role": "user", "content": "hi"}], mcp=LocalMCPInvoker(), cfg=cfg)
    assert res2.stopped_reason == "success"

