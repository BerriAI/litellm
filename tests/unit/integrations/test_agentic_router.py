# Golden parity tests for the agentic-router integration.
# Numeric cases are kept in lockstep with the cross-language reference
# implementation (https://github.com/lhwwxy/agentic-router-gateway):
# js/test/core.test.js / go/core/core_test.go.

import asyncio
import math

import pytest

from litellm.integrations.agentic_router import (
    AgenticRouterHook, Router, apply_reward, classify, correct_reward,
    features_of, fresh_state, recommend_rule, resolve_tier, reward_of, tier_of,
)

TIERS = {"providers": [{"provider": "a"}, {"provider": "b"}],
         "tiers": {"fast": ["a/flash"], "mid": ["b/mid"], "strong": ["a/pro"]}}


def test_classify_golden():
    assert classify("你好") == {"type": "other", "complexity": 0}
    assert classify("帮我总结这篇文章的要点")["type"] == "summarize"
    assert classify("帮我总结这篇文章的要点")["complexity"] == 0.05
    c = classify("def solve(x):\n  return x + 1\n```python")
    assert (c["type"], c["complexity"]) == ("code", 0.55)


def test_features_golden():
    text = "def solve(x):\n  return x + 1\n```python"
    assert features_of(classify(text), text) == [1.0, 0.55, 38 / 4000, 1.0, 0, 0, 0, 0, 0]


def test_tier_of_and_rule():
    assert tier_of("deepseek-v4-flash") == "fast"
    assert tier_of("gpt-4o") == "mid"
    assert tier_of("claude-pro") == "strong"
    assert recommend_rule({"complexity": 0.8})["tier"] == "strong"
    assert recommend_rule({"complexity": 0.2})["tier"] == "fast"


def test_resolve_tier_golden():
    assert resolve_tier("fast", "a", TIERS)["target"] == "flash"
    assert resolve_tier("mid", "a", TIERS)["note"] == "cross-provider"
    # mid with cross-provider disabled → falls back to strong
    assert resolve_tier("mid", "a", TIERS, {"crossProvider": False})["used"] == "strong"
    assert resolve_tier("mid", "a", None)["note"] == "tiers-not-ready"


def test_reward_golden():
    assert reward_of({}, "clean-close", 5000)[0] == 1
    assert reward_of({}, "failed", 5000)[0] == 0
    r, _ = reward_of({"toolErrors": 2, "retries": 1, "costYuan": 0.02}, "clean-close", 45000)
    assert r == 0  # 1 - 0.4 - 0.2 - 0.4 - 0.1 → clamped
    r, _ = reward_of({"toolErrors": 1, "costYuan": 0.002}, "clean-close", 5000)
    assert r == 0.76  # 1 - 0.2 - 0.04


def test_apply_reward_golden():
    state = fresh_state()
    rec = {"clsType": "code", "ucbTier": "fast", "linTier": "fast",
           "f": [1.0, 0.5, 0.1, 1.0, 0, 0, 0, 0, 0], "reward": 1,
           "chosen": "rule", "pChosen": 1, "pool": ["ucb", "linucb", "knn", "rule"]}
    apply_reward(state, rec)
    assert state["ucb"]["arms"]["code:fast"] == {"n": 1, "mean": 1}
    assert state["lin"]["tierW"]["fast"] == [1.0, 0.5, 0.1, 1.0, 0, 0, 0, 0, 0]
    assert state["exp3"]["weights"]["rule"] == pytest.approx(2.5 * math.exp(0.075))

    apply_reward(state, dict(rec, pChosen=0.5))
    assert state["ucb"]["arms"]["code:fast"] == {"n": 2, "mean": 1}
    # second linear update: lr=0.5, dot=2.26, err=-1.26
    assert state["lin"]["tierW"]["fast"][0] == pytest.approx(0.37)


def test_correct_reward_golden():
    state = fresh_state()
    rec = {"clsType": "code", "ucbTier": "fast", "linTier": "fast",
           "f": [1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0], "reward": 1,
           "chosen": "rule", "pChosen": 1, "pool": ["ucb", "linucb", "knn", "rule"]}
    apply_reward(state, rec)
    correct_reward(state, rec, 0)  # thumbs down: 1 → 0
    assert state["ucb"]["arms"]["code:fast"]["mean"] == 0  # n unchanged, mean corrected
    assert state["lin"]["tierW"]["fast"] == [0.0] * 9      # exact undo with recorded lr
    assert len(state["knn"]["pool"]) == 2                  # correction appended as new experience
    assert state["knn"]["pool"][1]["r"] == 0


def test_router_decide_settle(tmp_path):
    r = Router(home=str(tmp_path), mode="active", exploration=0, model_tiers={
        "providers": [{"provider": "p1"}],
        "tiers": {"fast": ["p1/lite"], "mid": ["p1/base"], "strong": ["p1/pro"]}})
    d = r.decide("帮我总结这篇文章的要点", "p1")
    assert d["type"] == "summarize"
    assert d["model"] in ("lite", "base", "pro")
    res = r.settle(d["id"], success=True, dt_ms=5000)
    assert res["reward"] == 1
    assert r.settle(d["id"], success=True) is None  # no double settle
    # state persisted across restarts
    r2 = Router(home=str(tmp_path), mode="active", exploration=0)
    assert r2.state["ucb"]["total"] == 1


def test_hook_active_mode_rewrites_model(tmp_path):
    hook = AgenticRouterHook(mode="active", home=str(tmp_path),
                             tier_models={"fast": "flash-model",
                                          "mid": "base-model",
                                          "strong": "pro-model"})
    data = {"model": "base-model", "litellm_call_id": "call-1",
            "messages": [{"role": "user", "content": "帮我总结这篇文章的要点"}]}
    out = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
    assert out["model"] in ("flash-model", "base-model", "pro-model")
    assert hook.stats()["pending"] == 1

    asyncio.run(hook.async_post_call_success_hook(
        None, {"litellm_call_id": "call-1", "response_cost": 0.002}, None,
        start_time=None, end_time=None))
    assert hook.stats()["pending"] == 0
    assert hook.stats()["linSamples"] == 1


def test_hook_shadow_mode_keeps_model(tmp_path):
    hook = AgenticRouterHook(mode="shadow", home=str(tmp_path),
                             tier_models={"fast": "f", "mid": "m", "strong": "s"})
    data = {"model": "m", "messages": [{"role": "user", "content": "请帮我重构这个模块并做性能优化"}]}
    out = asyncio.run(hook.async_pre_call_hook(None, None, data, "acompletion"))
    assert out["model"] == "m"  # shadow never rewrites
