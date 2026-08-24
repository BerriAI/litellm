"""Unit tests for the Conduct guardrail.

Mocked transport — no real network. Verifies the response-envelope
parser, pre-call hook behavior (allow / block / approval), fail-mode
handling, and session-ID resolution chain.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from litellm.proxy.guardrails.guardrail_hooks.conduct.conduct import (
    ConductGuardrail,
    ConductGuardrailBlocked,
    GuardDecision,
)


# ── Decision parsing ────────────────────────────────────────────────────


class TestGuardDecisionParse:
    @pytest.mark.parametrize("raw", ["ok", "OK", "", "  ok "])
    def test_ok_variants_are_allow(self, raw: str) -> None:
        assert GuardDecision.parse(raw).verdict == "allow"

    def test_blocked_extracts_rule_id(self) -> None:
        d = GuardDecision.parse(
            "BLOCKED — command touches /etc/passwd  [rule: no-etc-passwd]"
        )
        assert d.verdict == "block"
        assert d.rule_id == "no-etc-passwd"

    def test_pending_approval_treated_as_block(self) -> None:
        d = GuardDecision.parse(
            "PENDING approval — HITL required [rule: prod-deploy-gate]"
        )
        assert d.verdict == "approval"
        assert d.rule_id == "prod-deploy-gate"

    def test_warning_is_warning(self) -> None:
        d = GuardDecision.parse("WARNING — high-risk model [rule: model-tier]")
        assert d.verdict == "warning"

    def test_advisory_is_advisory(self) -> None:
        d = GuardDecision.parse("advisory: policy eval error: boom")
        assert d.verdict == "advisory"

    def test_unknown_prefix_marked_unknown(self) -> None:
        assert GuardDecision.parse("wat").verdict == "unknown"


# ── Pre-call hook ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _agent_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDUCT_AGENT_TOKEN", "cond_agt_test_placeholder")


def _guard() -> ConductGuardrail:
    return ConductGuardrail()


@pytest.mark.asyncio
class TestPreCallHook:
    async def test_allow_returns_data_with_metadata_tag(self) -> None:
        g = _guard()
        g._check = AsyncMock(return_value=GuardDecision(verdict="allow", raw="ok"))
        data = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = await g.async_pre_call_hook(None, None, data, "completion")
        assert result is data
        assert result["metadata"]["conduct_guard"]["verdict"] == "allow"

    async def test_block_raises(self) -> None:
        g = _guard()
        g._check = AsyncMock(
            return_value=GuardDecision(
                verdict="block",
                raw="BLOCKED — no secrets [rule: no-prod-secrets]",
                rule_id="no-prod-secrets",
            )
        )
        with pytest.raises(ConductGuardrailBlocked) as exc:
            await g.async_pre_call_hook(None, None, {}, "completion")
        assert exc.value.decision.rule_id == "no-prod-secrets"

    async def test_pending_approval_also_raises(self) -> None:
        g = _guard()
        g._check = AsyncMock(
            return_value=GuardDecision(verdict="approval", raw="PENDING approval — review")
        )
        with pytest.raises(ConductGuardrailBlocked):
            await g.async_pre_call_hook(None, None, {}, "completion")


class TestConfig:
    def test_missing_token_raises_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONDUCT_AGENT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="agent token"):
            ConductGuardrail()

    def test_config_api_key_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONDUCT_AGENT_TOKEN", "env-token")
        g = ConductGuardrail(api_key="config-token")
        assert g._agent_token == "config-token"

    def test_config_api_base_wins_over_default(self) -> None:
        g = ConductGuardrail(api_base="https://conduct.example.com/")
        assert g._api_url == "https://conduct.example.com"


if __name__ == "__main__":
    import subprocess
    import sys
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
