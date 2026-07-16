"""Unit tests for the shared `run_passthrough_cell` helper.

These tests inject a fake `run_models` callable and an explicit `env`
mapping (both are first-class parameters, no monkeypatching), so they
exercise the helper's branching -- env-missing guard, base-URL
assembly, extra-env forwarding, per-model pass/fail -- without
spawning the real CLI.

The env-builder tests pin the provider-mode contract itself: the
CLAUDE_CODE_USE_* / CLAUDE_CODE_SKIP_*_AUTH flags and the passthrough
route each mode must target. Those values are the feature -- e.g.
dropping the `/v1` from the vertex base URL produces a request Google
404s on -- so a mutation to any of them must fail here before it burns
a live matrix run.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import pytest

from claude_code._passthrough import (
    ANTHROPIC_PASSTHROUGH_BASE_PATH,
    CLIENT_SIDE_AWS_REGION,
    VERTEX_PLACEHOLDER_PROJECT,
    VERTEX_PLACEHOLDER_REGION,
    bedrock_extra_env,
    foundry_extra_env,
    run_passthrough_cell,
    vertex_extra_env,
)
from claude_code.cli_driver import ClaudeCLIError, DriverResult

PROXY_ENV = {
    "LITELLM_PROXY_BASE_URL": "http://localhost:4000",
    "LITELLM_PROXY_API_KEY": "sk-test",
}


class _FakeResult:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.single: Optional[Dict[str, Any]] = None

    def set(self, payload: Mapping[str, Any]) -> None:
        self.single = dict(payload)

    def add(self, payload: Mapping[str, Any]) -> None:
        self.rows.append(dict(payload))


def _fake_run_models(outcomes_by_model, captured: Dict[str, Any]):
    def fake(*, models, prompt, base_url, api_key, extra_env=None, **_kwargs):
        captured["models"] = list(models)
        captured["prompt"] = prompt
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["extra_env"] = dict(extra_env) if extra_env is not None else None
        return {model: outcomes_by_model[model] for model in models}

    return fake


def test_env_missing_guard_reports_fail_and_aborts():
    fake_result = _FakeResult()
    with pytest.raises(pytest.fail.Exception):
        run_passthrough_cell(
            compat_result=fake_result,
            models=["claude-haiku-4-5"],
            prompt="ping",
            env={},
        )
    assert fake_result.single is not None
    assert fake_result.single["status"] == "fail"
    assert "LITELLM_PROXY_BASE_URL" in fake_result.single["error"]


def test_anthropic_base_path_appended_to_normalized_proxy_url():
    fake_result = _FakeResult()
    captured: Dict[str, Any] = {}
    outcome = DriverResult(text="pong")

    run_passthrough_cell(
        compat_result=fake_result,
        models=["claude-haiku-4-5"],
        prompt="ping",
        passthrough_base_path=ANTHROPIC_PASSTHROUGH_BASE_PATH,
        run_models=_fake_run_models({"claude-haiku-4-5": outcome}, captured),
        env={**PROXY_ENV, "LITELLM_PROXY_BASE_URL": "http://localhost:4000/"},
    )

    assert captured["base_url"] == "http://localhost:4000/anthropic"
    assert captured["extra_env"] is None
    assert fake_result.rows == [{"status": "pass"}]


def test_extra_env_builder_receives_normalized_base_and_is_forwarded():
    fake_result = _FakeResult()
    captured: Dict[str, Any] = {}
    outcome = DriverResult(text="pong")
    seen_bases: List[str] = []

    def build(proxy_base: str) -> Dict[str, str]:
        seen_bases.append(proxy_base)
        return {"SOME_FLAG": "1"}

    run_passthrough_cell(
        compat_result=fake_result,
        models=["claude-haiku-4-5"],
        prompt="ping",
        build_extra_env=build,
        run_models=_fake_run_models({"claude-haiku-4-5": outcome}, captured),
        env={**PROXY_ENV, "LITELLM_PROXY_BASE_URL": "http://localhost:4000/"},
    )

    assert seen_bases == ["http://localhost:4000"]
    assert captured["extra_env"] == {"SOME_FLAG": "1"}
    assert captured["base_url"] == "http://localhost:4000"


def test_per_model_failures_reported_individually():
    fake_result = _FakeResult()
    captured: Dict[str, Any] = {}
    outcomes = {
        "claude-haiku-4-5": DriverResult(text="pong"),
        "claude-sonnet-4-6": ClaudeCLIError("claude CLI timed out after 120s"),
        "claude-opus-4-7": DriverResult(text="", exit_code=1),
    }

    with pytest.raises(pytest.fail.Exception):
        run_passthrough_cell(
            compat_result=fake_result,
            models=list(outcomes.keys()),
            prompt="ping",
            run_models=_fake_run_models(outcomes, captured),
            env=PROXY_ENV,
        )

    statuses = [row["status"] for row in fake_result.rows]
    assert statuses == ["pass", "fail", "fail"]
    assert "timed out" in fake_result.rows[1]["error"]
    assert "claude CLI failed" in fake_result.rows[2]["error"]


def test_empty_assistant_text_is_a_fail():
    fake_result = _FakeResult()
    captured: Dict[str, Any] = {}
    outcomes = {"claude-haiku-4-5": DriverResult(text="   ")}

    with pytest.raises(pytest.fail.Exception):
        run_passthrough_cell(
            compat_result=fake_result,
            models=["claude-haiku-4-5"],
            prompt="ping",
            run_models=_fake_run_models(outcomes, captured),
            env=PROXY_ENV,
        )

    assert fake_result.rows == [
        {
            "status": "fail",
            "error": "[claude-haiku-4-5] claude returned empty assistant text",
        }
    ]


def test_bedrock_extra_env_targets_proxy_bedrock_route():
    env = bedrock_extra_env("http://localhost:4000")
    assert env == {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
        "ANTHROPIC_BEDROCK_BASE_URL": "http://localhost:4000/bedrock",
        "AWS_REGION": CLIENT_SIDE_AWS_REGION,
    }


def test_vertex_extra_env_keeps_the_api_version_in_the_base_url():
    env = vertex_extra_env("http://localhost:4000")
    assert env == {
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_VERTEX_BASE_URL": "http://localhost:4000/vertex_ai/v1",
        "ANTHROPIC_VERTEX_PROJECT_ID": VERTEX_PLACEHOLDER_PROJECT,
        "CLOUD_ML_REGION": VERTEX_PLACEHOLDER_REGION,
    }


def test_foundry_extra_env_targets_proxy_azure_route():
    env = foundry_extra_env("http://localhost:4000")
    assert env == {
        "CLAUDE_CODE_USE_FOUNDRY": "1",
        "CLAUDE_CODE_SKIP_FOUNDRY_AUTH": "1",
        "ANTHROPIC_FOUNDRY_BASE_URL": "http://localhost:4000/azure",
    }
