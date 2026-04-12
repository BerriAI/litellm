"""
Tests for the pipeline executor.

Uses mock guardrails to validate pipeline execution without external services.
"""

from unittest.mock import MagicMock

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy.policy_engine.pipeline_executor import PipelineExecutor
from litellm.types.proxy.policy_engine.pipeline_types import (
    GuardrailPipeline,
    PipelineStep,
)

try:
    from fastapi.exceptions import HTTPException
except ImportError:
    HTTPException = None


# ─────────────────────────────────────────────────────────────────────────────
# Mock Guardrails
# ─────────────────────────────────────────────────────────────────────────────


class AlwaysFailGuardrail(CustomGuardrail):
    """Mock guardrail that always raises HTTPException(400)."""

    def __init__(self, guardrail_name: str):
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook="pre_call",
            default_on=True,
        )
        self.calls = 0

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        raise HTTPException(status_code=400, detail="Content policy violation")


class AlwaysPassGuardrail(CustomGuardrail):
    """Mock guardrail that always passes."""

    def __init__(self, guardrail_name: str):
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook="pre_call",
            default_on=True,
        )
        self.calls = 0

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        return None


class PiiMaskingGuardrail(CustomGuardrail):
    """Mock guardrail that masks PII in messages and returns modified data."""

    def __init__(self, guardrail_name: str):
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook="pre_call",
            default_on=True,
        )
        self.calls = 0
        self.received_messages = None

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        self.received_messages = data.get("messages", [])
        masked_messages = []
        for msg in data.get("messages", []):
            masked_msg = dict(msg)
            masked_msg["content"] = msg["content"].replace(
                "John Smith", "[REDACTED]"
            )
            masked_messages.append(masked_msg)
        return {"messages": masked_messages}


class ContentCheckGuardrail(CustomGuardrail):
    """Mock guardrail that records what messages it received."""

    def __init__(self, guardrail_name: str):
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook="pre_call",
            default_on=True,
        )
        self.calls = 0
        self.received_messages = None

    def should_run_guardrail(self, data, event_type) -> bool:
        return True

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        self.received_messages = data.get("messages", [])
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(HTTPException is None, reason="fastapi not installed")
@pytest.mark.asyncio
async def test_escalation_step1_fails_step2_blocks():
    """
    Pipeline: simple-filter (on_fail: next) -> advanced-filter (on_fail: block)
    Input: request that fails simple-filter
    Expected: simple-filter fails -> escalate -> advanced-filter fails -> block
    """
    simple_guard = AlwaysFailGuardrail(guardrail_name="simple-filter")
    advanced_guard = AlwaysFailGuardrail(guardrail_name="advanced-filter")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="simple-filter", on_fail="next", on_pass="allow"
            ),
            PipelineStep(
                guardrail="advanced-filter", on_fail="block", on_pass="allow"
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [simple_guard, advanced_guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "bad content"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="content-safety",
        )

        assert simple_guard.calls == 1
        assert advanced_guard.calls == 1
        assert result.terminal_action == "block"
        assert len(result.step_results) == 2
        assert result.step_results[0].guardrail_name == "simple-filter"
        assert result.step_results[0].outcome == "fail"
        assert result.step_results[0].action_taken == "next"
        assert result.step_results[1].guardrail_name == "advanced-filter"
        assert result.step_results[1].outcome == "fail"
        assert result.step_results[1].action_taken == "block"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.skipif(HTTPException is None, reason="fastapi not installed")
@pytest.mark.asyncio
async def test_early_allow_step1_passes_step2_skipped():
    """
    Pipeline: simple-filter (on_pass: allow) -> advanced-filter
    Input: clean request that passes simple-filter
    Expected: simple-filter passes -> allow (advanced-filter never called)
    """
    simple_guard = AlwaysPassGuardrail(guardrail_name="simple-filter")
    advanced_guard = AlwaysFailGuardrail(guardrail_name="advanced-filter")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="simple-filter", on_fail="next", on_pass="allow"
            ),
            PipelineStep(
                guardrail="advanced-filter", on_fail="block", on_pass="allow"
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [simple_guard, advanced_guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "clean content"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="content-safety",
        )

        assert simple_guard.calls == 1
        assert advanced_guard.calls == 0
        assert result.terminal_action == "allow"
        assert len(result.step_results) == 1
        assert result.step_results[0].outcome == "pass"
        assert result.step_results[0].action_taken == "allow"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.skipif(HTTPException is None, reason="fastapi not installed")
@pytest.mark.asyncio
async def test_escalation_step1_fails_step2_passes():
    """
    Pipeline: simple-filter (on_fail: next) -> advanced-filter (on_pass: allow)
    Input: request that fails simple but passes advanced
    Expected: simple-filter fails -> escalate -> advanced-filter passes -> allow
    """
    simple_guard = AlwaysFailGuardrail(guardrail_name="simple-filter")
    advanced_guard = AlwaysPassGuardrail(guardrail_name="advanced-filter")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="simple-filter", on_fail="next", on_pass="allow"
            ),
            PipelineStep(
                guardrail="advanced-filter", on_fail="block", on_pass="allow"
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [simple_guard, advanced_guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "borderline content"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="content-safety",
        )

        assert simple_guard.calls == 1
        assert advanced_guard.calls == 1
        assert result.terminal_action == "allow"
        assert len(result.step_results) == 2
        assert result.step_results[0].outcome == "fail"
        assert result.step_results[0].action_taken == "next"
        assert result.step_results[1].outcome == "pass"
        assert result.step_results[1].action_taken == "allow"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.skipif(HTTPException is None, reason="fastapi not installed")
@pytest.mark.asyncio
async def test_data_forwarding_pii_masking():
    """
    Pipeline: pii-masker (pass_data: true, on_pass: next) -> content-check (on_pass: allow)
    Input: "Hello John Smith"
    Expected: pii-masker masks -> content-check receives "[REDACTED]" -> allow
    """
    pii_guard = PiiMaskingGuardrail(guardrail_name="pii-masker")
    content_guard = ContentCheckGuardrail(guardrail_name="content-check")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="pii-masker",
                on_fail="block",
                on_pass="next",
                pass_data=True,
            ),
            PipelineStep(
                guardrail="content-check", on_fail="block", on_pass="allow"
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [pii_guard, content_guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={
                "messages": [{"role": "user", "content": "Hello John Smith"}]
            },
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="pii-then-safety",
        )

        assert pii_guard.calls == 1
        assert content_guard.calls == 1
        assert content_guard.received_messages[0]["content"] == "Hello [REDACTED]"
        assert result.terminal_action == "allow"
        assert result.modified_data is not None
        assert result.modified_data["messages"][0]["content"] == "Hello [REDACTED]"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_guardrail_not_found_uses_on_fail():
    """
    If a guardrail is not found, treat as error and use on_fail action.
    """
    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="nonexistent-guard",
                on_fail="block",
                on_pass="allow",
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = []

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="test-policy",
        )

        assert result.terminal_action == "block"
        assert result.step_results[0].outcome == "error"
        assert "not found" in result.step_results[0].error_detail
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_guardrail_not_found_with_next_continues():
    """
    If a guardrail is not found and on_fail is 'next', continue to next step.
    """
    pass_guard = AlwaysPassGuardrail(guardrail_name="fallback-guard")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[
            PipelineStep(
                guardrail="nonexistent-guard",
                on_fail="next",
                on_pass="allow",
            ),
            PipelineStep(
                guardrail="fallback-guard",
                on_fail="block",
                on_pass="allow",
            ),
        ],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [pass_guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="test-policy",
        )

        assert result.terminal_action == "allow"
        assert len(result.step_results) == 2
        assert result.step_results[0].outcome == "error"
        assert result.step_results[0].action_taken == "next"
        assert result.step_results[1].outcome == "pass"
        assert pass_guard.calls == 1
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.skipif(HTTPException is None, reason="fastapi not installed")
@pytest.mark.asyncio
async def test_single_step_pipeline_block():
    """Single step pipeline that blocks."""
    guard = AlwaysFailGuardrail(guardrail_name="blocker")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[PipelineStep(guardrail="blocker", on_fail="block")],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="test",
        )

        assert result.terminal_action == "block"
        assert guard.calls == 1
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_single_step_pipeline_allow():
    """Single step pipeline that allows."""
    guard = AlwaysPassGuardrail(guardrail_name="passer")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[PipelineStep(guardrail="passer", on_pass="allow")],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="test",
        )

        assert result.terminal_action == "allow"
        assert guard.calls == 1
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_step_results_include_duration():
    """Step results should include timing information."""
    guard = AlwaysPassGuardrail(guardrail_name="timed")

    pipeline = GuardrailPipeline(
        mode="pre_call",
        steps=[PipelineStep(guardrail="timed")],
    )

    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [guard]

    try:
        result = await PipelineExecutor.execute_steps(
            steps=pipeline.steps,
            mode=pipeline.mode,
            data={"messages": [{"role": "user", "content": "test"}]},
            user_api_key_dict=MagicMock(),
            call_type="completion",
            policy_name="test",
        )

        assert result.step_results[0].duration_seconds is not None
        assert result.step_results[0].duration_seconds >= 0
    finally:
        litellm.callbacks = original_callbacks
