"""
Compliance test for Block Code Execution guardrail.

Runs the code execution compliance dataset (from codeExecutionCompliancePrompts.ts)
against apply_guardrail and asserts 100% match: expected "fail" → guardrail blocks,
expected "pass" → guardrail allows.
"""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy.guardrails.guardrail_hooks.block_code_execution import (
    BlockCodeExecutionGuardrail,
)


def _load_compliance_dataset():
    path = (
        Path(__file__).resolve().parent
        / "code_execution_compliance_dataset.json"
    )
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def compliance_dataset():
    return _load_compliance_dataset()


@pytest.fixture(scope="module")
def guardrail():
    """Guardrail with block_all and execution intent detection (compliance mode)."""
    return BlockCodeExecutionGuardrail(
        guardrail_name="block_code_execution_compliance",
        blocked_languages=None,  # block all fenced code
        action="block",
        confidence_threshold=0.5,
        detect_execution_intent=True,
    )


@pytest.mark.asyncio
async def test_code_execution_compliance_dataset_scores_100_percent(
    guardrail, compliance_dataset
):
    """Run full compliance dataset against apply_guardrail; expect 100% match."""
    request_data = {}
    passed = 0
    failed = []
    for item in compliance_dataset:
        prompt = item["prompt"]
        expected = item["expected_result"]
        inputs = {"texts": [prompt]}
        try:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="request",
            )
            actual = "pass"
        except (HTTPException, ModifyResponseException):
            actual = "fail"
        if actual == expected:
            passed += 1
        else:
            failed.append(
                {
                    "id": item["id"],
                    "expected": expected,
                    "actual": actual,
                    "prompt_preview": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                }
            )
    total = len(compliance_dataset)
    pct = 100.0 * passed / total if total else 0
    assert failed == [], (
        f"Compliance score {passed}/{total} ({pct:.1f}%). Failures: {failed}"
    )
    assert pct == 100.0, f"Expected 100% compliance, got {pct:.1f}%"
