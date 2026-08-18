"""
e2b code execution sandbox - end-to-end integration tests.

These tests make REAL HTTP calls to the e2b API and are skipped automatically
unless E2B_API_KEY is set. Mock-only unit tests live in
tests/test_litellm/sandbox/test_e2b_sandbox.py.

Run only these tests:
    pytest tests/integration/sandbox/test_e2b_sandbox.py -v
"""

import os

import pytest

import litellm


@pytest.mark.skipif("E2B_API_KEY" not in os.environ, reason="needs a real E2B_API_KEY")
@pytest.mark.asyncio
async def test_integration_ephemeral_real_e2b():
    result = await litellm.acode_interpreter_tool(
        provider="e2b", code="print(sum(range(10)))"
    )
    assert result.stdout.strip() == "45"
    assert result.error is None


@pytest.mark.skipif("E2B_API_KEY" not in os.environ, reason="needs a real E2B_API_KEY")
@pytest.mark.asyncio
async def test_integration_lifecycle_roundtrip_real_e2b():
    container = await litellm.acreate_sandbox(provider="e2b")
    try:
        result = await litellm.arun_code(
            provider="e2b", container=container, code="print(6*7)"
        )
        assert result.stdout.strip() == "42"
    finally:
        assert await litellm.adelete_sandbox(provider="e2b", container=container)
