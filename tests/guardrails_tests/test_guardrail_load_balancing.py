"""
Test guardrail load balancing through the Router.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest
from litellm import Router


@pytest.mark.asyncio
async def test_guardrail_weighted_pick():
    """Test that aguardrail() respects weight parameter for weighted selection."""
    # guardrail-1 has weight=9, guardrail-2 has weight=1
    # So guardrail-1 should be selected ~90% of the time
    guardrail_list = [
        {
            "guardrail_name": "content-filter",
            "litellm_params": {"guardrail": "aporia", "mode": "pre_call", "weight": 9},
            "id": "guardrail-1",
        },
        {
            "guardrail_name": "content-filter",
            "litellm_params": {"guardrail": "bedrock", "mode": "pre_call", "weight": 1},
            "id": "guardrail-2",
        },
    ]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "fake-key"},
            }
        ],
        guardrail_list=guardrail_list,
    )

    selections = {}

    async def mock_guardrail_function(**kwargs):
        selected = kwargs.get("selected_guardrail", {})
        selected_id = selected.get("id", "unknown")
        selections[selected_id] = selections.get(selected_id, 0) + 1
        return {"status": "success"}

    # Run 100 times
    for _ in range(100):
        await router.aguardrail(
            guardrail_name="content-filter",
            original_function=mock_guardrail_function,
        )

    # guardrail-1 (weight=9) should be selected significantly more than guardrail-2 (weight=1)
    assert selections.get("guardrail-1", 0) > selections.get("guardrail-2", 0) * 2, \
        f"Expected guardrail-1 to be selected much more often. Got: {selections}"
