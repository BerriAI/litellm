import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.types.main import _CompletionDispatchContext


def _build_context() -> _CompletionDispatchContext:
    return _CompletionDispatchContext(
        _azure_detection_model="gpt-4o",
        acompletion=False,
        api_base=None,
        api_key=None,
        api_version=None,
        client=None,
        custom_llm_provider="openai",
        custom_prompt_dict={},
        extra_headers=None,
        headers={},
        hf_model_name=None,
        kwargs={},
        litellm_params={},
        logger_fn=None,
        logging=None,  # type: ignore[arg-type]
        max_retries=None,
        max_tokens=None,
        messages=[],
        metadata=None,
        model="gpt-4o",
        model_response=None,  # type: ignore[arg-type]
        optional_params={},
        organization=None,
        provider_config=None,
        shared_session=None,
        stream=None,
        temperature=None,
        text_completion=False,
        timeout=None,
        top_p=None,
    )


def test_dispatch_context_is_frozen():
    """A helper must not be able to re-route the call by rebinding a dispatch
    input mid-flight; this pins the frozen invariant the dispatch shape relies on."""
    ctx = _build_context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model = "claude-haiku-4-5"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.custom_llm_provider = "anthropic"  # type: ignore[misc]


def test_dispatch_context_uses_slots():
    """slots=True keeps the per-call context lightweight (no per-instance __dict__)."""
    ctx = _build_context()
    assert not hasattr(ctx, "__dict__")
    assert hasattr(type(ctx), "__slots__")
