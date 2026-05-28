"""
Regression tests for LIT-3095:
litellm_content_filter guardrail must raise HTTPException with status_code=400
(invalid request — content policy violation), NOT 403 (forbidden / auth failure)
when content is blocked.

This aligns the built-in `litellm_content_filter` with:
- `litellm.exceptions.GuardrailRaisedException` (default status_code=400)
- `litellm.exceptions.BlockedPiiEntityError` (default status_code=400)
- `litellm.integrations.custom_guardrail.CustomGuardrail._is_guardrail_intervention`
  which classifies HTTPException(status_code=400) as an intentional block
  (guardrail_intervened in logs) rather than guardrail_failed_to_respond.

Bug was reported on `nsfw-self-harm-filter-basic` (a basic harmful_self_harm
content filter) returning HTTP 403 to the client when a pre_call guardrail fired.
Fix: every `raise HTTPException(status_code=403, ...)` inside content_filter.py
was changed to `status_code=400`.
"""

import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import (  # noqa: E402
    ContentFilterGuardrail,
)


@pytest.mark.asyncio
async def test_lit_3095_category_keyword_block_returns_400():
    """
    Reproduces the ticket: pre_call content filter on `harmful_self_harm` blocks
    "I want to kill myself" and the raised HTTPException must be 400 (not 403).
    """
    guardrail = ContentFilterGuardrail(
        guardrail_name="nsfw-self-harm-filter-basic",
        categories=[
            {
                "category": "harmful_self_harm",
                "enabled": True,
                "action": "BLOCK",
                "severity_threshold": "medium",
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["I want to kill myself"]},
            request_data={},
            input_type="request",
        )

    assert exc_info.value.status_code == 400, (
        f"LIT-3095: expected 400 (invalid request — content policy violation), "
        f"got {exc_info.value.status_code}"
    )

    detail = exc_info.value.detail
    if isinstance(detail, dict):
        assert detail.get("category") == "harmful_self_harm"
        assert "kill myself" in str(detail.get("keyword", "")).lower()


@pytest.mark.asyncio
async def test_lit_3095_blocked_word_returns_400():
    """Blocked-word match path (blocked_words config) must raise 400."""
    guardrail = ContentFilterGuardrail(
        guardrail_name="word-block",
        blocked_words=[
            {
                "keyword": "forbidden-token-xyz",
                "action": "BLOCK",
                "description": "test sentinel",
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["this contains forbidden-token-xyz"]},
            request_data={},
            input_type="request",
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_lit_3095_pattern_match_returns_400():
    """Pattern-match block path must raise 400 (not 403)."""
    guardrail = ContentFilterGuardrail(
        guardrail_name="pattern-block",
        patterns=[
            {
                "pattern_type": "regex",
                "name": "test_pattern",
                "pattern": r"SECRET-\d{4}",
                "action": "BLOCK",
            }
        ],
    )

    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["please share SECRET-1234"]},
            request_data={},
            input_type="request",
        )

    assert exc_info.value.status_code == 400


def test_lit_3095_no_403_left_in_content_filter_raise_sites():
    """
    Static guard: there should be zero `raise HTTPException(... status_code=403 ...)`
    sites left in content_filter.py — neither multi-line, single-line, nor reordered.
    Any future regression that re-introduces a 403 will fail this test immediately,
    independent of category coverage.

    Uses an AST walk to find every `raise` statement whose value is a Call to
    `HTTPException` (or `fastapi.HTTPException`) with `status_code=403` passed
    either as a keyword argument or as the first positional argument. This catches
    multi-line, single-line, and any future stylistic reordering of the raise.
    """
    import ast

    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
        content_filter as _module,
    )

    src = open(_module.__file__).read()
    tree = ast.parse(src)

    def _is_http_exception(call_func):
        # Either `HTTPException(...)` or `fastapi.HTTPException(...)`
        if isinstance(call_func, ast.Name):
            return call_func.id == "HTTPException"
        if isinstance(call_func, ast.Attribute):
            return call_func.attr == "HTTPException"
        return False

    def _is_403(call):
        # status_code passed as kw
        for kw in call.keywords:
            if kw.arg == "status_code" and isinstance(kw.value, ast.Constant) and kw.value.value == 403:
                return True
        # HTTPException(status_code, detail) positional form
        if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == 403:
            return True
        return False

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            if _is_http_exception(node.exc.func) and _is_403(node.exc):
                offenders.append(node.lineno)

    assert offenders == [], (
        f"LIT-3095 regression: HTTPException(status_code=403) raised at line(s) "
        f"{offenders} in content_filter.py; expected 400 for content policy violations."
    )
