"""Test demonstrating LiteLLM's global warning filter pollution.

This test demonstrates that the global warning suppression introduced in commit
08b2b4f5f5 (June 19, 2025, PR #11895) affects non-LiteLLM code (e.g., user
application code) as well.

Specifically, litellm/llms/meta_llama/chat/transformation.py:12 installs:
    warnings.filterwarnings('ignore', message='Pydantic serializer warnings')

This is a GLOBAL filter that suppresses ANY warning matching that pattern,
regardless of whether it comes from LiteLLM code or user code.

EXPECTED: User warnings should be visible
ACTUAL: User warnings are SUPPRESSED by LiteLLM's global filter (test FAILS)

For full context, see: https://github.com/BerriAI/litellm/issues/11759#issuecomment-3494387017
"""
import sys
import warnings


def user_function_that_generates_warning(context: str):
    with warnings.catch_warnings(record=True) as w:
        # Application generates some pydantic warning completely unrelated to LiteLLM
        warnings.warn(f"Pydantic serializer warnings: context: {context}")

    assert len(w) > 0, f"User Pydantic warnings should be captured - context: {context}"
    assert "Pydantic serializer warnings" in str(w[0].message)

def test_1_user_warning_baseline():
    """Baseline: User warnings work without litellm imported.

    Only meaningful when run in isolation before litellm is imported.
    """
    if 'litellm' in sys.modules:
        return
    user_function_that_generates_warning("baseline")


def test_2_litellm_suppresses_user_warnings():
    """Like the baseline, LiteLLM should not suppress user warnings"""
    import litellm  # noqa: F401

    user_function_that_generates_warning("after litellm import")