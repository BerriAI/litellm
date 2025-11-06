"""Test demonstrating that Pydantic warnings leak through as visible output.

LiteLLM's Message and Choices classes use an anti-pattern: deleting fields after
initialization instead of using Field(exclude=True). This was introduced in:
- Message.audio: 13e0b3f626 (Oct 18, 2024) - PR #6304
- Message.thinking_blocks: ab7c4d1a0e (Feb 26, 2025) - PR #8843
- Message.annotations: 44f4c623e2 (Mar 22, 2025)

Instead of fixing the root cause, commit 08b2b4f5f5 (June 19, 2025, PR #11895)
added a global filter at transformation.py:12 to suppress these warnings.

However, the global filter doesn't prevent warnings from being generated - it only
suppresses them. Pytest still collects them and displays them in the "warnings
summary" section at the end of test runs, polluting test output.

Run this test with: pytest tests/test_warnings_leak_in_strict_mode.py -xvs
You'll see warnings in the output even though tests pass (without this fix).

For full context, see: https://github.com/BerriAI/litellm/issues/11759#issuecomment-3494387017
"""
from litellm.types.utils import Choices, Message, ModelResponse


def test_serialization_produces_visible_warnings(recwarn):
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                message=Message(
                    content="test response",
                    role="assistant",
                    audio=None,  # Triggers del self.audio
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Just do a normal model_dump
    response.model_dump()

    # Check if pytest collected any warnings
    warnings_list = list(recwarn)

    # Test FAILS if warnings were collected (leaked through despite global filter)
    # Test PASSES if no warnings (fix applied, no warnings generated)
    assert len(warnings_list) == 0, (
        f"Serialization produced {len(warnings_list)} warning(s) that leak into pytest output.\n"
        "\n"
        f"Warning message: {warnings_list[0].message if warnings_list else 'N/A'}"
    )
