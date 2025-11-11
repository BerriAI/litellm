"""Test demonstrating Pydantic serialization warnings from LiteLLM responses.

LiteLLM's Message and Choices classes use an anti-pattern: deleting fields after
initialization instead of using Field(exclude=True). This causes Pydantic to emit
serialization warnings when model_dump() is called on responses.

The anti-pattern was introduced in multiple commits:
- Message.audio: 13e0b3f626 (Oct 18, 2024) - PR #6304
- Message.thinking_blocks: ab7c4d1a0e (Feb 26, 2025) - PR #8843
- Message.annotations: 44f4c623e2 (Mar 22, 2025)

Instead of fixing the root cause, commit 08b2b4f5f5 (June 19, 2025, PR #11895)
added a GLOBAL warning filter that suppresses these warnings everywhere.

EXPECTED: No warnings when serializing LiteLLM responses
ACTUAL: Warnings ARE generated (but hidden by global filter)

For full context, see: https://github.com/BerriAI/litellm/issues/11759#issuecomment-3494387017
"""
import warnings

from litellm.types.utils import Choices, Message, ModelResponse


def test_response_serialization_warnings():
    """ModelResponse objects generate Pydantic warnings due to 'del self.field' anti-pattern."""
    # Create a response like LiteLLM would return
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                message=Message(
                    content="test response",
                    role="assistant",
                    audio=None,  # Triggers del self.audio in Message.__init__
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )

    # Explicitly capture warnings, overriding any global filters
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        response.model_dump()

    # Test FAILS when bug exists (warnings generated)
    # Test PASSES when bug is fixed (no warnings)
    assert len(w) == 0, (
        f"Pydantic serialization warnings detected ({len(w)} warning(s)).\n"
        "\n"
        f"Warning: {w[0].message if w else 'N/A'}\n"
        "\n"
        "Root cause: Message/Choices classes use 'del self.field' anti-pattern in __init__\n"
        "Fix: Use Field(exclude=True) instead of 'del self.field'\n"
        "\n"
        "Problematic fields in litellm/types/utils.py:\n"
        "  Message.__init__:\n"
        "    - audio (line 708): del self.audio\n"
        "    - images (line 712): del self.images\n"
        "    - annotations (line 718): del self.annotations\n"
        "    - reasoning_content (line 723): del self.reasoning_content\n"
        "    - thinking_blocks (line 728): del self.thinking_blocks\n"
        "  Choices.__init__:\n"
        "    - logprobs (line 897): del self.logprobs\n"
        "    - provider_specific_fields (similar pattern)"
    )
