from typing import Final

ANTHROPIC_DOCS_REFUSAL_URL: Final = "https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#refusal-response"

# Copied verbatim from ANTHROPIC_DOCS_REFUSAL_URL on 2026-10-12. Streams carry the same stop_details on message_delta.
ANTHROPIC_DOCS_REFUSAL_MESSAGE: Final = {
    "id": "msg_01XFUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_details": {
        "type": "refusal",
        "category": "cyber",
        "explanation": "This request was declined because it could enable cyber harm.",
    },
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
