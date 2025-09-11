from __future__ import annotations

from typing_extensions import TypeAlias

from ....types.chat import (
    ParsedChoice,
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
)

ParsedChatCompletionSnapshot: TypeAlias = ParsedChatCompletion[object]
"""Snapshot type representing an in-progress accumulation of
a `ParsedChatCompletion` object.
"""

ParsedChatCompletionMessageSnapshot: TypeAlias = ParsedChatCompletionMessage[object]
"""Snapshot type representing an in-progress accumulation of
a `ParsedChatCompletionMessage` object.

If the content has been fully accumulated, the `.parsed` content will be
the `response_format` instance, otherwise it'll be the raw JSON parsed version.
"""

ParsedChoiceSnapshot: TypeAlias = ParsedChoice[object]
