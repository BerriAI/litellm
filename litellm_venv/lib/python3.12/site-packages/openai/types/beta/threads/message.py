# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Union, Optional
from typing_extensions import Literal, TypeAlias

from ...._models import BaseModel
from .message_content import MessageContent
from ...shared.metadata import Metadata
from ..code_interpreter_tool import CodeInterpreterTool

__all__ = [
    "Message",
    "Attachment",
    "AttachmentTool",
    "AttachmentToolAssistantToolsFileSearchTypeOnly",
    "IncompleteDetails",
]


class AttachmentToolAssistantToolsFileSearchTypeOnly(BaseModel):
    type: Literal["file_search"]
    """The type of tool being defined: `file_search`"""


AttachmentTool: TypeAlias = Union[
    CodeInterpreterTool, AttachmentToolAssistantToolsFileSearchTypeOnly
]


class Attachment(BaseModel):
    file_id: Optional[str] = None
    """The ID of the file to attach to the message."""

    tools: Optional[List[AttachmentTool]] = None
    """The tools to add this file to."""


class IncompleteDetails(BaseModel):
    reason: Literal[
        "content_filter", "max_tokens", "run_cancelled", "run_expired", "run_failed"
    ]
    """The reason the message is incomplete."""


class Message(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    assistant_id: Optional[str] = None
    """
    If applicable, the ID of the
    [assistant](https://platform.openai.com/docs/api-reference/assistants) that
    authored this message.
    """

    attachments: Optional[List[Attachment]] = None
    """A list of files attached to the message, and the tools they were added to."""

    completed_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the message was completed."""

    content: List[MessageContent]
    """The content of the message in array of text and/or images."""

    created_at: int
    """The Unix timestamp (in seconds) for when the message was created."""

    incomplete_at: Optional[int] = None
    """The Unix timestamp (in seconds) for when the message was marked as incomplete."""

    incomplete_details: Optional[IncompleteDetails] = None
    """On an incomplete message, details about why the message is incomplete."""

    metadata: Optional[Metadata] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format, and querying for objects via API or the dashboard.

    Keys are strings with a maximum length of 64 characters. Values are strings with
    a maximum length of 512 characters.
    """

    object: Literal["thread.message"]
    """The object type, which is always `thread.message`."""

    role: Literal["user", "assistant"]
    """The entity that produced the message. One of `user` or `assistant`."""

    run_id: Optional[str] = None
    """
    The ID of the [run](https://platform.openai.com/docs/api-reference/runs)
    associated with the creation of this message. Value is `null` when messages are
    created manually using the create message or create thread endpoints.
    """

    status: Literal["in_progress", "incomplete", "completed"]
    """
    The status of the message, which can be either `in_progress`, `incomplete`, or
    `completed`.
    """

    thread_id: str
    """
    The [thread](https://platform.openai.com/docs/api-reference/threads) ID that
    this message belongs to.
    """
