from typing import (
    Optional,
    Union,
    Any,
    BinaryIO,
    Literal,
    Annotated,
    Iterable,
)
from typing_extensions import override
from pydantic import BaseModel

from openai.types.beta.threads.message_content import MessageContent
from openai.types.beta.threads.message_create_params import Attachment
from openai.types.beta.threads.message import Message as OpenAIMessage
from openai.types.beta.thread_create_params import (
    Message as OpenAICreateThreadParamsMessage,
    ToolResources as OpenAICreateThreadParamsToolResources,
)
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.threads.run import Run
from openai.types.beta.assistant import Assistant
from openai.pagination import SyncCursorPage

from typing import TypedDict, List, Optional


class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response:
        ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class MessageData(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    attachments: Optional[List[Attachment]]
    metadata: Optional[dict]


class Thread(BaseModel):
    id: str
    """The identifier, which can be referenced in API endpoints."""

    created_at: int
    """The Unix timestamp (in seconds) for when the thread was created."""

    metadata: Optional[object] = None
    """Set of 16 key-value pairs that can be attached to an object.

    This can be useful for storing additional information about the object in a
    structured format. Keys can be a maximum of 64 characters long and values can be
    a maxium of 512 characters long.
    """

    object: Literal["thread"]
    """The object type, which is always `thread`."""
