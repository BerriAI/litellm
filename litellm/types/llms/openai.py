from typing import (
    Optional,
    Union,
    Any,
    BinaryIO,
    Literal,
    Iterable,
)
from typing_extensions import override, Required, Dict
from pydantic import BaseModel
from openai.types.beta.threads.message_content import MessageContent
from openai.types.beta.threads.message import Message as OpenAIMessage
from openai.types.beta.thread_create_params import (
    Message as OpenAICreateThreadParamsMessage,
)
from openai.lib.streaming._assistants import (
    AssistantEventHandler,
    AssistantStreamManager,
    AsyncAssistantStreamManager,
    AsyncAssistantEventHandler,
)
from openai.types.beta.assistant_tool_param import AssistantToolParam
from openai.types.beta.threads.run import Run
from openai.types.beta.assistant import Assistant
from openai.pagination import SyncCursorPage, AsyncCursorPage
from os import PathLike
from openai.types import FileObject, Batch
from openai._legacy_response import HttpxBinaryResponseContent
from typing import TypedDict, List, Optional, Tuple, Mapping, IO

FileContent = Union[IO[bytes], bytes, PathLike]

FileTypes = Union[
    # file (or bytes)
    FileContent,
    # (filename, file (or bytes))
    Tuple[Optional[str], FileContent],
    # (filename, file (or bytes), content_type)
    Tuple[Optional[str], FileContent, Optional[str]],
    # (filename, file (or bytes), content_type, headers)
    Tuple[Optional[str], FileContent, Optional[str], Mapping[str, str]],
]


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


class ToolResourcesCodeInterpreter(TypedDict, total=False):
    file_ids: List[str]
    """
    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs made
    available to the `code_interpreter` tool. There can be a maximum of 20 files
    associated with the tool.
    """


class ToolResourcesFileSearchVectorStore(TypedDict, total=False):
    file_ids: List[str]
    """
    A list of [file](https://platform.openai.com/docs/api-reference/files) IDs to
    add to the vector store. There can be a maximum of 10000 files in a vector
    store.
    """

    metadata: object
    """Set of 16 key-value pairs that can be attached to a vector store.

    This can be useful for storing additional information about the vector store in
    a structured format. Keys can be a maximum of 64 characters long and values can
    be a maxium of 512 characters long.
    """


class ToolResourcesFileSearch(TypedDict, total=False):
    vector_store_ids: List[str]
    """
    The
    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)
    attached to this thread. There can be a maximum of 1 vector store attached to
    the thread.
    """

    vector_stores: Iterable[ToolResourcesFileSearchVectorStore]
    """
    A helper to create a
    [vector store](https://platform.openai.com/docs/api-reference/vector-stores/object)
    with file_ids and attach it to this thread. There can be a maximum of 1 vector
    store attached to the thread.
    """


class OpenAICreateThreadParamsToolResources(TypedDict, total=False):
    code_interpreter: ToolResourcesCodeInterpreter

    file_search: ToolResourcesFileSearch


class FileSearchToolParam(TypedDict, total=False):
    type: Required[Literal["file_search"]]
    """The type of tool being defined: `file_search`"""


class CodeInterpreterToolParam(TypedDict, total=False):
    type: Required[Literal["code_interpreter"]]
    """The type of tool being defined: `code_interpreter`"""


AttachmentTool = Union[CodeInterpreterToolParam, FileSearchToolParam]


class Attachment(TypedDict, total=False):
    file_id: str
    """The ID of the file to attach to the message."""

    tools: Iterable[AttachmentTool]
    """The tools to add this file to."""


class ImageFileObject(TypedDict):
    file_id: Required[str]
    detail: Optional[str]


class ImageURLObject(TypedDict):
    url: Required[str]
    detail: Optional[str]


class MessageContentTextObject(TypedDict):
    type: Required[Literal["text"]]
    text: str


class MessageContentImageFileObject(TypedDict):
    type: Literal["image_file"]
    image_file: ImageFileObject


class MessageContentImageURLObject(TypedDict):
    type: Required[str]
    image_url: ImageURLObject


class MessageData(TypedDict):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                MessageContentTextObject,
                MessageContentImageFileObject,
                MessageContentImageURLObject,
            ]
        ],
    ]
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


# OpenAI Files Types
class CreateFileRequest(TypedDict, total=False):
    """
    CreateFileRequest
    Used by Assistants API, Batches API, and Fine-Tunes API

    Required Params:
        file: FileTypes
        purpose: Literal['assistants', 'batch', 'fine-tune']

    Optional Params:
        extra_headers: Optional[Dict[str, str]]
        extra_body: Optional[Dict[str, str]] = None
        timeout: Optional[float] = None
    """

    file: FileTypes
    purpose: Literal["assistants", "batch", "fine-tune"]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class FileContentRequest(TypedDict, total=False):
    """
    FileContentRequest
    Used by Assistants API, Batches API, and Fine-Tunes API

    Required Params:
        file_id: str

    Optional Params:
        extra_headers: Optional[Dict[str, str]]
        extra_body: Optional[Dict[str, str]] = None
        timeout: Optional[float] = None
    """

    file_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


# OpenAI Batches Types
class CreateBatchRequest(TypedDict, total=False):
    """
    CreateBatchRequest
    """

    completion_window: Literal["24h"]
    endpoint: Literal["/v1/chat/completions", "/v1/embeddings"]
    input_file_id: str
    metadata: Optional[Dict[str, str]]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class RetrieveBatchRequest(TypedDict, total=False):
    """
    RetrieveBatchRequest
    """

    batch_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class CancelBatchRequest(TypedDict, total=False):
    """
    CancelBatchRequest
    """

    batch_id: str
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class ListBatchRequest(TypedDict, total=False):
    """
    ListBatchRequest - List your organization's batches
    Calls https://api.openai.com/v1/batches
    """

    after: Union[str, NotGiven]
    limit: Union[int, NotGiven]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]
    timeout: Optional[float]


class ChatCompletionToolCallFunctionChunk(TypedDict):
    name: Optional[str]
    arguments: str


class ChatCompletionToolCallChunk(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk


class ChatCompletionDeltaToolCallChunk(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionToolCallFunctionChunk
    index: int


class ChatCompletionDeltaChunk(TypedDict, total=False):
    content: Optional[str]
    tool_calls: List[ChatCompletionDeltaToolCallChunk]
    role: str


class ChatCompletionResponseMessage(TypedDict, total=False):
    content: Optional[str]
    tool_calls: List[ChatCompletionToolCallChunk]
    role: Literal["assistant"]


class ChatCompletionUsageBlock(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
