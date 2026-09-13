from enum import Enum
from typing import Any, Literal

from typing_extensions import TypedDict


class VectorStoreFileStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VectorStoreFileChunkingStrategyType(str, Enum):
    AUTO = "auto"
    STATIC = "static"


class VectorStoreFileStaticChunkingConfig(TypedDict, total=False):
    max_chunk_size_tokens: int
    chunk_overlap_tokens: int


class VectorStoreFileChunkingStrategy(TypedDict, total=False):
    type: Literal["auto", "static"]
    static: VectorStoreFileStaticChunkingConfig | None


class VectorStoreFileObject(TypedDict, total=False):
    id: str
    object: Literal["vector_store.file"]
    created_at: int
    usage_bytes: int | None
    vector_store_id: str
    status: VectorStoreFileStatus
    last_error: dict[str, Any] | None
    chunking_strategy: VectorStoreFileChunkingStrategy | None
    attributes: dict[str, str | int | float | bool] | None


class VectorStoreFileCreateRequest(TypedDict, total=False):
    file_id: str
    attributes: dict[str, str | int | float | bool] | None
    chunking_strategy: VectorStoreFileChunkingStrategy | None


class VectorStoreFileUpdateRequest(TypedDict, total=False):
    attributes: dict[str, str | int | float | bool]


class VectorStoreFileListQueryParams(TypedDict, total=False):
    after: str | None
    before: str | None
    filter: Literal["in_progress", "completed", "failed", "cancelled"] | None
    limit: int | None
    order: Literal["asc", "desc"] | None


class VectorStoreFileListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: list[VectorStoreFileObject]
    first_id: str | None
    last_id: str | None
    has_more: bool


class VectorStoreFileDeleteResponse(TypedDict, total=False):
    id: str
    object: Literal["vector_store.file.deleted"]
    deleted: bool


class VectorStoreFileContentTextPart(TypedDict, total=False):
    type: Literal["text"]
    text: str


class VectorStoreFileContentResponse(TypedDict, total=False):
    file_id: str
    filename: str | None
    attributes: dict[str, str | int | float | bool] | None
    content: list[VectorStoreFileContentTextPart]


class VectorStoreFileAuthCredentials(TypedDict, total=False):
    headers: dict[str, Any]
    query_params: dict[str, Any]
