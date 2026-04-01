from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel
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
    static: Optional[VectorStoreFileStaticChunkingConfig]


class VectorStoreFileObject(TypedDict, total=False):
    id: str
    object: Literal["vector_store.file"]
    created_at: int
    usage_bytes: Optional[int]
    vector_store_id: str
    status: VectorStoreFileStatus
    last_error: Optional[Dict[str, Any]]
    chunking_strategy: Optional[VectorStoreFileChunkingStrategy]
    attributes: Optional[Dict[str, Union[str, int, float, bool]]]


class VectorStoreFileCreateRequest(TypedDict, total=False):
    file_id: str
    attributes: Optional[Dict[str, Union[str, int, float, bool]]]
    chunking_strategy: Optional[VectorStoreFileChunkingStrategy]


class VectorStoreFileUpdateRequest(TypedDict, total=False):
    attributes: Dict[str, Union[str, int, float, bool]]


class VectorStoreFileListQueryParams(TypedDict, total=False):
    after: Optional[str]
    before: Optional[str]
    filter: Optional[Literal["in_progress", "completed", "failed", "cancelled"]]
    limit: Optional[int]
    order: Optional[Literal["asc", "desc"]]


class VectorStoreFileListResponse(TypedDict, total=False):
    object: Literal["list"]
    data: List[VectorStoreFileObject]
    first_id: Optional[str]
    last_id: Optional[str]
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
    filename: Optional[str]
    attributes: Optional[Dict[str, Union[str, int, float, bool]]]
    content: List[VectorStoreFileContentTextPart]


class VectorStoreFileAuthCredentials(TypedDict, total=False):
    headers: Dict[str, Any]
    query_params: Dict[str, Any]
