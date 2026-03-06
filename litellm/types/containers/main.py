from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from typing_extensions import TypedDict


class ExpiresAfter(BaseModel):
    """Container expiration settings."""
    anchor: Literal["last_active_at"]
    minutes: int


class ContainerObject(BaseModel):
    """Represents a container object."""
    id: str
    object: Literal["container"]
    created_at: int
    status: str
    expires_after: Optional[ExpiresAfter] = None
    last_active_at: Optional[int] = None
    name: Optional[str] = None
    _hidden_params: Dict[str, Any] = {}

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict()


class DeleteContainerResult(BaseModel):
    """Result of a delete container request."""
    id: str
    object: Literal["container.deleted"]
    deleted: bool

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerListResponse(BaseModel):
    """Response object for list containers request."""
    object: Literal["list"]
    data: List[ContainerObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerCreateOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's container creation API.
    
    Params here: https://platform.openai.com/docs/api-reference/containers/create
    """
    expires_after: Optional[Dict[str, Any]]  # ExpiresAfter object
    file_ids: Optional[List[str]]
    extra_headers: Optional[Dict[str, str]]
    extra_body: Optional[Dict[str, str]]


class ContainerCreateRequestParams(ContainerCreateOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's container creation API.
    
    Params here: https://platform.openai.com/docs/api-reference/containers/create
    """
    name: str


class ContainerListOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's container list API.
    
    Params here: https://platform.openai.com/docs/api-reference/containers/list
    """
    after: Optional[str]
    limit: Optional[int]
    order: Optional[str]
    extra_headers: Optional[Dict[str, str]]
    extra_query: Optional[Dict[str, str]]


class ContainerFileObject(BaseModel):
    """Represents a container file object."""
    id: str
    object: Literal["container.file", "container_file"]  # OpenAI returns "container.file"
    container_id: str
    bytes: Optional[int] = None  # Can be null for some files
    created_at: int
    path: str
    source: str
    _hidden_params: Dict[str, Any] = {}

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerFileListResponse(BaseModel):
    """Response object for list container files request."""
    object: Literal["list"]
    data: List[ContainerFileObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class DeleteContainerFileResponse(BaseModel):
    """Response object for delete container file request."""
    id: str
    object: Literal["container_file.deleted"]
    deleted: bool

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()

