from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel


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

