from typing import Any, Literal

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
    expires_after: ExpiresAfter | None = None
    last_active_at: int | None = None
    name: str | None = None
    _hidden_params: dict[str, Any] = {}

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
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

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerListResponse(BaseModel):
    """Response object for list containers request."""

    object: Literal["list"]
    data: list[ContainerObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerCreateOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's container creation API.

    Params here: https://platform.openai.com/docs/api-reference/containers/create
    """

    expires_after: dict[str, Any] | None  # ExpiresAfter object
    file_ids: list[str] | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, str] | None


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

    after: str | None
    limit: int | None
    order: str | None
    extra_headers: dict[str, str] | None
    extra_query: dict[str, str] | None


class ContainerFileObject(BaseModel):
    """Represents a container file object."""

    id: str
    object: Literal["container.file", "container_file"]  # OpenAI returns "container.file"
    container_id: str
    bytes: int | None = None  # Can be null for some files
    created_at: int
    path: str
    source: str
    _hidden_params: dict[str, Any] = {}

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerFileListResponse(BaseModel):
    """Response object for list container files request."""

    object: Literal["list"]
    data: list[ContainerFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class DeleteContainerFileResponse(BaseModel):
    """Response object for delete container file request."""

    id: str
    # OpenAI / Azure wire format uses dots; keep underscore variant for compatibility.
    object: Literal["container.file.deleted", "container_file.deleted"]
    deleted: bool

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()
