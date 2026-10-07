import builtins
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import PrivateAttr
from typing_extensions import ReadOnly, TypedDict

from litellm.types.llms.base import LiteLLMBaseModel


class ExpiresAfter(LiteLLMBaseModel):
    """Container expiration settings."""

    anchor: Literal["last_active_at"]
    minutes: int


class ContainerObject(LiteLLMBaseModel):
    """Represents a container object."""

    id: str
    object: Literal["container"]
    created_at: int
    status: str
    expires_after: ExpiresAfter | None = None
    last_active_at: int | None = None
    name: str | None = None
    _hidden_params: dict[str, Any] = PrivateAttr(default={})

    @property
    def hidden_params(self) -> dict[str, builtins.object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict()


class DeleteContainerResult(LiteLLMBaseModel):
    """Result of a delete container request."""

    id: str
    object: Literal["container.deleted"]
    deleted: bool

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerListResponse(LiteLLMBaseModel):
    """Response object for list containers request."""

    object: Literal["list"]
    data: list[ContainerObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
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

    expires_after: ReadOnly[Mapping[str, object] | None]  # ExpiresAfter object
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


class ContainerFileObject(LiteLLMBaseModel):
    """Represents a container file object."""

    id: str
    object: Literal["container.file", "container_file"]  # OpenAI returns "container.file"
    container_id: str
    bytes: int | None = None  # Can be null for some files
    created_at: int
    path: str
    source: str
    _hidden_params: dict[str, builtins.object] = PrivateAttr(default={})

    @property
    def hidden_params(self) -> dict[str, builtins.object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class ContainerFileListResponse(LiteLLMBaseModel):
    """Response object for list container files request."""

    object: Literal["list"]
    data: list[ContainerFileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()


class DeleteContainerFileResponse(LiteLLMBaseModel):
    """Response object for delete container file request."""

    id: str
    # OpenAI / Azure wire format uses dots; keep underscore variant for compatibility.
    object: Literal["container.file.deleted", "container_file.deleted"]
    deleted: bool

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()
