import builtins
from collections.abc import Callable, Mapping
from typing import Final, Literal

from pydantic import PrivateAttr, TypeAdapter
from pydantic.main import IncEx
from typing_extensions import ReadOnly, TypedDict, Unpack

from litellm.types.llms.base import LiteLLMBaseModel

_ATTRIBUTE_VALUE: Final = TypeAdapter(builtins.object)


class _ModelDumpOptions(TypedDict, total=False):
    """Keyword arguments `json` forwards to pydantic's `BaseModel.model_dump`."""

    mode: ReadOnly[str]
    include: ReadOnly[IncEx | None]
    exclude: ReadOnly[IncEx | None]
    context: ReadOnly[builtins.object]
    by_alias: ReadOnly[bool | None]
    exclude_unset: ReadOnly[bool]
    exclude_defaults: ReadOnly[bool]
    exclude_none: ReadOnly[bool]
    exclude_computed_fields: ReadOnly[bool]
    round_trip: ReadOnly[bool]
    warnings: ReadOnly[bool | Literal["none", "warn", "error"]]
    fallback: ReadOnly[Callable[[builtins.object], builtins.object] | None]
    serialize_as_any: ReadOnly[bool]
    polymorphic_serialization: ReadOnly[bool | None]


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
    _hidden_params: dict[str, builtins.object] = PrivateAttr(default={})

    @property
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        # Allow dictionary-style access to attributes
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
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
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
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
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
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
    def hidden_params(self) -> dict[str, builtins.object]:  # mutable-ok: API requires mutation
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, builtins.object]) -> None:  # mutable-ok: API requires mutation
        self._hidden_params = hidden_params

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: builtins.object = None) -> builtins.object:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> builtins.object:
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
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
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
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
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def json(self, **kwargs: Unpack[_ModelDumpOptions]):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            return self.dict()
