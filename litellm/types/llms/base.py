import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from openai._models import BaseModel as OpenAIObject
from pydantic import BaseModel, ConfigDict

from litellm.constants import DEFER_PYDANTIC_BUILD

_SCHEMA_BUILD_LOCK: Final = threading.RLock()


class LiteLLMBaseModel(BaseModel):
    model_config = ConfigDict(defer_build=DEFER_PYDANTIC_BUILD)

    if TYPE_CHECKING:

        def __init__(self, /, **data: object) -> None: ...

    @classmethod
    def model_rebuild(
        cls,
        *,
        force: bool = False,
        raise_errors: bool = True,
        _parent_namespace_depth: int = 2,
        _types_namespace: Mapping[str, object] | None = None,
    ) -> bool | None:
        # Resolve names from the model's own module, never a caller frame: a deferred first-use build
        # reads f_locals 5 frames up, and on Python < 3.13 that rewrites the dict the caller's locals() returned
        with _SCHEMA_BUILD_LOCK:
            return super().model_rebuild(
                force=force,
                raise_errors=raise_errors,
                _parent_namespace_depth=0,
                _types_namespace=_types_namespace,
            )

    def model_post_init(self, context: object, /) -> None:
        # Instances built by a parent's validator or by model_construct skip this class's own
        # first-use build, leaving a MockValSer that fails when the instance is serialized as Any
        # TODO: drop once pydantic ships a fix for https://github.com/pydantic/pydantic/issues/13647
        model = type(self)
        if not model.__pydantic_complete__:
            model.model_rebuild(raise_errors=False)


class LiteLLMPydanticObjectBase(LiteLLMBaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except Exception:
            # if using pydantic v1
            return self.dict(**kwargs)

    def fields_set(self):
        try:
            return self.model_fields_set
        except Exception:
            # if using pydantic v1
            return self.__fields_set__

    model_config = ConfigDict(protected_namespaces=())


class BaseLiteLLMOpenAIResponseObject(LiteLLMBaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key) -> bool:
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()


class HiddenParams(OpenAIObject):
    original_response: str | Any | None = None
    model_id: str | None = None  # used in Router for individual deployments
    api_base: str | None = None  # returns api base used for making completion call
    _response_ms: float | None = None
    response_cost: float | None = None

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):
        try:
            return self.model_dump()
        except Exception:
            # if using pydantic v1
            return self.dict()

    def model_dump(self, **kwargs):
        # Override model_dump to include private attributes
        data: Final = super().model_dump(**kwargs)
        data["_response_ms"] = self._response_ms
        return data


class CachedTokensDetails(LiteLLMBaseModel):
    text_tokens: int | None = None
    audio_tokens: int | None = None
    image_tokens: int | None = None
