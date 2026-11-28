from enum import Enum
from typing import Annotated, Final, TypeAlias

from openai._models import BaseModel as OpenAIObject
from pydantic import ConfigDict, TypeAdapter, ValidatorFunctionWrapHandler, WrapValidator

from litellm.types.llms.base_model import LiteLLMBaseModel

_ATTRIBUTE_VALUE: Final = TypeAdapter(object)


def _str_as_given(value: object, handler: ValidatorFunctionWrapHandler) -> object:
    return value if isinstance(value, str) and not isinstance(value, Enum) else handler(value)


StrAsGiven: TypeAlias = Annotated[str, WrapValidator(_str_as_given)]


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

    def __getitem__(self, key: str) -> object:
        return _ATTRIBUTE_VALUE.validate_python(self.__dict__[key])

    def get(self, key: str, default: object = None):
        return self.__dict__.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()


class HiddenParams(OpenAIObject):
    original_response: str | object | None = None
    model_id: str | None = None  # used in Router for individual deployments
    api_base: str | None = None  # returns api base used for making completion call
    _response_ms: float | None = None
    response_cost: float | None = None

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def get(self, key: str, default: object = None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> object:
        # Allow dictionary-style access to attributes
        return _ATTRIBUTE_VALUE.validate_python(getattr(self, key))

    def __setitem__(self, key: str, value: object) -> None:
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs: object):
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
