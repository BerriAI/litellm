from typing import Any, Optional, Union

from openai._models import BaseModel as OpenAIObject
from pydantic import BaseModel, ConfigDict


class LiteLLMPydanticObjectBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict(**kwargs)

    def fields_set(self):
        try:
            return self.model_fields_set  # noqa
        except Exception:
            # if using pydantic v1
            return self.__fields_set__

    model_config = ConfigDict(protected_namespaces=())


class BaseLiteLLMOpenAIResponseObject(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()


class HiddenParams(OpenAIObject):
    original_response: Optional[Union[str, Any]] = None
    model_id: Optional[str] = None  # used in Router for individual deployments
    api_base: Optional[str] = None  # returns api base used for making completion call
    _response_ms: Optional[float] = None
    response_cost: Optional[float] = None

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def __setitem__(self, key, value):
        # Allow dictionary-style assignment of attributes
        setattr(self, key, value)

    def json(self, **kwargs):  # type: ignore
        try:
            return self.model_dump()  # noqa
        except Exception:
            # if using pydantic v1
            return self.dict()

    def model_dump(self, **kwargs):
        # Override model_dump to include private attributes
        data = super().model_dump(**kwargs)
        data["_response_ms"] = self._response_ms
        return data