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
