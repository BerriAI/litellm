from pydantic import BaseModel


class BaseLiteLLMOpenAIResponseObject(BaseModel):
    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __contains__(self, key):
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()
