# File generated from our OpenAPI spec by Stainless.

from typing import Optional

from ..._models import BaseModel
from .function_parameters import FunctionParameters

__all__ = ["FunctionDefinition"]


class FunctionDefinition(BaseModel):
    name: str
    """The name of the function to be called.

    Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length
    of 64.
    """

    parameters: FunctionParameters
    """The parameters the functions accepts, described as a JSON Schema object.

    See the [guide](https://platform.openai.com/docs/guides/gpt/function-calling)
    for examples, and the
    [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
    documentation about the format.

    To describe a function that accepts no parameters, provide the value
    `{"type": "object", "properties": {}}`.
    """

    description: Optional[str] = None
    """
    A description of what the function does, used by the model to choose when and
    how to call the function.
    """
