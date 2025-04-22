from typing import List, Literal, Optional, TypedDict, Union


class FunctionCall(TypedDict):
    name: Optional[str]
    arguments: Optional[Union[str, dict]]


class MistralToolCallMessage(TypedDict):
    id: Optional[str]
    type: Literal["function"]
    function: Optional[FunctionCall]
