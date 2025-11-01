from typing import Union, Literal

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    type_: Literal["text"] = Field(default="text", alias="type")
    text: str


class ImageURLContent(BaseModel):
    url: str
    detail: str = "auto"


class ImageContent(BaseModel):
    type_: Literal["image_url"] = Field(default="image_url", alias="type")
    image_url: ImageURLContent


class FunctionObj(BaseModel):
    name: str
    arguments: str


class FunctionTool(BaseModel):
    description: str = ""
    name: str
    parameters: dict = {}
    strict: bool = False


class ChatCompletionTool(BaseModel):
    type_: Literal["function"] = Field(default="function", alias="type")
    function: FunctionTool


class MessageToolCall(BaseModel):
    id: str
    type_: Literal["function"] = Field(default="function", alias="type")
    function: FunctionObj


class SAPMessage(BaseModel):
    """
    Model for SystemChatMessage and DeveloperChatMessage
    """

    role: Literal["system", "developer"] = "system"
    content: str


class SAPUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: Union[
        str, TextContent, ImageContent, list[Union[TextContent, ImageContent]]
    ]


class SAPAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""
    refusal: str = ""
    tool_calls: list[MessageToolCall] = []


class SAPToolChatMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


class ResponseFormat(BaseModel):
    type_: Literal["text", "json_object"] = Field(default="text", alias="type")


class JSONResponseSchema(BaseModel):
    description: str = ""
    name: str
    schema_: dict = Field(default_factory=dict, alias="schema")
    strict: bool = False


class ResponseFormatJSONSchema(BaseModel):
    type_: Literal["json_schema"] = Field(default="json_schema", alias="type")
    json_schema: JSONResponseSchema
