from typing import Union, Literal

from pydantic import BaseModel, Field

class TextContent(BaseModel):
    type: Literal["text"] = ["text"]
    text: str

class ImageURLContent(BaseModel):
    url: str
    detail: str = "auto"

class UserMessageContent(BaseModel):
    type: Literal["text", "image_url"] = ["text"]
    text: str = ""
    image_url: ImageURLContent = {}

class FunctionObj(BaseModel):
    name: str
    arguments: str

class FunctionTool(BaseModel):
    description: str = ""
    name: str
    parameters: dict = {}
    strict: bool = False

class ChatCompletionTool(BaseModel):
    type: Literal["function"] = ["function"]
    function: FunctionTool

class MessageToolCall(BaseModel):
    id: str
    type: Literal["function"] = ["function"]
    function: FunctionObj

class SAPMessage(BaseModel):
    """
    Model for UserChatMessage SystemChatMessage and DeveloperChatMessage
    """
    role: Literal["user", "system", "developer"] = "user"
    content: str

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
    type: Literal["text", "json_object"] = ["text"]

class JSONResponseSchema(BaseModel):
    description: str = ""
    name: str
    schema_: dict = Field(default_factory=dict, alias="schema")
    strict: bool = False

class ResponseFormatJSONSchema(BaseModel):
    type: Literal["json_schema"] = ["json_schema"]
    json_schema: JSONResponseSchema
