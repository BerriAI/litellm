from typing import Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


def validate_different_content(v: Union[str, dict, list]) -> str:
    if v in ((), {}, []):
        return ""
    elif isinstance(v, dict) and "text" in v:
        return v['text']
    elif isinstance(v, list):
        new_v = []
        for item in v:
            if isinstance(item, dict) and "text" in item:
                if item['text']:
                    new_v.append(item['text'])
            elif isinstance(item, str):
                new_v.append(item)
        return '\n'.join(new_v)
    elif isinstance(v, str):
        return v
    raise ValueError("Content must be a string")
    return v

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

    _content_validator = field_validator("content", mode="before")(validate_different_content)


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

    _content_validator = field_validator("content", mode="before")(validate_different_content)



class SAPToolChatMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

    _content_validator = field_validator("content", mode="before")(validate_different_content)

ChatMessage = Union[SAPUserMessage, SAPAssistantMessage, SAPToolChatMessage, SAPMessage]


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


class KeyValueListPair(BaseModel):
    key: str
    value: list[str]


class DocumentMetadataKeyValueListPairs(KeyValueListPair):
    select_mode: list[Literal['ignoreIfKeyAbsent']] = None


class GroundingSearchConfig(BaseModel):
    max_chunk_count: int = Field(default=None, ge=0)
    max_document_count: int = Field(default=None, ge=0)

    @model_validator(mode='after')
    def validate_max_chunk_count_and_max_document_count(self):
        if self.max_chunk_count and self.max_document_count:
            raise ValidationError("Cannot specify both maxChunkCount and maxDocumentCount.")
        return self


class DocumentGroundingFilter(BaseModel):
    id_: str = Field(default=None, alias="id")
    data_repository_type: Literal["vector", "help.sap.com"]
    search_config: GroundingSearchConfig = None
    data_repositories: list[str] = None
    data_repository_metadata: list[KeyValueListPair] = None
    document_metadata: list[DocumentMetadataKeyValueListPairs] = None
    chunk_metadata: list[KeyValueListPair] = None


class DocumentGroundingPlaceholders(BaseModel):
    input: list[str] = Field(min_length=1)
    output: str


class DocumentGroundingConfig(BaseModel):
    filters: list[DocumentGroundingFilter] = None
    placeholders: DocumentGroundingPlaceholders
    metadata_params: list[str] = None


class GroundingModuleConfig(BaseModel):
    type_: Literal["document_grounding_service"] = Field(default="document_grounding_service", alias="type")
    config: DocumentGroundingConfig


class Template(BaseModel):
    template: list[ChatMessage]
    defaults: dict = None
    response_format: ResponseFormat | ResponseFormatJSONSchema = None
    tools: list[FunctionTool] = None


class LLMModelDetails(BaseModel):
    name: str
    version: str = "latest"
    params: dict = None


class PromptTemplatingModuleConfig(BaseModel):
    prompt: Template
    model: LLMModelDetails


class ModuleConfig(BaseModel):
    prompt_templating: PromptTemplatingModuleConfig
    # filtering: Optional[FilteringModuleConfig] = None
    # masking: Optional[MaskingModuleConfig] = None
    grounding: GroundingModuleConfig = None
    # translation: Optional[TranslationModuleConfig] = None


class GlobalStreamOptions(BaseModel):
    enabled: bool = False
    chunk_size: int = 100
    delimiters: list[str] = None

    @model_validator(mode='after')
    def validate_streaming_params(self):
        """Validate that chunk_size and delimiters are not set when enabled is False."""
        if not self.enabled:
            if self.chunk_size != 100:  # Check if chunk_size was explicitly set
                raise ValueError("chunk_size cannot be set when enabled is False")
            if self.delimiters is not None:
                raise ValueError("delimiters cannot be set when enabled is False")
        return self

    def model_dump(self, **kwargs):
        """Override model_dump to exclude chunk_size and delimiters when enabled is False."""
        data = super().model_dump(**kwargs)
        if not self.enabled:
            # Remove chunk_size and delimiters from output when streaming is disabled
            data.pop('chunk_size', None)
            data.pop('delimiters', None)
        return data


class OrchestrationConfig(BaseModel):
    modules: ModuleConfig
    stream: GlobalStreamOptions = None


class OrchestrationRequest(BaseModel):
    config: OrchestrationConfig
    placeholder_values: dict = None
