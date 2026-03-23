from typing import Union, Literal, Optional
from enum import Enum
import warnings

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_different_content(v: Union[str, dict, list]) -> str:
    if v in ((), {}, []):
        return ""
    elif isinstance(v, dict) and "text" in v:
        return v["text"]
    elif isinstance(v, list):
        new_v = []
        for item in v:
            if isinstance(item, dict) and "text" in item:
                if item["text"]:
                    new_v.append(item["text"])
            elif isinstance(item, str):
                new_v.append(item)
        return "\n".join(new_v)
    elif isinstance(v, str):
        return v
    raise ValueError("Content must be a string")


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
    parameters: dict = {"type": "object", "properties": {}}
    strict: bool = False

    def model_dump(self, **kwargs) -> dict:
        kwargs["exclude_unset"] = False
        return super().model_dump(**kwargs)

    @field_validator("parameters", mode="before")
    @classmethod
    def ensure_object_type(cls, v: dict) -> dict:
        """Ensure parameters has type='object' as required by SAP Orchestration Service."""
        if not v:
            return {"type": "object", "properties": {}}
        if "type" not in v:
            v = {"type": "object", **v}
        if "properties" not in v:
            v["properties"] = {}
        return v


class ChatCompletionTool(BaseModel):
    type_: Literal["function"] = Field(default="function", alias="type")
    function: FunctionTool

    def model_dump(self, **kwargs) -> dict:
        kwargs["exclude_unset"] = False
        return super().model_dump(**kwargs)


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

    _content_validator = field_validator("content", mode="before")(
        validate_different_content
    )


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

    _content_validator = field_validator("content", mode="before")(
        validate_different_content
    )


class SAPToolChatMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

    _content_validator = field_validator("content", mode="before")(
        validate_different_content
    )


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
    select_mode: Optional[list[Literal["ignoreIfKeyAbsent"]]] = None


class GroundingSearchConfig(BaseModel):
    max_chunk_count: Optional[int] = Field(default=None, ge=0)
    max_document_count: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_max_chunk_count_and_max_document_count(self):
        if self.max_chunk_count is not None and self.max_document_count is not None:
            raise ValueError("Cannot specify both maxChunkCount and maxDocumentCount.")
        return self


class DocumentGroundingFilter(BaseModel):
    id_: Optional[str] = Field(default=None, alias="id")
    data_repository_type: Literal["vector", "help.sap.com"]
    search_config: Optional[GroundingSearchConfig] = None
    data_repositories: Optional[list[str]] = None
    data_repository_metadata: Optional[list[KeyValueListPair]] = None
    document_metadata: Optional[list[DocumentMetadataKeyValueListPairs]] = None
    chunk_metadata: Optional[list[KeyValueListPair]] = None


class DocumentGroundingPlaceholders(BaseModel):
    input: list[str] = Field(min_length=1)
    output: str


class DocumentGroundingConfig(BaseModel):
    filters: Optional[list[DocumentGroundingFilter]] = None
    placeholders: DocumentGroundingPlaceholders
    metadata_params: Optional[list[str]] = None


class GroundingModuleConfig(BaseModel):
    type_: Literal["document_grounding_service"] = Field(
        default="document_grounding_service", alias="type"
    )
    config: DocumentGroundingConfig


class Template(BaseModel):
    template: list[ChatMessage]
    defaults: Optional[dict[str, str]] = None
    response_format: Optional[Union[ResponseFormat, ResponseFormatJSONSchema]] = None
    tools: Optional[list[ChatCompletionTool]] = None


class LLMModelDetails(BaseModel):
    name: str
    version: str = "latest"
    params: Optional[dict] = None


class PromptTemplatingModuleConfig(BaseModel):
    prompt: Template
    model: LLMModelDetails


class SAPMaskingProfileEntity(str, Enum):
    """
    Enumerates the entity categories that can be masked by the SAP Data Privacy Integration service.

    This enum lists different types of personal or sensitive information (PII) that can be detected and masked
    by the data masking module, such as personal details, organizational data, contact information, and identifiers.

    Values:
        PERSON: Represents personal names.

        ORG: Represents organizational names.

        UNIVERSITY: Represents educational institutions.

        LOCATION: Represents geographical locations.

        EMAIL: Represents email addresses.

        PHONE: Represents phone numbers.

        ADDRESS: Represents physical addresses.

        SAP_IDS_INTERNAL: Represents internal SAP identifiers.

        SAP_IDS_PUBLIC: Represents public SAP identifiers.

        URL: Represents URLs.

        USERNAME_PASSWORD: Represents usernames and passwords.

        NATIONAL_ID: Represents national identification numbers.

        IBAN: Represents International Bank Account Numbers.

        SSN: Represents Social Security Numbers.

        CREDIT_CARD_NUMBER: Represents credit card numbers.

        PASSPORT: Represents passport numbers.

        DRIVING_LICENSE: Represents driving license numbers.

        NATIONALITY: Represents nationality information.

        RELIGIOUS_GROUP: Represents religious group affiliation.

        POLITICAL_GROUP: Represents political group affiliation.

        PRONOUNS_GENDER: Represents pronouns and gender identity.

        GENDER: Represents gender information.

        SEXUAL_ORIENTATION: Represents sexual orientation.

        TRADE_UNION: Represents trade union membership.

        SENSITIVE_DATA: Represents any other sensitive information.
    """

    PERSON = "profile-person"
    ORG = "profile-org"
    UNIVERSITY = "profile-university"
    LOCATION = "profile-location"
    EMAIL = "profile-email"
    PHONE = "profile-phone"
    ADDRESS = "profile-address"
    SAP_IDS_INTERNAL = "profile-sapids-internal"
    SAP_IDS_PUBLIC = "profile-sapids-public"
    URL = "profile-url"
    USERNAME_PASSWORD = "profile-username-password"
    NATIONAL_ID = "profile-nationalid"
    IBAN = "profile-iban"
    SSN = "profile-ssn"
    CREDIT_CARD_NUMBER = "profile-credit-card-number"
    PASSPORT = "profile-passport"
    DRIVING_LICENSE = "profile-driverlicense"
    NATIONALITY = "profile-nationality"
    RELIGIOUS_GROUP = "profile-religious-group"
    POLITICAL_GROUP = "profile-political-group"
    PRONOUNS_GENDER = "profile-pronouns-gender"
    GENDER = "profile-gender"
    SEXUAL_ORIENTATION = "profile-sexual-orientation"
    TRADE_UNION = "profile-trade-union"
    SENSITIVE_DATA = "profile-sensitive-data"
    ETHNICITY = "profile-ethnicity"


class DPIMethodConstant(BaseModel):
    """
    Replaces the entity with the specified value followed by an incrementing number
    """

    method: Literal["constant"] = "constant"
    value: str


class DPIMethodFabricatedData(BaseModel):
    """
    Replaces the entity with a randomly generated value appropriate to its type.
    """

    method: Literal["fabricated_data"] = "fabricated_data"


class DPICustomEntity(BaseModel):
    """
    regex: Regular expression to match the entity
    replacement_strategy: Replacement strategy to be used for the entity
    """

    regex: str
    replacement_strategy: DPIMethodConstant


class DPIStandardEntity(BaseModel):
    """
    type: Standard entity type to be masked
    replacement_strategy: Replacement strategy to be used for the entity
    """

    type_: SAPMaskingProfileEntity = Field(..., alias="type")
    replacement_strategy: Optional[
        Union[DPIMethodConstant, DPIMethodFabricatedData]
    ] = None


class MaskGroundingInput(BaseModel):
    """
    Controls whether the input to the grounding module will be masked with the configuration
    supplied in the masking module
    """

    enabled: bool = False


class MaskingProviderConfig(BaseModel):
    """
    SAP Data Privacy Integration provider for data masking.

    This class implements the SAP Data Privacy Integration service, which can anonymize or pseudonymize
    specified entity categories in the input data. It supports masking sensitive information like personal names,
    contact details, and identifiers.

    Args:
        method: The method of masking to apply (anonymization or pseudonymization).

        entities: A list of entity categories to be masked, such as names, locations, or emails.

        allowlist: A list of strings that should not be masked.

        mask_grounding_input: A flag indicating whether to mask input to the grounding module.
    """

    type_: Literal["sap_data_privacy_integration"] = Field(
        default="sap_data_privacy_integration", alias="type"
    )
    method: Literal["anonymization", "pseudonymization"]
    entities: list[Union[DPIStandardEntity, DPICustomEntity]]
    allowlist: Optional[list[str]] = None
    mask_grounding_input: Optional[MaskGroundingInput] = None


class MaskingModuleConfig(BaseModel):
    """
    Configuration for the data masking module.

    Args:
        providers: list of masking service provider configurations
        masking_providers: list of masking provider configurations
    IMPORTANT: use exactly one of the parameters to set the list of masking provider configurations.
    DEPRECATED: parameter 'masking_providers' will be removed Sept 15, 2026. Use 'providers' instead.
    """

    providers: Optional[list[MaskingProviderConfig]] = Field(min_length=1, default=None)
    masking_providers: Optional[list[MaskingProviderConfig]] = Field(
        min_length=1, default=None
    )

    @model_validator(mode="after")
    def enforce_exactly_one_provider_list(self):
        has_providers = self.providers is not None
        has_masking_providers = self.masking_providers is not None

        if has_providers == has_masking_providers:
            raise ValueError(
                "For SAP Masking Module Config must set exactly one of: 'providers' or 'masking_providers' "
                "DEPRECATED: parameter 'masking_providers' will be removed Sept 15, 2026. Use 'providers' instead."
            )

        if has_masking_providers:
            warnings.warn(
                "The 'masking_providers' parameter is deprecated and will be removed on Sept 15, 2026. "
                "Use 'providers' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        return self


class AzureThreshold(int, Enum):
    """
    Enumerates the threshold levels for the Azure Content Safety service.

    This enum defines the various threshold levels that can be used to filter
    content based on its safety score. Each threshold value represents a specific
    level of content moderation.

    Values:
        ALLOW_SAFE: Allows only Safe content.

        ALLOW_SAFE_LOW: Allows Safe and Low content.

        ALLOW_SAFE_LOW_MEDIUM: Allows Safe, Low, and Medium content.

        ALLOW_ALL: Allows all content (Safe, Low, Medium, and High).
    """

    ALLOW_SAFE = 0
    ALLOW_SAFE_LOW = 2
    ALLOW_SAFE_LOW_MEDIUM = 4
    ALLOW_ALL = 6


class AzureContentFilter(BaseModel):
    """
    Specific filter configuration for Azure Content Safety.

    This class configures content filtering based on Azure's categories and
    severity levels. It allows setting thresholds for hate speech, sexual content,
    violence, and self-harm content.

    Values:
        hate: Threshold for hate speech content.

        sexual: Threshold for sexual content.

        violence: Threshold for violent content.

        self_harm: Threshold for self-harm content.
    """

    hate: Optional[Union[AzureThreshold, Literal[0, 2, 4, 6]]] = None
    sexual: Optional[Union[AzureThreshold, Literal[0, 2, 4, 6]]] = None
    violence: Optional[Union[AzureThreshold, Literal[0, 2, 4, 6]]] = None
    self_harm: Optional[Union[AzureThreshold, Literal[0, 2, 4, 6]]] = None


class AzureContentSafetyInput(AzureContentFilter):
    """
    Filter configuration for Azure Content Safety Input

        Values:
            hate: Threshold for hate speech content.

            sexual: Threshold for sexual content.

            violence: Threshold for violent content.

            self_harm: Threshold for self-harm content.

            prompt_shield: A flag to use prompt shield
    """

    prompt_shield: Optional[bool] = False


class AzureContentSafetyOutput(AzureContentFilter):
    """
    Filter configuration for Azure Content Safety Output

    Values:
        hate: Threshold for hate speech content.

        sexual: Threshold for sexual content.

        violence: Threshold for violent content.

        self_harm: Threshold for self-harm content.

        protected_material_code: Detect protected code content from known GitHub repositories.
                    The scan includes software libraries, source code, algorithms,
                    and other proprietary programming content.
    """

    protected_material_code: Optional[bool] = False


class LlamaGuard38bFilter(BaseModel):
    """
    Specific implementation of ContentFilter for Llama Guard 3. Llama Guard 3 is a
    Llama-3.1-8B pretrained model, fine-tuned for content safety classification.

    Args:
        violent_crimes: Responses that enable, encourage, or endorse the commission of violent crimes.

        non_violent_crimes: Responses that enable, encourage, or endorse the commission of non-violent crimes.

        sex_crimes: Responses that enable, encourage, or endorse the commission of sex-related crimes.

        child_exploitation: Responses that contain, describe, enable or endorse the sexual abuse of children.

        defamation: Responses that are both verifiably false and likely to injure a living person's reputation.

        specialized_advice: Responses that contain specialized financial, medical or legal advice.

        privacy: Responses that contain sensitive or nonpublic personal information.

        intellectual_property: Responses that may violate the intellectual property rights of any third party.

        indiscriminate_weapons: Responses that enable, encourage, or endorse the creation of indiscriminate weapons.

        hate: Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics.

        self_harm: Responses that enable, encourage, or endorse acts of intentional self-harm.

        sexual_content: Responses that contain erotica.

        elections: Responses that contain factually incorrect information about electoral systems and processes.

        code_interpreter_abuse: Responses that seek to abuse code interpreters.
    """

    violent_crimes: bool = Field(default=False)
    non_violent_crimes: bool = Field(default=False)
    sex_crimes: bool = Field(default=False)
    child_exploitation: bool = Field(default=False)
    defamation: bool = Field(default=False)
    specialized_advice: bool = Field(default=False)
    privacy: bool = Field(default=False)
    intellectual_property: bool = Field(default=False)
    indiscriminate_weapons: bool = Field(default=False)
    hate: bool = Field(default=False)
    self_harm: bool = Field(default=False)
    sexual_content: bool = Field(default=False)
    elections: bool = Field(default=False)
    code_interpreter_abuse: bool = Field(default=False)


class LlamaGuard38bFilterConfig(BaseModel):
    type_: Literal["llama_guard_3_8b"] = Field(default="llama_guard_3_8b", alias="type")
    config: LlamaGuard38bFilter


class AzureContentSafetyInputFilterConfig(BaseModel):
    type_: Literal["azure_content_safety"] = Field(
        default="azure_content_safety", alias="type"
    )
    config: Optional[AzureContentSafetyInput] = None


class AzureContentSafetyOutputFilterConfig(BaseModel):
    type_: Literal["azure_content_safety"] = Field(
        default="azure_content_safety", alias="type"
    )
    config: Optional[AzureContentSafetyOutput] = None


class FilteringStreamOptions(BaseModel):
    """
    overlap: Number of characters that should be additionally sent to content filtering services
    from previous chunks as additional context.
    """

    overlap: Optional[int] = Field(default=0, ge=0, le=10000)


class InputFiltering(BaseModel):
    """Module for managing and applying input content filters.

    Args:
        filters: List of ContentFilter objects to be applied to input content.
    """

    filters: list[
        Union[AzureContentSafetyInputFilterConfig, LlamaGuard38bFilterConfig]
    ] = Field(min_length=1)


class OutputFiltering(BaseModel):
    """Module for managing and applying output content filters.

    Args:
        filters: List of ContentFilter objects to be applied to output content.

        stream_options: Module-specific streaming options.
    """

    filters: list[
        Union[AzureContentSafetyOutputFilterConfig, LlamaGuard38bFilterConfig]
    ] = Field(min_length=1)
    stream_options: Optional[FilteringStreamOptions] = None


class FilteringModuleConfig(BaseModel):
    """Module for managing and applying content filters.

    Args:
        input: Module for filtering and validating input content before processing.

        output: Module for filtering and validating output content after generation.
    """

    input: Optional[InputFiltering] = None
    output: Optional[OutputFiltering] = None

    @model_validator(mode="after")
    def enforce_min_properties(self) -> "FilteringModuleConfig":
        """
        Ensure at least one of input or output filtering is provided.
        """
        if self.input is None and self.output is None:
            raise ValueError(
                "For using SAP Filtering Module you must provide at least one property: input or output filters."
            )
        return self


class SAPDocumentTranslationApplyToSelector(BaseModel):
    """
    This selector allows you to define the scope of translation, such as specific placeholders or
    messages with specific roles.
    For example, {"category": "placeholders",
                "items": ["user_input"],
                "source_language": "de-DE"}
                targets the value of "user_input" in placeholder_values specified in the request payload;
                and considers the value to be in German.
    """

    category: Literal["placeholders", "template_roles"]
    items: list[str]
    source_language: str


class InputTranslationConfig(BaseModel):
    """
    Configuration for input translation.

    Args:
            source_language: Language of the text to be translated. Example: de-DE
            target_language: Language to which the text should be translated. Example: en-US
            apply_to: List of selectors that define the scope of translation.
    """

    source_language: Optional[str] = None
    target_language: str
    apply_to: Optional[list[SAPDocumentTranslationApplyToSelector]] = None


class OutputTranslationConfig(BaseModel):
    source_language: Optional[str] = None
    target_language: Union[str, SAPDocumentTranslationApplyToSelector]


class SAPDocumentTranslationInput(BaseModel):
    """
    Configuration for input translation

    Args:
        type: The type of translation module (e.g., 'sap_document_translation').

        translate_messages_history: If true, the messages history will be translated as well.

        config: Configuration object for the translation module.
    """

    type_: Literal["sap_document_translation"] = Field(
        default="sap_document_translation", alias="type"
    )
    translate_messages_history: Optional[bool] = None
    config: InputTranslationConfig


class SAPDocumentTranslationOutput(BaseModel):
    """
    Configuration for output translation

    Args:
         type: The type of translation module (e.g., 'sap_document_translation').

        config: Configuration object for the translation module.
    """

    type_: Literal["sap_document_translation"] = Field(
        default="sap_document_translation", alias="type"
    )
    config: OutputTranslationConfig


class TranslationModuleConfig(BaseModel):
    """
    Configuration for translation module

    Args:
        input: Configuration for input translation

        output: Configuration for output translation
    """

    input: Optional[SAPDocumentTranslationInput] = None
    output: Optional[SAPDocumentTranslationOutput] = None

    @model_validator(mode="after")
    def enforce_min_properties(self) -> "TranslationModuleConfig":
        if self.input is None and self.output is None:
            raise ValueError(
                "TranslationModuleConfig requires at least one of 'input' or 'output'."
            )
        return self


class ModuleConfig(BaseModel):
    prompt_templating: PromptTemplatingModuleConfig
    filtering: Optional[FilteringModuleConfig] = None
    masking: Optional[MaskingModuleConfig] = None
    grounding: Optional[GroundingModuleConfig] = None
    translation: Optional[TranslationModuleConfig] = None


class GlobalStreamOptions(BaseModel):
    enabled: bool = False
    chunk_size: Optional[int] = Field(default=None, ge=1)
    delimiters: Optional[list[str]] = None


class OrchestrationConfig(BaseModel):
    modules: Union[ModuleConfig, list[ModuleConfig]]
    stream: Optional[GlobalStreamOptions] = None


class OrchestrationRequest(BaseModel):
    config: OrchestrationConfig
    placeholder_values: Optional[dict[str, str]] = None
