import warnings
from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def validate_different_content(v: str | dict | list) -> str:
    if v in ((), {}, []):
        return ""
    elif isinstance(v, dict) and "text" in v:
        return v["text"]
    elif isinstance(v, list):
        new_v: Final = []
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


class FileContent(BaseModel):
    type_: Literal["file"] = Field(default="file", alias="type")
    file_data: str
    filename: str = ""


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

    _content_validator = field_validator("content", mode="before")(validate_different_content)


class SAPUserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | TextContent | ImageContent | FileContent | list[TextContent | ImageContent | FileContent]


class ReasoningBlock(BaseModel):
    content: str = ""
    signature: str = ""


class SAPAssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""
    refusal: str = ""
    tool_calls: list[MessageToolCall] = []
    reasoning_content: list[ReasoningBlock] | None = None

    _content_validator = field_validator("content", mode="before")(validate_different_content)


class SAPToolChatMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str

    _content_validator = field_validator("content", mode="before")(validate_different_content)


ChatMessage = SAPMessage | SAPUserMessage | SAPAssistantMessage | SAPToolChatMessage


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
    select_mode: list[Literal["ignoreIfKeyAbsent"]] | None = None


class GroundingSearchConfig(BaseModel):
    max_chunk_count: int | None = Field(default=None, ge=0)
    max_document_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_max_chunk_count_and_max_document_count(self):
        if self.max_chunk_count is not None and self.max_document_count is not None:
            raise ValueError("Cannot specify both maxChunkCount and maxDocumentCount.")
        return self


class DocumentGroundingFilter(BaseModel):
    id_: str | None = Field(default=None, alias="id")
    data_repository_type: Literal["vector", "help.sap.com"]
    search_config: GroundingSearchConfig | None = None
    data_repositories: list[str] | None = None
    data_repository_metadata: list[KeyValueListPair] | None = None
    document_metadata: list[DocumentMetadataKeyValueListPairs] | None = None
    chunk_metadata: list[KeyValueListPair] | None = None


class DocumentGroundingPlaceholders(BaseModel):
    input: list[str] = Field(min_length=1)
    output: str


class DocumentGroundingConfig(BaseModel):
    filters: list[DocumentGroundingFilter] | None = None
    placeholders: DocumentGroundingPlaceholders
    metadata_params: list[str] | None = None


class GroundingModuleConfig(BaseModel):
    type_: Literal["document_grounding_service"] = Field(default="document_grounding_service", alias="type")
    config: DocumentGroundingConfig


class Template(BaseModel):
    template: list[ChatMessage]
    defaults: dict[str, str] | None = None
    response_format: ResponseFormat | ResponseFormatJSONSchema | None = None
    tools: list[ChatCompletionTool] | None = None


class LLMModelDetails(BaseModel):
    name: str
    version: str = "latest"
    params: dict | None = None


class PromptTemplatingModuleConfig(BaseModel):
    prompt: Template | None = None
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
    replacement_strategy: DPIMethodConstant | DPIMethodFabricatedData | None = None


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

    type_: Literal["sap_data_privacy_integration"] = Field(default="sap_data_privacy_integration", alias="type")
    method: Literal["anonymization", "pseudonymization"]
    entities: list[DPIStandardEntity | DPICustomEntity]
    allowlist: list[str] | None = None
    mask_grounding_input: MaskGroundingInput | None = None


class MaskingModuleConfig(BaseModel):
    """
    Configuration for the data masking module.

    Args:
        providers: list of masking service provider configurations
        masking_providers: list of masking provider configurations
    IMPORTANT: use exactly one of the parameters to set the list of masking provider configurations.
    DEPRECATED: parameter 'masking_providers' will be removed Sept 15, 2026. Use 'providers' instead.
    """

    providers: list[MaskingProviderConfig] | None = Field(min_length=1, default=None)
    masking_providers: list[MaskingProviderConfig] | None = Field(min_length=1, default=None)

    @model_validator(mode="after")
    def enforce_exactly_one_provider_list(self):
        has_providers: Final = self.providers is not None
        has_masking_providers: Final = self.masking_providers is not None

        if not has_providers and not has_masking_providers:
            raise ValueError("For SAP Masking Module Config you must provide 'providers'.")
        if has_providers and has_masking_providers:
            raise ValueError(
                "For SAP Masking Module Config you must set exactly one of: 'providers' or 'masking_providers', not both."
            )

        if has_masking_providers:
            warnings.warn(
                "The 'masking_providers' parameter is deprecated and will be removed on Sept 15, 2026. "
                "Use 'providers' instead.",
                DeprecationWarning,
                stacklevel=5,
            )

        return self


class AfterLastRoleTargetSelector(BaseModel):
    """
    Scopes input filtering to all messages after the last message with a given role.

    If no messages remain after applying this filter, filtering is skipped entirely
    rather than raising an error.

    Args:
        after_last_role: The role used as the anchor. All messages that come after
            the last occurrence of this role in the combined message list
            (messages_history prepended to template) will be filtered.
            One of: 'system', 'user', 'assistant', 'developer', 'tool'.
    """

    after_last_role: Literal["system", "user", "assistant", "developer", "tool"]


class LastMessagesTargetSelector(BaseModel):
    """
    Scopes input filtering to the last N messages of the combined message list.

    Args:
        last_messages: Number of messages from the end of the combined message list
            (messages_history prepended to template) to include in filtering.
            Must be >= 1 (0 is not allowed and returns 400 Bad Request).
            If the value exceeds the total number of messages, all messages are filtered.
    """

    last_messages: int = Field(ge=1)


# InputFilterTargetSelector is a discriminated union: exactly one of the two selector
# shapes must be present. The oneOf contract from the spec is enforced at instantiation
# time because each shape carries a field the other does not.
InputFilterTargetSelector = AfterLastRoleTargetSelector | LastMessagesTargetSelector


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

    hate: AzureThreshold | Literal[0, 2, 4, 6] | None = None
    sexual: AzureThreshold | Literal[0, 2, 4, 6] | None = None
    violence: AzureThreshold | Literal[0, 2, 4, 6] | None = None
    self_harm: AzureThreshold | Literal[0, 2, 4, 6] | None = None


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

    prompt_shield: bool | None = False


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

    protected_material_code: bool | None = False


class LlamaGuard38bFilter(BaseModel):
    """
    Specific implementation of ContentFilter for Llama Guard 3. Llama Guard 3 is a
    Llama-3.1-8B pretrained model, fine-tuned for content safety classification.

    At least one category must be set to True (spec: minProperties: 1).

    .. deprecated::
        Will be removed 2027-09-20. Use :class:`GraniteGuardian41` instead.

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

    @model_validator(mode="after")
    def enforce_min_one_category(self) -> "LlamaGuard38bFilter":
        """At least one category must be enabled (spec: minProperties: 1)."""
        if not any(
            [
                self.violent_crimes,
                self.non_violent_crimes,
                self.sex_crimes,
                self.child_exploitation,
                self.defamation,
                self.specialized_advice,
                self.privacy,
                self.intellectual_property,
                self.indiscriminate_weapons,
                self.hate,
                self.self_harm,
                self.sexual_content,
                self.elections,
                self.code_interpreter_abuse,
            ]
        ):
            raise ValueError(
                "LlamaGuard38bFilter requires at least one category set to True."
            )
        return self


class LlamaGuard38bFilterConfig(BaseModel):
    type_: Literal["llama_guard_3_8b"] = Field(default="llama_guard_3_8b", alias="type")
    config: LlamaGuard38bFilter
    target_selector: InputFilterTargetSelector | None = None


class GraniteGuardian41Categories(BaseModel):
    """
    Content categories evaluated by IBM Granite Guardian 4.1.

    At least one category must be set to True (minProperties: 1 in the spec).
    Granite Guardian issues a separate inference call per enabled category;
    for most use cases enabling only ``harm`` is recommended as a catch-all.

    Args:
        harm: Catch-all criterion for generally harmful content.
        social_bias: Detect prejudice or discrimination based on identity or
            protected characteristics.
        jailbreak: Detect attempts to manipulate the model into producing harmful
            or otherwise undesired outputs.
        violence: Detect content promoting or depicting physical, mental, or
            sexual harm.
        profanity: Detect offensive language or insults.
        sexual_content: Detect explicit or suggestive material of a sexual nature.
        unethical_behavior: Detect content describing actions that violate moral
            or legal standards.
    """

    harm: bool = Field(default=False)
    social_bias: bool = Field(default=False)
    jailbreak: bool = Field(default=False)
    violence: bool = Field(default=False)
    profanity: bool = Field(default=False)
    sexual_content: bool = Field(default=False)
    unethical_behavior: bool = Field(default=False)

    @model_validator(mode="after")
    def enforce_min_one_category(self) -> "GraniteGuardian41Categories":
        """At least one category must be enabled (spec: minProperties: 1)."""
        if not any(
            [
                self.harm,
                self.social_bias,
                self.jailbreak,
                self.violence,
                self.profanity,
                self.sexual_content,
                self.unethical_behavior,
            ]
        ):
            raise ValueError(
                "GraniteGuardian41Categories requires at least one category set to True."
            )
        return self


class GraniteGuardian41(BaseModel):
    """
    Configuration for IBM Granite Guardian 4.1 filter provider.

    Args:
        enable_reasoning: Enable reasoning (think) mode. When True, the model returns
            an explanation alongside each verdict, e.g.
            ``{'verdict': True, 'reasoning': '...'}``. Applies to every configured
            category. Defaults to False.
        categories: Content criteria to evaluate. At least one category must be
            enabled. Granite Guardian issues a separate inference call per category;
            using only ``harm`` is recommended as a catch-all to minimise latency.
    """

    enable_reasoning: bool = Field(default=False)
    categories: GraniteGuardian41Categories


class AzureContentSafetyInputFilterConfig(BaseModel):
    type_: Literal["azure_content_safety"] = Field(default="azure_content_safety", alias="type")
    config: AzureContentSafetyInput | None = None
    target_selector: InputFilterTargetSelector | None = None


class AzureContentSafetyOutputFilterConfig(BaseModel):
    type_: Literal["azure_content_safety"] = Field(default="azure_content_safety", alias="type")
    config: AzureContentSafetyOutput | None = None


class GraniteGuardianFilterConfig(BaseModel):
    """
    Filter configuration for the IBM Granite Guardian 4.1 provider.

    Args:
        type_: Provider discriminator — always ``'granite_guardian_4_1'``.
        config: Category and reasoning settings for Granite Guardian.
        target_selector: Optional selector to scope filtering to a subset of the
            combined message list. When absent, all input content is filtered.
    """

    type_: Literal["granite_guardian_4_1"] = Field(default="granite_guardian_4_1", alias="type")
    config: GraniteGuardian41
    target_selector: InputFilterTargetSelector | None = None


class FilteringStreamOptions(BaseModel):
    """
    overlap: Number of characters that should be additionally sent to content filtering services
    from previous chunks as additional context.
    """

    overlap: int | None = Field(default=0, ge=0, le=10000)


class InputFiltering(BaseModel):
    """Module for managing and applying input content filters.

    Args:
        filters: List of filter provider configurations to be applied to input content.
            Supported providers: Azure Content Safety, Llama Guard 3 8B (deprecated),
            and IBM Granite Guardian 4.1.
    """

    filters: list[AzureContentSafetyInputFilterConfig | LlamaGuard38bFilterConfig | GraniteGuardianFilterConfig] = Field(min_length=1)


class OutputFiltering(BaseModel):
    """Module for managing and applying output content filters.

    Args:
        filters: List of filter provider configurations to be applied to output content.
            Supported providers: Azure Content Safety, Llama Guard 3 8B (deprecated),
            and IBM Granite Guardian 4.1.
        stream_options: Module-specific streaming options. Ignored when streaming is
            disabled.
    """

    filters: list[AzureContentSafetyOutputFilterConfig | LlamaGuard38bFilterConfig | GraniteGuardianFilterConfig] = Field(min_length=1)
    stream_options: FilteringStreamOptions | None = None


class FilteringModuleConfig(BaseModel):
    """Module for managing and applying content filters.

    Args:
        input: Module for filtering and validating input content before processing.

        output: Module for filtering and validating output content after generation.
    """

    input: InputFiltering | None = None
    output: OutputFiltering | None = None

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

    source_language: str | None = None
    target_language: str
    apply_to: list[SAPDocumentTranslationApplyToSelector] | None = None


class OutputTranslationConfig(BaseModel):
    source_language: str | None = None
    target_language: str | SAPDocumentTranslationApplyToSelector


class SAPDocumentTranslationInput(BaseModel):
    """
    Configuration for input translation

    Args:
        type: The type of translation module (e.g., 'sap_document_translation').

        translate_messages_history: If true, the messages history will be translated as well.

        config: Configuration object for the translation module.
    """

    type_: Literal["sap_document_translation"] = Field(default="sap_document_translation", alias="type")
    translate_messages_history: bool | None = None
    config: InputTranslationConfig


class SAPDocumentTranslationOutput(BaseModel):
    """
    Configuration for output translation

    Args:
         type: The type of translation module (e.g., 'sap_document_translation').

        config: Configuration object for the translation module.
    """

    type_: Literal["sap_document_translation"] = Field(default="sap_document_translation", alias="type")
    config: OutputTranslationConfig


class TranslationModuleConfig(BaseModel):
    """
    Configuration for translation module

    Args:
        input: Configuration for input translation

        output: Configuration for output translation
    """

    input: SAPDocumentTranslationInput | None = None
    output: SAPDocumentTranslationOutput | None = None

    @model_validator(mode="after")
    def enforce_min_properties(self) -> "TranslationModuleConfig":
        if self.input is None and self.output is None:
            raise ValueError("TranslationModuleConfig requires at least one of 'input' or 'output'.")
        return self


class ModuleConfig(BaseModel):
    prompt_templating: PromptTemplatingModuleConfig
    filtering: FilteringModuleConfig | None = None
    masking: MaskingModuleConfig | None = None
    grounding: GroundingModuleConfig | None = None
    translation: TranslationModuleConfig | None = None


class GlobalStreamOptions(BaseModel):
    enabled: bool = False
    chunk_size: int | None = Field(default=None, ge=1)
    delimiters: list[str] | None = None


class OrchestrationConfig(BaseModel):
    modules: ModuleConfig | list[ModuleConfig]
    stream: GlobalStreamOptions | None = None


class OrchestrationRequest(BaseModel):
    config: OrchestrationConfig
    placeholder_values: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Partial config models — used by the config_ref request variants.
# All fields are optional so callers only supply what they want to override.
# ---------------------------------------------------------------------------


class PartialPromptTemplatingModuleConfig(BaseModel):
    """Partial prompt-templating override for config_ref requests.

    Both fields are optional: omit ``prompt`` to keep the referenced template,
    omit ``model`` to keep the referenced model.
    """

    prompt: Template | None = None
    model: LLMModelDetails | None = None


class PartialModuleConfigs(BaseModel):
    """Partial module configuration for config_ref overrides.

    Only specify the modules you want to override; the remaining configuration
    is taken from the referenced orchestration config.
    """

    prompt_templating: PartialPromptTemplatingModuleConfig | None = None
    filtering: FilteringModuleConfig | None = None
    masking: MaskingModuleConfig | None = None
    grounding: GroundingModuleConfig | None = None
    translation: TranslationModuleConfig | None = None


class PartialOrchestrationConfig(BaseModel):
    """Partial orchestration configuration for config_ref overrides.

    All fields are optional.  Supply only the parts that should be overridden;
    the rest is taken from the referenced configuration stored in SAP AI Core.
    """

    modules: PartialModuleConfigs | None = None
    stream: GlobalStreamOptions | None = None


# ---------------------------------------------------------------------------
# config_ref discriminated shapes (spec: CompletionPostRequest oneOf variants)
# ---------------------------------------------------------------------------


class CompletionRequestConfigurationReferenceByIdConfigRef(BaseModel):
    """Reference an SAP AI Core orchestration configuration by its UUID."""

    id: str


class CompletionRequestConfigurationReferenceById(BaseModel):
    """POST /v2/completion body variant: reference a saved config by ID.

    The optional ``config`` field carries a partial override that is merged
    on top of the referenced configuration.  ``placeholder_values`` and
    ``messages_history`` work the same as in the full-config variant.
    """

    config_ref: CompletionRequestConfigurationReferenceByIdConfigRef
    config: PartialOrchestrationConfig | None = None
    placeholder_values: dict[str, str] | None = None
    messages_history: list[ChatMessage] | None = None


class CompletionRequestConfigurationReferenceByNameScenarioVersionConfigRef(BaseModel):
    """Reference an SAP AI Core orchestration configuration by name + scenario + version."""

    scenario: str
    name: str
    version: str


class CompletionRequestConfigurationReferenceByNameScenarioVersion(BaseModel):
    """POST /v2/completion body variant: reference a saved config by name/scenario/version.

    The optional ``config`` field carries a partial override that is merged
    on top of the referenced configuration.  ``placeholder_values`` and
    ``messages_history`` work the same as in the full-config variant.
    """

    config_ref: CompletionRequestConfigurationReferenceByNameScenarioVersionConfigRef
    config: PartialOrchestrationConfig | None = None
    placeholder_values: dict[str, str] | None = None
    messages_history: list[ChatMessage] | None = None
