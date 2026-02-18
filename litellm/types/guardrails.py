from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Required, TypedDict

from litellm.types.proxy.guardrails.guardrail_hooks.enkryptai import (
    EnkryptAIGuardrailConfigs,
)
from litellm.types.proxy.guardrails.guardrail_hooks.grayswan import (
    GraySwanGuardrailConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.ibm import (
    IBMGuardrailsBaseConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.litellm_content_filter import (
    ContentFilterCategoryConfig,
)
from litellm.types.proxy.guardrails.guardrail_hooks.qualifire import (
    QualifireGuardrailConfigModel,
)
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrailConfigModel,
)

"""
Pydantic object defining how to set guardrails on litellm proxy

guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock  # supported values: "aporia", "bedrock", "lakera", "zscaler_ai_guard"
      mode: "during_call"
      guardrailIdentifier: ff6ujrregl1q
      guardrailVersion: "DRAFT"
      default_on: true
"""


class SupportedGuardrailIntegrations(Enum):
    APORIA = "aporia"
    BEDROCK = "bedrock"
    GUARDRAILS_AI = "guardrails_ai"
    LAKERA = "lakera"
    LAKERA_V2 = "lakera_v2"
    PRESIDIO = "presidio"
    HIDE_SECRETS = "hide-secrets"
    HIDDENLAYER = "hiddenlayer"
    AIM = "aim"
    PANGEA = "pangea"
    LASSO = "lasso"
    PILLAR = "pillar"
    GRAYSWAN = "grayswan"
    PANW_PRISMA_AIRS = "panw_prisma_airs"
    AZURE_PROMPT_SHIELD = "azure/prompt_shield"
    AZURE_TEXT_MODERATIONS = "azure/text_moderations"
    MODEL_ARMOR = "model_armor"
    OPENAI_MODERATION = "openai_moderation"
    NOMA = "noma"
    TOOL_PERMISSION = "tool_permission"
    ZSCALER_AI_GUARD = "zscaler_ai_guard"
    JAVELIN = "javelin"
    ENKRYPTAI = "enkryptai"
    IBM_GUARDRAILS = "ibm_guardrails"
    LITELLM_CONTENT_FILTER = "litellm_content_filter"
    MCP_SECURITY = "mcp_security"
    ONYX = "onyx"
    PROMPT_SECURITY = "prompt_security"
    GENERIC_GUARDRAIL_API = "generic_guardrail_api"
    QUALIFIRE = "qualifire"
    CUSTOM_CODE = "custom_code"


class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


default_roles = [Role.SYSTEM, Role.ASSISTANT, Role.USER]


class GuardrailItemSpec(TypedDict, total=False):
    callbacks: Required[List[str]]
    default_on: bool
    logging_only: Optional[bool]
    enabled_roles: Optional[List[Role]]
    callback_args: Dict[str, Dict]


class GuardrailItem(BaseModel):
    callbacks: List[str]
    default_on: bool
    logging_only: Optional[bool]
    guardrail_name: str
    callback_args: Dict[str, Dict]
    enabled_roles: Optional[List[Role]]

    model_config = ConfigDict(use_enum_values=True)

    def __init__(
        self,
        callbacks: List[str],
        guardrail_name: str,
        default_on: bool = False,
        logging_only: Optional[bool] = None,
        enabled_roles: Optional[List[Role]] = default_roles,
        callback_args: Dict[str, Dict] = {},
    ):
        super().__init__(
            callbacks=callbacks,
            default_on=default_on,
            logging_only=logging_only,
            guardrail_name=guardrail_name,
            enabled_roles=enabled_roles,
            callback_args=callback_args,
        )


# Define the TypedDicts
class LakeraCategoryThresholds(TypedDict, total=False):
    prompt_injection: float
    jailbreak: float


class PiiAction(str, Enum):
    BLOCK = "BLOCK"
    MASK = "MASK"


class PiiEntityCategory(str, Enum):
    GENERAL = "General"
    FINANCE = "Finance"
    USA = "USA"
    UK = "UK"
    SPAIN = "Spain"
    ITALY = "Italy"
    POLAND = "Poland"
    SINGAPORE = "Singapore"
    AUSTRALIA = "Australia"
    INDIA = "India"
    FINLAND = "Finland"


class PiiEntityType(str, Enum):
    # General
    CREDIT_CARD = "CREDIT_CARD"
    CRYPTO = "CRYPTO"
    DATE_TIME = "DATE_TIME"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    NRP = "NRP"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    URL = "URL"
    # USA
    US_BANK_NUMBER = "US_BANK_NUMBER"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_ITIN = "US_ITIN"
    US_PASSPORT = "US_PASSPORT"
    US_SSN = "US_SSN"
    # UK
    UK_NHS = "UK_NHS"
    UK_NINO = "UK_NINO"
    # Spain
    ES_NIF = "ES_NIF"
    ES_NIE = "ES_NIE"
    # Italy
    IT_FISCAL_CODE = "IT_FISCAL_CODE"
    IT_DRIVER_LICENSE = "IT_DRIVER_LICENSE"
    IT_VAT_CODE = "IT_VAT_CODE"
    IT_PASSPORT = "IT_PASSPORT"
    IT_IDENTITY_CARD = "IT_IDENTITY_CARD"
    # Poland
    PL_PESEL = "PL_PESEL"
    # Singapore
    SG_NRIC_FIN = "SG_NRIC_FIN"
    SG_UEN = "SG_UEN"
    # Australia
    AU_ABN = "AU_ABN"
    AU_ACN = "AU_ACN"
    AU_TFN = "AU_TFN"
    AU_MEDICARE = "AU_MEDICARE"
    # India
    IN_PAN = "IN_PAN"
    IN_AADHAAR = "IN_AADHAAR"
    IN_VEHICLE_REGISTRATION = "IN_VEHICLE_REGISTRATION"
    IN_VOTER = "IN_VOTER"
    IN_PASSPORT = "IN_PASSPORT"
    # Finland
    FI_PERSONAL_IDENTITY_CODE = "FI_PERSONAL_IDENTITY_CODE"


# Define mappings of PII entity types by category
PII_ENTITY_CATEGORIES_MAP = {
    PiiEntityCategory.GENERAL: [
        PiiEntityType.DATE_TIME,
        PiiEntityType.EMAIL_ADDRESS,
        PiiEntityType.IP_ADDRESS,
        PiiEntityType.NRP,
        PiiEntityType.LOCATION,
        PiiEntityType.PERSON,
        PiiEntityType.PHONE_NUMBER,
        PiiEntityType.MEDICAL_LICENSE,
        PiiEntityType.URL,
    ],
    PiiEntityCategory.FINANCE: [
        PiiEntityType.CREDIT_CARD,
        PiiEntityType.CRYPTO,
        PiiEntityType.IBAN_CODE,
    ],
    PiiEntityCategory.USA: [
        PiiEntityType.US_BANK_NUMBER,
        PiiEntityType.US_DRIVER_LICENSE,
        PiiEntityType.US_ITIN,
        PiiEntityType.US_PASSPORT,
        PiiEntityType.US_SSN,
    ],
    PiiEntityCategory.UK: [PiiEntityType.UK_NHS, PiiEntityType.UK_NINO],
    PiiEntityCategory.SPAIN: [PiiEntityType.ES_NIF, PiiEntityType.ES_NIE],
    PiiEntityCategory.ITALY: [
        PiiEntityType.IT_FISCAL_CODE,
        PiiEntityType.IT_DRIVER_LICENSE,
        PiiEntityType.IT_VAT_CODE,
        PiiEntityType.IT_PASSPORT,
        PiiEntityType.IT_IDENTITY_CARD,
    ],
    PiiEntityCategory.POLAND: [PiiEntityType.PL_PESEL],
    PiiEntityCategory.SINGAPORE: [PiiEntityType.SG_NRIC_FIN, PiiEntityType.SG_UEN],
    PiiEntityCategory.AUSTRALIA: [
        PiiEntityType.AU_ABN,
        PiiEntityType.AU_ACN,
        PiiEntityType.AU_TFN,
        PiiEntityType.AU_MEDICARE,
    ],
    PiiEntityCategory.INDIA: [
        PiiEntityType.IN_PAN,
        PiiEntityType.IN_AADHAAR,
        PiiEntityType.IN_VEHICLE_REGISTRATION,
        PiiEntityType.IN_VOTER,
        PiiEntityType.IN_PASSPORT,
    ],
    PiiEntityCategory.FINLAND: [PiiEntityType.FI_PERSONAL_IDENTITY_CODE],
}


class PiiEntityCategoryMap(TypedDict):
    category: str
    entities: List[str]


class GuardrailParamUITypes(str, Enum):
    BOOL = "bool"
    STR = "str"


class PresidioPresidioConfigModelUserInterface(BaseModel):
    """Configuration parameters for the Presidio PII masking guardrail on LiteLLM UI"""

    presidio_analyzer_api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Presidio analyzer API",
    )
    presidio_anonymizer_api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Presidio anonymizer API",
    )
    presidio_filter_scope: Optional[Literal["input", "output", "both"]] = Field(
        default=None,
        description=(
            "Where to apply Presidio checks: 'input' (user -> model), "
            "'output' (model -> user), or 'both' (default)."
        ),
    )
    output_parse_pii: Optional[bool] = Field(
        default=None,
        description="When True, LiteLLM will replace the masked text with the original text in the response",
        # extra param to let the ui know this is a boolean
        json_schema_extra={"ui_type": GuardrailParamUITypes.BOOL},
    )
    presidio_language: Optional[str] = Field(
        default="en",
        description="Language code for Presidio PII analysis (e.g., 'en', 'de', 'es', 'fr')",
    )
    presidio_run_on: Optional[Literal["input", "output", "both"]] = Field(
        default=None,
        description="Where to apply Presidio checks: input, output, or both (default).",
    )


class PresidioConfigModel(PresidioPresidioConfigModelUserInterface):
    """Configuration parameters for the Presidio PII masking guardrail"""

    pii_entities_config: Optional[Dict[Union[PiiEntityType, str], PiiAction]] = Field(
        default=None, description="Configuration for PII entity types and actions"
    )

    presidio_score_thresholds: Optional[Dict[Union[PiiEntityType, str], float]] = Field(
        default=None,
        description=(
            "Optional per-entity minimum confidence scores for Presidio detections. "
            "Entities below the threshold are ignored."
        ),
    )
    presidio_ad_hoc_recognizers: Optional[str] = Field(
        default=None,
        description="Path to a JSON file containing ad-hoc recognizers for Presidio",
    )
    mock_redacted_text: Optional[dict] = Field(
        default=None, description="Mock redacted text for testing"
    )


class BedrockGuardrailConfigModel(BaseModel):
    """Configuration parameters for the AWS Bedrock guardrail"""

    guardrailIdentifier: Optional[str] = Field(
        default=None, description="The ID of your guardrail on Bedrock"
    )
    guardrailVersion: Optional[str] = Field(
        default=None,
        description="The version of your Bedrock guardrail (e.g., DRAFT or version number)",
    )
    disable_exception_on_block: Optional[bool] = Field(
        default=False,
        description="If True, will not raise an exception when the guardrail is blocked. Useful for OpenWebUI where exceptions can end the chat flow.",
    )
    aws_region_name: Optional[str] = Field(
        default=None, description="AWS region where your guardrail is deployed"
    )
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key ID for authentication"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key for authentication"
    )
    aws_session_token: Optional[str] = Field(
        default=None, description="AWS session token for temporary credentials"
    )
    aws_session_name: Optional[str] = Field(
        default=None, description="Name of the AWS session"
    )
    aws_profile_name: Optional[str] = Field(
        default=None, description="AWS profile name for credential retrieval"
    )
    aws_role_name: Optional[str] = Field(
        default=None, description="AWS role name for assuming roles"
    )
    aws_web_identity_token: Optional[str] = Field(
        default=None, description="Web identity token for AWS role assumption"
    )
    aws_sts_endpoint: Optional[str] = Field(
        default=None, description="AWS STS endpoint URL"
    )
    aws_bedrock_runtime_endpoint: Optional[str] = Field(
        default=None, description="AWS Bedrock runtime endpoint URL"
    )


class LakeraV2GuardrailConfigModel(BaseModel):
    """Configuration parameters for the Lakera AI v2 guardrail"""

    api_key: Optional[str] = Field(
        default=None, description="API key for the Lakera AI service"
    )
    api_base: Optional[str] = Field(
        default=None, description="Base URL for the Lakera AI API"
    )
    project_id: Optional[str] = Field(
        default=None, description="Project ID for the Lakera AI project"
    )
    payload: Optional[bool] = Field(
        default=True, description="Whether to include payload in the response"
    )
    breakdown: Optional[bool] = Field(
        default=True, description="Whether to include breakdown in the response"
    )
    metadata: Optional[Dict] = Field(
        default=None, description="Additional metadata to include in the request"
    )
    dev_info: Optional[bool] = Field(
        default=True,
        description="Whether to include developer information in the response",
    )
    on_flagged: Optional[Literal["block", "monitor"]] = Field(
        default="block",
        description="Action to take when content is flagged: 'block' (raise exception) or 'monitor' (log only)",
    )


class LassoGuardrailConfigModel(BaseModel):
    """Configuration parameters for the Lasso guardrail"""

    lasso_user_id: Optional[str] = Field(
        default=None, description="User ID for the Lasso guardrail"
    )
    lasso_conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for the Lasso guardrail"
    )
    mask: Optional[bool] = Field(
        default=False, description="Enable content masking using Lasso classifix API"
    )


class PillarGuardrailConfigModel(BaseModel):
    """Configuration parameters for the Pillar Security guardrail"""

    on_flagged_action: Optional[str] = Field(
        default="monitor",
        description="Action to take when content is flagged: 'block' (raise exception) or 'monitor' (log only)",
    )
    async_mode: Optional[bool] = Field(
        default=None,
        description="Set to True to request asynchronous analysis (sets `plr_async` header). Defaults to provider behaviour when omitted.",
    )
    persist_session: Optional[bool] = Field(
        default=None,
        description="Controls Pillar session persistence (sets `plr_persist` header). Set to False to disable persistence.",
    )
    include_scanners: Optional[bool] = Field(
        default=True,
        description="Include scanner category summaries in responses (sets `plr_scanners` header).",
    )
    include_evidence: Optional[bool] = Field(
        default=True,
        description="Include detailed evidence payloads in responses (sets `plr_evidence` header).",
    )


class NomaGuardrailConfigModel(BaseModel):
    """Configuration parameters for the Noma Security guardrail"""

    application_id: Optional[str] = Field(
        default=None,
        description="Application ID for Noma Security. Defaults to 'litellm' if not provided",
    )
    monitor_mode: Optional[bool] = Field(
        default=None,
        description="If True, logs violations without blocking. Defaults to False if not provided",
    )
    block_failures: Optional[bool] = Field(
        default=None,
        description="If True, blocks requests on API failures. Defaults to True if not provided",
    )
    anonymize_input: Optional[bool] = Field(
        default=None,
        description="If True, replaces sensitive content with anonymized version when only PII/PCI/secrets are detected. Only applies in blocking mode. Defaults to False if not provided",
    )


class ZscalerAIGuardConfigModel(BaseModel):
    """Configuration parameters for the Zscaler AI Guard guardrail"""

    policy_id: Optional[int] = Field(
        default=None,
        description="Policy ID for Zscaler AI Guard. Can also be set via ZSCALER_AI_GUARD_POLICY_ID environment variable",
    )
    send_user_api_key_alias: Optional[bool] = Field(
        default=False, description="Whether to send user_API_key_alias in headers"
    )
    send_user_api_key_user_id: Optional[bool] = Field(
        default=False, description="Whether to send user_API_key_user_id in headers"
    )
    send_user_api_key_team_id: Optional[bool] = Field(
        default=False, description="Whether to send user_API_key_team_id in headers"
    )


class JavelinGuardrailConfigModel(BaseModel):
    """Configuration parameters for the Javelin guardrail"""

    guard_name: Optional[str] = Field(
        default=None, description="Name of the Javelin guard to use"
    )
    api_version: Optional[str] = Field(
        default="v1", description="API version for Javelin service"
    )
    metadata: Optional[Dict] = Field(
        default=None, description="Additional metadata to send with requests"
    )
    application: Optional[str] = Field(
        default=None, description="Application name for Javelin service"
    )
    config: Optional[Dict] = Field(
        default=None, description="Additional configuration for the guardrail"
    )


class ContentFilterAction(str, Enum):
    """Action to take when content filter detects a match"""

    BLOCK = "BLOCK"
    MASK = "MASK"


class BlockedWord(BaseModel):
    """Represents a blocked word with its action and optional description"""

    keyword: str = Field(description="The keyword to block or mask")
    action: ContentFilterAction = Field(
        description="Action to take when keyword is detected (BLOCK or MASK)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description explaining why this keyword is sensitive",
    )


class ContentFilterPattern(BaseModel):
    """Represents a content filter pattern (prebuilt or custom regex)"""

    pattern_type: Literal["prebuilt", "regex"] = Field(
        description="Type of pattern: 'prebuilt' for predefined patterns or 'regex' for custom"
    )
    pattern_name: Optional[str] = Field(
        default=None,
        description="Name of prebuilt pattern (e.g., 'us_ssn', 'credit_card'). Required if pattern_type is 'prebuilt'",
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Custom regex pattern. Required if pattern_type is 'regex'",
    )
    name: Optional[str] = Field(
        default=None,
        description="Name for this pattern (used in logging and error messages)",
    )
    action: ContentFilterAction = Field(
        description="Action to take when pattern matches (BLOCK or MASK)"
    )


class ContentFilterConfigModel(BaseModel):
    """Configuration parameters for the content filter guardrail"""

    patterns: Optional[List[ContentFilterPattern]] = Field(
        default=None,
        description="List of patterns (prebuilt or custom regex) to detect",
    )
    blocked_words: Optional[List[BlockedWord]] = Field(
        default=None, description="List of blocked words with individual actions"
    )
    blocked_words_file: Optional[str] = Field(
        default=None, description="Path to YAML file containing blocked_words list"
    )
    categories: Optional[List[ContentFilterCategoryConfig]] = Field(
        default=None,
        description="List of prebuilt categories to enable (harmful_*, bias_*)",
    )
    severity_threshold: Optional[str] = Field(
        default=None,
        description="Minimum severity to block (high, medium, low)",
    )
    pattern_redaction_format: Optional[str] = Field(
        default=None,
        description="Format string for pattern redaction (use {pattern_name} placeholder)",
    )
    keyword_redaction_tag: Optional[str] = Field(
        default=None,
        description="Tag to use for keyword redaction",
    )


class BaseLitellmParams(
    ContentFilterConfigModel
):  # works for new and patch update guardrails
    api_key: Optional[str] = Field(
        default=None, description="API key for the guardrail service"
    )
    api_base: Optional[str] = Field(
        default=None, description="Base URL for the guardrail service API"
    )

    experimental_use_latest_role_message_only: Optional[bool] = Field(
        default=False,
        description="When True, guardrails only receive the latest message for the relevant role (e.g., newest user input pre-call, newest assistant output post-call)",
    )

    # Lakera specific params
    category_thresholds: Optional[LakeraCategoryThresholds] = Field(
        default=None,
        description="Threshold configuration for Lakera guardrail categories",
    )

    # hide secrets params
    detect_secrets_config: Optional[dict] = Field(
        default=None, description="Configuration for detect-secrets guardrail"
    )

    # guardrails ai params
    guard_name: Optional[str] = Field(
        default=None, description="Name of the guardrail in guardrails.ai"
    )
    default_on: Optional[bool] = Field(
        default=None, description="Whether the guardrail is enabled by default"
    )

    ################## PII control params #################
    ########################################################
    mask_request_content: Optional[bool] = Field(
        default=None,
        description="Will mask request content if guardrail makes any changes",
    )
    mask_response_content: Optional[bool] = Field(
        default=None,
        description="Will mask response content if guardrail makes any changes",
    )

    # pangea params
    pangea_input_recipe: Optional[str] = Field(
        default=None, description="Recipe for input (LLM request)"
    )

    pangea_output_recipe: Optional[str] = Field(
        default=None, description="Recipe for output (LLM response)"
    )

    model: Optional[str] = Field(
        default=None,
        description="Optional field if guardrail requires a 'model' parameter",
    )

    violation_message_template: Optional[str] = Field(
        default=None,
        description="Custom message when a guardrail blocks an action. Supports placeholders like {tool_name}, {rule_id}, and {default_message}.",
    )

    # Model Armor params
    template_id: Optional[str] = Field(
        default=None, description="The ID of your Model Armor template"
    )
    location: Optional[str] = Field(
        default=None, description="Google Cloud location/region (e.g., us-central1)"
    )
    credentials: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud credentials JSON file or JSON string",
    )
    api_endpoint: Optional[str] = Field(
        default=None, description="Optional custom API endpoint for Model Armor"
    )
    fail_on_error: Optional[bool] = Field(
        default=True,
        description="Whether to fail the request if Model Armor encounters an error",
    )

    additional_provider_specific_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific parameters for generic guardrail APIs",
    )

    # Custom code guardrail params
    custom_code: Optional[str] = Field(
        default=None,
        description="Python-like code containing the apply_guardrail function for custom guardrail logic",
    )

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class Mode(BaseModel):
    tags: Dict[str, str] = Field(description="Tags for the guardrail mode")
    default: Optional[str] = Field(
        default=None, description="Default mode when no tags match"
    )


class LitellmParams(
    PresidioConfigModel,
    BedrockGuardrailConfigModel,
    LakeraV2GuardrailConfigModel,
    LassoGuardrailConfigModel,
    PillarGuardrailConfigModel,
    GraySwanGuardrailConfigModel,
    NomaGuardrailConfigModel,
    ToolPermissionGuardrailConfigModel,
    ZscalerAIGuardConfigModel,
    JavelinGuardrailConfigModel,
    BaseLitellmParams,
    EnkryptAIGuardrailConfigs,
    IBMGuardrailsBaseConfigModel,
    QualifireGuardrailConfigModel,
):
    guardrail: str = Field(description="The type of guardrail integration to use")
    mode: Union[str, List[str], Mode] = Field(
        description="When to apply the guardrail (pre_call, post_call, during_call, logging_only)"
    )

    @field_validator(
        "mode",
        "default_action",
        "on_disallowed_action",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def normalize_lowercase(cls, v):
        """Normalize string and list fields to lowercase for ALL guardrail types."""
        if isinstance(v, str):
            return v.lower()
        if isinstance(v, list):
            return [x.lower() if isinstance(x, str) else x for x in v]
        return v

    def __init__(self, **kwargs):
        default_on = kwargs.pop("default_on", None)
        if default_on is not None:
            kwargs["default_on"] = default_on
        else:
            kwargs["default_on"] = False

        super().__init__(**kwargs)

    def __contains__(self, key):
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)


class Guardrail(TypedDict, total=False):
    guardrail_id: Optional[str]
    guardrail_name: Required[str]
    litellm_params: Required[LitellmParams]
    guardrail_info: Optional[Dict]
    policy_template: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class guardrailConfig(TypedDict):
    guardrails: List[Guardrail]


class GuardrailEventHooks(str, Enum):
    pre_call = "pre_call"
    post_call = "post_call"
    during_call = "during_call"
    logging_only = "logging_only"
    pre_mcp_call = "pre_mcp_call"
    during_mcp_call = "during_mcp_call"


class DynamicGuardrailParams(TypedDict):
    extra_body: Dict[str, Any]


class GUARDRAIL_DEFINITION_LOCATION(str, Enum):
    DB = "db"
    CONFIG = "config"


class GuardrailInfoResponse(BaseModel):
    guardrail_id: Optional[str] = None
    guardrail_name: str
    litellm_params: Optional[BaseLitellmParams] = None
    guardrail_info: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    guardrail_definition_location: GUARDRAIL_DEFINITION_LOCATION = (
        GUARDRAIL_DEFINITION_LOCATION.CONFIG
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ListGuardrailsResponse(BaseModel):
    guardrails: List[GuardrailInfoResponse]


class GuardrailUIAddGuardrailSettings(BaseModel):
    supported_entities: List[str]
    supported_actions: List[str]
    supported_modes: List[str]
    pii_entity_categories: List[PiiEntityCategoryMap]
    content_filter_settings: Optional[Dict[str, Any]] = None


class PresidioPerRequestConfig(BaseModel):
    """
    presdio params that can be controlled per request, api key
    """

    language: Optional[str] = None
    entities: Optional[List[PiiEntityType]] = None


class ApplyGuardrailRequest(BaseModel):
    guardrail_name: str
    text: str
    language: Optional[str] = None
    entities: Optional[List[PiiEntityType]] = None


class ApplyGuardrailResponse(BaseModel):
    response_text: str


class PatchGuardrailRequest(BaseModel):
    guardrail_name: Optional[str] = None
    litellm_params: Optional[BaseLitellmParams] = None
    guardrail_info: Optional[Dict[str, Any]] = None
