from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import Required, TypedDict

"""
Pydantic object defining how to set guardrails on litellm proxy

guardrails:
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      guardrailIdentifier: ff6ujrregl1q
      guardrailVersion: "DRAFT"
      default_on: true
"""


class SupportedGuardrailIntegrations(Enum):
    APORIA = "aporia"
    BEDROCK = "bedrock"
    GURDRAILS_AI = "guardrails_ai"
    LAKERA = "lakera"
    LAKERA_V2 = "lakera_v2"
    PRESIDIO = "presidio"
    HIDE_SECRETS = "hide-secrets"
    AIM = "aim"
    PANGEA = "pangea"
    LASSO = "lasso"
    PANW_PRISMA_AIRS = "panw_prisma_airs"
    AZURE_PROMPT_SHIELD = "azure/prompt_shield"
    AZURE_TEXT_MODERATIONS = "azure/text_moderations"


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
    category: PiiEntityCategory
    entities: List[PiiEntityType]


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


class PresidioConfigModel(PresidioPresidioConfigModelUserInterface):
    """Configuration parameters for the Presidio PII masking guardrail"""

    pii_entities_config: Optional[Dict[PiiEntityType, PiiAction]] = Field(
        default=None, description="Configuration for PII entity types and actions"
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


class LassoGuardrailConfigModel(BaseModel):
    """Configuration parameters for the Lasso guardrail"""

    lasso_user_id: Optional[str] = Field(
        default=None, description="User ID for the Lasso guardrail"
    )
    lasso_conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for the Lasso guardrail"
    )


class AzureContentSafetyConfigModel(BaseModel):
    """Configuration parameters for the Azure Content Safety Prompt Shield guardrail"""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for the Azure Content Safety Prompt Shield guardrail",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the Azure Content Safety Prompt Shield guardrail",
    )
    api_version: Optional[str] = Field(
        default=None,
        description="API version for the Azure Content Safety Prompt Shield guardrail",
    )


class AzureContentSafetyPromptShieldConfigModel(AzureContentSafetyConfigModel):
    """Configuration parameters for the Azure Content Safety Prompt Shield guardrail"""

    pass


class AzureContentSafetyTextModerationConfigModel(AzureContentSafetyConfigModel):

    severity_threshold: Optional[int] = Field(
        default=None,
        description="Severity threshold for the Azure Content Safety Text Moderation guardrail across all categories",
    )
    severity_threshold_by_category: Optional[Dict[str, int]] = Field(
        default=None,
        description="Severity threshold by category for the Azure Content Safety Text Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )

    categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to scan for the Azure Content Safety Text Moderation guardrail. See list of categories - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/harm-categories?tabs=warning",
    )
    blocklistNames: Optional[List[str]] = Field(
        default=None,
        description="Blocklist names to scan for the Azure Content Safety Text Moderation guardrail. Learn more - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text",
    )
    haltOnBlocklistHit: Optional[bool] = Field(
        default=None,
        description="Whether to halt the request if a blocklist hit is detected",
    )
    outputType: Optional[Literal["FourSeverityLevels", "EightSeverityLevels"]] = Field(
        default=None,
        description="Output type for the Azure Content Safety Text Moderation guardrail. Learn more - https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text",
    )


class LitellmParams(
    PresidioConfigModel,
    BedrockGuardrailConfigModel,
    LakeraV2GuardrailConfigModel,
    LassoGuardrailConfigModel,
):
    guardrail: str = Field(description="The type of guardrail integration to use")
    mode: Union[str, List[str]] = Field(
        description="When to apply the guardrail (pre_call, post_call, during_call, logging_only)"
    )
    api_key: Optional[str] = Field(
        default=None, description="API key for the guardrail service"
    )
    api_base: Optional[str] = Field(
        default=None, description="Base URL for the guardrail service API"
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


class Guardrail(TypedDict, total=False):
    guardrail_id: Optional[str]
    guardrail_name: str
    litellm_params: LitellmParams
    guardrail_info: Optional[Dict]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class guardrailConfig(TypedDict):
    guardrails: List[Guardrail]


class GuardrailEventHooks(str, Enum):
    pre_call = "pre_call"
    post_call = "post_call"
    during_call = "during_call"
    logging_only = "logging_only"


class DynamicGuardrailParams(TypedDict):
    extra_body: Dict[str, Any]


class GuardrailInfoLiteLLMParamsResponse(BaseModel):
    """The returned LiteLLM Params object for /guardrails/list"""

    guardrail: str
    mode: Union[str, List[str]]
    default_on: Optional[bool] = False
    pii_entities_config: Optional[Dict[PiiEntityType, PiiAction]] = None

    def __init__(self, **kwargs):
        default_on = kwargs.get("default_on")
        if default_on is None:
            default_on = False

        super().__init__(**kwargs)


class GuardrailInfoResponse(BaseModel):
    guardrail_id: Optional[str] = None
    guardrail_name: str
    litellm_params: Optional[GuardrailInfoLiteLLMParamsResponse] = None
    guardrail_info: Optional[Dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    guardrail_definition_location: Literal["config", "db"] = "config"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ListGuardrailsResponse(BaseModel):
    guardrails: List[GuardrailInfoResponse]


class GuardrailUIAddGuardrailSettings(BaseModel):
    supported_entities: List[PiiEntityType]
    supported_actions: List[PiiAction]
    supported_modes: List[GuardrailEventHooks]
    pii_entity_categories: List[PiiEntityCategoryMap]


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


class PatchGuardrailLitellmParams(BaseModel):
    default_on: Optional[bool] = None
    pii_entities_config: Optional[Dict[PiiEntityType, PiiAction]] = None


class PatchGuardrailRequest(BaseModel):
    guardrail_name: Optional[str] = None
    litellm_params: Optional[PatchGuardrailLitellmParams] = None
    guardrail_info: Optional[Dict[str, Any]] = None
