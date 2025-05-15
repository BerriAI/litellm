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
    PRESIDIO = "presidio"
    HIDE_SECRETS = "hide-secrets"
    AIM = "aim"


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


class LitellmParams(TypedDict, total=False):
    guardrail: str
    mode: str
    api_key: Optional[str]
    api_base: Optional[str]

    # Lakera specific params
    category_thresholds: Optional[LakeraCategoryThresholds]

    # Bedrock specific params
    guardrailIdentifier: Optional[str]
    guardrailVersion: Optional[str]

    # Presidio params
    output_parse_pii: Optional[bool]
    presidio_ad_hoc_recognizers: Optional[str]
    mock_redacted_text: Optional[dict]
    # PII control params
    pii_entities_config: Optional[Dict[PiiEntityType, PiiAction]]

    # hide secrets params
    detect_secrets_config: Optional[dict]

    # guardrails ai params
    guard_name: Optional[str]
    default_on: Optional[bool]

    ################## PII control params #################
    ########################################################
    mask_request_content: Optional[
        bool
    ]  # will mask request content if guardrail makes any changes
    mask_response_content: Optional[
        bool
    ]  # will mask response content if guardrail makes any changes


class Guardrail(TypedDict, total=False):
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


class GuardrailLiteLLMParamsResponse(BaseModel):
    """The returned LiteLLM Params object for /guardrails/list"""

    guardrail: str
    mode: Union[str, List[str]]
    default_on: bool = Field(default=False)

    def __init__(self, **kwargs):
        default_on = kwargs.get("default_on")
        if default_on is None:
            default_on = False

        super().__init__(**kwargs)


class GuardrailInfoResponse(BaseModel):
    guardrail_name: str
    litellm_params: GuardrailLiteLLMParamsResponse
    guardrail_info: Optional[Dict]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ListGuardrailsResponse(BaseModel):
    guardrails: List[GuardrailInfoResponse]


class GuardrailUIAddGuardrailSettings(BaseModel):
    supported_entities: List[PiiEntityType]
    supported_actions: List[PiiAction]
    supported_modes: List[GuardrailEventHooks]
    pii_entity_categories: List[PiiEntityCategoryMap]
