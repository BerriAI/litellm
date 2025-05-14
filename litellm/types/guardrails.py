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


class LitellmParams(TypedDict):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ListGuardrailsResponse(BaseModel):
    guardrails: List[GuardrailInfoResponse]
