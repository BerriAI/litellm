from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, model_validator

from litellm.types.utils import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm_proxy_extras.litellm_proxy._types import (
        AllowedModelRegion,
        LitellmUserRoles,
        Member,
        Span,
    )
else:
    Member = Any
    LitellmUserRoles = Any
    AllowedModelRegion = Any
    Span = Any


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


class LiteLLM_VerificationToken(LiteLLMPydanticObjectBase):
    token: Optional[str] = None
    key_name: Optional[str] = None
    key_alias: Optional[str] = None
    spend: float = 0.0
    max_budget: Optional[float] = None
    expires: Optional[Union[str, datetime]] = None
    models: List = []
    aliases: Dict = {}
    config: Dict = {}
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Dict = {}
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    allowed_cache_controls: Optional[list] = []
    allowed_routes: Optional[list] = []
    permissions: Dict = {}
    model_spend: Dict = {}
    model_max_budget: Dict = {}
    soft_budget_cooldown: bool = False
    blocked: Optional[bool] = None
    litellm_budget_table: Optional[dict] = None
    org_id: Optional[str] = None  # org id for a given key
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class LiteLLM_VerificationTokenView(LiteLLM_VerificationToken):
    """
    Combined view of litellm verification token + litellm team table (select values)
    """

    team_spend: Optional[float] = None
    team_alias: Optional[str] = None
    team_tpm_limit: Optional[int] = None
    team_rpm_limit: Optional[int] = None
    team_max_budget: Optional[float] = None
    team_models: List = []
    team_blocked: bool = False
    soft_budget: Optional[float] = None
    team_model_aliases: Optional[Dict] = None
    team_member_spend: Optional[float] = None
    team_member: Optional[Member] = None
    team_metadata: Optional[Dict] = None

    # End User Params
    end_user_id: Optional[str] = None
    end_user_tpm_limit: Optional[int] = None
    end_user_rpm_limit: Optional[int] = None
    end_user_max_budget: Optional[float] = None

    # Time stamps
    last_refreshed_at: Optional[float] = None  # last time joint view was pulled from db

    def __init__(self, **kwargs):
        # Handle litellm_budget_table_* keys
        for key, value in list(kwargs.items()):
            if key.startswith("litellm_budget_table_") and value is not None:
                # Extract the corresponding attribute name
                attr_name = key.replace("litellm_budget_table_", "")
                # Check if the value is None and set the corresponding attribute
                if getattr(self, attr_name, None) is None:
                    kwargs[attr_name] = value
            if key == "end_user_id" and value is not None and isinstance(value, int):
                kwargs[key] = str(value)
        # Initialize the superclass
        super().__init__(**kwargs)


class UserAPIKeyAuth(
    LiteLLM_VerificationTokenView
):  # the expected response object for user api key auth
    """
    Return the row in the db
    """

    api_key: Optional[str] = None
    user_role: Optional[LitellmUserRoles] = None
    allowed_model_region: Optional[AllowedModelRegion] = None
    parent_otel_span: Optional[Span] = None
    rpm_limit_per_model: Optional[Dict[str, int]] = None
    tpm_limit_per_model: Optional[Dict[str, int]] = None
    user_tpm_limit: Optional[int] = None
    user_rpm_limit: Optional[int] = None
    user_email: Optional[str] = None
    request_route: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def check_api_key(cls, values):
        if values.get("api_key") is not None:
            values.update({"token": hash_token(values.get("api_key"))})
            if isinstance(values.get("api_key"), str) and values.get(
                "api_key"
            ).startswith("sk-"):
                values.update({"api_key": hash_token(values.get("api_key"))})
        return values
