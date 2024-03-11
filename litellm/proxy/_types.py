from pydantic import BaseModel, Extra, Field, root_validator, Json
import enum
from typing import Optional, List, Union, Dict, Literal, Any
from datetime import datetime
import uuid, json, sys, os


def hash_token(token: str):
    import hashlib

    # Hash the string using SHA-256
    hashed_token = hashlib.sha256(token.encode()).hexdigest()

    return hashed_token


class LiteLLMBase(BaseModel):
    """
    Implements default functions, all pydantic objects should have.
    """

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)  # noqa
        except Exception as e:
            # if using pydantic v1
            return self.dict(**kwargs)

    def fields_set(self):
        try:
            return self.model_fields_set  # noqa
        except:
            # if using pydantic v1
            return self.__fields_set__

    class Config:
        protected_namespaces = ()


######### Request Class Definition ######
class ProxyChatCompletionRequest(LiteLLMBase):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[str]] = None
    tool_choice: Optional[str] = None
    functions: Optional[List[str]] = None  # soon to be deprecated
    function_call: Optional[str] = None  # soon to be deprecated

    # Optional LiteLLM params
    caching: Optional[bool] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    num_retries: Optional[int] = None
    context_window_fallback_dict: Optional[Dict[str, str]] = None
    fallbacks: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = {}
    deployment_id: Optional[str] = None
    request_timeout: Optional[int] = None

    class Config:
        extra = "allow"  # allow params not defined here, these fall in litellm.completion(**kwargs)


class ModelInfoDelete(LiteLLMBase):
    id: Optional[str]


class ModelInfo(LiteLLMBase):
    id: Optional[str]
    mode: Optional[Literal["embedding", "chat", "completion"]]
    input_cost_per_token: Optional[float] = 0.0
    output_cost_per_token: Optional[float] = 0.0
    max_tokens: Optional[int] = 2048  # assume 2048 if not set

    # for azure models we need users to specify the base model, one azure you can call deployments - azure/my-random-model
    # we look up the base model in model_prices_and_context_window.json
    base_model: Optional[
        Literal[
            "gpt-4-1106-preview",
            "gpt-4-32k",
            "gpt-4",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo",
            "text-embedding-ada-002",
        ]
    ]

    class Config:
        extra = Extra.allow  # Allow extra fields
        protected_namespaces = ()

    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("id") is None:
            values.update({"id": str(uuid.uuid4())})
        if values.get("mode") is None:
            values.update({"mode": None})
        if values.get("input_cost_per_token") is None:
            values.update({"input_cost_per_token": None})
        if values.get("output_cost_per_token") is None:
            values.update({"output_cost_per_token": None})
        if values.get("max_tokens") is None:
            values.update({"max_tokens": None})
        if values.get("base_model") is None:
            values.update({"base_model": None})
        return values


class BlockUsers(LiteLLMBase):
    user_ids: List[str]  # required


class ModelParams(LiteLLMBase):
    model_name: str
    litellm_params: dict
    model_info: ModelInfo

    class Config:
        protected_namespaces = ()

    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("model_info") is None:
            values.update({"model_info": ModelInfo()})
        return values


class GenerateRequestBase(LiteLLMBase):
    """
    Overlapping schema between key and user generate/update requests
    """

    models: Optional[list] = []
    spend: Optional[float] = 0
    max_budget: Optional[float] = None
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    max_parallel_requests: Optional[int] = None
    metadata: Optional[dict] = {}
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    budget_duration: Optional[str] = None
    allowed_cache_controls: Optional[list] = []
    soft_budget: Optional[float] = None


class GenerateKeyRequest(GenerateRequestBase):
    key_alias: Optional[str] = None
    duration: Optional[str] = None
    aliases: Optional[dict] = {}
    config: Optional[dict] = {}
    permissions: Optional[dict] = {}
    model_max_budget: Optional[dict] = (
        {}
    )  # {"gpt-4": 5.0, "gpt-3.5-turbo": 5.0}, defaults to {}

    class Config:
        protected_namespaces = ()


class GenerateKeyResponse(GenerateKeyRequest):
    key: str
    key_name: Optional[str] = None
    expires: Optional[datetime]
    user_id: str

    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("token") is not None:
            values.update({"key": values.get("token")})
        dict_fields = [
            "metadata",
            "aliases",
            "config",
            "permissions",
            "model_max_budget",
        ]
        for field in dict_fields:
            value = values.get(field)
            if value is not None and isinstance(value, str):
                try:
                    values[field] = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"Field {field} should be a valid dictionary")

        return values


class UpdateKeyRequest(GenerateKeyRequest):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    key: str
    duration: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None


class KeyRequest(LiteLLMBase):
    keys: List[str]


class LiteLLM_ModelTable(LiteLLMBase):
    model_aliases: Optional[str] = None  # json dump the dict
    created_by: str
    updated_by: str


class NewUserRequest(GenerateKeyRequest):
    max_budget: Optional[float] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None


class NewUserResponse(GenerateKeyResponse):
    max_budget: Optional[float] = None


class UpdateUserRequest(GenerateRequestBase):
    # Note: the defaults of all Params here MUST BE NONE
    # else they will get overwritten
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    spend: Optional[float] = None
    metadata: Optional[dict] = None
    user_role: Optional[str] = None
    max_budget: Optional[float] = None

    @root_validator(pre=True)
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class Member(LiteLLMBase):
    role: Literal["admin", "user"]
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    @root_validator(pre=True)
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class TeamBase(LiteLLMBase):
    team_alias: Optional[str] = None
    team_id: Optional[str] = None
    organization_id: Optional[str] = None
    admins: list = []
    members: list = []
    members_with_roles: List[Member] = []
    metadata: Optional[dict] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    max_budget: Optional[float] = None
    models: list = []


class NewTeamRequest(TeamBase):
    model_aliases: Optional[dict] = None


class GlobalEndUsersSpend(LiteLLMBase):
    api_key: Optional[str] = None


class TeamMemberAddRequest(LiteLLMBase):
    team_id: str
    member: Member


class TeamMemberDeleteRequest(LiteLLMBase):
    team_id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None

    @root_validator(pre=True)
    def check_user_info(cls, values):
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class UpdateTeamRequest(LiteLLMBase):
    team_id: str  # required
    team_alias: Optional[str] = None
    admins: Optional[list] = None
    members: Optional[list] = None
    members_with_roles: Optional[List[Member]] = None
    metadata: Optional[dict] = None


class DeleteTeamRequest(LiteLLMBase):
    team_ids: List[str]  # required


class LiteLLM_TeamTable(TeamBase):
    spend: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    model_id: Optional[int] = None

    @root_validator(pre=True)
    def set_model_info(cls, values):
        dict_fields = [
            "metadata",
            "aliases",
            "config",
            "permissions",
            "model_max_budget",
            "model_aliases",
        ]
        for field in dict_fields:
            value = values.get(field)
            if value is not None and isinstance(value, str):
                try:
                    values[field] = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"Field {field} should be a valid dictionary")

        return values


class TeamRequest(LiteLLMBase):
    teams: List[str]


class LiteLLM_BudgetTable(LiteLLMBase):
    """Represents user-controllable params for a LiteLLM_BudgetTable record"""

    soft_budget: Optional[float] = None
    max_budget: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    model_max_budget: Optional[dict] = None
    budget_duration: Optional[str] = None


class NewOrganizationRequest(LiteLLM_BudgetTable):
    organization_alias: str
    models: List = []
    budget_id: Optional[str] = None


class LiteLLM_OrganizationTable(LiteLLMBase):
    """Represents user-controllable params for a LiteLLM_OrganizationTable record"""

    organization_alias: Optional[str] = None
    budget_id: str
    metadata: Optional[dict] = None
    models: List[str]
    created_by: str
    updated_by: str


class NewOrganizationResponse(LiteLLM_OrganizationTable):
    organization_id: str
    created_at: datetime
    updated_at: datetime


class OrganizationRequest(LiteLLMBase):
    organizations: List[str]


class BudgetRequest(LiteLLMBase):
    budgets: List[str]


class KeyManagementSystem(enum.Enum):
    GOOGLE_KMS = "google_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    LOCAL = "local"


class TeamDefaultSettings(LiteLLMBase):
    team_id: str

    class Config:
        extra = "allow"  # allow params not defined here, these fall in litellm.completion(**kwargs)


class DynamoDBArgs(LiteLLMBase):
    billing_mode: Literal["PROVISIONED_THROUGHPUT", "PAY_PER_REQUEST"]
    read_capacity_units: Optional[int] = None
    write_capacity_units: Optional[int] = None
    ssl_verify: Optional[bool] = None
    region_name: str
    user_table_name: str = "LiteLLM_UserTable"
    key_table_name: str = "LiteLLM_VerificationToken"
    config_table_name: str = "LiteLLM_Config"
    spend_table_name: str = "LiteLLM_SpendLogs"
    aws_role_name: Optional[str] = None
    aws_session_name: Optional[str] = None
    aws_web_identity_token: Optional[str] = None
    aws_provider_id: Optional[str] = None
    aws_policy_arns: Optional[List[str]] = None
    aws_policy: Optional[str] = None
    aws_duration_seconds: Optional[int] = None
    assume_role_aws_role_name: Optional[str] = None
    assume_role_aws_session_name: Optional[str] = None


class ConfigGeneralSettings(LiteLLMBase):
    """
    Documents all the fields supported by `general_settings` in config.yaml
    """

    completion_model: Optional[str] = Field(
        None, description="proxy level default model for all chat completion calls"
    )
    key_management_system: Optional[KeyManagementSystem] = Field(
        None, description="key manager to load keys from / decrypt keys with"
    )
    use_google_kms: Optional[bool] = Field(
        None, description="decrypt keys with google kms"
    )
    use_azure_key_vault: Optional[bool] = Field(
        None, description="load keys from azure key vault"
    )
    master_key: Optional[str] = Field(
        None, description="require a key for all calls to proxy"
    )
    database_url: Optional[str] = Field(
        None,
        description="connect to a postgres db - needed for generating temporary keys + tracking spend / key",
    )
    database_connection_pool_limit: Optional[int] = Field(
        100,
        description="default connection pool for prisma client connecting to postgres db",
    )
    database_connection_timeout: Optional[float] = Field(
        60, description="default timeout for a connection to the database"
    )
    database_type: Optional[Literal["dynamo_db"]] = Field(
        None, description="to use dynamodb instead of postgres db"
    )
    database_args: Optional[DynamoDBArgs] = Field(
        None,
        description="custom args for instantiating dynamodb client - e.g. billing provision",
    )
    otel: Optional[bool] = Field(
        None,
        description="[BETA] OpenTelemetry support - this might change, use with caution.",
    )
    custom_auth: Optional[str] = Field(
        None,
        description="override user_api_key_auth with your own auth script - https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth",
    )
    max_parallel_requests: Optional[int] = Field(
        None, description="maximum parallel requests for each api key"
    )
    infer_model_from_keys: Optional[bool] = Field(
        None,
        description="for `/models` endpoint, infers available model based on environment keys (e.g. OPENAI_API_KEY)",
    )
    background_health_checks: Optional[bool] = Field(
        None, description="run health checks in background"
    )
    health_check_interval: int = Field(
        300, description="background health check interval in seconds"
    )
    alerting: Optional[List] = Field(
        None,
        description="List of alerting integrations. Today, just slack - `alerting: ['slack']`",
    )
    alerting_threshold: Optional[int] = Field(
        None,
        description="sends alerts if requests hang for 5min+",
    )
    ui_access_mode: Optional[Literal["admin_only", "all"]] = Field(
        "all", description="Control access to the Proxy UI"
    )


class ConfigYAML(LiteLLMBase):
    """
    Documents all the fields supported by the config.yaml
    """

    environment_variables: Optional[dict] = Field(
        None,
        description="Object to pass in additional environment variables via POST request",
    )
    model_list: Optional[List[ModelParams]] = Field(
        None,
        description="List of supported models on the server, with model-specific configs",
    )
    litellm_settings: Optional[dict] = Field(
        None,
        description="litellm Module settings. See __init__.py for all, example litellm.drop_params=True, litellm.set_verbose=True, litellm.api_base, litellm.cache",
    )
    general_settings: Optional[ConfigGeneralSettings] = None

    class Config:
        protected_namespaces = ()


class LiteLLM_VerificationToken(LiteLLMBase):
    token: Optional[str] = None
    key_name: Optional[str] = None
    key_alias: Optional[str] = None
    spend: float = 0.0
    max_budget: Optional[float] = None
    expires: Optional[str] = None
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
    permissions: Dict = {}
    model_spend: Dict = {}
    model_max_budget: Dict = {}

    # hidden params used for parallel request limiting, not required to create a token
    user_id_rate_limits: Optional[dict] = None
    team_id_rate_limits: Optional[dict] = None

    class Config:
        protected_namespaces = ()


class LiteLLM_VerificationTokenView(LiteLLM_VerificationToken):
    """
    Combined view of litellm verification token + litellm team table (select values)
    """

    team_spend: Optional[float] = None
    team_tpm_limit: Optional[int] = None
    team_rpm_limit: Optional[int] = None
    team_max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    team_model_aliases: Optional[Dict] = None


class UserAPIKeyAuth(
    LiteLLM_VerificationTokenView
):  # the expected response object for user api key auth
    """
    Return the row in the db
    """

    api_key: Optional[str] = None
    user_role: Optional[Literal["proxy_admin", "app_owner", "app_user"]] = None

    @root_validator(pre=True)
    def check_api_key(cls, values):
        if values.get("api_key") is not None:
            values.update({"token": hash_token(values.get("api_key"))})
        return values


class LiteLLM_Config(LiteLLMBase):
    param_name: str
    param_value: Dict


class LiteLLM_UserTable(LiteLLMBase):
    user_id: str
    max_budget: Optional[float]
    spend: float = 0.0
    model_max_budget: Optional[Dict] = {}
    model_spend: Optional[Dict] = {}
    user_email: Optional[str]
    models: list = []

    @root_validator(pre=True)
    def set_model_info(cls, values):
        if values.get("spend") is None:
            values.update({"spend": 0.0})
        if values.get("models") is None:
            values.update({"models": []})
        return values

    class Config:
        protected_namespaces = ()


class LiteLLM_SpendLogs(LiteLLMBase):
    request_id: str
    api_key: str
    model: Optional[str] = ""
    api_base: Optional[str] = ""
    call_type: str
    spend: Optional[float] = 0.0
    total_tokens: Optional[int] = 0
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    startTime: Union[str, datetime, None]
    endTime: Union[str, datetime, None]
    user: Optional[str] = ""
    metadata: Optional[dict] = {}
    cache_hit: Optional[str] = "False"
    cache_key: Optional[str] = None
    request_tags: Optional[Json] = None


class LiteLLM_SpendLogs_ResponseObject(LiteLLMBase):
    response: Optional[List[Union[LiteLLM_SpendLogs, Any]]] = None
