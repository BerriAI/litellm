"""Shared pydantic request/response models for the e2e gateway.

Only the fields the tests read are modelled; pydantic ignores the rest, so a
response validates without mirroring every proxy field. No untyped dicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel, model_validator

# ---------- keys ----------


class ModelBudgetEntry(BaseModel):
    budget_limit: float
    time_period: str


class BudgetWindow(BaseModel):
    budget_duration: str
    max_budget: float


class BudgetWindowState(BudgetWindow):
    reset_at: datetime | None = None


class KeyLoggingCallbackVars(BaseModel):
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


class KeyLoggingCallback(BaseModel):
    callback_name: str
    callback_type: str = "success_and_failure"
    callback_vars: KeyLoggingCallbackVars


class KeyMetadata(BaseModel):
    logging: list[KeyLoggingCallback] | None = None


class ObjectPermission(BaseModel):
    mcp_servers: list[str] | None = None


class KeyGenerateBody(BaseModel):
    models: list[str] = []
    duration: str | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    organization_id: str | None = None
    budget_id: str | None = None
    key_alias: str | None = None
    model_max_budget: dict[str, ModelBudgetEntry] | None = None
    budget_fallbacks: dict[str, list[str]] | None = None
    budget_limits: list[BudgetWindow] | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_routes: list[str] | None = None
    metadata: KeyMetadata | None = None
    object_permission: ObjectPermission | None = None


class KeyGenerateResponse(BaseModel):
    key: str


class KeyDeleteBody(BaseModel):
    keys: list[str]


class KeyInfoParams(BaseModel):
    key: str


class LiteLLMBudgetTable(BaseModel):
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_reset_at: str | None = None


class KeyInfo(BaseModel):
    key_alias: str | None = None
    models: list[str] = []
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    team_id: str | None = None
    spend: float | None = None
    max_budget: float | None = None
    budget_reset_at: str | None = None
    budget_id: str | None = None
    litellm_budget_table: LiteLLMBudgetTable | None = None
    budget_limits: list[BudgetWindowState] | None = None


class KeyInfoResponse(BaseModel):
    info: KeyInfo


# ---------- customers ----------


class CustomerDeleteBody(BaseModel):
    user_ids: list[str]


# ---------- chat / embeddings ----------


class ChatMetadata(BaseModel):
    tags: list[str] | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ThinkingParam(BaseModel):
    """Extended-thinking control shared by Anthropic and DeepSeek reasoner models.
    DeepSeek accepts only ``type`` (enabled/disabled) and ignores budget_tokens;
    Anthropic also honors budget_tokens. Sending ``type="disabled"`` is the
    product-facing way a caller turns reasoning off (LIT-3686 / GH #27453)."""

    type: Literal["enabled", "disabled"]
    budget_tokens: int | None = None


class ChatToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None


class ChatTool(BaseModel):
    type: str = "function"
    function: ChatToolFunction


class ChatBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    user: str | None = None
    metadata: ChatMetadata | None = None
    reasoning_effort: str | None = None
    thinking: ThinkingParam | None = None
    service_tier: str | None = None
    tools: list[ChatTool] | None = None
    tool_choice: str | None = None
    guardrails: list[str] | None = None


class OutMessage(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None


class ChatChoice(BaseModel):
    message: OutMessage | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


class ChatResponse(BaseModel):
    id: str | None = None
    model: str | None = None
    choices: list[ChatChoice] = []
    usage: Usage | None = None
    service_tier: str | None = None


# ---------- anthropic /v1/messages + count_tokens ----------


class JsonSchemaProperty(BaseModel):
    """One property in a tool's JSON-Schema `input_schema`. Only `type` is
    modelled; the endpoints under test read no further into the schema."""

    type: str


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, JsonSchemaProperty] = {}
    required: list[str] = []


class AnthropicToolSearchTool(BaseModel):
    """The tool_search discovery tool. `type` carries the SDK-version-pinned
    suffix (e.g. ``tool_search_tool_regex_20251119``) that LiteLLM keys its
    per-provider beta-header translation on; `name` is the unsuffixed
    canonical name the upstream accepts."""

    type: str
    name: str


class AnthropicCustomTool(BaseModel):
    name: str
    description: str
    input_schema: ToolInputSchema


type AnthropicTool = AnthropicToolSearchTool | AnthropicCustomTool


class AnthropicMessagesBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int
    stream: bool | None = None
    tools: list[AnthropicTool] | None = None


class CountTokensBody(BaseModel):
    """POST /v1/messages/count_tokens body: the /v1/messages shape minus
    max_tokens (the endpoint only counts the prompt)."""

    model: str
    messages: list[ChatMessage]


class AnthropicContentBlock(BaseModel):
    type: str | None = None
    text: str | None = None


class AnthropicMessagesResponse(BaseModel):
    """A /v1/messages answer. `content` is the Anthropic-native passthrough
    shape; `choices` is the OpenAI-normalized shape LiteLLM emits for some
    providers (e.g. Bedrock Converse). Presence of either proves the proxy
    accepted and round-tripped the request. `extra="allow"` keeps the other
    top-level keys so a shape-check failure can report the actual response keys
    for triage."""

    model_config = ConfigDict(extra="allow")
    model: str | None = None
    content: list[AnthropicContentBlock] | None = None
    choices: list[ChatChoice] | None = None


class CountTokensResponse(BaseModel):
    """`/v1/messages/count_tokens` answer. `input_tokens` is required so a 200
    whose body lacks it fails validation instead of passing vacuously."""

    input_tokens: int


class EmbedBody(BaseModel):
    model: str
    input: str


class EmbedResponse(BaseModel):
    model: str | None = None


# ---------- ocr ----------


class OcrDocument(BaseModel):
    """A document for /v1/ocr in Mistral OCR format: a document_url for PDFs/docs
    or an image_url for images. exclude_none on serialize drops the unset one."""

    type: str
    document_url: str | None = None
    image_url: str | None = None


class OcrBody(BaseModel):
    model: str
    document: OcrDocument


class OcrPage(BaseModel):
    index: int
    markdown: str


class OcrResponse(BaseModel):
    object: str | None = None
    model: str | None = None
    pages: list[OcrPage] = []


# ---------- spend logs ----------


class SpendLogRow(BaseModel):
    request_id: str | None = None
    api_key: str | None = None
    model: str | None = None
    spend: float | None = None
    status: str | None = None
    cache_hit: str | None = None
    call_type: str | None = None
    custom_llm_provider: str | None = None
    team_id: str | None = None
    user: str | None = None
    end_user: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_tags: list[str] | None = None


class SpendLogs(RootModel[list[SpendLogRow]]):
    pass


class SpendLogsParams(BaseModel):
    request_id: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def require_filter(self) -> SpendLogsParams:
        if self.request_id is None and self.api_key is None:
            raise ValueError(
                "unfiltered /spend/logs returns the entire spend table and OOMs the "
                "runner on long-lived environments; filter by request_id or api_key, "
                "or use ProxyClient.spend_logs_window for a bounded /spend/logs/v2 read"
            )
        return self


class SpendLogsPageParams(BaseModel):
    """Query for /spend/logs/v2, which requires an explicit date window and
    serves pages of at most 100 rows."""

    start_date: str
    end_date: str
    page: int
    page_size: int
    api_key: str | None = None


class SpendLogsPage(BaseModel):
    data: list[SpendLogRow] = []
    total: int
    page: int
    page_size: int
    total_pages: int


# ---------- spend calculate ----------


class SpendCalculateBody(BaseModel):
    model: str
    messages: list[ChatMessage]


class SpendCalculateResponse(BaseModel):
    cost: float


# ---------- spend tags ----------


class TagSpend(BaseModel):
    individual_request_tag: str | None = None
    log_count: int | None = None
    total_spend: float | None = None


class SpendTagsResponse(RootModel[list[TagSpend]]):
    """GET /spend/tags answers with a bare array of per-tag aggregates, not an
    object wrapping them (that's /global/spend/tags). Read the rows off .root."""


# ---------- route probing ----------


class DateRangeParams(BaseModel):
    start_date: str
    end_date: str


class RouteSpec(RootModel[dict[str, object]]):
    """One /openapi.json path entry: a map of HTTP method -> operation. Only the
    method names are read, so the operation specs stay opaque."""

    @property
    def methods(self) -> frozenset[str]:
        return frozenset(method.lower() for method in self.root)


class OpenAPISchema(BaseModel):
    paths: dict[str, RouteSpec] = {}


# ---------- model info / custom pricing ----------


class CustomPricing(BaseModel):
    """The per-token custom-pricing fields a deployment can override in
    litellm_params - the token-cost subset of litellm's CustomPricingLiteLLMParams
    the proxy applies to chat spend. All optional: a config sets only what it
    overrides, and /model/info echoes the rates the proxy resolved."""

    model_config = ConfigDict(extra="ignore")
    mode: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None

    def overrides(self) -> dict[str, float]:
        """The rates actually declared (non-null) - e.g. those a config.yml sets."""
        declared = {
            "input_cost_per_token": self.input_cost_per_token,
            "output_cost_per_token": self.output_cost_per_token,
            "cache_read_input_token_cost": self.cache_read_input_token_cost,
            "cache_creation_input_token_cost": self.cache_creation_input_token_cost,
        }
        return {field: rate for field, rate in declared.items() if rate is not None}

    def token_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Spend for a fresh (uncached) call under these rates: the proxy's
        custom-pricing formula (prompt * input + completion * output)."""
        assert self.input_cost_per_token is not None and self.output_cost_per_token is not None, (
            "custom pricing has no per-token rates"
        )
        return prompt_tokens * self.input_cost_per_token + completion_tokens * self.output_cost_per_token


class ModelInfoEntry(BaseModel):
    """One /model/info row. `litellm_params` is the configured deployment (carries
    any custom-pricing override); `model_info` is the price the proxy resolved for
    it - the override merged over the cost-map defaults."""

    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    litellm_params: CustomPricing = CustomPricing()
    model_info: CustomPricing = CustomPricing()


class ModelInfoResponse(BaseModel):
    data: list[ModelInfoEntry] = []


class FileEntry(BaseModel):
    id: str


class FileListResponse(BaseModel):
    """GET /files answer. `data` is required on purpose: a 200 whose body lacks
    the OpenAI-format file list must fail validation, not pass vacuously."""

    data: list[FileEntry]


class FineTuningJobsParams(BaseModel):
    custom_llm_provider: Literal["openai", "azure"]


class FineTuningJobEntry(BaseModel):
    id: str


class FineTuningJobsResponse(BaseModel):
    """GET /fine_tuning/jobs answer; `data` required for the same reason as
    FileListResponse."""

    data: list[FineTuningJobEntry]


# ---------- model management ----------


class LiteLLMParamsBody(BaseModel):
    """POST /model/new litellm_params: `model` is the only required field; `api_key`
    et al may be an `os.environ/FOO` reference the proxy resolves at call time.
    `input_cost_per_token`/`output_cost_per_token` register a per-deployment custom
    pricing override; left None (and dropped from the body) the deployment keeps the
    backend's canonical rate."""

    model: str
    api_key: str | None = None
    litellm_credential_name: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    realtime_protocol: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    vertex_credentials: str | None = None
    gcs_bucket_name: str | None = None
    bucket_name: str | None = None
    s3_bucket_name: str | None = None
    s3_region_name: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    aws_batch_role_arn: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    extra_headers: dict[str, str] | None = None
    use_in_pass_through: bool | None = None
    complexity_router_config: dict[str, object] | None = None
    mock_response: str | None = None


ModelMode = Literal["batch", "realtime", "image_generation"]


class ModelInfoBody(BaseModel):
    # id is left unset so the proxy assigns a unique model_id per deployment.
    # Pinning it to the model_name made re-registrations of a fixed-name model
    # (e.g. the batch suite's openai-batch) collide on the model_id unique
    # constraint when a prior run's teardown had not removed the row.
    id: str | None = None
    mode: ModelMode | None = None


class ModelNewBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    litellm_params: LiteLLMParamsBody
    model_info: ModelInfoBody


class ModelNewResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str


class ModelListEntry(BaseModel):
    id: str


class ModelsListResponse(BaseModel):
    """GET /v1/models on the data plane: the deployments the gateway can actually
    serve right now. Used to confirm a freshly created model has propagated from
    the control plane before a test calls it."""

    data: tuple[ModelListEntry, ...] = ()


class ModelDeleteBody(BaseModel):
    id: str


class CredentialCreateBody(BaseModel):
    credential_name: str
    credential_values: dict[str, str]
    credential_info: dict[str, str] = {}


class CredentialCreateResponse(BaseModel):
    success: bool


# ---------- key / team / user / organization management ----------


class KeyUpdateBody(BaseModel):
    key: str
    models: list[str]


class KeyListParams(BaseModel):
    key_alias: str


class KeyListResponse(BaseModel):
    total_count: int


class TeamMemberEntry(BaseModel):
    role: Literal["admin", "user"]
    user_id: str


class TeamNewBody(BaseModel):
    team_alias: str
    models: list[str] = []
    team_id: str | None = None
    organization_id: str | None = None


class TeamNewResponse(BaseModel):
    team_id: str


class TeamInfoParams(BaseModel):
    team_id: str


class TeamData(BaseModel):
    team_alias: str | None = None
    models: list[str] = []
    members_with_roles: list[TeamMemberEntry] = []


class TeamInfoResponse(BaseModel):
    team_id: str
    team_info: TeamData


class TeamMemberAddBody(BaseModel):
    team_id: str
    member: TeamMemberEntry


class TeamMemberDeleteBody(BaseModel):
    team_id: str
    user_id: str


class TeamDeleteBody(BaseModel):
    team_ids: list[str]


UserRole = Literal["proxy_admin", "proxy_admin_viewer", "internal_user", "internal_user_viewer"]


class UserNewBody(BaseModel):
    user_email: str
    user_role: UserRole
    user_id: str | None = None


class UserNewResponse(BaseModel):
    user_id: str


class UserInfoParams(BaseModel):
    user_id: str


class UserData(BaseModel):
    user_id: str | None = None
    user_email: str | None = None
    user_role: str | None = None


class UserInfoResponse(BaseModel):
    user_id: str
    user_info: UserData


class UserDeleteBody(BaseModel):
    user_ids: list[str]


class UserListParams(BaseModel):
    user_ids: str


class UserListResponse(BaseModel):
    total: int


class OrgNewBody(BaseModel):
    organization_alias: str
    models: list[str] = []


class OrgNewResponse(BaseModel):
    organization_id: str


class OrgInfoParams(BaseModel):
    organization_id: str


class OrgInfoResponse(BaseModel):
    organization_id: str
    organization_alias: str | None = None
    models: list[str] = []


class OrgDeleteBody(BaseModel):
    organization_ids: list[str]
