"""Shared pydantic request/response models for the e2e gateway.

Only the fields the tests read are modelled; pydantic ignores the rest, so a
response validates without mirroring every proxy field. No untyped dicts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel

# ---------- keys ----------


class ModelBudgetEntry(BaseModel):
    budget_limit: float
    time_period: str


class BudgetWindow(BaseModel):
    budget_duration: str
    max_budget: float


class KeyGenerateBody(BaseModel):
    models: list[str] = []
    duration: str | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    budget_id: str | None = None
    key_alias: str | None = None
    model_max_budget: dict[str, ModelBudgetEntry] | None = None
    budget_fallbacks: dict[str, list[str]] | None = None
    budget_limits: list[BudgetWindow] | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_routes: list[str] | None = None


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
    spend: float | None = None
    max_budget: float | None = None
    budget_reset_at: str | None = None
    budget_id: str | None = None
    litellm_budget_table: LiteLLMBudgetTable | None = None


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


class AnthropicMessagesBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int


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


class DeploymentParams(BaseModel):
    """The configured litellm_params a /model/info row reports for a deployment,
    mirroring litellm's LiteLLM_Params (litellm/types/router.py) field for field
    so tests can assert any configured knob. Same pricing field names as
    CustomPricing, so pricing tests read them unchanged. Secrets (api_key et al)
    come back encrypted, so tests assert presence, never the value. Deliberately
    omitted because they have no typed JSON shape to pin: mock_response,
    model_info (surfaced as ModelInfoEntry.model_info), the *_router_config
    blobs, and configurable_clientside_auth_params."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    model: str | None = None
    custom_llm_provider: str | None = None

    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    organization: str | None = None
    litellm_credential_name: str | None = None

    tpm: int | None = None
    rpm: int | None = None
    itpm: int | None = None
    otpm: int | None = None
    max_parallel_requests: int | None = None
    order: int | None = None
    weight: int | None = None

    timeout: float | str | None = None
    stream_timeout: float | str | None = None
    max_retries: int | None = None

    max_budget: float | None = None
    budget_duration: str | None = None
    default_api_key_tpm_limit: int | None = None
    default_api_key_rpm_limit: int | None = None

    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    input_cost_per_second: float | None = None
    output_cost_per_second: float | None = None
    cache_read_input_token_cost: float | None = None
    cache_creation_input_token_cost: float | None = None

    tags: list[str] | None = None
    tag_regex: list[str] | None = None

    use_in_pass_through: bool | None = None
    use_litellm_proxy: bool | None = None
    use_chat_completions_api: bool | None = None
    use_xai_oauth: bool | None = None
    merge_reasoning_content_in_choices: bool | None = None

    region_name: str | None = None
    aws_region_name: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    watsonx_region_name: str | None = None

    max_file_size_mb: float | None = None
    litellm_trace_id: str | None = None


class ModelInfoEntry(BaseModel):
    """One /model/info row. `litellm_params` is the configured deployment (carries
    any custom-pricing override); `model_info` is the price the proxy resolved for
    it - the override merged over the cost-map defaults."""

    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    litellm_params: DeploymentParams = DeploymentParams()
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
    api_base: str | None = None
    api_version: str | None = None
    aws_region_name: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None
    vertex_credentials: str | None = None
    bucket_name: str | None = None
    s3_bucket_name: str | None = None
    s3_region_name: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    aws_batch_role_arn: str | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    tpm: int | None = None


ModelMode = Literal["batch", "realtime", "image_generation"]


class ModelInfoBody(BaseModel):
    id: str
    mode: ModelMode | None = None


class ModelNewBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    litellm_params: LiteLLMParamsBody
    model_info: ModelInfoBody


class ModelNewResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str


class ModelDeleteBody(BaseModel):
    id: str


class ModelUpdateParams(BaseModel):
    """POST /model/update litellm_params: only the fields being changed; the proxy
    merges them over the deployment's stored params."""

    tpm: int | None = None


class ModelUpdateBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    litellm_params: ModelUpdateParams
    model_info: ModelInfoBody
