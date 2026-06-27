"""Shared pydantic request/response models for the e2e gateway.

Only the fields the tests read are modelled; pydantic ignores the rest, so a
response validates without mirroring every proxy field. No untyped dicts.
"""

from __future__ import annotations

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
    model_max_budget: dict[str, ModelBudgetEntry] | None = None
    budget_limits: list[BudgetWindow] | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None


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


class ChatBody(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    user: str | None = None
    metadata: ChatMetadata | None = None


class OutMessage(BaseModel):
    content: str | None = None


class ChatChoice(BaseModel):
    message: OutMessage | None = None


class Usage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    id: str | None = None
    model: str | None = None
    choices: list[ChatChoice] = []
    usage: Usage | None = None


class EmbedBody(BaseModel):
    model: str
    input: str


class EmbedResponse(BaseModel):
    model: str | None = None


# ---------- spend logs ----------


class SpendLogRow(BaseModel):
    request_id: str | None = None
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


# ---------- spend calculate ----------


class SpendCalculateBody(BaseModel):
    model: str
    messages: list[ChatMessage]


class SpendCalculateResponse(BaseModel):
    cost: float


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
        assert (
            self.input_cost_per_token is not None
            and self.output_cost_per_token is not None
        ), "custom pricing has no per-token rates"
        return (
            prompt_tokens * self.input_cost_per_token
            + completion_tokens * self.output_cost_per_token
        )


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
