"""Shared pydantic request/response models for the e2e gateway.

Only the fields the tests read are modelled; pydantic ignores the rest, so a
response validates without mirroring every proxy field. No untyped dicts.
"""

from __future__ import annotations

from pydantic import BaseModel, RootModel

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


# ---------- spend tags ----------


class TagSpend(BaseModel):
    individual_request_tag: str
    log_count: int | None = None
    total_spend: float | None = None


class TagSpends(RootModel[list[TagSpend]]):
    pass


class SpendTagsResponse(BaseModel):
    spend_per_tag: list[TagSpend] | None = None


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
