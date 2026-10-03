from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelInsightsMetric = Literal["requests", "spend", "tokens"]


class ModelInsightMetric(BaseModel):
    model_group: str
    model: str
    provider: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    requests: int
    successful_requests: int
    failed_requests: int


class ModelInsightDailyMetric(ModelInsightMetric):
    date: str


class ModelInsightDailyTotal(BaseModel):
    date: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    requests: int


class ModelInsightTask(BaseModel):
    task_type: str
    label: str
    category: str


class ModelInsightTaskSummary(ModelInsightTask):
    value: float
    share: float
    leader: str
    provider: str


class ModelInsightsResponse(BaseModel):
    start_date: str
    end_date: str
    daily: list[ModelInsightDailyMetric]
    daily_totals: tuple[ModelInsightDailyTotal, ...]
    top_models: list[ModelInsightMetric]


class ModelInsightTasksResponse(BaseModel):
    start_date: str
    end_date: str
    tasks: list[ModelInsightTaskSummary]


class TaskClassifierModel(BaseModel):
    id: str
    name: str
    provider: Literal["typesafe", "laya", "bespoke"]
    model: str


class TaskClassifierResponse(BaseModel):
    enabled: bool
    model_id: str | None
    models: list[TaskClassifierModel]
    batch_size: int
    message_logging_enabled: bool


class TaskClassifierUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    model_id: str | None = Field(default=None, min_length=1, max_length=256)
