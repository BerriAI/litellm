from typing import Literal

from litellm.types.llms.base import LiteLLMBaseModel

ModelInsightsMetric = Literal["requests", "spend", "tokens"]


class ModelInsightMetric(LiteLLMBaseModel):
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


class ModelInsightDailyTotal(LiteLLMBaseModel):
    date: str
    spend: float
    prompt_tokens: int
    completion_tokens: int
    requests: int


class ModelInsightTask(LiteLLMBaseModel):
    task_type: str
    label: str
    category: str


class ModelInsightTaskSummary(ModelInsightTask):
    value: float
    share: float
    leader: str
    provider: str


class ModelInsightsResponse(LiteLLMBaseModel):
    start_date: str
    end_date: str
    daily: list[ModelInsightDailyMetric]
    daily_totals: tuple[ModelInsightDailyTotal, ...]
    top_models: list[ModelInsightMetric]


class ModelInsightTasksResponse(LiteLLMBaseModel):
    start_date: str
    end_date: str
    tasks: list[ModelInsightTaskSummary]
