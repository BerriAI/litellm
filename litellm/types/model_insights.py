from typing import Literal

from pydantic import BaseModel

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


class ModelInsightTask(BaseModel):
    task_type: str
    label: str
    category: str


class ModelInsightTaskMetric(ModelInsightMetric):
    task_type: str


class ModelInsightsResponse(BaseModel):
    start_date: str
    end_date: str
    daily: list[ModelInsightDailyMetric]
    top_models: list[ModelInsightMetric]
    by_task: list[ModelInsightTaskMetric]
    tasks: list[ModelInsightTask]
