from pydantic import BaseModel


class HourlySavingsBucket(BaseModel):
    bucket_start: str
    compression_savings_spend: float
    prompt_caching_savings_spend: float


class HourlySavingsResponse(BaseModel):
    buckets: list[HourlySavingsBucket]
    start_date: str
    end_date: str
    timezone: str
    spend_logs_disabled: bool
