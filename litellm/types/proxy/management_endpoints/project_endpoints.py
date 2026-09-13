from pydantic import BaseModel


class ProjectDailySpendRow(BaseModel):
    date: str
    project_id: str
    project_alias: str | None = None
    spend: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0


class ProjectDailySpendResponse(BaseModel):
    start_date: str
    end_date: str
    results: tuple[ProjectDailySpendRow, ...]
