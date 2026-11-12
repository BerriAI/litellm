from pydantic import BaseModel, ConfigDict


class CustomerOrganizationUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    alias: str | None = None
    spend: float
    requests: int
    tokens: int
