from pydantic import field_validator

from litellm.types.llms.base import LiteLLMBaseModel


class WorkerRegistryEntry(LiteLLMBaseModel):
    worker_id: str
    name: str
    url: str

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("Worker URL must start with http:// or https://")
        return v
