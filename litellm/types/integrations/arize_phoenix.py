from litellm.types.llms.base import LiteLLMBaseModel

from .arize import Protocol


class ArizePhoenixConfig(LiteLLMBaseModel):
    otlp_auth_headers: str | None = None
    protocol: Protocol
    endpoint: str
    project_name: str | None = None
