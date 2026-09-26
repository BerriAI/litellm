from typing import Final

from litellm.llms.base_llm.systemone import HttpJevClassifierClient
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.secret_managers.main import get_secret_str


def create_typesafe_client(
    api_base: str | None, api_key: str | None, http_client: AsyncHTTPHandler
) -> HttpJevClassifierClient:
    resolved_key: Final = api_key or get_secret_str("TYPESAFE_API_KEY")
    if not resolved_key:
        raise ValueError("jev_classifier_config.api_key or TYPESAFE_API_KEY is required for classifier_type 'jev'")
    resolved_base: Final = api_base or get_secret_str("TYPESAFE_API_BASE") or "https://api.typesafe.ai"
    return HttpJevClassifierClient(api_key=resolved_key, api_base=resolved_base, http_client=http_client)
