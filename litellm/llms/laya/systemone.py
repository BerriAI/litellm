from litellm.llms.base_llm.systemone import HttpJevClassifierClient
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


def create_laya_client(
    api_base: str | None, api_key: str | None, http_client: AsyncHTTPHandler
) -> HttpJevClassifierClient:
    if not api_base or not api_base.strip():
        raise ValueError("jev_classifier_config.laya_api_base is required for Laya")
    return HttpJevClassifierClient(
        api_base=api_base,
        api_key=api_key,
        http_client=http_client,
        custom_llm_provider="laya",
        use_requested_model=True,
    )
